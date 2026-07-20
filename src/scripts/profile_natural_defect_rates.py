"""
profile_natural_defect_rates.py

Data-profiling script for D4 (Severity Grid). Pulls the two categories of
real, measured data D4's missingness and validity anchors (D1 Sec 10) need
but aren't yet tabulated anywhere in this project:

  1. Per-dataset, per-column NATURAL missingness rates
     -- for D4 Sec 2.1's missingness alpha_max anchor.
  2. Per-dataset, per-column NATURAL constraint-violation rates -- how often
     a column's ALREADY-EXISTING (un-corrupted) values already violate their
     own confirmed domain constraint from validity_constraints_annotated.csv
     -- for D4 Sec 2.5's validity alpha_max anchor.

This is a profiling script, not part of the corruption engine. It reads the
FULL processed dataset (not a train/test split) for the most stable natural-
rate estimate -- D2 Decision 4 governs corruption PARAMETERS estimated per
experiment from the train fold only; this script runs once, ahead of time,
purely to inform D4's grid design, so the train/test distinction doesn't
apply here the same way.

Reuses validity_violations.py's own load_constraints/parse_constraint
functions rather than re-implementing constraint parsing, so this profiling
is guaranteed consistent with what the corruption engine actually checks
against -- not a second, potentially-drifting interpretation of the
annotation file.

Usage:
    python -m src.scripts.profile_natural_defect_rates --all
    python -m src.scripts.profile_natural_defect_rates --datasets heloc jm1
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.validity_violations import load_constraints, parse_constraint

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
CONSTRAINTS_CSV = PROJECT_ROOT / "datasets" / "metadata" / "validity_constraints_annotated.csv"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MISSINGNESS_CSV = OUT_DIR / "natural_missingness_profile.csv"
OUT_VALIDITY_CSV = OUT_DIR / "natural_validity_violation_profile.csv"
OUT_LOG = OUT_DIR / "profile_natural_defect_rates.log"

# Same target-column mapping used by every D3 runner -- excluded from
# profiling since the label column isn't a feature these anchors apply to.
# Duplicated here rather than imported from a single source; same
# consolidation opportunity flagged repeatedly for the runner scripts
# (natural home: registry.py), not yet acted on.
TARGET_COLUMNS = {
    ("openml", "phoneme"):              "Class",
    ("openml", "bank-marketing"):       "Class",
    ("openml", "MagicTelescope"):       "class:",
    ("openml", "churn"):                "class",
    ("openml", "heloc"):                "RiskPerformance",
    ("openml", "electricity"):          "class",
    ("openml", "jm1"):                  "defects",
    ("openml", "road-safety"):          "Sex_of_Driver",
    ("openml", "kick"):                 "IsBadBuy",
    ("uci", "default_credit"):          "Y",
    ("uci", "online_shoppers"):         "Revenue",
    ("uci", "diabetes_readmission"):    "readmitted",
    ("folktables", "ACSIncome"):        "target",
    ("folktables", "ACSPublicCoverage"):"target",
    ("cleanml", "Airbnb"):              "Rating",
    ("cleanml", "EEG"):                 "Eye",
    ("tableshift", "assistments"):      "correct",
}

_log_file = None


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)
    if _log_file:
        _log_file.write(msg + end)
        _log_file.flush()


def ts():
    return datetime.now().strftime("%H:%M:%S")


def discover_datasets():
    found = []
    for source_dir in sorted(PROCESSED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for f in sorted(source_dir.glob("*.parquet")):
            found.append((source_dir.name, f.stem, f))
    return found


def profile_missingness(df: pd.DataFrame, target_col: str, source: str, name: str) -> list:
    """Per-column natural missingness rate, excluding the target column."""
    rows = []
    for col in df.columns:
        if col == target_col:
            continue
        rate = float(df[col].isna().mean())
        rows.append({
            "source": source, "dataset": name, "column": col,
            "dtype": str(df[col].dtype), "natural_missingness_rate": rate,
        })
    return rows


def profile_validity(df: pd.DataFrame, constraints: dict, source: str, name: str) -> list:
    """
    Per-column natural constraint-violation rate: what fraction of a column's
    ALREADY-EXISTING (non-NaN) values already violate its own confirmed
    constraint. Only meaningful for no_negative and range constraints --
    'none' has no defined bound to check against here (that's an IQR-based
    injection-time rule, not a natural-violation measurement), and
    'categorical' has no numeric magnitude to violate.
    """
    rows = []
    for col, constraint_str in constraints.items():
        if col not in df.columns:
            continue
        try:
            ctype, bounds = parse_constraint(constraint_str)
        except ValueError:
            continue
        if ctype not in ("no_negative", "range"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        values = df[col].dropna()
        if len(values) == 0:
            continue

        if ctype == "no_negative":
            violation_mask = values < 0
        else:  # range
            a, b = bounds
            violation_mask = (values < a) | (values > b)

        rows.append({
            "source": source, "dataset": name, "column": col,
            "constraint_type": constraint_str,
            "n_non_null": int(len(values)),
            "n_natural_violations": int(violation_mask.sum()),
            "natural_violation_rate": float(violation_mask.mean()),
        })
    return rows


def run_one_dataset(source, name, path, missingness_rows, validity_rows, dataset_summary):
    key = (source, name)
    target_col = TARGET_COLUMNS.get(key)

    log()
    log(f"{'─'*70}")
    log(f"[{ts()}] {source}/{name}")
    log(f"{'─'*70}")

    if target_col is None:
        log(f"  !! No target-column mapping for {key} — skipping.")
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log(f"  !! READ ERROR: {e}")
        return

    log(f"  rows={len(df)}, columns={len(df.columns)}")

    # --- Missingness profile ---
    m_rows = profile_missingness(df, target_col, source, name)
    missingness_rows.extend(m_rows)
    if m_rows:
        rates = [r["natural_missingness_rate"] for r in m_rows]
        max_rate = max(rates)
        max_col = m_rows[int(np.argmax(rates))]["column"]
        mean_rate = float(np.mean(rates))
        log(f"  missingness: max={max_rate:.4f} (column='{max_col}'), mean={mean_rate:.4f}")
    else:
        max_rate = mean_rate = 0.0
        max_col = None

    # --- Validity profile ---
    v_rows = []
    try:
        constraints = load_constraints(str(CONSTRAINTS_CSV), name)
        v_rows = profile_validity(df, constraints, source, name)
        validity_rows.extend(v_rows)
        if v_rows:
            vrates = [r["natural_violation_rate"] for r in v_rows]
            v_max = max(vrates)
            v_max_col = v_rows[int(np.argmax(vrates))]["column"]
            v_mean = float(np.mean(vrates))
            log(f"  validity: {len(v_rows)} constrained numeric columns checked, "
                f"max_violation_rate={v_max:.4f} (column='{v_max_col}'), mean={v_mean:.4f}")
        else:
            v_max = v_mean = 0.0
            v_max_col = None
            log(f"  validity: no no_negative/range columns to check")
    except (FileNotFoundError, ValueError) as e:
        log(f"  !! No usable constraint annotations: {str(e)[:150]}")
        v_max = v_mean = 0.0
        v_max_col = None

    dataset_summary.append({
        "source": source, "dataset": name, "n_rows": len(df),
        "missingness_max_rate": max_rate, "missingness_max_column": max_col,
        "missingness_mean_rate": mean_rate,
        "validity_max_violation_rate": v_max, "validity_max_violation_column": v_max_col,
        "validity_mean_violation_rate": v_mean,
    })


def main():
    parser = argparse.ArgumentParser(description="Profile natural missingness and validity-violation rates for D4.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUT_LOG, "w", buffering=1)

    if not CONSTRAINTS_CSV.exists():
        log(f"Warning: constraints file not found at {CONSTRAINTS_CSV} — "
            f"validity profiling will be skipped, missingness profiling will still run.")

    discovered = discover_datasets()
    if not discovered:
        log(f"No parquet files found under {PROCESSED_DIR}.")
        sys.exit(1)

    if args.all:
        selected = discovered
    elif args.datasets:
        selected = [d for d in discovered if d[1] in args.datasets]
        missing = set(args.datasets) - {d[1] for d in selected}
        if missing:
            log(f"Warning: requested datasets not found: {missing}")
    else:
        selected = discovered
        log("No --datasets or --all given; defaulting to all discovered datasets.")

    log(f"NATURAL DEFECT RATE PROFILING — {len(selected)} dataset(s)")
    log(f"Constraints file: {CONSTRAINTS_CSV}")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    missingness_rows, validity_rows, dataset_summary = [], [], []
    for source, name, path in selected:
        run_one_dataset(source, name, path, missingness_rows, validity_rows, dataset_summary)

    pd.DataFrame(missingness_rows).to_csv(
        OUT_DIR / "natural_missingness_profile_percolumn.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(
        OUT_DIR / "natural_validity_violation_profile_percolumn.csv", index=False)
    summary_df = pd.DataFrame(dataset_summary)
    summary_df.to_csv(OUT_DIR / "natural_defect_rate_summary.csv", index=False)

    log()
    log("=" * 70)
    log("SUMMARY (per-dataset, for D4 Sec 2.1 / 2.5 anchor tables)")
    log("=" * 70)
    if not summary_df.empty:
        log(summary_df[["dataset", "missingness_max_rate", "missingness_mean_rate",
                         "validity_max_violation_rate", "validity_mean_violation_rate"]]
            .to_string(index=False))
    log()
    log(f"Per-column missingness detail -> {OUT_DIR / 'natural_missingness_profile_percolumn.csv'}")
    log(f"Per-column validity detail     -> {OUT_DIR / 'natural_validity_violation_profile_percolumn.csv'}")
    log(f"Per-dataset summary            -> {OUT_DIR / 'natural_defect_rate_summary.csv'}")
    log(f"Log                            -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()