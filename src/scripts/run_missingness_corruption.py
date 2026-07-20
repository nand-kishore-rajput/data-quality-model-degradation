"""
run_missingness_corruption.py

Smoke-test / validation runner for D3's missingness.py module (MCAR/MAR/MNAR)
against the real processed corpus.

NOT the final Phase 2 experiment runner. D4's severity grid isn't frozen yet,
so PLACEHOLDER_SEVERITIES below is illustrative only, for validating the
corruption engine mechanically — swap in D4's real grid once it exists.

What it does, per dataset:
  1. Load datasets/processed/<source>/<name>.parquet
  2. Split into train/test folds (D2 Decision 1: split before corrupt).
     Split ratio and stratification (SPLIT_RATIO, SPLIT_SEED below) are new
     decisions this script makes explicitly — not yet specified anywhere in
     D1/D2/D4 — flagged here rather than silently assumed.
  3. For each mechanism (mcar, mar, mnar) x placeholder severity x seed:
     corrupt the test fold via src.corruption_engine.missingness.corrupt,
     using the train fold as train_fold_reference (D2 Decision 4).
  4. Log every result (achieved rate, validation status, errors) to a
     manifest CSV, immediately after each call — matching the live-logging
     pattern already used in preprocess_to_processed.py.

Usage:
    python -m src.scripts.run_missingness_corruption --datasets phoneme churn
    python -m src.scripts.run_missingness_corruption --all
    python -m src.scripts.run_missingness_corruption --all --severities 0.1 0.2 --seeds 1 2 3
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.missingness import corrupt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MANIFEST = OUT_DIR / "missingness_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "missingness_smoketest_run.log"

# ----------------------------------------------------------------------
# Target-column registry. Mirrors preprocess_to_processed.py's DATASETS
# list (dataset name -> target column). Kept here as a minimal, explicit
# mapping rather than re-importing the full preprocessing registry, since
# this script only needs the target column, not the transform steps.
# TODO: this duplicates information that also lives in preprocess_to_processed.py
# and conceptually belongs in registry.py — worth consolidating there so
# there's a single source of truth instead of two registries that could drift.
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Placeholder config — explicitly NOT canonical, see module docstring.
# ----------------------------------------------------------------------
SPLIT_RATIO = 0.7          # 70% train / 30% test, new decision, flag for confirmation
SPLIT_SEED = 42            # matches the project's standing seed=42 convention
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]  # NOT D4's real grid
PLACEHOLDER_SEEDS = [1, 2, 3]                            # NOT D1's real 5-seed budget
MECHANISMS = ["mcar", "mar", "mnar"]
MAX_CORRUPTED_COLUMNS = 5  # cap columns touched per call, keeps the smoke test fast;
                           # remove this cap for the real experiment run

_log_file = None


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)
    if _log_file:
        _log_file.write(msg + end)
        _log_file.flush()


def ts():
    return datetime.now().strftime("%H:%M:%S")


def discover_datasets():
    """Find every processed parquet file, matched against TARGET_COLUMNS."""
    found = []
    for source_dir in sorted(PROCESSED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for f in sorted(source_dir.glob("*.parquet")):
            key = (source_dir.name, f.stem)
            found.append((source_dir.name, f.stem, f))
    return found


def split_train_test(df: pd.DataFrame, target_col: str, ratio: float, seed: int):
    """
    D2 Decision 1 (split before corrupt), stratified by target to preserve
    class balance in both folds. Fixed once per dataset (D2's split is not
    redrawn per corruption seed — D11 Decision 8's same reasoning applies here).
    """
    rng = np.random.default_rng(seed)
    train_idx_parts, test_idx_parts = [], []
    for cls, group in df.groupby(target_col):
        idx = group.index.to_numpy().copy()  # .copy() required: to_numpy() can
                                              # return a read-only view, which
                                              # rng.shuffle() cannot operate on in-place
        rng.shuffle(idx)
        cut = int(len(idx) * ratio)
        train_idx_parts.append(idx[:cut])
        test_idx_parts.append(idx[cut:])
    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    train_fold = df.loc[train_idx].reset_index(drop=True)
    test_fold = df.loc[test_idx].reset_index(drop=True)
    return train_fold, test_fold


MIN_COLUMN_STD = 1e-8  # matches missingness.py's MIN_REFERENCE_STD — columns
    # below this are excluded from the candidate pool entirely, rather than
    # selected and then reliably failing MNAR every time this script runs
    # (surfaced by EEG: real datasets can have near-constant/dead-sensor
    # columns that are legitimately unsuitable for a z-score-based mechanism).


def eligible_columns(train_fold: pd.DataFrame, target_col: str, max_cols: int):
    """Numeric, non-target, non-near-constant columns. See script-level note
    on the separate categorical-column limitation this inherits from
    missingness.py."""
    cols = []
    skipped_constant = []
    for c in train_fold.columns:
        if c == target_col or not pd.api.types.is_numeric_dtype(train_fold[c]):
            continue
        if train_fold[c].std() < MIN_COLUMN_STD:
            skipped_constant.append(c)
            continue
        cols.append(c)
    if skipped_constant:
        log(f"  Skipping near-constant column(s) (train-fold std < {MIN_COLUMN_STD}): {skipped_constant}")
    return cols[:max_cols]


def run_one_dataset(source: str, name: str, path: Path, results: list):
    key = (source, name)
    target_col = TARGET_COLUMNS.get(key)

    log()
    log(f"{'─'*70}")
    log(f"[{ts()}] {source}/{name}")
    log(f"{'─'*70}")

    if target_col is None:
        log(f"  !! No target-column mapping for {key} — skipping. "
            f"Add it to TARGET_COLUMNS.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": "SKIPPED_NO_TARGET_MAPPING"})
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log(f"  !! READ ERROR: {e}")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": f"READ_ERROR: {e}"})
        return

    if target_col not in df.columns:
        log(f"  !! target column '{target_col}' not found in {path.name} "
            f"(columns: {list(df.columns)[:5]}...) — skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": "TARGET_COL_NOT_FOUND"})
        return

    train_fold, test_fold = split_train_test(df, target_col, SPLIT_RATIO, SPLIT_SEED)
    columns = eligible_columns(train_fold, target_col, MAX_CORRUPTED_COLUMNS)

    if not columns:
        log(f"  !! No eligible numeric columns to corrupt — skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": "NO_ELIGIBLE_COLUMNS"})
        return

    log(f"  train={len(train_fold)} test={len(test_fold)} "
        f"columns={columns}")

    for mechanism in MECHANISMS:
        for severity in PLACEHOLDER_SEVERITIES:
            for seed in PLACEHOLDER_SEEDS:
                row = {
                    "source": source, "dataset": name, "mechanism": mechanism,
                    "severity": severity, "seed": seed,
                }
                try:
                    _, meta = corrupt(
                        test_fold, mechanism, severity, seed,
                        train_fold_reference=train_fold,
                        columns=columns,
                        dataset_name=name,
                    )
                    # Per-column granularity (2026-07-19): a call can partially
                    # succeed — some columns corrupted, others excluded for
                    # failing their own mechanism-fidelity check. That's the
                    # normal outcome now, not an error; PARTIAL_SUCCESS is
                    # informational, distinct from the rarer case below where
                    # every requested column failed.
                    row["status"] = "PARTIAL_SUCCESS" if meta.excluded_columns else "OK"
                    row["validation_passed"] = meta.validation_passed
                    row["achieved_rate"] = str(meta.achieved_rate)
                    row["excluded_columns"] = str(meta.excluded_columns) if meta.excluded_columns else None
                    if meta.excluded_columns:
                        log(f"    -- {mechanism} sev={severity} seed={seed}: excluded {list(meta.excluded_columns.keys())}")
                except RuntimeError as e:
                    # Only reached when EVERY requested column failed its
                    # mechanism-fidelity check — a genuinely rare, worth-
                    # investigating outcome now that partial success is handled
                    # above, not the common case it used to be.
                    row["status"] = "ALL_COLUMNS_FAILED"
                    row["validation_passed"] = False
                    row["achieved_rate"] = None
                    row["error"] = str(e)[:500]
                    log(f"    !! {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                except Exception as e:
                    row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                    row["validation_passed"] = False
                    row["achieved_rate"] = None
                    row["error"] = str(e)[:500]

                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism and r["status"] in ("OK", "PARTIAL_SUCCESS"))
        n_total = len(PLACEHOLDER_SEVERITIES) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's missingness module against real processed data.")
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Dataset names to run (e.g. phoneme churn). Default: first 3 discovered.")
    parser.add_argument("--all", action="store_true", help="Run all discovered datasets.")
    parser.add_argument("--severities", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    global PLACEHOLDER_SEVERITIES, PLACEHOLDER_SEEDS, _log_file
    if args.severities:
        PLACEHOLDER_SEVERITIES = args.severities
    if args.seeds:
        PLACEHOLDER_SEEDS = args.seeds

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUT_LOG, "w", buffering=1)

    discovered = discover_datasets()
    if not discovered:
        log(f"No parquet files found under {PROCESSED_DIR}. Check the path.")
        sys.exit(1)

    if args.all:
        selected = discovered
    elif args.datasets:
        selected = [d for d in discovered if d[1] in args.datasets]
        missing = set(args.datasets) - {d[1] for d in selected}
        if missing:
            log(f"Warning: requested datasets not found: {missing}")
    else:
        selected = discovered[:3]
        log(f"No --datasets or --all given; defaulting to first 3: {[d[1] for d in selected]}")

    log(f"MISSINGNESS SMOKE TEST — {len(selected)} dataset(s), "
        f"{len(MECHANISMS)} mechanisms, {len(PLACEHOLDER_SEVERITIES)} severities, "
        f"{len(PLACEHOLDER_SEEDS)} seeds")
    log(f"Severities (PLACEHOLDER, not D4's real grid): {PLACEHOLDER_SEVERITIES}")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for source, name, path in selected:
        run_one_dataset(source, name, path, results)
        pd.DataFrame(results).to_csv(OUT_MANIFEST, index=False)

    log()
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    df_results = pd.DataFrame(results)
    if not df_results.empty and "status" in df_results.columns:
        log(df_results["status"].value_counts().to_string())
        n_ok = (df_results["status"] == "OK").sum()
        n_partial = (df_results["status"] == "PARTIAL_SUCCESS").sum()
        n_all_failed = (df_results["status"] == "ALL_COLUMNS_FAILED").sum()
        n_unexpected = df_results["status"].astype(str).str.startswith("UNEXPECTED_ERROR").sum()
        log()
        log(f"OK: {n_ok} | Partial success (some column(s) excluded, expected occasionally): {n_partial} "
            f"| All columns failed (investigate these): {n_all_failed} "
            f"| Unexpected errors (investigate these): {n_unexpected}")
    log(f"Manifest -> {OUT_MANIFEST}")
    log(f"Log      -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()