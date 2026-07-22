"""
generate_severity_table.py

Generates the actual 17-dataset x 14-sub-mechanism x 10-severity-level table
from D4's finalized calibration formulas, replacing the placeholder grid
D3's six runners have been using for smoke-testing (`[0.05, 0.10, 0.15, 0.20, 0.25]`).

D4's mechanisms split into two categories:

  CORPUS-WIDE FIXED alpha_max (13 of 14 sub-mechanisms): severity(l) =
    (l/10) * alpha_max, identical across all 17 datasets. No data pull
    needed -- these are the fixed constants D4 already decided (missingness
    0.0865, feature noise 0.30, quantization's implicit 1.0, label noise
    0.10, distribution shift covariate 0.25 / concept 0.5, validity 0.1088,
    uniqueness 0.06).

    Label noise was ORIGINALLY per-dataset (Cleanlab-estimated), but real-
    corpus testing of that approach produced implausible rates (13%-56%,
    inconsistent with Phase 0's own learnability screening) -- diagnosed as
    probe underfitting from the logistic-regression probe, not genuine label
    noise. Reverted to a fixed 0.10 default per David's review (2026-07-20).
    See D4 Sec 2.3 and Future Work item 8.

  PER-DATASET (1 of 14 sub-mechanisms): distribution_shift/label needs each
    dataset's natural class prior (directly computable, no Cleanlab) to set
    rho_target per D4 Sec 2.4's rule. Severity itself (l/10) is still
    corpus-wide; only rho_target varies per dataset.

Usage:
    python -m src.scripts.generate_severity_table --all
    python -m src.scripts.generate_severity_table --datasets heloc jm1
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_TABLE = OUT_DIR / "severity_table.csv"
OUT_LOG = OUT_DIR / "generate_severity_table.log"

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

SEVERITY_LEVELS = list(range(1, 11))  # 1-10, per D4 Sec 1

# D4's finalized corpus-wide alpha_max values -- Decision Log entries 2, 3,
# 6, 8, 9, 10. Fixed constants, no data pull required.
ALPHA_MAX = {
    "missingness": 0.0865,           # D4 Sec 2.1, road-safety anchor
    "feature_noise_rate": 0.30,      # D4 Sec 2.2, gaussian + categorical_swap
    "feature_noise_quantization": 0.99,  # D4 Sec 2.2 -- fixed 2026-07-21:
                                      # originally 1.0, which made severity
                                      # level 10 compute to EXACTLY 1.0 --
                                      # violating D1 Sec 3's "severity never
                                      # approaches 1" rule that every
                                      # corrupt() implementation enforces
                                      # (severity must be < 1). Surfaced when
                                      # wiring this table into the runners:
                                      # every quantization call at level 10
                                      # raised ValueError. 0.99 keeps level
                                      # 10 just under the boundary while
                                      # leaving every other level unaffected.
    "label_noise": 0.10,             # D4 Sec 2.3 -- REVISED (David,
                                      # 2026-07-20) from a per-dataset
                                      # Cleanlab estimate to a fixed corpus-
                                      # wide default. Real-corpus testing of
                                      # the original approach produced
                                      # implausible per-dataset rates
                                      # (13%-56%, inconsistent with Phase 0's
                                      # learnability screening) -- see D4
                                      # Sec 2.3 for the full diagnosis. This
                                      # is a stopgap, not a measured anchor;
                                      # D4 Future Work item 8 covers fixing
                                      # the estimation methodology properly.
    "distribution_shift_covariate": 0.25,   # D4 Sec 2.4
    "distribution_shift_concept": 0.5,      # D4 Sec 2.4
    "validity_violations": 0.1088,   # D4 Sec 2.5, HELOC anchor
    "uniqueness": 0.06,              # D4 Sec 2.6, HELOC anchor (Q2)
}

LABEL_SHIFT_MINORITY_FLOOR_FRACTION = 0.05  # D4 Sec 2.4, per David's review

# The 14 sub-mechanisms, tagged with which alpha_max category they use.
SUB_MECHANISMS = [
    ("missingness", "mcar", "corpus_wide", "missingness"),
    ("missingness", "mar", "corpus_wide", "missingness"),
    ("missingness", "mnar", "corpus_wide", "missingness"),
    ("feature_noise", "gaussian", "corpus_wide", "feature_noise_rate"),
    ("feature_noise", "categorical_swap", "corpus_wide", "feature_noise_rate"),
    ("feature_noise", "quantization", "corpus_wide", "feature_noise_quantization"),
    ("label_noise", "uniform", "corpus_wide", "label_noise"),
    ("label_noise", "class_conditional", "corpus_wide", "label_noise"),
    ("distribution_shift", "covariate", "corpus_wide", "distribution_shift_covariate"),
    ("distribution_shift", "label", "per_dataset_prior", None),
    ("distribution_shift", "concept", "corpus_wide", "distribution_shift_concept"),
    ("validity_violations", "validity_violations", "corpus_wide", "validity_violations"),
    ("uniqueness", "global", "corpus_wide", "uniqueness"),
    ("uniqueness", "slice_conditional", "corpus_wide", "uniqueness"),
]

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


def compute_natural_prior_and_target(df: pd.DataFrame, target_col: str):
    """
    Natural class prior rho = P(Y=1) (using whichever class is coded 1, per
    D3's binary-corpus convention), and rho_target per D4 Sec 2.4: push
    toward 0.9 if rho < 0.5, else 0.1, clamped so neither class falls below
    LABEL_SHIFT_MINORITY_FLOOR_FRACTION of the fold.
    """
    y = df[target_col].dropna()
    classes = sorted(y.unique())
    if len(classes) != 2:
        return None, None
    c1 = classes[1]
    rho = float((y == c1).mean())

    rho_target = 0.9 if rho < 0.5 else 0.1
    floor = LABEL_SHIFT_MINORITY_FLOOR_FRACTION
    rho_target = min(max(rho_target, floor), 1 - floor)
    return rho, rho_target


def build_rows_for_dataset(source, name, df, target_col, natural_rho, rho_target):
    rows = []
    for mech_class, sub_mech, category, alpha_key in SUB_MECHANISMS:
        for level in SEVERITY_LEVELS:
            frac = level / 10.0
            row = {
                "source": source, "dataset": name,
                "mechanism_class": mech_class, "sub_mechanism": sub_mech,
                "severity_level": level,
            }
            if category == "corpus_wide":
                row["severity_value"] = round(frac * ALPHA_MAX[alpha_key], 6)
                row["alpha_max_used"] = ALPHA_MAX[alpha_key]
                row["alpha_max_source"] = "corpus_wide_fixed_D4"
                row["note"] = None
            elif category == "per_dataset_prior":
                # Fixed 2026-07-21: cap at 0.99, same boundary issue as
                # quantization above -- frac=1.0 at level 10 would violate
                # D1 Sec 3's severity<1 requirement, which distribution_shift.py
                # enforces directly.
                row["severity_value"] = min(frac, 0.99)
                row["alpha_max_used"] = None
                row["alpha_max_source"] = "not_applicable_uses_rho_target_instead"
                row["note"] = (f"natural_rho={natural_rho:.4f}, rho_target={rho_target:.4f}"
                               if natural_rho is not None else "target column not binary, skipped")
            rows.append(row)
    return rows


def run_one_dataset(source, name, path, all_rows):
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

    if target_col not in df.columns:
        log(f"  !! target column '{target_col}' not found — skipping.")
        return

    feature_cols = [c for c in df.columns if c != target_col]

    # Distribution shift / label: natural prior + rho_target.
    natural_rho, rho_target = compute_natural_prior_and_target(df, target_col)
    if natural_rho is not None:
        log(f"  distribution_shift/label: natural_rho={natural_rho:.4f}, rho_target={rho_target:.4f}")
    else:
        log(f"  !! distribution_shift/label: target not binary — rows will note this.")

    rows = build_rows_for_dataset(source, name, df, target_col, natural_rho, rho_target)
    all_rows.extend(rows)
    log(f"  {len(rows)} rows generated (14 sub-mechanisms x 10 severity levels)")


def main():
    parser = argparse.ArgumentParser(description="Generate the real severity table from D4's finalized formulas.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUT_LOG, "w", buffering=1)

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

    log(f"SEVERITY TABLE GENERATION — {len(selected)} dataset(s), "
        f"{len(SUB_MECHANISMS)} sub-mechanisms, {len(SEVERITY_LEVELS)} severity levels")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows = []
    for source, name, path in selected:
        run_one_dataset(source, name, path, all_rows)

    table = pd.DataFrame(all_rows)
    table.to_csv(OUT_TABLE, index=False)

    log()
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total rows generated: {len(table)}")
    if not table.empty:
        log()
        log("Sample (severity_level=5, one row per sub-mechanism, first dataset):")
        first_dataset = table["dataset"].iloc[0]
        sample = table[(table["dataset"] == first_dataset) & (table["severity_level"] == 5)]
        log(sample[["mechanism_class", "sub_mechanism", "severity_value", "note"]].to_string(index=False))
    log()
    log(f"Full table -> {OUT_TABLE}")
    log(f"Log        -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()