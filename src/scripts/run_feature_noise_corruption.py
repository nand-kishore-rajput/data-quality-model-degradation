"""
run_feature_noise_corruption.py

Smoke-test / validation runner for D3's feature_noise.py module
(Additive Gaussian Noise, Categorical Swap, Quantization) against the real
processed corpus. Companion to run_missingness_corruption.py; same structure,
same caveats.

NOT the final Phase 2 experiment runner — D4's severity grid isn't frozen, so
PLACEHOLDER_SEVERITIES is illustrative. Swap in D4's real grid once it exists.

Key structural difference from the missingness runner: feature_noise
mechanisms are feature-TYPE-specific. gaussian and quantization need
continuous columns; categorical_swap needs categorical columns. This runner
splits each dataset's columns by inferred type and routes each mechanism to
the columns it can actually operate on, rather than throwing every numeric
column at every mechanism (which is what the missingness runner does, since
missingness is type-agnostic).

Usage:
    python -m src.scripts.run_feature_noise_corruption --datasets phoneme churn
    python -m src.scripts.run_feature_noise_corruption --all
    python -m src.scripts.run_feature_noise_corruption --all --severities 0.1 0.2 --seeds 1 2 3
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.feature_noise import corrupt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MANIFEST = OUT_DIR / "feature_noise_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "feature_noise_smoketest_run.log"

# ----------------------------------------------------------------------
# Target-column registry — same mapping as the missingness runner. Kept in
# sync deliberately; see that script's TODO about consolidating into
# registry.py as a single source of truth.
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
# Placeholder config — NOT canonical, see module docstring.
# ----------------------------------------------------------------------
SPLIT_RATIO = 0.7
SPLIT_SEED = 42
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
PLACEHOLDER_SEEDS = [1, 2, 3]
MECHANISMS = ["gaussian", "categorical_swap", "quantization"]
MAX_COLUMNS_PER_MECHANISM = 5  # cap columns per mechanism per call for speed;
                               # remove for the real experiment run.

# Column-type inference thresholds (runner-level heuristic, not a D-doc
# decision — this is just how the smoke test decides which columns to feed
# which mechanism; the real Phase 2 pipeline will use D6's dataset profiles).
CATEGORICAL_MAX_UNIQUE = 20  # a numeric column with <= this many distinct
    # values is treated as categorical-like for routing purposes; object/string
    # columns are always categorical.
MIN_STD_FLOOR = 1e-8  # skip near-constant columns before they reach the engine
                      # (the engine excludes them too, this just avoids noise).

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


def split_train_test(df, target_col, ratio, seed):
    """D2 Decision 1, stratified by target — identical to the missingness runner."""
    rng = np.random.default_rng(seed)
    train_idx_parts, test_idx_parts = [], []
    for cls, group in df.groupby(target_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(len(idx) * ratio)
        train_idx_parts.append(idx[:cut])
        test_idx_parts.append(idx[cut:])
    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    return (df.loc[train_idx].reset_index(drop=True),
            df.loc[test_idx].reset_index(drop=True))


def classify_columns(train_fold, target_col):
    """
    Split non-target columns into continuous vs categorical, for routing to
    the type-appropriate mechanism. Returns (continuous_cols, categorical_cols).

    Heuristic (runner-level, not a D-doc decision):
      - object/string/category dtype        -> categorical
      - numeric with <= CATEGORICAL_MAX_UNIQUE distinct -> categorical-like
      - numeric with more distinct values   -> continuous
      - near-constant (std < MIN_STD_FLOOR)  -> skipped entirely
    """
    continuous, categorical = [], []
    for c in train_fold.columns:
        if c == target_col:
            continue
        s = train_fold[c]
        if not pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) >= 2:
                categorical.append(c)
            continue
        # numeric
        if s.std() < MIN_STD_FLOOR:
            continue  # near-constant, skip
        if s.nunique(dropna=True) <= CATEGORICAL_MAX_UNIQUE:
            categorical.append(c)
        else:
            continuous.append(c)
    return continuous, categorical


def columns_for_mechanism(mechanism, continuous, categorical):
    if mechanism in ("gaussian", "quantization"):
        return continuous[:MAX_COLUMNS_PER_MECHANISM]
    elif mechanism == "categorical_swap":
        return categorical[:MAX_COLUMNS_PER_MECHANISM]
    return []


def run_one_dataset(source, name, path, results):
    key = (source, name)
    target_col = TARGET_COLUMNS.get(key)

    log()
    log(f"{'─'*70}")
    log(f"[{ts()}] {source}/{name}")
    log(f"{'─'*70}")

    if target_col is None:
        log(f"  !! No target-column mapping for {key} — skipping.")
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
        log(f"  !! target column '{target_col}' not found — skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": "TARGET_COL_NOT_FOUND"})
        return

    train_fold, test_fold = split_train_test(df, target_col, SPLIT_RATIO, SPLIT_SEED)
    continuous, categorical = classify_columns(train_fold, target_col)

    log(f"  train={len(train_fold)} test={len(test_fold)}")
    log(f"  continuous={continuous[:MAX_COLUMNS_PER_MECHANISM]}")
    log(f"  categorical={categorical[:MAX_COLUMNS_PER_MECHANISM]}")

    for mechanism in MECHANISMS:
        cols = columns_for_mechanism(mechanism, continuous, categorical)
        if not cols:
            log(f"  {mechanism}: SKIPPED (no eligible columns of the right type)")
            results.append({"source": source, "dataset": name, "mechanism": mechanism,
                             "severity": None, "seed": None,
                             "status": "SKIPPED_NO_ELIGIBLE_COLUMNS"})
            continue

        for severity in PLACEHOLDER_SEVERITIES:
            for seed in PLACEHOLDER_SEEDS:
                row = {"source": source, "dataset": name, "mechanism": mechanism,
                       "severity": severity, "seed": seed}
                try:
                    _, meta = corrupt(test_fold, mechanism, severity, seed,
                                      train_fold_reference=train_fold,
                                      columns=cols, dataset_name=name)
                    row["status"] = "PARTIAL_SUCCESS" if meta.excluded_columns else "OK"
                    row["validation_passed"] = meta.validation_passed
                    row["achieved_rate"] = str(meta.achieved_rate)
                    row["excluded_columns"] = str(meta.excluded_columns) if meta.excluded_columns else None
                    if meta.excluded_columns:
                        log(f"    -- {mechanism} sev={severity} seed={seed}: excluded {list(meta.excluded_columns.keys())}")
                except RuntimeError as e:
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
                    log(f"    !! UNEXPECTED {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism
                   and r["status"] in ("OK", "PARTIAL_SUCCESS"))
        n_total = len(PLACEHOLDER_SEVERITIES) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's feature_noise module against real processed data.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
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
        selected = discovered[:3]
        log(f"No --datasets or --all given; defaulting to first 3: {[d[1] for d in selected]}")

    log(f"FEATURE NOISE SMOKE TEST — {len(selected)} dataset(s), "
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
        log(f"OK: {n_ok} | Partial success: {n_partial} "
            f"| All columns failed (investigate): {n_all_failed} "
            f"| Unexpected errors (investigate): {n_unexpected}")
    log(f"Manifest -> {OUT_MANIFEST}")
    log(f"Log      -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()