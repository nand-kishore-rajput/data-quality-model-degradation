"""
run_feature_noise_corruption.py

Full-scale runner for D3's feature_noise.py module (Additive Gaussian Noise,
Categorical Swap, Quantization) against the real processed corpus.

CHANGES FROM SMOKETEST VERSION (D7 pre-flight review):
  1. MAX_COLUMNS_PER_MECHANISM cap REMOVED. The smoketest capped every call
     at 5 columns (continuous or categorical) regardless of dataset size —
     same issue as the missingness runner, same fix.
  2. conditioning_feature and validation_details_raw added to the manifest
     row (conditioning_feature is None for every feature_noise call — this
     module has no conditioning structure — but the field is included for
     schema consistency across all six manifests, per D7).
  3. Seed count supplied at the command line: --seeds 1 2 3 4 5.

Key structural note (unchanged from smoketest): feature_noise mechanisms are
feature-TYPE-specific. gaussian and quantization need continuous columns;
categorical_swap needs categorical columns. This runner splits each
dataset's columns by inferred type and routes each mechanism to the columns
it can actually operate on.

Usage (full-scale):
    python -m src.scripts.run_feature_noise_corruption --all --seeds 1 2 3 4 5
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.feature_noise import corrupt
from src.corruption_engine.base import derive_call_seed

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
SEVERITY_TABLE_PATH = OUT_DIR / "severity_table.csv"
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
# Config
# ----------------------------------------------------------------------
SPLIT_RATIO = 0.7
SPLIT_SEED = 42
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
PLACEHOLDER_SEEDS = [1, 2, 3]  # overridden via --seeds 1 2 3 4 5 for the real run
MECHANISMS = ["gaussian", "categorical_swap", "quantization"]
# MAX_COLUMNS_PER_MECHANISM removed (was: 5). Every eligible column of the
# right type is now corrupted per mechanism per call.

# Column-type inference thresholds (runner-level heuristic, not a D-doc
# decision — this is just how the runner decides which columns to feed
# which mechanism; the real Phase 2 pipeline will use D6's dataset profiles).
CATEGORICAL_MAX_UNIQUE = 20  # a numeric column with <= this many distinct
    # values is treated as categorical-like for routing purposes; object/string
    # columns are always categorical.
MIN_STD_FLOOR = 1e-8  # skip near-constant columns before they reach the engine
                      # (the engine excludes them too, this just avoids noise).

_severity_table_cache = None
_severities_overridden = False


def load_severity_table():
    """Loads D4's real severity table (generate_severity_table.py's output),
    cached after first read. Raises clearly if it doesn't exist rather than
    silently falling back -- D4's real grid should be used whenever
    available; the placeholder grid is a fallback for a missing file or a
    manual --severities override only."""
    global _severity_table_cache
    if _severity_table_cache is None:
        if not SEVERITY_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"D4's severity table not found at {SEVERITY_TABLE_PATH}. "
                f"Run `python -m src.scripts.generate_severity_table --all` first, "
                f"or pass --severities to use the placeholder grid explicitly."
            )
        _severity_table_cache = pd.read_csv(SEVERITY_TABLE_PATH)
    return _severity_table_cache


def get_severities_for(dataset_name: str, sub_mechanism: str) -> list:
    """Returns the 10 real severity values (level 1-10, in order) for this
    dataset/sub_mechanism from D4's table."""
    table = load_severity_table()
    rows = table[(table["dataset"] == dataset_name) & (table["sub_mechanism"] == sub_mechanism)]
    rows = rows.sort_values("severity_level")
    if rows.empty:
        raise ValueError(
            f"No severity table rows for dataset='{dataset_name}', "
            f"sub_mechanism='{sub_mechanism}'. Check severity_table.csv covers this combination."
        )
    return rows["severity_value"].tolist()

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


def split_train_test(df, target_col, ratio, seed, dataset_name):
    """D2 Decision 1, stratified by target. Uses derive_call_seed (fixed
    2026-07-21) rather than the raw seed directly -- same fix as
    missingness.py's corruption-engine seed-correlation bug, applied here
    since SPLIT_SEED=42 is reused identically across all 17 datasets."""
    call_seed = derive_call_seed(seed, dataset_name, "train_test_split", 0.0)
    rng = np.random.default_rng(call_seed)
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
    """No cap applied — every eligible column of the right type is corrupted
    (cap removed for the full-scale run; see module docstring)."""
    if mechanism in ("gaussian", "quantization"):
        return continuous
    elif mechanism == "categorical_swap":
        return categorical
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

    train_fold, test_fold = split_train_test(df, target_col, SPLIT_RATIO, SPLIT_SEED, name)
    continuous, categorical = classify_columns(train_fold, target_col)

    log(f"  train={len(train_fold)} test={len(test_fold)}")
    log(f"  continuous ({len(continuous)})={continuous}")
    log(f"  categorical ({len(categorical)})={categorical}")

    for mechanism in MECHANISMS:
        cols = columns_for_mechanism(mechanism, continuous, categorical)
        if not cols:
            log(f"  {mechanism}: SKIPPED (no eligible columns of the right type)")
            results.append({"source": source, "dataset": name, "mechanism": mechanism,
                             "severity": None, "seed": None,
                             "status": "SKIPPED_NO_ELIGIBLE_COLUMNS"})
            continue

        severities = PLACEHOLDER_SEVERITIES if _severities_overridden else get_severities_for(name, mechanism)
        for severity in severities:
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
                    # --- D7 pre-flight additions: previously computed but discarded ---
                    row["conditioning_feature"] = meta.conditioning_feature
                    row["validation_details_raw"] = str(meta.validation_details)
                    # -------------------------------------------------------------------
                    if meta.excluded_columns:
                        log(f"    -- {mechanism} sev={severity} seed={seed}: excluded {list(meta.excluded_columns.keys())}")
                except RuntimeError as e:
                    row["status"] = "ALL_COLUMNS_FAILED"
                    row["validation_passed"] = False
                    row["achieved_rate"] = None
                    row["conditioning_feature"] = None
                    row["validation_details_raw"] = None
                    row["error"] = str(e)[:500]
                    log(f"    !! {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                except Exception as e:
                    row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                    row["validation_passed"] = False
                    row["achieved_rate"] = None
                    row["conditioning_feature"] = None
                    row["validation_details_raw"] = None
                    row["error"] = str(e)[:500]
                    log(f"    !! UNEXPECTED {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism
                   and r["status"] in ("OK", "PARTIAL_SUCCESS"))
        n_total = len(severities) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Run D3's feature_noise module against real processed data (full scale).")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--severities", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    global PLACEHOLDER_SEVERITIES, PLACEHOLDER_SEEDS, _log_file, _severities_overridden
    _severities_overridden = bool(args.severities)
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

    log(f"FEATURE NOISE FULL RUN — {len(selected)} dataset(s), "
        f"{len(MECHANISMS)} mechanisms, 10 severity levels per D4's real grid, "
        f"{len(PLACEHOLDER_SEEDS)} seeds, NO column cap")
    if _severities_overridden:
        log(f"Severities: OVERRIDDEN via --severities, NOT D4's real grid: {PLACEHOLDER_SEVERITIES}")
    else:
        log(f"Severities: D4's real per-dataset grid, loaded from {SEVERITY_TABLE_PATH}")
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