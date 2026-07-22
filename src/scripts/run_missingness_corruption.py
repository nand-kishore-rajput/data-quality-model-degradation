"""
run_missingness_corruption.py

Full-scale runner for D3's missingness.py module (MCAR/MAR/MNAR) against the
real processed corpus.

CHANGES FROM SMOKETEST VERSION (D7 pre-flight review):
  1. MAX_CORRUPTED_COLUMNS cap REMOVED. The smoketest capped every call at 5
     columns regardless of dataset size, so high-feature-count datasets
     (road-safety: 61 features, Airbnb: 39) never had more than 5 columns
     corrupted. D10's mean/max aggregation metrics need the whole dataset,
     not an arbitrary 5-column subset — required before this run counts as
     real experiment data, not optional polish.
  2. conditioning_feature and validation_details_raw added to the manifest
     row. base.py's CorruptionMetadata computes both (MAR's conditioning
     feature, per-column association diagnostics) but the smoketest version
     of this script discarded them before they reached CSV.
  3. Seed count supplied at the command line for the real run:
     --seeds 1 2 3 4 5 (D1's 5-seed budget).
  4. BUG FOUND AND FIXED (2026-07-22, first full-scale run): online_shoppers'
     'Weekend' column is boolean dtype. pandas' is_numeric_dtype() returns
     True for bool, so it was passed to missingness.corrupt() as an eligible
     column -- but bool columns cannot hold NaN without an explicit cast,
     raising "Invalid value 'nan' for dtype 'bool'". All 150 online_shoppers
     rows (3 mechanisms x 10 severities x 5 seeds) failed with this exact
     error on the first full-scale run. Invisible under the old column cap,
     since 'Weekend' is the 14th of 14 columns and the cap never reached it.
     Fixed by excluding bool-dtype columns explicitly in eligible_columns().

Usage (full-scale):
    python -m src.scripts.run_missingness_corruption --all --seeds 1 2 3 4 5
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.missingness import corrupt
from src.corruption_engine.base import derive_call_seed

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
SEVERITY_TABLE_PATH = OUT_DIR / "severity_table.csv"
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
# Config
# ----------------------------------------------------------------------
SPLIT_RATIO = 0.7          # 70% train / 30% test
SPLIT_SEED = 42            # matches the project's standing seed=42 convention
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]  # fallback ONLY if
    # severity_table.csv is missing, or --severities overrides it for a quick
    # manual test. D4's real grid (loaded below) is used by default.
PLACEHOLDER_SEEDS = [1, 2, 3]  # overridden via --seeds 1 2 3 4 5 for the real run
MECHANISMS = ["mcar", "mar", "mnar"]

# MAX_CORRUPTED_COLUMNS removed (was: 5). Every eligible numeric column is
# now corrupted per call — required for D10's mean/max aggregation metrics
# to reflect the whole dataset rather than an arbitrary 5-column subset.

_severity_table_cache = None
_severities_overridden = False


def load_severity_table():
    """Loads D4's real severity table (generate_severity_table.py's output),
    cached after first read. Raises clearly if it doesn't exist rather than
    silently falling back — D4's real grid should be used whenever available;
    the placeholder grid is a fallback for a missing file or a manual
    --severities override only."""
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
    """Find every processed parquet file, matched against TARGET_COLUMNS."""
    found = []
    for source_dir in sorted(PROCESSED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for f in sorted(source_dir.glob("*.parquet")):
            found.append((source_dir.name, f.stem, f))
    return found


def split_train_test(df: pd.DataFrame, target_col: str, ratio: float, seed: int, dataset_name: str):
    """
    D2 Decision 1 (split before corrupt), stratified by target to preserve
    class balance in both folds. Fixed once per dataset (D2's split is not
    redrawn per corruption seed — D11 Decision 8's same reasoning applies here).

    Uses derive_call_seed rather than np.random.default_rng(seed) directly
    -- fixed 2026-07-21, same root cause as the corruption engine's per-call
    seeding bug: SPLIT_SEED=42 is the SAME literal value reused across all
    17 datasets, and a fresh Generator from that literal seed each time
    means every dataset's split draws from the same starting PRNG prefix,
    just shuffling arrays of different sizes.
    """
    call_seed = derive_call_seed(seed, dataset_name, "train_test_split", 0.0)
    rng = np.random.default_rng(call_seed)
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


def eligible_columns(train_fold: pd.DataFrame, target_col: str):
    """Numeric, non-target, non-near-constant columns. No cap applied — every
    eligible column is corrupted per call (cap removed for the full-scale
    run; see module docstring).

    Excludes boolean-dtype columns explicitly (2026-07-22 fix): pandas'
    is_numeric_dtype() returns True for bool columns, but a bool column
    cannot hold NaN without an explicit cast -- injecting missingness into
    one raises "Invalid value 'nan' for dtype 'bool'" rather than silently
    upcasting. Confirmed on online_shoppers' 'Weekend' column, which is bool
    and sits last (14th of 14) in that dataset's column order -- the
    MAX_CORRUPTED_COLUMNS=5 cap in the smoketest version of this script
    never reached it, so this bug was invisible until the cap was removed
    for the full-scale run. All 150 online_shoppers failures (3 mechanisms
    x 10 severities x 5 seeds) traced to this single root cause."""
    cols = []
    skipped_constant = []
    skipped_bool = []
    for c in train_fold.columns:
        if c == target_col or not pd.api.types.is_numeric_dtype(train_fold[c]):
            continue
        if pd.api.types.is_bool_dtype(train_fold[c]):
            skipped_bool.append(c)
            continue
        if train_fold[c].std() < MIN_COLUMN_STD:
            skipped_constant.append(c)
            continue
        cols.append(c)
    if skipped_constant:
        log(f"  Skipping near-constant column(s) (train-fold std < {MIN_COLUMN_STD}): {skipped_constant}")
    if skipped_bool:
        log(f"  Skipping boolean-dtype column(s) (cannot hold NaN without an explicit cast): {skipped_bool}")
    return cols


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

    train_fold, test_fold = split_train_test(df, target_col, SPLIT_RATIO, SPLIT_SEED, name)
    columns = eligible_columns(train_fold, target_col)

    if not columns:
        log(f"  !! No eligible numeric columns to corrupt — skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": "NO_ELIGIBLE_COLUMNS"})
        return

    log(f"  train={len(train_fold)} test={len(test_fold)} "
        f"n_columns={len(columns)} columns={columns}")

    for mechanism in MECHANISMS:
        severities = PLACEHOLDER_SEVERITIES if _severities_overridden else get_severities_for(name, mechanism)
        for severity in severities:
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
                    # --- D7 pre-flight additions: previously computed but discarded ---
                    row["conditioning_feature"] = meta.conditioning_feature
                    row["validation_details_raw"] = str(meta.validation_details)
                    # -------------------------------------------------------------------
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

                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism and r["status"] in ("OK", "PARTIAL_SUCCESS"))
        n_total = len(severities) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Run D3's missingness module against real processed data (full scale).")
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Dataset names to run (e.g. phoneme churn). Default: first 3 discovered.")
    parser.add_argument("--all", action="store_true", help="Run all discovered datasets.")
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

    log(f"MISSINGNESS FULL RUN — {len(selected)} dataset(s), "
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
        log(f"OK: {n_ok} | Partial success (some column(s) excluded, expected occasionally): {n_partial} "
            f"| All columns failed (investigate these): {n_all_failed} "
            f"| Unexpected errors (investigate these): {n_unexpected}")
    log(f"Manifest -> {OUT_MANIFEST}")
    log(f"Log      -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()