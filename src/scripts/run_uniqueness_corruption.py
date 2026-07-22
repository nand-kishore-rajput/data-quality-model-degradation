"""
run_uniqueness_corruption.py

Full-scale runner for D3's uniqueness.py module (Global 9.1, Slice-
Conditional 9.2) against the real processed corpus.

CHANGE FROM SMOKETEST VERSION (D7 pre-flight review): conditioning_feature
and validation_details_raw added to the manifest row. conditioning_feature
is populated for slice_conditional (the frozen slice's feature and bounds,
e.g. "slice_S(feature in [lo, hi])") — confirmed directly in uniqueness.py's
corrupt_slice_conditional — but was being discarded before reaching CSV.
It will be None for the global mechanism, which has no slice structure.

Structural note specific to this runner: uniqueness's output fold is LARGER
than the input (rows are appended, not transformed) — every other runner's
implicit assumption of same-size output does not hold here. The manifest
records n_original/n_final explicitly so this is visible, not just inferred.

Usage (full-scale):
    python -m src.scripts.run_uniqueness_corruption --all --seeds 1 2 3 4 5
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.uniqueness import corrupt
from src.corruption_engine.base import derive_call_seed

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
SEVERITY_TABLE_PATH = OUT_DIR / "severity_table.csv"
OUT_MANIFEST = OUT_DIR / "uniqueness_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "uniqueness_smoketest_run.log"

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

SPLIT_RATIO = 0.7
SPLIT_SEED = 42
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
PLACEHOLDER_SEEDS = [1, 2, 3]  # overridden via --seeds 1 2 3 4 5 for the real run
MECHANISMS = ["global", "slice_conditional"]

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
    """
    D2 Decision 1 (split before corrupt), stratified by target.

    Uses derive_call_seed rather than np.random.default_rng(seed) directly
    -- fixed 2026-07-21, same root cause as the corruption engine's per-call
    seeding bug: SPLIT_SEED=42 is the SAME literal value reused across all
    17 datasets, and a fresh Generator created from that literal seed each
    time means every dataset's split draws from the same starting PRNG
    prefix, just shuffling arrays of different sizes. Structurally identical
    to the missingness/*.py seed-correlation bug found via real-corpus
    testing -- fixed the same way, by deriving a seed unique to
    (seed, dataset_name) rather than reusing the raw seed value directly.
    """
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
    pre_existing_dup_rate = float(test_fold.duplicated().mean())
    log(f"  train={len(train_fold)} test={len(test_fold)} "
        f"pre_existing_duplicate_rate={pre_existing_dup_rate:.4f}")

    for mechanism in MECHANISMS:
        severities = PLACEHOLDER_SEVERITIES if _severities_overridden else get_severities_for(name, mechanism)
        for severity in severities:
            for seed in PLACEHOLDER_SEEDS:
                row = {"source": source, "dataset": name, "mechanism": mechanism,
                       "severity": severity, "seed": seed}
                try:
                    corrupted, meta = corrupt(test_fold, mechanism, severity, seed,
                                              train_fold_reference=train_fold, dataset_name=name)
                    row["status"] = "OK"
                    row["validation_passed"] = meta.validation_passed
                    row["achieved_rate"] = str(meta.achieved_rate)
                    row["n_original"] = len(test_fold)
                    row["n_final"] = len(corrupted)
                    # --- D7 pre-flight additions: previously computed but discarded ---
                    row["conditioning_feature"] = meta.conditioning_feature
                    row["validation_details_raw"] = str(meta.validation_details)
                    # -------------------------------------------------------------------
                except RuntimeError as e:
                    row["status"] = "CORRUPTION_ERROR"
                    row["validation_passed"] = False
                    row["conditioning_feature"] = None
                    row["validation_details_raw"] = None
                    row["error"] = str(e)[:400]
                    log(f"    !! {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                except Exception as e:
                    row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                    row["validation_passed"] = False
                    row["conditioning_feature"] = None
                    row["validation_details_raw"] = None
                    row["error"] = str(e)[:400]
                    log(f"    !! UNEXPECTED {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism and r["status"] == "OK")
        n_total = len(severities) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Run D3's uniqueness module against real processed data (full scale).")
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

    log(f"UNIQUENESS FULL RUN — {len(selected)} dataset(s), "
        f"{len(MECHANISMS)} mechanisms, 10 severity levels per D4's real grid, "
        f"{len(PLACEHOLDER_SEEDS)} seeds")
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
        n_err = (df_results["status"] == "CORRUPTION_ERROR").sum()
        n_unexpected = df_results["status"].astype(str).str.startswith("UNEXPECTED_ERROR").sum()
        log()
        log(f"OK: {n_ok} | Corruption errors (investigate): {n_err} "
            f"| Unexpected errors (investigate): {n_unexpected}")
    log(f"Manifest -> {OUT_MANIFEST}")
    log(f"Log      -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()