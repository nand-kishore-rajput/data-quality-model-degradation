"""
run_uniqueness_corruption.py

Smoke-test / validation runner for D3's uniqueness.py module (Global 9.1,
Slice-Conditional 9.2) against the real processed corpus. Sixth and final
runner, companion to the other five.

NOT the final Phase 2 experiment runner — PLACEHOLDER_SEVERITIES is
illustrative, not D4's frozen grid.

Structural note specific to this runner: uniqueness's output fold is LARGER
than the input (rows are appended, not transformed) — every other runner's
implicit assumption of same-size output does not hold here. The manifest
records n_original/n_appended/n_final explicitly so this is visible, not
just inferred.

Usage:
    python -m src.scripts.run_uniqueness_corruption --datasets phoneme churn
    python -m src.scripts.run_uniqueness_corruption --all
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.uniqueness import corrupt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
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
PLACEHOLDER_SEEDS = [1, 2, 3]
MECHANISMS = ["global", "slice_conditional"]

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
    pre_existing_dup_rate = float(test_fold.duplicated().mean())
    log(f"  train={len(train_fold)} test={len(test_fold)} "
        f"pre_existing_duplicate_rate={pre_existing_dup_rate:.4f}")

    for mechanism in MECHANISMS:
        for severity in PLACEHOLDER_SEVERITIES:
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
                except RuntimeError as e:
                    row["status"] = "CORRUPTION_ERROR"
                    row["validation_passed"] = False
                    row["error"] = str(e)[:400]
                    log(f"    !! {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                except Exception as e:
                    row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                    row["validation_passed"] = False
                    row["error"] = str(e)[:400]
                    log(f"    !! UNEXPECTED {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism and r["status"] == "OK")
        n_total = len(PLACEHOLDER_SEVERITIES) * len(PLACEHOLDER_SEEDS)
        log(f"  {mechanism}: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's uniqueness module against real processed data.")
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

    log(f"UNIQUENESS SMOKE TEST — {len(selected)} dataset(s), "
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