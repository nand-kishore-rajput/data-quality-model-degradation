"""
run_validity_violations_corruption.py

Smoke-test / validation runner for D3's validity_violations.py module (D1
Sec 8) against the real processed corpus. Companion to the other four
runners; same structure, same caveats.

NOT the final Phase 2 experiment runner — PLACEHOLDER_SEVERITIES is
illustrative, not D4's frozen grid.

Structural difference from the other runners: this module needs the
confirmed domain-constraint annotation file (validity_constraints_annotated.csv,
Kishore, confirmed 2026-07-20) to know what counts as a violation per column.
There is no sensible default eligible-column selection the way missingness.py
has — every column corrupted here is read directly from that file's
CONFIRMED annotations, never guessed.

Usage:
    python -m src.scripts.run_validity_violations_corruption --datasets phoneme churn
    python -m src.scripts.run_validity_violations_corruption --all
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.validity_violations import corrupt, load_constraints, parse_constraint

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
CONSTRAINTS_CSV = PROJECT_ROOT / "datasets/metadata/validity_constraints_annotated.csv"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MANIFEST = OUT_DIR / "validity_violations_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "validity_violations_smoketest_run.log"

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
        results.append({"source": source, "dataset": name, "severity": None,
                         "seed": None, "status": "SKIPPED_NO_TARGET_MAPPING"})
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log(f"  !! READ ERROR: {e}")
        results.append({"source": source, "dataset": name, "severity": None,
                         "seed": None, "status": f"READ_ERROR: {e}"})
        return

    try:
        constraints = load_constraints(str(CONSTRAINTS_CSV), name)
    except (FileNotFoundError, ValueError) as e:
        log(f"  !! No usable constraint annotations for '{name}': {str(e)[:200]}")
        results.append({"source": source, "dataset": name, "severity": None,
                         "seed": None, "status": "NO_CONSTRAINT_ANNOTATIONS"})
        return

    train_fold, test_fold = split_train_test(df, target_col, SPLIT_RATIO, SPLIT_SEED)
    numeric_annotated = [
        c for c in constraints
        if c in test_fold.columns and parse_constraint(constraints[c])[0] != "categorical"
        and pd.api.types.is_numeric_dtype(test_fold[c])
    ]
    log(f"  train={len(train_fold)} test={len(test_fold)}")
    log(f"  {len(constraints)} columns annotated, {len(numeric_annotated)} eligible "
        f"(non-categorical, numeric): {numeric_annotated}")

    if not numeric_annotated:
        log(f"  !! No eligible (non-categorical, numeric) annotated columns — skipping.")
        results.append({"source": source, "dataset": name, "severity": None,
                         "seed": None, "status": "SKIPPED_NO_ELIGIBLE_COLUMNS"})
        return

    for severity in PLACEHOLDER_SEVERITIES:
        for seed in PLACEHOLDER_SEEDS:
            row = {"source": source, "dataset": name, "severity": severity, "seed": seed}
            try:
                _, meta = corrupt(test_fold, severity, seed, train_fold_reference=train_fold,
                                  constraints=constraints, columns=numeric_annotated,
                                  dataset_name=name)
                row["status"] = "PARTIAL_SUCCESS" if meta.excluded_columns else "OK"
                row["validation_passed"] = meta.validation_passed
                row["achieved_rate"] = str(meta.achieved_rate)
                row["excluded_columns"] = str(meta.excluded_columns) if meta.excluded_columns else None
            except RuntimeError as e:
                row["status"] = "ALL_COLUMNS_FAILED"
                row["validation_passed"] = False
                row["error"] = str(e)[:400]
                log(f"    !! sev={severity} seed={seed}: {str(e)[:150]}")
            except Exception as e:
                row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                row["validation_passed"] = False
                row["error"] = str(e)[:400]
                log(f"    !! UNEXPECTED sev={severity} seed={seed}: {str(e)[:150]}")
            results.append(row)

    n_ok = sum(1 for r in results if r["dataset"] == name and r["status"] in ("OK", "PARTIAL_SUCCESS"))
    n_total = len(PLACEHOLDER_SEVERITIES) * len(PLACEHOLDER_SEEDS)
    log(f"  validity_violations: {n_ok}/{n_total} OK")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's validity_violations module against real processed data.")
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

    if not CONSTRAINTS_CSV.exists():
        log(f"Constraint annotation file not found at {CONSTRAINTS_CSV}. "
            f"This module cannot run without it (D1 Decision Log item 6).")
        sys.exit(1)

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

    log(f"VALIDITY VIOLATIONS SMOKE TEST — {len(selected)} dataset(s), "
        f"{len(PLACEHOLDER_SEVERITIES)} severities, {len(PLACEHOLDER_SEEDS)} seeds")
    log(f"Severities (PLACEHOLDER, not D4's real grid): {PLACEHOLDER_SEVERITIES}")
    log(f"Constraints file: {CONSTRAINTS_CSV}")
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