"""
run_label_noise_corruption.py

Smoke-test / validation runner for D3's label_noise.py module (Uniform,
Class-Conditional) against the real processed corpus. Companion to the
missingness and feature_noise runners; same structure, same caveats.

NOT the final Phase 2 experiment runner — PLACEHOLDER_SEVERITIES is
illustrative, not D4's frozen grid.

Structural differences from the other two runners:
  - Label noise operates on the single target column, not a set of feature
    columns — so there's no per-column type routing or column cap.
  - Severity ceiling is 0.5 for binary labels (a 50% flip rate destroys the
    signal), so the placeholder severities stay well below that.
  - Class-conditional needs Cleanlab to estimate the asymmetry ratio (D1 Sec
    6.2). If Cleanlab isn't installed, this runner catches the ImportError
    and records CLEANLAB_UNAVAILABLE for those rows rather than crashing the
    whole run — the uniform-noise rows still complete. This makes the
    Cleanlab dependency visible in the manifest instead of silent.

Usage:
    python -m src.scripts.run_label_noise_corruption --datasets phoneme churn
    python -m src.scripts.run_label_noise_corruption --all
    python -m src.scripts.run_label_noise_corruption --all --severities 0.05 0.1 --seeds 1 2 3
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.label_noise import corrupt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")  # adjust to your environment
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MANIFEST = OUT_DIR / "label_noise_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "label_noise_smoketest_run.log"

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
# Kept below the 0.5 binary ceiling with margin — label noise's usable range
# is tighter than feature corruption's (see label_noise.py).
PLACEHOLDER_SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.25]
PLACEHOLDER_SEEDS = [1, 2, 3]
MECHANISMS = ["uniform", "class_conditional"]  # boundary_aware deferred (D1 6.3)

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
    n_classes = df[target_col].nunique(dropna=True)
    log(f"  train={len(train_fold)} test={len(test_fold)} target='{target_col}' n_classes={n_classes}")

    if n_classes != 2:
        log(f"  !! target is not binary ({n_classes} classes) — label_noise implements binary only; skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": f"NON_BINARY_TARGET_{n_classes}"})
        return

    feature_cols = [c for c in train_fold.columns if c != target_col]

    for mechanism in MECHANISMS:
        cleanlab_unavailable_logged = False
        for severity in PLACEHOLDER_SEVERITIES:
            for seed in PLACEHOLDER_SEEDS:
                row = {"source": source, "dataset": name, "mechanism": mechanism,
                       "severity": severity, "seed": seed}
                try:
                    _, meta = corrupt(test_fold, mechanism, severity, seed,
                                      train_fold_reference=train_fold,
                                      target_col=target_col,
                                      feature_cols=feature_cols,
                                      dataset_name=name)
                    row["status"] = "OK"
                    row["validation_passed"] = meta.validation_passed
                    row["achieved_rate"] = str(meta.achieved_rate)
                    if mechanism == "class_conditional":
                        row["asymmetry_ratio"] = meta.validation_details.get("asymmetry_ratio")
                        row["ratio_source"] = meta.validation_details.get("asymmetry_ratio_source")
                except ImportError as e:
                    # Cleanlab missing — class_conditional only. Record it as a
                    # distinct status so the dependency gap is visible in the
                    # manifest, and don't spam the log for every severity/seed.
                    row["status"] = "CLEANLAB_UNAVAILABLE"
                    row["validation_passed"] = None
                    row["error"] = str(e)[:300]
                    if not cleanlab_unavailable_logged:
                        log(f"  {mechanism}: Cleanlab unavailable — all {mechanism} rows recorded as CLEANLAB_UNAVAILABLE. "
                            f"Install cleanlab or pass an explicit asymmetry_ratio.")
                        cleanlab_unavailable_logged = True
                except Exception as e:
                    row["status"] = f"UNEXPECTED_ERROR: {type(e).__name__}"
                    row["validation_passed"] = False
                    row["error"] = str(e)[:500]
                    log(f"    !! UNEXPECTED {mechanism} sev={severity} seed={seed}: {str(e)[:150]}")
                results.append(row)

        n_ok = sum(1 for r in results if r["dataset"] == name and r["mechanism"] == mechanism and r["status"] == "OK")
        n_total = len(PLACEHOLDER_SEVERITIES) * len(PLACEHOLDER_SEEDS)
        status_note = ""
        if any(r["dataset"] == name and r["mechanism"] == mechanism and r["status"] == "CLEANLAB_UNAVAILABLE" for r in results):
            status_note = " (Cleanlab unavailable)"
        log(f"  {mechanism}: {n_ok}/{n_total} OK{status_note}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's label_noise module against real processed data.")
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

    log(f"LABEL NOISE SMOKE TEST — {len(selected)} dataset(s), "
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
        n_cleanlab = (df_results["status"] == "CLEANLAB_UNAVAILABLE").sum()
        n_unexpected = df_results["status"].astype(str).str.startswith("UNEXPECTED_ERROR").sum()
        log()
        log(f"OK: {n_ok} | Cleanlab unavailable (class_conditional rows): {n_cleanlab} "
            f"| Unexpected errors (investigate): {n_unexpected}")
    log(f"Manifest -> {OUT_MANIFEST}")
    log(f"Log      -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()