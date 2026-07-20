"""
run_distribution_shift_corruption.py

Smoke-test / validation runner for D3's distribution_shift.py module
(Covariate 7.1, Label 7.2, Concept 7.3) against the real processed corpus.
Companion to the missingness / feature_noise / label_noise runners.

NOT the final Phase 2 experiment runner — PLACEHOLDER_SEVERITIES is
illustrative, not D4's frozen grid, and target_prior for label shift is left
at the module default (D4 supplies the real rho_target).

Structural notes specific to distribution shift:
  - All three sub-mechanisms operate on the whole fold (not per-column), so
    there's no column-type routing or per-column exclusion here.
  - Covariate (7.1) and Label (7.2) shift RESAMPLE rows — the output row
    multiset differs from the input. Concept (7.3) preserves rows and changes
    only some labels within the frozen slice S. The runner records which
    mode each mechanism used so the manifest is unambiguous.
  - 7.1's core validation property (PSI rises monotonically with severity) is
    a CROSS-severity property, not checkable from a single call — so this
    runner additionally checks per-dataset PSI monotonicity across the
    severity sweep and logs it explicitly.

Usage:
    python -m src.scripts.run_distribution_shift_corruption --datasets phoneme churn
    python -m src.scripts.run_distribution_shift_corruption --all
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruption_engine.distribution_shift import corrupt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_MANIFEST = OUT_DIR / "distribution_shift_smoketest_manifest.csv"
OUT_LOG = OUT_DIR / "distribution_shift_smoketest_run.log"

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
MECHANISMS = ["covariate", "label", "concept"]

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
        log(f"  !! target not binary ({n_classes} classes) — label/concept shift are binary-only; skipping.")
        results.append({"source": source, "dataset": name, "mechanism": None,
                         "severity": None, "seed": None, "status": f"NON_BINARY_TARGET_{n_classes}"})
        return

    for mechanism in MECHANISMS:
        # For covariate shift, track PSI across severities (seed=1) to verify
        # the monotonicity property D1 Sec 7.1 requires.
        covariate_psi_by_severity = {}

        for severity in PLACEHOLDER_SEVERITIES:
            for seed in PLACEHOLDER_SEEDS:
                row = {"source": source, "dataset": name, "mechanism": mechanism,
                       "severity": severity, "seed": seed}
                try:
                    _, meta = corrupt(test_fold, mechanism, severity, seed,
                                      train_fold_reference=train_fold,
                                      target_col=target_col, dataset_name=name)
                    row["status"] = "OK"
                    row["validation_passed"] = meta.validation_passed
                    row["achieved_rate"] = str(meta.achieved_rate)
                    if mechanism == "covariate":
                        d = meta.validation_details
                        row["effective_sample_fraction"] = d.get("effective_sample_fraction")
                        row["degenerate_resample"] = d.get("degenerate_resample")
                        if d.get("degenerate_resample"):
                            log(f"    -- covariate sev={severity} seed={seed}: degenerate resample, "
                                f"effective_sample_fraction={d.get('effective_sample_fraction'):.3f}")
                    if mechanism == "covariate" and seed == 1:
                        covariate_psi_by_severity[severity] = meta.validation_details["psi_pc1_score"]
                except NotImplementedError as e:
                    row["status"] = "NOT_IMPLEMENTED"
                    row["validation_passed"] = None
                    row["error"] = str(e)[:300]
                except (ValueError, RuntimeError) as e:
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

        # Covariate-specific: verify PSI monotonicity across the severity sweep.
        if mechanism == "covariate" and len(covariate_psi_by_severity) >= 2:
            sevs = sorted(covariate_psi_by_severity)
            psis = [covariate_psi_by_severity[s] for s in sevs]
            monotone = all(psis[i] <= psis[i+1] + 0.01 for i in range(len(psis)-1))
            log(f"    PSI vs severity (seed=1): " +
                ", ".join(f"{s}:{covariate_psi_by_severity[s]:.3f}" for s in sevs))
            log(f"    PSI monotonically increases with severity (D1 Sec 7.1): {monotone}")
            if not monotone:
                log(f"    !! PSI NON-MONOTONE — investigate: covariate shift should "
                    f"produce more drift at higher severity.")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test D3's distribution_shift module against real processed data.")
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

    log(f"DISTRIBUTION SHIFT SMOKE TEST — {len(selected)} dataset(s), "
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