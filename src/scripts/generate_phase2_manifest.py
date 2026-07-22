"""
generate_phase2_manifest.py

D7 — Phase-2-Ready Manifest generator.

Joins:
  - D3's corruption-run outputs (per-mechanism manifest CSVs, e.g. missingness_manifest.csv)
  - D10's quality/drift metric computations (per corrupted fold)
  - D11's target computation (Relative Delta M)

Produces: phase2_manifest.csv, one row per (dataset, mechanism, severity, model_family, seed).

Expected row count: 17 datasets x 14 sub-mechanisms x 10 severities x 4 model families x 5 seeds = 47,600

NOTE: This script assumes the six D3 runner outputs and D4's severity_table.csv already exist
on disk. Paths below are placeholders -- adjust to your actual workspace layout
(/workspace/results/... per prior sessions) before running.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config -- adjust paths to match actual workspace layout
# ---------------------------------------------------------------------------

CORRUPTION_MANIFEST_DIR = Path("/workspace/results/corruption_manifests")
SEVERITY_TABLE_PATH = Path("/workspace/results/severity_table.csv")
QUALITY_METRICS_DIR = Path("/workspace/results/quality_metrics")
MODEL_EVAL_DIR = Path("/workspace/results/model_evaluations")
OUTPUT_PATH = Path("/workspace/results/phase2_manifest.csv")

MODEL_FAMILIES = ["xgboost", "random_forest", "logistic_regression", "mlp"]
SEEDS = [1, 2, 3, 4, 5]

DEGENERATE_RESAMPLE_THRESHOLD = 0.10   # from D3 -- effective_sample_fraction below this = degenerate
SEVERITY_INFEASIBLE_RATIO = 0.9        # D7.4 -- achieved_rate < 0.9 * requested -> infeasible flag

MECHANISM_RUNNER_MAP = {
    "mcar": "missingness", "mar": "missingness", "mnar": "missingness",
    "gaussian": "feature_noise", "categorical_swap": "feature_noise", "quantization": "feature_noise",
    "uniform": "label_noise", "class_conditional": "label_noise",
    "covariate": "distribution_shift", "label": "distribution_shift", "concept": "distribution_shift",
    "validity_violations": "validity_violations",
    "global": "uniqueness", "slice_conditional": "uniqueness",
}

ROW_COUNT_CHANGING_MECHANISMS = {"global", "slice_conditional"}  # uniqueness sub-mechanisms


# ---------------------------------------------------------------------------
# Step 1 -- experiment_id (deterministic, reproducible from inputs alone)
# ---------------------------------------------------------------------------

def make_experiment_id(dataset, mechanism, severity_level, model_family, seed):
    return f"{dataset}__{mechanism}__{severity_level}__{model_family}__{seed}"


# ---------------------------------------------------------------------------
# Step 2 -- load D3's per-mechanism corruption manifests
# ---------------------------------------------------------------------------

def load_corruption_manifests():
    """
    Loads all six D3 runner-script manifest CSVs and concatenates them.
    Expected columns per D3's CorruptionMetadata schema (base.py):
      dataset, mechanism, severity, seed, n_original, n_final,
      achieved_rate, validation_passed, excluded_columns,
      effective_sample_fraction (covariate only), ...
    Column names below are best-guesses based on D3's build report --
    VERIFY against the actual CSV headers before trusting this join.
    """
    frames = []
    for csv_path in CORRUPTION_MANIFEST_DIR.glob("*_manifest.csv"):
        df = pd.read_csv(csv_path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No corruption manifest CSVs found in {CORRUPTION_MANIFEST_DIR}. "
            "Check that D3's six runner scripts have been run and wrote their manifests here."
        )

    combined = pd.concat(frames, ignore_index=True)

    # n_final may be missing for row-count-preserving mechanisms if the runner
    # only wrote it for uniqueness -- backfill assuming n_final == n_original
    # for every mechanism that doesn't grow rows.
    if "n_final" not in combined.columns:
        combined["n_final"] = np.nan
    mask_missing_n_final = combined["n_final"].isna() & ~combined["mechanism"].isin(ROW_COUNT_CHANGING_MECHANISMS)
    combined.loc[mask_missing_n_final, "n_final"] = combined.loc[mask_missing_n_final, "n_original"]

    # KNOWN RISK: if any row-count-changing mechanism (uniqueness) is missing n_final,
    # that's a real bug in the runner, not something to silently backfill -- surface it.
    still_missing = combined["n_final"].isna()
    if still_missing.any():
        bad = combined.loc[still_missing, ["dataset", "mechanism", "seed"]]
        raise ValueError(
            f"n_final missing for {still_missing.sum()} rows, including row-count-changing "
            f"mechanisms where it cannot be safely backfilled:\n{bad.head(20)}"
        )

    combined["row_count_changed"] = combined["n_final"] != combined["n_original"]
    return combined


# ---------------------------------------------------------------------------
# Step 3 -- load D4's severity table and merge severity_value
# ---------------------------------------------------------------------------

def load_severity_table():
    """
    D4's finalized severity_table.csv, 2,380 rows (17 x 14 x 10).
    Expected columns: dataset, sub_mechanism, severity_level, severity_value
    """
    df = pd.read_csv(SEVERITY_TABLE_PATH)
    expected_rows = 17 * 14 * 10
    if len(df) != expected_rows:
        # Not fatal -- D4's doc already noted this is exactly 2,380 rows with zero nulls.
        # A mismatch here means either D4's table changed or the file is stale. Warn loudly.
        print(
            f"WARNING: severity_table.csv has {len(df)} rows, expected {expected_rows}. "
            "Check whether D4's table has been regenerated since this script was written."
        )
    if df["severity_value"].isna().any():
        raise ValueError("severity_table.csv contains null severity_value entries -- D4 claims zero nulls.")
    return df


# ---------------------------------------------------------------------------
# Step 4 -- load D10's quality/drift metrics per corrupted fold
# ---------------------------------------------------------------------------

def load_quality_metrics():
    """
    D10's ~20 metric columns, computed per (dataset, mechanism, severity, seed) corrupted fold.
    NOTE: quality metrics likely do NOT vary by model_family (they're properties of the
    corrupted data, not the model) -- this join is expected to be many-to-one when later
    joined against model_family-level rows. Handled explicitly in Step 6, not assumed silently.
    """
    frames = []
    for csv_path in QUALITY_METRICS_DIR.glob("*_metrics.csv"):
        frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(f"No quality metric CSVs found in {QUALITY_METRICS_DIR}.")

    combined = pd.concat(frames, ignore_index=True)

    # duplicate_rate_scope: per D7.3, standardize on test_fold_only for manifest consistency.
    # If D10's metrics pipeline already computed duplicate_rate on the full dataset anywhere
    # (as D6's profiling did), that's the WRONG population for this table -- flag, don't silently trust.
    if "duplicate_rate_scope" not in combined.columns:
        combined["duplicate_rate_scope"] = "test_fold_only"
        print(
            "NOTE: duplicate_rate_scope not found in quality metrics output -- defaulting to "
            "'test_fold_only' per D7.3. VERIFY this matches how duplicate_rate was actually computed."
        )

    return combined


# ---------------------------------------------------------------------------
# Step 5 -- load model evaluation results and compute D11's Relative Delta M
# ---------------------------------------------------------------------------

def load_model_evaluations():
    """
    Expected columns: dataset, mechanism, severity_level, model_family, seed,
                      M_clean, M_corrupted, evaluation_metric
    M_clean per D11: single deterministic value per (dataset, model_family), not averaged
    across seeds -- so the same M_clean should repeat identically across all seed rows
    for a given (dataset, model_family). Verified below, not assumed.
    """
    frames = []
    for csv_path in MODEL_EVAL_DIR.glob("*_eval.csv"):
        frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(f"No model evaluation CSVs found in {MODEL_EVAL_DIR}.")

    combined = pd.concat(frames, ignore_index=True)

    # Sanity check: M_clean should be constant per (dataset, model_family) across all rows,
    # per D11's "single deterministic value" decision. A violation here means either the
    # evaluation pipeline recomputed M_clean per corruption call (wrong) or there's a real bug.
    grouped = combined.groupby(["dataset", "model_family"])["M_clean"].nunique()
    violations = grouped[grouped > 1]
    if len(violations) > 0:
        raise ValueError(
            f"M_clean is not constant per (dataset, model_family) for:\n{violations}\n"
            "This violates D11's decision that M_clean is a single deterministic value, "
            "not averaged or recomputed per call. Fix the evaluation pipeline before proceeding."
        )

    combined["relative_delta_m"] = (combined["M_clean"] - combined["M_corrupted"]) / combined["M_clean"]
    return combined


# ---------------------------------------------------------------------------
# Step 6 -- join everything into the final manifest
# ---------------------------------------------------------------------------

def build_manifest():
    corruption_df = load_corruption_manifests()
    severity_df = load_severity_table()
    quality_df = load_quality_metrics()
    eval_df = load_model_evaluations()

    # Join 1: corruption manifest + severity table (adds severity_value)
    # Keying on dataset + mechanism/sub_mechanism + severity_level.
    # NOTE: corruption_df's "mechanism" column needs to match severity_df's "sub_mechanism" --
    # VERIFY naming consistency (D3 sometimes says "mechanism", D4 says "sub_mechanism") before running.
    merged = corruption_df.merge(
        severity_df.rename(columns={"sub_mechanism": "mechanism"}),
        on=["dataset", "mechanism", "severity_level"],
        how="left",
        validate="many_to_one",
    )
    if merged["severity_value"].isna().any():
        missing = merged[merged["severity_value"].isna()][["dataset", "mechanism", "severity_level"]].drop_duplicates()
        raise ValueError(f"Severity value join failed for these (dataset, mechanism, severity_level) combos:\n{missing}")

    # Join 2: + quality metrics (many-to-one from model_family perspective --
    # quality metrics don't vary by model_family, so this join happens BEFORE
    # the model_family cross-join in join 3)
    merged = merged.merge(
        quality_df,
        on=["dataset", "mechanism", "severity_level", "seed"],
        how="left",
        validate="one_to_one",
    )

    # Join 3: + model evaluations (this is where model_family enters --
    # corruption/quality data doesn't depend on model_family, but every
    # corrupted fold gets evaluated by all 4 families)
    final = merged.merge(
        eval_df,
        on=["dataset", "mechanism", "severity_level", "seed"],
        how="inner",  # inner: a row without a completed model evaluation isn't ready for Phase 2
        validate="one_to_many",
    )

    # experiment_id
    final["experiment_id"] = final.apply(
        lambda row: make_experiment_id(
            row["dataset"], row["mechanism"], row["severity_level"], row["model_family"], row["seed"]
        ),
        axis=1,
    )

    # D7.4 -- severity_infeasible_flag
    # Only meaningful for rate-based mechanisms with an achieved_rate column.
    # Quantization (bin-count based) and uniqueness (deterministic count) don't have
    # a comparable "achieved_rate" -- excluded from this check, flagged False by default
    # rather than silently computing a nonsensical ratio.
    RATE_BASED_MECHANISMS = {"mcar", "mar", "mnar", "gaussian", "categorical_swap",
                             "uniform", "class_conditional", "validity_violations"}
    final["severity_infeasible_flag"] = False
    rate_mask = final["mechanism"].isin(RATE_BASED_MECHANISMS)
    if "achieved_rate" in final.columns:
        infeasible = rate_mask & (
            final["achieved_rate"] < SEVERITY_INFEASIBLE_RATIO * final["severity_value"]
        )
        final.loc[infeasible, "severity_infeasible_flag"] = True
    else:
        print("WARNING: 'achieved_rate' column not found -- severity_infeasible_flag left as all-False. "
              "Check the corruption manifest join in Step 6, Join 1.")

    # D7 diagnostic columns already present from D3's own outputs, just renamed/passed through:
    # validation_passed, excluded_columns, effective_sample_fraction (covariate only) -- verify presence.
    for col in ["validation_passed", "excluded_columns", "effective_sample_fraction"]:
        if col not in final.columns:
            print(f"WARNING: expected diagnostic column '{col}' not found in joined output.")

    if "effective_sample_fraction" in final.columns:
        final["degenerate_resample_flag"] = (
            final["effective_sample_fraction"].notna()
            & (final["effective_sample_fraction"] < DEGENERATE_RESAMPLE_THRESHOLD)
        )
    else:
        final["degenerate_resample_flag"] = False

    # kick's high-cardinality flag -- hardcoded per D7.5's decision to keep this simple.
    # OPTIMIZATION NOTE: this is currently a single hardcoded dataset name. A more general
    # version would derive this from D6's categorical_max_cardinality column against a
    # threshold (e.g. > 50), catching future high-cardinality datasets automatically rather
    # than requiring each one to be manually added here. Deferred -- not needed for the
    # current 17-dataset corpus where only kick actually has this property.
    final["high_cardinality_dataset"] = final["dataset"] == "kick"

    return final


# ---------------------------------------------------------------------------
# Step 7 -- validate and write
# ---------------------------------------------------------------------------

def validate_manifest(df):
    expected_rows = 17 * 14 * 10 * len(MODEL_FAMILIES) * len(SEEDS)
    print(f"Manifest rows: {len(df)} (expected: {expected_rows})")
    if len(df) != expected_rows:
        print(
            "MISMATCH -- do not assume this is an error automatically. Check for:\n"
            "  - incomplete model evaluation runs (inner join drops incomplete rows -- likely cause)\n"
            "  - a mechanism/dataset combo that failed validation entirely in D3 and was excluded upstream\n"
            "  - severity table row count (already checked against 2,380 above)"
        )

    dupes = df.duplicated(subset=["experiment_id"]).sum()
    if dupes > 0:
        raise ValueError(f"{dupes} duplicate experiment_id values found -- join produced unexpected fan-out.")

    print(f"severity_infeasible_flag=True count: {df['severity_infeasible_flag'].sum()}")
    print(f"degenerate_resample_flag=True count: {df['degenerate_resample_flag'].sum()}")
    print(f"row_count_changed=True count: {df['row_count_changed'].sum()}")


def main():
    manifest = build_manifest()
    validate_manifest(manifest)
    manifest.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote manifest to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()