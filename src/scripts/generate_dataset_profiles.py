"""
generate_dataset_profiles.py  (D6 -- Dataset Profiles)

Computes a standard structural/statistical profile for each of the 17
processed corpus datasets, entirely FRESH from the processed parquet files
-- deliberately not reusing any of D3/D4/D5's prior computations or
assumptions, so D6 stands as an independent, self-contained reference
rather than inheriting any other deliverable's specific implementation
choices.

Per dataset, computes:
  - Size: row count, feature count (numeric / categorical / total)
  - Target: class balance, imbalance ratio (majority:minority)
  - Missingness: overall cell-missing rate, max per-column rate, count of
    columns with any missingness
  - Duplication: full-row duplicate rate
  - Numeric features: median absolute skewness across columns, count of
    highly-skewed columns (|skew| > 2), min/max column-level std (a cheap
    scale-heterogeneity indicator -- large ratios flag the kind of
    disproportionate-feature-scale issue found during D3's EEG investigation)
  - Categorical features: median cardinality, max cardinality

Outputs:
  - dataset_profiles_summary.csv -- one row per dataset (the actual D6 table)
  - dataset_profiles_numeric_detail.csv -- per-numeric-column skew/scale detail
  - dataset_profiles_categorical_detail.csv -- per-categorical-column cardinality detail

Usage:
    python -m src.scripts.generate_dataset_profiles
    python -m src.scripts.generate_dataset_profiles --datasets heloc jm1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "dataset_profiles"

# Same target-column registry used throughout D3/D4/D5 -- kept consistent
# with every other deliverable rather than re-deriving target identity here.
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

HIGH_SKEW_THRESHOLD = 2.0  # |skew| beyond this is conventionally "highly skewed"

_log_lines = []


def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(msg)


def discover_datasets():
    found = []
    for source_dir in sorted(PROCESSED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for f in sorted(source_dir.glob("*.parquet")):
            found.append((source_dir.name, f.stem, f))
    return found


def profile_dataset(source, name, path, numeric_rows, categorical_rows):
    log(f"\n{'─'*70}\n{source}/{name}\n{'─'*70}")

    target_col = TARGET_COLUMNS.get((source, name))
    if target_col is None:
        log(f"  !! No target-column mapping -- skipping.")
        return None

    df = pd.read_parquet(path)
    if target_col not in df.columns:
        log(f"  !! Target column '{target_col}' not found -- skipping.")
        return None

    n_rows = len(df)
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # --- Target / class balance ---
    class_counts = df[target_col].value_counts(dropna=True)
    n_classes = df[target_col].nunique(dropna=True)
    imbalance_ratio = (class_counts.max() / class_counts.min()) if len(class_counts) >= 2 else float("nan")

    # --- Missingness ---
    missing_per_col = df[feature_cols].isna().mean()
    overall_missing_rate = float(missing_per_col.mean())
    max_missing_rate = float(missing_per_col.max()) if len(missing_per_col) else 0.0
    n_cols_with_missing = int((missing_per_col > 0).sum())

    # --- Duplication ---
    duplicate_rate = float(df.duplicated().mean())

    # --- Numeric feature summary ---
    numeric_skews, numeric_stds = {}, {}
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 2 and vals.std() > 0:
            s = float(skew(vals))
        else:
            s = float("nan")
        numeric_skews[col] = s
        numeric_stds[col] = float(vals.std()) if len(vals) else float("nan")
        numeric_rows.append({"source": source, "dataset": name, "column": col,
                              "skew": s, "std": numeric_stds[col], "n_unique": int(df[col].nunique(dropna=True))})

    valid_skews = [s for s in numeric_skews.values() if not np.isnan(s)]
    median_abs_skew = float(np.median(np.abs(valid_skews))) if valid_skews else float("nan")
    n_highly_skewed = int(sum(1 for s in valid_skews if abs(s) > HIGH_SKEW_THRESHOLD))

    valid_stds = [s for s in numeric_stds.values() if not np.isnan(s) and s > 0]
    scale_ratio = (max(valid_stds) / min(valid_stds)) if len(valid_stds) >= 2 else float("nan")

    # --- Categorical feature summary ---
    cat_cardinalities = {}
    for col in categorical_cols:
        n_unique = int(df[col].nunique(dropna=True))
        cat_cardinalities[col] = n_unique
        categorical_rows.append({"source": source, "dataset": name, "column": col, "n_unique": n_unique})

    median_cardinality = float(np.median(list(cat_cardinalities.values()))) if cat_cardinalities else float("nan")
    max_cardinality = max(cat_cardinalities.values()) if cat_cardinalities else 0

    log(f"  rows={n_rows}, features={len(feature_cols)} (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})")
    log(f"  target='{target_col}', n_classes={n_classes}, imbalance_ratio={imbalance_ratio:.2f}")
    log(f"  missingness: overall={overall_missing_rate:.4f}, max_col={max_missing_rate:.4f}, "
        f"n_cols_with_missing={n_cols_with_missing}")
    log(f"  duplicate_rate={duplicate_rate:.4f}")
    log(f"  numeric: median_abs_skew={median_abs_skew:.2f}, n_highly_skewed={n_highly_skewed}, "
        f"scale_ratio(max_std/min_std)={scale_ratio:.1f}")
    log(f"  categorical: median_cardinality={median_cardinality:.1f}, max_cardinality={max_cardinality}")

    return {
        "source": source, "dataset": name, "n_rows": n_rows,
        "n_features_total": len(feature_cols), "n_features_numeric": len(numeric_cols),
        "n_features_categorical": len(categorical_cols),
        "target_col": target_col, "n_classes": n_classes, "imbalance_ratio": imbalance_ratio,
        "missing_rate_overall": overall_missing_rate, "missing_rate_max_col": max_missing_rate,
        "n_cols_with_missing": n_cols_with_missing,
        "duplicate_rate": duplicate_rate,
        "numeric_median_abs_skew": median_abs_skew, "numeric_n_highly_skewed": n_highly_skewed,
        "numeric_scale_ratio": scale_ratio,
        "categorical_median_cardinality": median_cardinality, "categorical_max_cardinality": max_cardinality,
    }


def main():
    parser = argparse.ArgumentParser(description="D6: generate fresh dataset profiles for the real corpus.")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    discovered = discover_datasets()
    if not discovered:
        log(f"No parquet files found under {PROCESSED_DIR}.")
        return

    selected = [d for d in discovered if d[1] in args.datasets] if args.datasets else discovered

    log(f"D6 DATASET PROFILING -- {len(selected)} dataset(s)")

    summary_rows, numeric_rows, categorical_rows = [], [], []
    for source, name, path in selected:
        result = profile_dataset(source, name, path, numeric_rows, categorical_rows)
        if result:
            summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "dataset_profiles_summary.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(OUT_DIR / "dataset_profiles_numeric_detail.csv", index=False)
    pd.DataFrame(categorical_rows).to_csv(OUT_DIR / "dataset_profiles_categorical_detail.csv", index=False)

    log(f"\n{'='*70}\nSUMMARY TABLE ({len(summary_df)} datasets)\n{'='*70}")
    if not summary_df.empty:
        log(summary_df[["dataset", "n_rows", "n_features_total", "imbalance_ratio",
                         "missing_rate_overall", "duplicate_rate", "numeric_n_highly_skewed"]]
            .to_string(index=False))

    log(f"\nOutputs:")
    log(f"  {OUT_DIR / 'dataset_profiles_summary.csv'}")
    log(f"  {OUT_DIR / 'dataset_profiles_numeric_detail.csv'}")
    log(f"  {OUT_DIR / 'dataset_profiles_categorical_detail.csv'}")

    (OUT_DIR / "dataset_profiles_run.log").write_text("\n".join(_log_lines))


if __name__ == "__main__":
    main()