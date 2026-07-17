# import pandas as pd
# from pathlib import Path
# from datetime import datetime

# # ==================================================
# # CONFIG
# # ==================================================

# DATASET_PATH = Path("datasets/raw")
# LOG_FILE = Path("datasets/metadata/dataset_inspection.log")

# # ==================================================
# # START LOG
# # ==================================================

# with open(LOG_FILE, "w", encoding="utf-8") as log:

#     log.write("=" * 120 + "\n")
#     log.write("DATASET INSPECTION REPORT\n")
#     log.write(f"Generated: {datetime.now()}\n")
#     log.write("=" * 120 + "\n\n")

#     parquet_files = sorted(DATASET_PATH.rglob("*.parquet"))

#     log.write(f"Datasets Found: {len(parquet_files)}\n\n")

#     # ==================================================
#     # INSPECT EACH DATASET
#     # ==================================================

#     for file_path in parquet_files:

#         source = file_path.parent.name
#         dataset_name = file_path.stem

#         log.write("\n")
#         log.write("=" * 120 + "\n")
#         log.write(f"DATASET : {dataset_name}\n")
#         log.write(f"SOURCE  : {source}\n")
#         log.write(f"FILE    : {file_path}\n")
#         log.write("=" * 120 + "\n")

#         try:

#             df = pd.read_parquet(file_path)

#             # ------------------------------------------
#             # Shape
#             # ------------------------------------------

#             log.write("\n[SHAPE]\n")
#             log.write(f"Rows    : {len(df):,}\n")
#             log.write(f"Columns : {len(df.columns)}\n")

#             # ------------------------------------------
#             # Missingness
#             # ------------------------------------------

#             total_cells = df.shape[0] * df.shape[1]
#             missing_cells = df.isna().sum().sum()

#             log.write("\n[MISSINGNESS]\n")
#             log.write(f"Missing Cells   : {missing_cells:,}\n")
#             log.write(
#                 f"Missing Percent : "
#                 f"{100 * missing_cells / total_cells:.4f}%\n"
#             )

#             # ------------------------------------------
#             # Numeric/Categorical
#             # ------------------------------------------

#             numeric_cols = df.select_dtypes(
#                 include=["number"]
#             ).columns

#             categorical_cols = df.select_dtypes(
#                 exclude=["number"]
#             ).columns

#             log.write("\n[FEATURE TYPES]\n")
#             log.write(
#                 f"Numeric Features     : {len(numeric_cols)}\n"
#             )
#             log.write(
#                 f"Categorical Features : {len(categorical_cols)}\n"
#             )

#             # ------------------------------------------
#             # Column List
#             # ------------------------------------------

#             log.write("\n[COLUMNS]\n")

#             for idx, col in enumerate(df.columns, start=1):
#                 log.write(f"{idx:3d}. {col}\n")

#             # ------------------------------------------
#             # Potential Targets
#             # ------------------------------------------

#             log.write("\n[POTENTIAL TARGET CANDIDATES]\n")

#             candidates = []

#             for col in df.columns:

#                 try:
#                     unique_count = df[col].nunique(
#                         dropna=False
#                     )

#                     if 2 <= unique_count <= 20:
#                         candidates.append(
#                             (col, unique_count)
#                         )

#                 except Exception:
#                     pass

#             candidates = sorted(
#                 candidates,
#                 key=lambda x: x[1]
#             )

#             for col, unique_count in candidates[:20]:

#                 log.write(
#                     f"{col:<50}"
#                     f"unique_values={unique_count}\n"
#                 )

#             # ------------------------------------------
#             # Last Column Analysis
#             # ------------------------------------------

#             target = df.columns[-1]

#             log.write("\n[LAST COLUMN ANALYSIS]\n")
#             log.write(f"Column : {target}\n")

#             unique_count = df[target].nunique(
#                 dropna=False
#             )

#             log.write(
#                 f"Unique Values : {unique_count}\n"
#             )

#             log.write("\nTop 10 Values:\n")

#             value_counts = (
#                 df[target]
#                 .astype(str)
#                 .value_counts(dropna=False)
#                 .head(10)
#             )

#             for value, count in value_counts.items():

#                 pct = (
#                     count / len(df)
#                 ) * 100

#                 log.write(
#                     f"{value:<30}"
#                     f"{count:>12,}"
#                     f" ({pct:.2f}%)\n"
#                 )

#         except Exception as e:

#             log.write("\n[ERROR]\n")
#             log.write(str(e) + "\n")

# print(f"\nInspection report written to:\n{LOG_FILE}")




"""
inspect_datasets.py

Iterates through all 17 selected corpus datasets and writes a detailed
inspection log covering:
  - Shape and basic stats
  - Target column analysis (raw and post-binarization)
  - Column types (numeric / categorical / high-cardinality)
  - Missingness per column (top offenders)
  - Duplicate row count
  - Constant / near-constant columns
  - Columns requiring special handling (leakage drops, sentinel values, encoding)
  - AUC screening pre-processing requirements summary

Output: datasets/metadata/dataset_inspection1.log

Nothing is written to parquet files. Read-only inspection only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
RAW_DIR      = PROJECT_ROOT / "datasets" / "raw"
OUT_PATH     = PROJECT_ROOT / "datasets" / "metadata" / "dataset_inspection1.log"

# ----------------------------------------------------------------------
# Dataset registry — all 17 selected datasets
# (source, dataset_name, parquet_path, target, special_handling)
# ----------------------------------------------------------------------
DATASETS = [
    {
        "source": "openml",
        "name": "phoneme",
        "path": RAW_DIR / "openml" / "phoneme.parquet",
        "target": "Class",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "",
    },
    {
        "source": "openml",
        "name": "bank-marketing",
        "path": RAW_DIR / "openml" / "bank_marketing.parquet",
        "target": "Class",
        "binarize": None,
        "drop_cols": ["V12"],   # duration — leakage
        "sentinels": {},
        "notes": "V12=duration is leakage variable; drop before modeling",
    },
    {
        "source": "openml",
        "name": "MagicTelescope",
        "path": RAW_DIR / "openml" / "MagicTelescope.parquet",
        "target": "class:",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "",
    },
    {
        "source": "openml",
        "name": "churn",
        "path": RAW_DIR / "openml" / "churn.parquet",
        "target": "class",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "",
    },
    {
        "source": "openml",
        "name": "heloc",
        "path": RAW_DIR / "openml" / "heloc.parquet",
        "target": "RiskPerformance",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {-9: "nan", -8: "nan", -7: "category"},
        "notes": (
            "Sentinel encoding: -9/-8=missing (map to NaN), "
            "-7=structural category 'condition not met' (retain). "
            "588 duplicate rows documented natural defect."
        ),
    },
    {
        "source": "openml",
        "name": "electricity",
        "path": RAW_DIR / "openml" / "electricity.parquet",
        "target": "class",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "Temporal dataset — blocked CV required in Phase 1",
    },
    {
        "source": "openml",
        "name": "jm1",
        "path": RAW_DIR / "openml" / "jm1.parquet",
        "target": "defects",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "2061 duplicate rows — natural uniqueness defect. Published label noise (Shepperd et al. 2013).",
    },
    {
        "source": "openml",
        "name": "road-safety",
        "path": RAW_DIR / "openml" / "road_safety.parquet",
        "target": "Sex_of_Driver",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": (
            "Subsampled 50k parquet — already filtered to binary target "
            "['1.0','2.0'] (Male/Female). Target stored as ArrowString '1.0'/'2.0'."
        ),
    },
    {
        "source": "openml",
        "name": "kick",
        "path": RAW_DIR / "openml" / "kick.parquet",
        "target": "IsBadBuy",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "Needs subsample to 50k before modeling",
    },
    {
        "source": "uci",
        "name": "default_credit",
        "path": RAW_DIR / "uci" / "default_credit.parquet",
        "target": "Y",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "",
    },
    {
        "source": "uci",
        "name": "online_shoppers",
        "path": RAW_DIR / "uci" / "online_shoppers.parquet",
        "target": "Revenue",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "",
    },
    {
        "source": "uci",
        "name": "diabetes_readmission",
        "path": RAW_DIR / "uci" / "diabetes_readmission.parquet",
        "target": "readmitted",
        "binarize": lambda y: (y == "<30").astype(int),
        "drop_cols": [],
        "sentinels": {},
        "notes": (
            "Target binarized: <30 days=1, else=0. "
            "weight col 96.86% missing — documented natural defect. "
            "payer_code 39.6% missing, medical_specialty 49.1% missing."
        ),
    },
    {
        "source": "folktables",
        "name": "ACSIncome",
        "path": None,   # path resolved at runtime by glob
        "target": "target",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "Folktables ACS survey — check exact parquet filename at runtime",
    },
    {
        "source": "folktables",
        "name": "ACSPublicCoverage",
        "path": None,
        "target": "target",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "Folktables ACS survey — check exact parquet filename at runtime",
    },
    {
        "source": "cleanml",
        "name": "Airbnb",
        "path": None,
        "target": "Rating",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "CleanML — check exact parquet filename at runtime",
    },
    {
        "source": "cleanml",
        "name": "EEG",
        "path": None,
        "target": "Eye",
        "binarize": None,
        "drop_cols": [],
        "sentinels": {},
        "notes": "Temporal EEG dataset — blocked CV required in Phase 1",
    },
    {
        "source": "tableshift",
        "name": "assistments",
        "path": RAW_DIR / "tableshift" / "assistments.parquet",
        "target": "correct",
        "binarize": lambda y: (y == 1).astype(int),
        "drop_cols": [
            # Hard leakage exclusions
            "hint_count", "bottom_hint", "actions", "attempt_count",
            "ms_first_response", "overlap_time", "answer_id", "answer_text",
            "first_action", "end_time",
            # Proxy leakage (affect detectors)
            "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
            "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
            # High-cardinality IDs — drop per Phase 0 decision
            "user_id", "problem_id", "template_id", "assistment_id",
            "assignment_id", "sequence_id", "base_sequence_id",
        ],
        "sentinels": {},
        "notes": (
            "Target binarized: correct==1 → 1, else → 0. "
            "Extensive leakage exclusion list applied. "
            "Subsampled to 25k before modeling."
        ),
    },
]

# High-cardinality threshold — columns with more distinct values than this
# will be flagged for encoding attention
HIGH_CARD_THRESHOLD = 50

# Near-constant threshold — columns where one value covers > this % of rows
NEAR_CONSTANT_THRESHOLD = 99.0


# ----------------------------------------------------------------------
# Path resolver for datasets without explicit paths
# ----------------------------------------------------------------------
def _norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_path(ds):
    if ds["path"] is not None:
        return ds["path"]
    src_dir = RAW_DIR / ds["source"]
    if not src_dir.exists():
        return None
    want = _norm(ds["name"])
    candidates = list(src_dir.glob("*.parquet"))
    for c in candidates:
        stem = _norm(c.stem)
        if want == stem or want in stem or stem in want:
            return c
    return None


# ----------------------------------------------------------------------
# Per-dataset inspection
# ----------------------------------------------------------------------
def inspect(ds, log):
    def w(line=""):
        log.write(line + "\n")

    name = ds["name"]
    w("=" * 80)
    w(f"DATASET: {name}  |  source={ds['source']}")
    w("=" * 80)

    path = resolve_path(ds)
    if path is None or not Path(path).exists():
        w(f"  !! FILE NOT FOUND — expected: {path}")
        w()
        return

    w(f"  Path   : {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        w(f"  !! READ ERROR: {e}")
        w()
        return

    target = ds["target"]
    drop_cols = ds["drop_cols"]
    sentinels = ds["sentinels"]
    binarize  = ds["binarize"]

    # --- Shape ---
    w()
    w(f"  SHAPE (raw)       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    w(f"  Target column     : {target}")
    w(f"  Drop cols         : {drop_cols if drop_cols else 'none'}")
    w(f"  Sentinel rules    : {sentinels if sentinels else 'none'}")
    w(f"  Notes             : {ds['notes'] if ds['notes'] else 'none'}")

    # --- Column types ---
    w()
    w("  COLUMN TYPES:")
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    other_cols = [c for c in df.columns if c not in num_cols and c not in cat_cols]
    w(f"    Numeric    : {len(num_cols)}  cols")
    w(f"    Categorical: {len(cat_cols)}  cols")
    w(f"    Other/Arrow: {len(other_cols)}  cols  {other_cols if other_cols else ''}")

    # --- Categorical columns detail ---
    if cat_cols:
        w()
        w("  CATEGORICAL COLUMNS (distinct values | encoding need):")
        for c in cat_cols:
            if c == target:
                continue
            n_distinct = df[c].nunique(dropna=True)
            flag = "HIGH-CARDINALITY" if n_distinct > HIGH_CARD_THRESHOLD else "ok"
            sample = df[c].dropna().unique()[:5].tolist()
            w(f"    {c:45s}  distinct={n_distinct:5d}  [{flag}]  sample={sample}")

    # --- Missingness ---
    w()
    w("  MISSINGNESS (raw — before sentinel decode):")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    w(f"    Total missing cells : {total_missing:,} / {total_cells:,} ({100*total_missing/total_cells:.4f}%)")

    missing_by_col = df.isna().sum()
    missing_cols   = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if len(missing_cols) == 0:
        w("    No missing values in any column.")
    else:
        w(f"    Columns with missing values ({len(missing_cols)}):")
        for col, cnt in missing_cols.items():
            pct = 100 * cnt / df.shape[0]
            flag = " !! SEVERE (>50%)" if pct > 50 else (" ! HIGH (>20%)" if pct > 20 else "")
            w(f"      {col:45s}  {cnt:7,}  ({pct:6.2f}%){flag}")

    # --- Sentinel decode impact (if applicable) ---
    if sentinels:
        w()
        w("  SENTINEL ANALYSIS:")
        nan_sentinels = [v for v, a in sentinels.items() if a == "nan"]
        cat_sentinels = [v for v, a in sentinels.items() if a == "category"]
        for s in nan_sentinels + cat_sentinels:
            action = sentinels[s]
            count = (df[num_cols] == s).sum().sum()
            cols_hit = (df[num_cols] == s).any()
            cols_hit = cols_hit[cols_hit].index.tolist()
            w(f"    Sentinel {s:4d} → {action:10s} : {count:,} cells across {len(cols_hit)} cols")
        df_decoded = df.copy()
        if nan_sentinels:
            df_decoded[num_cols] = df_decoded[num_cols].replace(nan_sentinels, np.nan)
        decoded_missing = df_decoded.isna().sum().sum()
        w(f"    Decoded missing cells: {decoded_missing:,} (delta = +{decoded_missing - total_missing:,})")

    # --- Duplicates ---
    w()
    w("  DUPLICATES:")
    feature_cols = [c for c in df.columns if c != target and c not in drop_cols]
    dupes = df[feature_cols].duplicated().sum()
    w(f"    Duplicate rows (feature cols only): {dupes:,} ({100*dupes/df.shape[0]:.2f}%)")

    # --- Constant / near-constant columns ---
    w()
    w("  CONSTANT / NEAR-CONSTANT COLUMNS:")
    const_found = False
    for c in df.columns:
        if c == target:
            continue
        try:
            top_freq = df[c].value_counts(dropna=False, normalize=True).iloc[0] * 100
            if top_freq >= NEAR_CONSTANT_THRESHOLD:
                w(f"    !! {c:45s}  top value covers {top_freq:.1f}% of rows")
                const_found = True
        except Exception:
            pass
    if not const_found:
        w("    None detected.")

    # --- Target analysis ---
    w()
    w("  TARGET ANALYSIS:")
    if target not in df.columns:
        w(f"    !! Target column '{target}' NOT FOUND in file")
    else:
        y_raw = df[target].dropna()
        raw_counts = y_raw.value_counts(dropna=False)
        w(f"    Raw distinct values : {y_raw.nunique()}")
        w(f"    Raw value counts:")
        for val, cnt in raw_counts.items():
            w(f"      {str(val):20s}  {cnt:8,}  ({100*cnt/len(y_raw):.2f}%)")

        if binarize is not None:
            y_bin = binarize(y_raw)
            bin_counts = y_bin.value_counts()
            minority = round(100 * bin_counts.min() / bin_counts.sum(), 2)
            majority = round(100 * bin_counts.max() / bin_counts.sum(), 2)
            w(f"    After binarization:")
            for val, cnt in bin_counts.items():
                w(f"      {str(val):20s}  {cnt:8,}  ({100*cnt/bin_counts.sum():.2f}%)")
            w(f"    Minority % (binarized): {minority}%  |  Majority %: {majority}%")
            w(f"    Minority pass (>=10%)  : {minority >= 10.0}")
        else:
            counts = y_raw.value_counts()
            minority = round(100 * counts.min() / counts.sum(), 2)
            w(f"    Minority %: {minority}%  |  Binary: {y_raw.nunique() == 2}")

    # --- AUC screening requirements summary ---
    w()
    w("  AUC SCREENING PRE-PROCESSING REQUIREMENTS:")
    reqs = []
    if cat_cols:
        high_card = [c for c in cat_cols if c != target and df[c].nunique() > HIGH_CARD_THRESHOLD]
        if high_card:
            reqs.append(f"Label-encode categoricals (HIGH-CARD cols need attention: {high_card})")
        else:
            reqs.append(f"Label-encode categoricals ({len([c for c in cat_cols if c != target])} cols)")
    if sentinels:
        reqs.append("Apply sentinel decode (-9/-8 → NaN; -7 → retain category)")
    if drop_cols:
        reqs.append(f"Drop leakage/excluded columns: {drop_cols}")
    if binarize is not None:
        reqs.append("Binarize target before training")
    if df.shape[0] > 50000:
        reqs.append(f"Subsample to 50k rows (current: {df.shape[0]:,})")
    if total_missing > 0:
        reqs.append(f"NaNs present ({total_missing:,} cells) — XGBoost handles natively, no imputation needed")
    if not reqs:
        reqs.append("No special preprocessing required — read and train directly")
    for r in reqs:
        w(f"    - {r}")

    w()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as log:
        log.write(f"DATASET INSPECTION LOG\n")
        log.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Purpose   : Pre-AUC screening analysis — understand each dataset before writing the screener\n")
        log.write(f"Datasets  : {len(DATASETS)} selected corpus datasets\n")
        log.write(f"Project   : When Does Data Quality Predict Model Failure?\n")
        log.write("\n")

        ok = 0
        failed = 0
        for ds in DATASETS:
            try:
                inspect(ds, log)
                ok += 1
            except Exception as e:
                log.write(f"\n!! UNEXPECTED ERROR inspecting {ds['name']}: {e}\n\n")
                failed += 1

        log.write("=" * 80 + "\n")
        log.write(f"SUMMARY: {ok} datasets inspected, {failed} errors\n")
        log.write("=" * 80 + "\n")

    print(f"Done. Log written → {OUT_PATH}")
    print(f"Inspected: {ok} | Errors: {failed}")


if __name__ == "__main__":
    main()