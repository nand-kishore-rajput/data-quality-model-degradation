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


import pandas as pd
from pathlib import Path

# ==================================================
# LOAD DATASET
# ==================================================

file_path = "datasets/raw/openml/heloc.parquet"

df = pd.read_parquet(file_path)

print("=" * 80)
print("HELOC DATASET AUDIT")
print("=" * 80)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ==================================================
# CHECK FOR SENTINEL VALUES
# ==================================================

sentinel_values = [-7, -8, -9]

print("\n" + "=" * 80)
print("SENTINEL VALUE ANALYSIS")
print("=" * 80)

results = []

for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):

        for sentinel in sentinel_values:

            count = (df[col] == sentinel).sum()

            if count > 0:

                results.append({
                    "column": col,
                    "sentinel": sentinel,
                    "count": count,
                    "percent": round(
                        100 * count / len(df),
                        4
                    )
                })

if len(results) == 0:

    print("No sentinel values found.")

else:

    sentinel_df = pd.DataFrame(results)

    print(sentinel_df)

# ==================================================
# MISSING VALUES BEFORE CONVERSION
# ==================================================

print("\n" + "=" * 80)
print("CURRENT MISSINGNESS")
print("=" * 80)

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No NaN values currently present.")
else:
    print(missing.sort_values(ascending=False))

# ==================================================
# FEATURE SUMMARY
# ==================================================

print("\n" + "=" * 80)
print("COLUMN SUMMARY")
print("=" * 80)

for col in df.columns:

    unique_count = df[col].nunique(dropna=False)

    print(
        f"{col:<40}"
        f"unique={unique_count}"
    )

# ==================================================
# TARGET DISTRIBUTION
# ==================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

target = "RiskPerformance"

print(df[target].value_counts(dropna=False))

print("\nDone.")