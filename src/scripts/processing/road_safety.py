import pandas as pd

# Load parquet file
df = pd.read_parquet(
    "datasets/raw/openml/road_safety.parquet"
)

target = "Sex_of_Driver"

# ----------------------------------------------------------------------
# Raw missingness
# ----------------------------------------------------------------------
total_cells = df.shape[0] * df.shape[1]
raw_missing = df.isna().sum().sum()
print(f"Raw missing cells : {raw_missing}")
print(f"Raw missing %     : {100 * raw_missing / total_cells:.4f}%")

# ----------------------------------------------------------------------
# Filter target: '1.0'=Male, '2.0'=Female; drop '3.0'=Unknown
# ----------------------------------------------------------------------
df_filtered = df[df[target].isin(["1.0", "2.0"])].reset_index(drop=True)
print(f"Rows after target filter: {len(df_filtered)}")

counts = df_filtered[target].value_counts()
total = counts.sum()
print(f"Class counts : {counts.to_dict()}")
print(f"Minority %   : {round(100 * counts.min() / total, 2)}")

# ----------------------------------------------------------------------
# All data is 2015 only — year stratification not applicable
# Simple seeded random subsample to 50,000
# ----------------------------------------------------------------------
TARGET_N = 50_000
df_sample = df_filtered.sample(n=TARGET_N, random_state=42).reset_index(drop=True)

print(f"\nSubsample shape : {df_sample.shape}")
counts_s = df_sample[target].value_counts()
total_s = counts_s.sum()
print(f"Subsample class counts : {counts_s.to_dict()}")
print(f"Subsample minority %   : {round(100 * counts_s.min() / total_s, 2)}")

# ----------------------------------------------------------------------
# Duplicates in subsample
# ----------------------------------------------------------------------
feature_cols = [c for c in df_sample.columns if c != target]
dupes = df_sample[feature_cols].duplicated().sum()
print(f"Duplicate rows : {dupes}")

# ----------------------------------------------------------------------
# Save subsample to raw folder
# ----------------------------------------------------------------------
out_path = "datasets/processed/openml/road_safety.parquet"
df_sample.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
print("Done.")