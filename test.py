"""
D7 manifest schema verification — round 2.
Round 1 (Airbnb only) confirmed the base schema but left open whether
excluded_columns / achieved_rate shapes hold on datasets that actually
trigger exclusions or use categorical mechanisms.

Run this in the same container and paste back the full output.
"""
import pandas as pd
import ast

BASE = "/workspace/results/corruption_engine"

files = {
    "validity_violations": f"{BASE}/validity_violations_smoketest_manifest.csv",
    "missingness": f"{BASE}/missingness_smoketest_manifest.csv",
    "distribution_shift": f"{BASE}/distribution_shift_smoketest_manifest.csv",
    "label_noise": f"{BASE}/label_noise_smoketest_manifest.csv",
    "feature_noise": f"{BASE}/feature_noise_smoketest_manifest.csv",
    "uniqueness": f"{BASE}/uniqueness_smoketest_manifest.csv",
}

TARGET_DATASETS = ["EEG", "bank-marketing", "diabetes_readmission"]
# EEG: should trigger excluded_columns (borderline channels, per D3 findings)
# bank-marketing / diabetes_readmission: have categorical features, to check
# feature_noise's categorical_swap achieved_rate dict shape vs gaussian's.

for mech, path in files.items():
    print(f"\n{'='*70}\n{mech}  ({path})\n{'='*70}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("FILE NOT FOUND")
        continue

    print("dtypes:")
    print(df.dtypes)

    print("\nUnique datasets present in this file:")
    print(sorted(df["dataset"].unique().tolist()))

    # excluded_columns: check if ANY row, in ANY dataset, has a non-null value
    if "excluded_columns" in df.columns:
        non_null = df[df["excluded_columns"].notna()]
        print(f"\nexcluded_columns non-null row count: {len(non_null)}")
        if len(non_null) > 0:
            print("dtype when populated:", type(non_null["excluded_columns"].iloc[0]))
            print("sample value:", non_null["excluded_columns"].iloc[0])
        else:
            print("excluded_columns is NaN for every row in this file — "
                  "check whether EEG rows exist here at all (see dataset list above).")

    # For EEG specifically, if present, show full rows
    for ds in TARGET_DATASETS:
        sub = df[df["dataset"] == ds]
        if len(sub) > 0:
            print(f"\n--- {mech} rows for dataset={ds} (first 3) ---")
            print(sub.head(3).to_string())
        else:
            print(f"\n(no rows for dataset={ds} in this file)")

    # achieved_rate shape check: parse a couple of examples and print keys
    if "achieved_rate" in df.columns:
        print("\nachieved_rate parse check (first 2 non-null rows):")
        sample = df[df["achieved_rate"].notna()].head(2)
        for i, row in sample.iterrows():
            raw = row["achieved_rate"]
            try:
                parsed = ast.literal_eval(raw)
                print(f"  row {i}: keys={list(parsed.keys())[:5]}... "
                      f"n_keys={len(parsed)}, "
                      f"sample_values={list(parsed.values())[:3]}")
            except Exception as e:
                print(f"  row {i}: FAILED TO PARSE — {e}")
                print(f"  raw string (first 200 chars): {str(raw)[:200]}")

print(f"\n{'='*70}\nSeverity table check\n{'='*70}")
sev = pd.read_csv("/workspace/results/corruption_engine/severity_table.csv")
print("Datasets in severity_table.csv:", sorted(sev["dataset"].unique().tolist()))
print("Row count:", len(sev))
print("Any nulls in severity_value?", sev["severity_value"].isna().any())

# Confirm float-join feasibility: for one mechanism, check whether runner's
# `severity` column values match severity_table's `severity_value` values
# closely enough for a rounded join to work (no drift/precision mismatch).
missingness = pd.read_csv(files["missingness"])
merge_check = missingness[["dataset", "mechanism", "severity"]].drop_duplicates()
sev_mcar = sev[sev["sub_mechanism"] == "mcar"][["dataset", "severity_value", "severity_level"]]
test_join = merge_check.merge(
    sev_mcar,
    left_on=["dataset", "severity"],
    right_on=["dataset", "severity_value"],
    how="left"
)
print("\nFloat-exact join test (missingness/mcar vs severity_table):")
print(test_join.head(10).to_string())
print(f"\nUnmatched rows (severity_level is null): {test_join['severity_level'].isna().sum()} / {len(test_join)}")