"""
D7 Section 7 cross-manifest validation.

Run this after all six corruption runners have completed their full-scale
(5-seed, real D4 severity grid) runs. Checks the six manifests together
against D7's stated validation requirements, plus a few cross-cutting
sanity checks that only make sense once all six exist side by side.

Run in the same container and paste back the full output.
"""
import pandas as pd
import ast

BASE = "/workspace/results/corruption_engine"

FILES = {
    "missingness": f"{BASE}/missingness_smoketest_manifest.csv",
    "feature_noise": f"{BASE}/feature_noise_smoketest_manifest.csv",
    "label_noise": f"{BASE}/label_noise_smoketest_manifest.csv",
    "distribution_shift": f"{BASE}/distribution_shift_smoketest_manifest.csv",
    "validity_violations": f"{BASE}/validity_violations_smoketest_manifest.csv",
    "uniqueness": f"{BASE}/uniqueness_smoketest_manifest.csv",
}

SEVERITY_TABLE_PATH = f"results/corruption_engine/severity_table.csv"

EXPECTED_DATASETS = {
    "Airbnb", "EEG", "ACSIncome", "ACSPublicCoverage", "MagicTelescope",
    "bank-marketing", "churn", "electricity", "heloc", "jm1", "kick",
    "phoneme", "road-safety", "assistments", "default_credit",
    "diabetes_readmission", "online_shoppers",
}


def parse_dict_safe(x):
    if pd.isna(x):
        return None
    try:
        return ast.literal_eval(x)
    except Exception:
        return None


print("=" * 70)
print("1. LOAD EACH MANIFEST, BASIC SHAPE CHECK")
print("=" * 70)

dfs = {}
for name, path in FILES.items():
    try:
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"\n{name}: {len(df)} rows, {len(df.columns)} columns")
        print(f"  columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"\n{name}: FILE NOT FOUND at {path}")

print("\n" + "=" * 70)
print("2. STATUS DISTRIBUTION PER MANIFEST")
print("=" * 70)

for name, df in dfs.items():
    print(f"\n--- {name} ---")
    if "status" in df.columns:
        print(df["status"].value_counts().to_string())
    else:
        print("  NO 'status' COLUMN FOUND")

print("\n" + "=" * 70)
print("3. DATASET COVERAGE — every manifest should reference all 17 datasets")
print("=" * 70)

for name, df in dfs.items():
    present = set(df["dataset"].dropna().unique())
    missing = EXPECTED_DATASETS - present
    extra = present - EXPECTED_DATASETS
    print(f"\n--- {name} ---")
    print(f"  datasets present: {len(present)} / 17")
    if missing:
        print(f"  MISSING: {missing}")
    if extra:
        print(f"  UNEXPECTED EXTRA: {extra}")

print("\n" + "=" * 70)
print("4. conditioning_feature / validation_details_raw PRESENCE CHECK")
print("=" * 70)
print("(confirms the D7 pre-flight patch actually took effect on the full run)")

for name, df in dfs.items():
    print(f"\n--- {name} ---")
    for col in ["conditioning_feature", "validation_details_raw"]:
        if col not in df.columns:
            print(f"  {col}: COLUMN MISSING ENTIRELY")
            continue
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null} / {len(df)} non-null")

print("\n  Expected non-null conditioning_feature counts (approximate, from module logic):")
print("    missingness: >0 for 'mar' rows only (MNAR/MCAR leave it null)")
print("    uniqueness: >0 for 'slice_conditional' rows only ('global' leaves it null)")
print("    all others: 0 (no conditioning structure in those mechanisms)")

print("\n" + "=" * 70)
print("5. ROW COUNT RECONCILIATION AGAINST severity_table.csv")
print("=" * 70)

try:
    sev = pd.read_csv(SEVERITY_TABLE_PATH)
    print(f"severity_table.csv: {len(sev)} rows total")
    print("\nRows per mechanism_class in severity_table.csv:")
    print(sev.groupby("mechanism_class").size().to_string())

    print("\nExpected manifest row count = severity_table rows-for-that-mechanism-class x 5 seeds")
    class_map = {
        "missingness": "missingness",
        "feature_noise": "feature_noise",
        "label_noise": "label_noise",
        "distribution_shift": "distribution_shift",
        "validity_violations": "validity",
        "uniqueness": "uniqueness",
    }
    for name, df in dfs.items():
        sev_class = class_map.get(name)
        sev_rows = sev[sev["mechanism_class"].str.contains(sev_class, case=False, na=False)] if sev_class else None
        expected = len(sev_rows) * 5 if sev_rows is not None else "?"
        print(f"  {name}: actual={len(df)}, severity_table-implied expected={expected}")
except FileNotFoundError:
    print(f"severity_table.csv not found at {SEVERITY_TABLE_PATH}")

print("\n" + "=" * 70)
print("6. EEG EXCLUSION CONSISTENCY CHECK (missingness vs feature_noise)")
print("=" * 70)
print("(EEG's borderline channels should show up consistently, not contradictorily)")

if "missingness" in dfs:
    eeg_miss = dfs["missingness"][dfs["missingness"]["dataset"] == "EEG"]
    if "excluded_columns" in eeg_miss.columns:
        excl = eeg_miss["excluded_columns"].dropna()
        all_excluded_cols = set()
        for v in excl:
            parsed = parse_dict_safe(v)
            if parsed:
                all_excluded_cols.update(parsed.keys())
        print(f"  missingness: EEG columns ever excluded = {sorted(all_excluded_cols)}")

print("\n" + "=" * 70)
print("7. UNIQUENESS ROW-COUNT GROWTH SANITY CHECK")
print("=" * 70)
print("(n_final should exceed n_original for every OK row)")

if "uniqueness" in dfs:
    uniq = dfs["uniqueness"]
    ok_rows = uniq[uniq["status"] == "OK"]
    if "n_original" in ok_rows.columns and "n_final" in ok_rows.columns:
        bad = ok_rows[ok_rows["n_final"] <= ok_rows["n_original"]]
        print(f"  OK rows where n_final <= n_original (should be 0): {len(bad)}")
        if len(bad) > 0:
            print(bad[["dataset", "mechanism", "severity", "seed", "n_original", "n_final"]].to_string())

print("\n" + "=" * 70)
print("8. VALIDATION_PASSED NaN / FALSE AUDIT PER MANIFEST")
print("=" * 70)

for name, df in dfs.items():
    if "validation_passed" not in df.columns:
        continue
    print(f"\n--- {name} ---")
    print(f"  dtype: {df['validation_passed'].dtype}")
    vc = df["validation_passed"].value_counts(dropna=False)
    print(vc.to_string())

print("\n" + "=" * 70)
print("9. GRAND TOTAL ROW COUNT ACROSS ALL SIX MANIFESTS")
print("=" * 70)

total = sum(len(df) for df in dfs.values())
print(f"Total rows across all six manifests: {total}")
print("(This is NOT the same as the 47,600-row Phase 2 manifest -- that")
print(" number comes later, after generate_phase2_manifest.py multiplies")
print(" each corruption row by 4 model families.)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)