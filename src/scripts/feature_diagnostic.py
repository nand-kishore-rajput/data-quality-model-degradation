"""
Two outstanding follow-ups from the D7 cross-manifest validation.
Run in the same container and paste back the output.
"""
import pandas as pd

BASE = "/workspace/results/corruption_engine"

print("=" * 70)
print("FOLLOW-UP 1: distribution_shift's conditioning_feature by mechanism")
print("=" * 70)

ds = pd.read_csv(f"{BASE}/distribution_shift_smoketest_manifest.csv")
print("\nRows with non-null conditioning_feature, grouped by mechanism:")
print(ds[ds["conditioning_feature"].notna()]["mechanism"].value_counts().to_string())

print("\nRows with NULL conditioning_feature, grouped by mechanism:")
print(ds[ds["conditioning_feature"].isna()]["mechanism"].value_counts().to_string())

print("\nSample non-null conditioning_feature values (first 3 per mechanism):")
for mech in ds["mechanism"].unique():
    sub = ds[(ds["mechanism"] == mech) & (ds["conditioning_feature"].notna())]
    if len(sub) > 0:
        print(f"\n  mechanism={mech}:")
        for v in sub["conditioning_feature"].head(3):
            print(f"    {v}")


print("\n" + "=" * 70)
print("FOLLOW-UP 2: feature_noise's validation_passed=False breakdown")
print("=" * 70)

fn = pd.read_csv(f"{BASE}/feature_noise_smoketest_manifest.csv")
false_rows = fn[fn["validation_passed"] == False]
print(f"\nTotal False rows: {len(false_rows)}")

print("\nBreakdown by (mechanism, severity) -- concentration check:")
print(false_rows.groupby(["mechanism", "severity"]).size().to_string())

print("\nBreakdown by mechanism only:")
print(false_rows["mechanism"].value_counts().to_string())

print("\nBreakdown by dataset:")
print(false_rows["dataset"].value_counts().to_string())

print("\nSeverity distribution of False rows vs. ALL rows (mean/median comparison):")
print(f"  False rows -- mean severity: {false_rows['severity'].mean():.4f}, median: {false_rows['severity'].median():.4f}")
print(f"  All rows   -- mean severity: {fn['severity'].mean():.4f}, median: {fn['severity'].median():.4f}")

print("\nAre False rows concentrated at the top 20% of severity values?")
top20_threshold = fn["severity"].quantile(0.8)
n_false_in_top20 = (false_rows["severity"] >= top20_threshold).sum()
print(f"  Top-20% severity threshold: {top20_threshold:.4f}")
print(f"  False rows with severity >= threshold: {n_false_in_top20} / {len(false_rows)} "
      f"({100*n_false_in_top20/len(false_rows):.1f}%)")

print("\nSample of 5 False rows with their achieved_rate and validation_details_raw:")
for _, row in false_rows.head(5).iterrows():
    print(f"\n  dataset={row['dataset']} mechanism={row['mechanism']} severity={row['severity']} seed={row['seed']}")
    print(f"    achieved_rate: {row['achieved_rate']}")
    print(f"    validation_details_raw: {str(row['validation_details_raw'])[:300]}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)