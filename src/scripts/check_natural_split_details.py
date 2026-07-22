"""
check_natural_split_details.py

Resolves D5's open Action Item 1 (which admission_source_id group is larger
in diabetes_readmission, to finalize which population is "train" vs
"evaluate") and sanity-checks both anchor datasets' natural splits before
building the full D5 training/evaluation pipeline.

Specifically checks:

  1. diabetes_readmission's admission_source_id: full value distribution,
     plus the resulting group sizes under a standard Emergency vs
     Non-emergency binarization (per the UCI dataset's documented code
     mapping: 7 = Emergency Room; everything else = non-emergency, with
     ambiguous/unknown codes -- 9, 15, 17, 20, 21 -- reported separately
     rather than silently bucketed either way).

  2. road-safety's Urban_or_Rural_Area: re-confirms group sizes directly
     against the real data (already seen once via find_natural_shift_candidates.py,
     re-checked here for completeness alongside the diabetes_readmission check).

  3. Target-column prevalence within each natural-split group, for both
     datasets -- a sanity check before building the full train/evaluate
     pipeline. If either split group has near-zero representation of a
     target class, D2's stratified-split convention (used within each
     population) would struggle, and that's worth knowing before writing
     the pipeline, not after.

Usage:
    python -m src.scripts.check_natural_split_details
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/workspace")  # for local dev, override with env var if needed
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

ROAD_SAFETY_PATH = PROCESSED_DIR / "openml" / "road-safety.parquet"
DIABETES_PATH = PROCESSED_DIR / "uci" / "diabetes_readmission.parquet"

ROAD_SAFETY_TARGET = "Sex_of_Driver"  # matches TARGET_COLUMNS used throughout D3's runners
DIABETES_TARGET = "readmitted"

# Standard UCI diabetes_readmission admission_source_id code mapping.
# 7 = Emergency Room is the one code that unambiguously means "emergency".
# Everything else is a referral/transfer/delivery-related code (non-emergency
# in the sense D5 cares about: was this patient's entry into care an acute,
# unplanned emergency event or not). Codes 9, 15, 17, 20, 21 are documented
# as "Not Available"/"NULL"/"Not Mapped"/"Unknown" in the UCI codebook --
# reported separately rather than assigned to either bucket, since silently
# putting "Unknown" into "Non-emergency" would misrepresent the split.
EMERGENCY_CODE = 7
AMBIGUOUS_CODES = {9, 15, 17, 20, 21}


def check_diabetes_admission_source():
    print("=" * 70)
    print("diabetes_readmission: admission_source_id")
    print("=" * 70)
    df = pd.read_parquet(DIABETES_PATH)

    print(f"Full value distribution (n={len(df)}):")
    counts = df["admission_source_id"].value_counts(dropna=False).sort_index()
    print(counts.to_string())
    print()

    emergency_mask = df["admission_source_id"] == EMERGENCY_CODE
    ambiguous_mask = df["admission_source_id"].isin(AMBIGUOUS_CODES)
    non_emergency_mask = ~emergency_mask & ~ambiguous_mask

    n_emergency = emergency_mask.sum()
    n_non_emergency = non_emergency_mask.sum()
    n_ambiguous = ambiguous_mask.sum()

    print(f"Emergency (code={EMERGENCY_CODE}):        {n_emergency} rows")
    print(f"Non-emergency (all other defined codes): {n_non_emergency} rows")
    print(f"Ambiguous/unknown (codes {sorted(AMBIGUOUS_CODES)}): {n_ambiguous} rows "
          f"(excluded from both groups)")
    print()
    larger = "Emergency" if n_emergency > n_non_emergency else "Non-emergency"
    print(f"LARGER GROUP (proposed 'train' population per Decision 3): {larger}")
    print()

    if DIABETES_TARGET in df.columns:
        print(f"Target '{DIABETES_TARGET}' prevalence within each group:")
        print(f"  Emergency:     {df.loc[emergency_mask, DIABETES_TARGET].value_counts(normalize=True).to_dict()}")
        print(f"  Non-emergency: {df.loc[non_emergency_mask, DIABETES_TARGET].value_counts(normalize=True).to_dict()}")
    else:
        print(f"!! Target column '{DIABETES_TARGET}' not found in this dataset's columns: {list(df.columns)}")
    print()


def check_road_safety_urban_rural():
    print("=" * 70)
    print("road-safety: Urban_or_Rural_Area")
    print("=" * 70)
    df = pd.read_parquet(ROAD_SAFETY_PATH)

    counts = df["Urban_or_Rural_Area"].value_counts(dropna=False)
    print(f"Full value distribution (n={len(df)}):")
    print(counts.to_string())
    print()

    # Standard road-safety coding: 1 = Urban, 2 = Rural (per the dataset's
    # documented convention) -- confirmed here against real data, not assumed.
    urban_mask = df["Urban_or_Rural_Area"] == 1
    rural_mask = df["Urban_or_Rural_Area"] == 2
    n_urban = urban_mask.sum()
    n_rural = rural_mask.sum()
    larger = "Urban" if n_urban > n_rural else "Rural"
    print(f"Urban (code=1): {n_urban} rows")
    print(f"Rural (code=2): {n_rural} rows")
    print(f"LARGER GROUP (proposed 'train' population per Decision 3): {larger}")
    print()

    if ROAD_SAFETY_TARGET in df.columns:
        print(f"Target '{ROAD_SAFETY_TARGET}' prevalence within each group:")
        print(f"  Urban: {df.loc[urban_mask, ROAD_SAFETY_TARGET].value_counts(normalize=True).to_dict()}")
        print(f"  Rural: {df.loc[rural_mask, ROAD_SAFETY_TARGET].value_counts(normalize=True).to_dict()}")
    else:
        print(f"!! Target column '{ROAD_SAFETY_TARGET}' not found in this dataset's columns: {list(df.columns)}")
    print()


def main():
    if DIABETES_PATH.exists():
        check_diabetes_admission_source()
    else:
        print(f"!! diabetes_readmission not found at {DIABETES_PATH}")

    if ROAD_SAFETY_PATH.exists():
        check_road_safety_urban_rural()
    else:
        print(f"!! road-safety not found at {ROAD_SAFETY_PATH}")


if __name__ == "__main__":
    main()