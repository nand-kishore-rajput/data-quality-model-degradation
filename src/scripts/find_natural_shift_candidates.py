"""
find_natural_shift_candidates.py

D5 (Natural-Shift Validation) needs a genuine, real split within road-safety
and diabetes_readmission -- D1's two named anchor datasets -- that
represents naturally-occurring shift (most plausibly temporal, given both
datasets' known origins: road-safety spans multiple years of UK accident
data; diabetes_readmission spans 1999-2008 across 130 US hospitals).

This script does NOT assume which column is the right split -- it reports
every column in the FULL processed dataset (not just the "eligible numeric"
subset other D3 modules filtered to, which would hide a categorical year/
date/region field even if one exists), flags likely temporal/grouping
candidates by name pattern and cardinality, and reports each candidate's
value distribution so a real decision can be made from actual data, not
assumption.

Usage:
    python -m src.scripts.find_natural_shift_candidates
    python -m src.scripts.find_natural_shift_candidates --datasets road-safety diabetes_readmission
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/workspace")  # for local dev, override with env var if needed
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

# The two D5 anchor datasets D1 names explicitly. Included as a dict so the
# source subdirectory is known without a full corpus scan.
ANCHOR_DATASETS = {
    "road-safety": ("openml", "road-safety.parquet"),
    "diabetes_readmission": ("uci", "diabetes_readmission.parquet"),
}

# Name patterns suggesting a temporal or grouping column worth checking as a
# natural-shift split candidate -- a heuristic filter to highlight likely
# candidates first, not an exhaustive or authoritative classification.
CANDIDATE_KEYWORDS = [
    "year", "date", "time", "month", "day", "period",
    "hospital", "region", "area", "location", "site", "state", "county",
    "police", "force",  # road-safety specific (UK police force area is a
                          # standard field in this dataset's original source)
]


def find_candidates(df: pd.DataFrame) -> list:
    hits = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in CANDIDATE_KEYWORDS):
            hits.append(col)
    return hits


def profile_column(df: pd.DataFrame, col: str):
    s = df[col]
    n_unique = s.nunique(dropna=True)
    print(f"    '{col}': dtype={s.dtype}, n_unique={n_unique}, "
          f"n_missing={s.isna().sum()}")
    if n_unique <= 30:
        print(f"      value_counts:\n" +
              "\n".join(f"        {v}: {c}" for v, c in s.value_counts().head(15).items()))
    else:
        print(f"      range: [{s.min()}, {s.max()}] (too many distinct values to list)")


def main():
    parser = argparse.ArgumentParser(description="Find natural-shift split candidates in D5's anchor datasets.")
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Subset of anchor datasets to check (default: both).")
    args = parser.parse_args()

    targets = args.datasets if args.datasets else list(ANCHOR_DATASETS.keys())

    for name in targets:
        if name not in ANCHOR_DATASETS:
            print(f"!! '{name}' is not a known D5 anchor dataset. "
                  f"Known: {list(ANCHOR_DATASETS.keys())}")
            continue

        source, filename = ANCHOR_DATASETS[name]
        path = PROCESSED_DIR / source / filename
        print(f"{'='*70}")
        print(f"{name}  ({path})")
        print(f"{'='*70}")

        if not path.exists():
            print(f"  !! File not found at {path}")
            continue

        df = pd.read_parquet(path)
        print(f"  rows={len(df)}, columns={len(df.columns)}")
        print(f"  ALL columns: {list(df.columns)}")
        print()

        candidates = find_candidates(df)
        if not candidates:
            print("  No column names matched temporal/grouping keywords. "
                  "A natural split may not be directly available as a labeled "
                  "column -- worth checking whether row order or an index "
                  "carries implicit chronological information, or whether the "
                  "RAW (pre-processing) source file has a field that was "
                  "dropped during processing.")
        else:
            print(f"  Candidate columns (name-matched): {candidates}")
            print()
            for col in candidates:
                profile_column(df, col)
        print()


if __name__ == "__main__":
    main()