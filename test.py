import pandas as pd
from pathlib import Path

METADATA_DIR = Path("datasets/metadata")
RAW_DIR = Path("datasets/raw")       # adjust to Path("data-quality-model-degradation/datasets/raw")
                                      # if this script runs from one level above the project root
MINORITY_FLOOR = 10.0                 # % — balance rule; set to 5.0 to test the looser floor

files = [
    "openml_metadata.csv",
    "uci_metadata.csv",
    "folktables_metadata.csv",
    "cleanml_metadata.csv",
    "tableshift_metadata.csv",
]

canonical_columns = [
    "source",
    "dataset_id",
    "dataset_name",
    "rows",
    "columns",
    "feature_count",
    "target",
    "missing_cells",
    "missing_percent",
]

catalog = []
for file in files:
    df = pd.read_csv(METADATA_DIR / file)
    for col in canonical_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[canonical_columns]
    catalog.append(df)

benchmark_catalog = pd.concat(catalog, ignore_index=True)

# drop fully-empty trailing rows some source CSVs leave behind
benchmark_catalog = benchmark_catalog.dropna(how="all").reset_index(drop=True)


# ----------------------------------------------------------------------
# Minority-class enrichment
# ----------------------------------------------------------------------
# Metadata files don't carry class balance, so we open each raw file and
# read ONLY the designated target column (never the last column — that
# was the earlier bug). min(class_count)/total is invariant to label
# recoding (1/2 vs 0/1 vs yes/no), so no recoding is needed here.

# Explicit overrides for rows where the catalog name doesn't match the
# filename, or the file lives outside its nominal source directory.
# These take priority over fuzzy matching — an explicit path is always
# safer than a guess, which is exactly the failure mode that produced
# the seismic-bumps/Seeds mismatch earlier in this project.
FILE_OVERRIDES = {
    ("uci", "Default of Credit Card Clients"): RAW_DIR / "uci" / "default_credit.parquet",
    ("uci", "Polish Companies Bankruptcy"):    RAW_DIR / "uci" / "polish_bankruptcy.parquet",
    ("tableshift", "assistments"):             RAW_DIR / "tableshift" / "assistments.parquet",
}

def _norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")

def resolve_path(source, dataset_name):
    """Find the raw file for a catalog row. Checks explicit overrides
    first, then falls back to normalized fuzzy matching within the
    source's raw subdirectory."""
    override = FILE_OVERRIDES.get((str(source), str(dataset_name)))
    if override is not None:
        return override if override.exists() else None

    src_dir = RAW_DIR / str(source)
    if not src_dir.exists():
        return None

    want = _norm(dataset_name)
    candidates = list(src_dir.glob("*.parquet"))

    # exact match first (with/without a trailing 'raw' suffix)
    for c in candidates:
        stem = _norm(c.stem)
        if want == stem or want == stem.removesuffix("raw"):
            return c

    # then containment, longest stem match wins to avoid greedy false hits
    hits = [
        c for c in candidates
        if want in _norm(c.stem) or _norm(c.stem).removesuffix("raw") in want
    ]
    return max(hits, key=lambda c: len(_norm(c.stem))) if hits else None


def target_stats(row):
    """Return (minority_percent, n_classes, majority_percent, note)."""
    target = row["target"]
    if pd.isna(target):
        return pd.NA, pd.NA, pd.NA, "NO_TARGET_IN_CATALOG"

    path = resolve_path(row["source"], row["dataset_name"])
    if path is None:
        return pd.NA, pd.NA, pd.NA, "FILE_NOT_FOUND"

    try:
        # read only the target column — cheap even for the 6M-row files
        y = pd.read_parquet(path, columns=[target])[target].dropna()
    except Exception as e:
        return pd.NA, pd.NA, pd.NA, f"READ_ERROR:{type(e).__name__}"

    if len(y) == 0:
        return pd.NA, pd.NA, pd.NA, "TARGET_ALL_NULL"

    counts = y.value_counts()
    total = counts.sum()
    minority = round(100 * counts.min() / total, 2)
    majority = round(100 * counts.max() / total, 2)
    n_classes = len(counts)
    note = "OK" if n_classes == 2 else "MULTICLASS_NEEDS_COLLAPSE_RULE"
    return minority, n_classes, majority, note


stats = benchmark_catalog.apply(target_stats, axis=1, result_type="expand")
stats.columns = ["minority_percent", "n_classes", "majority_percent", "balance_note"]
benchmark_catalog = pd.concat([benchmark_catalog, stats], axis=1)

# pass/fail against the floor — only meaningful for genuinely binary targets
benchmark_catalog["minority_pass"] = benchmark_catalog.apply(
    lambda r: pd.NA if pd.isna(r["minority_percent"]) or r["n_classes"] != 2
              else r["minority_percent"] >= MINORITY_FLOOR,
    axis=1,
)

benchmark_catalog.to_csv(METADATA_DIR / "benchmark_catalog.csv", index=False)

print("Saved benchmark_catalog.csv")
print("Rows:", len(benchmark_catalog))
print(f"\nBalance summary (floor = {MINORITY_FLOOR:.0f}%):")
print(benchmark_catalog[
    ["source", "dataset_name", "target", "minority_percent",
     "n_classes", "minority_pass", "balance_note"]
].to_string(index=False))

# quick visibility into anything still unresolved
unresolved = benchmark_catalog[benchmark_catalog["balance_note"] != "OK"]
if len(unresolved):
    print(f"\n{len(unresolved)} rows still unresolved — check balance_note:")
    print(unresolved[["source", "dataset_name", "balance_note"]].to_string(index=False))