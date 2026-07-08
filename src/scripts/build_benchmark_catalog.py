"""
build_benchmark_catalog.py

Builds the unified benchmark catalog from all per-source metadata files,
then enriches it with measured minority-class statistics read directly
from the raw parquet files (metadata alone doesn't carry class balance).

Special case: 'assistments' target ('correct') is not binary — it stores
partial credit. Per the ASSISTments data dictionary, it must be binarized
as (correct == 1) before balance can be measured. This script applies
that rule automatically for that one dataset; every other dataset is
read as-is from its designated target column.
"""

import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../data-quality-model-degradation
METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"

MINORITY_FLOOR = 10.0   # % — balance rule; set to 5.0 to test the looser floor

METADATA_FILES = [
    "openml_metadata.csv",
    "uci_metadata.csv",
    "folktables_metadata.csv",
    "cleanml_metadata.csv",
    "tableshift_metadata.csv",
]

CANONICAL_COLUMNS = [
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

# Explicit file-path overrides for rows where the catalog name doesn't
# match the filename, or the file lives outside its nominal source dir.
# Explicit path always wins over fuzzy matching — a guessed match is
# exactly the failure mode that produced the seismic-bumps/Seeds mismatch
# earlier in this project.
FILE_OVERRIDES = {
    ("uci", "Default of Credit Card Clients"): RAW_DIR / "uci" / "default_credit.parquet",
    ("uci", "Polish Companies Bankruptcy"):    RAW_DIR / "uci" / "polish_bankruptcy.parquet",
    ("tableshift", "assistments"):             RAW_DIR / "tableshift" / "assistments.parquet",
}

# Datasets whose target needs a transform before balance is meaningful.
# Currently only assistments: 1 = correct, anything else = incorrect
# (per official ASSISTments documentation).
BINARIZE_RULES = {
    ("tableshift", "assistments"): lambda y: (y == 1).astype(int),
}


# ----------------------------------------------------------------------
# Step 1: build the unified catalog from per-source metadata files
# ----------------------------------------------------------------------
def build_catalog() -> pd.DataFrame:
    frames = []
    for file in METADATA_FILES:
        df = pd.read_csv(METADATA_DIR / file)
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        frames.append(df[CANONICAL_COLUMNS])

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.dropna(how="all").reset_index(drop=True)
    return catalog


# ----------------------------------------------------------------------
# Step 2: minority-class enrichment (reads raw parquet files)
# ----------------------------------------------------------------------
def _norm(s: str) -> str:
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_path(source, dataset_name) -> Path | None:
    """Find the raw file for a catalog row: explicit override first,
    then normalized fuzzy match within the source's raw subdirectory."""
    override = FILE_OVERRIDES.get((str(source), str(dataset_name)))
    if override is not None:
        return override if override.exists() else None

    src_dir = RAW_DIR / str(source)
    if not src_dir.exists():
        return None

    want = _norm(dataset_name)
    candidates = list(src_dir.glob("*.parquet"))

    for c in candidates:
        stem = _norm(c.stem)
        if want == stem or want == stem.removesuffix("raw"):
            return c

    hits = [
        c for c in candidates
        if want in _norm(c.stem) or _norm(c.stem).removesuffix("raw") in want
    ]
    return max(hits, key=lambda c: len(_norm(c.stem))) if hits else None


def target_stats(row) -> tuple:
    """Return (minority_percent, n_classes, majority_percent, note)."""
    target = row["target"]
    if pd.isna(target):
        return pd.NA, pd.NA, pd.NA, "NO_TARGET_IN_CATALOG"

    path = resolve_path(row["source"], row["dataset_name"])
    if path is None:
        return pd.NA, pd.NA, pd.NA, "FILE_NOT_FOUND"

    try:
        y = pd.read_parquet(path, columns=[target])[target].dropna()
    except Exception as e:
        return pd.NA, pd.NA, pd.NA, f"READ_ERROR:{type(e).__name__}"

    if len(y) == 0:
        return pd.NA, pd.NA, pd.NA, "TARGET_ALL_NULL"

    # apply a binarization rule if this dataset needs one, BEFORE counting
    key = (str(row["source"]), str(row["dataset_name"]))
    note_suffix = ""
    if key in BINARIZE_RULES:
        y = BINARIZE_RULES[key](y)
        note_suffix = "_BINARIZED"

    counts = y.value_counts()
    total = counts.sum()
    minority = round(100 * counts.min() / total, 2)
    majority = round(100 * counts.max() / total, 2)
    n_classes = len(counts)

    note = ("OK" if n_classes == 2 else "MULTICLASS_NEEDS_COLLAPSE_RULE") + note_suffix
    return minority, n_classes, majority, note


def enrich_with_balance(catalog: pd.DataFrame) -> pd.DataFrame:
    stats = catalog.apply(target_stats, axis=1, result_type="expand")
    stats.columns = ["minority_percent", "n_classes", "majority_percent", "balance_note"]
    catalog = pd.concat([catalog, stats], axis=1)

    catalog["minority_pass"] = catalog.apply(
        lambda r: pd.NA if pd.isna(r["minority_percent"]) or r["n_classes"] != 2
                  else r["minority_percent"] >= MINORITY_FLOOR,
        axis=1,
    )
    return catalog


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    catalog = build_catalog()
    catalog = enrich_with_balance(catalog)

    out_path = METADATA_DIR / "benchmark_catalog.csv"
    catalog.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    print("Rows:", len(catalog))
    print(f"\nBalance summary (floor = {MINORITY_FLOOR:.0f}%):")
    print(catalog[
        ["source", "dataset_name", "target", "minority_percent",
         "n_classes", "minority_pass", "balance_note"]
    ].to_string(index=False))

    unresolved = catalog[~catalog["balance_note"].str.startswith("OK", na=False)]
    if len(unresolved):
        print(f"\n{len(unresolved)} rows still unresolved — check balance_note:")
        print(unresolved[["source", "dataset_name", "balance_note"]].to_string(index=False))


if __name__ == "__main__":
    main()