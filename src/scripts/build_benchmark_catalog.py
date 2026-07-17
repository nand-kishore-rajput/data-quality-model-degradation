"""
build_benchmark_catalog.py

Builds the unified benchmark catalog from all per-source metadata files,
then enriches it with:
  - Minority-class balance statistics (read from raw parquet files)
  - Raw vs decoded missingness (sentinel-aware for HELOC)
  - Duplicate row counts (natural uniqueness defect measurement)
  - Curation status, effective features, subsample protocol columns

Special cases:
  - 'assistments' target ('correct') is binarized as (correct == 1)
    per the ASSISTments data dictionary before balance is measured.
  - 'diabetes-readmission' target ('readmitted') is 3-class; binarized
    as (<30 days = 1, else = 0) per standard literature convention.
  - 'heloc' (46932) encodes missing values as sentinel integers:
      -9, -8  → true missing (map to NaN for decoded measurement)
      -7      → structural category "condition not met" (NOT missing,
                 retain as a real category; affects only
                 MSinceMostRecentDelq and MSinceMostRecentInqexcl7days)
    missing_percent_raw reflects what naive tools see (0.0 for HELOC).
    missing_percent_decoded reflects true missingness after sentinel
    decoding. The delta is a documented natural data quality defect.

Nothing in this script modifies raw files. All output is catalog metadata only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"
RAW_DIR      = PROJECT_ROOT / "datasets" / "raw"

MINORITY_FLOOR = 10.0   # % — balance inclusion rule

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

# ----------------------------------------------------------------------
# Explicit file-path overrides
# A guessed match is exactly the failure mode that produced the
# seismic-bumps/Seeds mismatch earlier in this project — explicit wins.
# ----------------------------------------------------------------------
FILE_OVERRIDES = {
    ("uci",        "Default of Credit Card Clients"):                          RAW_DIR / "uci"        / "default_credit.parquet",
    ("uci",        "Polish Companies Bankruptcy"):                             RAW_DIR / "uci"        / "polish_bankruptcy.parquet",
    ("uci",        "Diabetes 130-US Hospitals for Years 1999-2008"):           RAW_DIR / "uci"        / "diabetes_readmission.parquet",
    ("tableshift", "assistments"):                                             RAW_DIR / "tableshift" / "assistments.parquet",
    ("cleanml",    "Movie"):                                                   RAW_DIR / "cleanml"    / "Movie_raw.parquet",
    ("cleanml",    "Restaurant"):                                              RAW_DIR / "cleanml"    / "Restaurant_raw.parquet",
}

# ----------------------------------------------------------------------
# Sentinel rules
# Keys: (source, dataset_name)
# Values: dict mapping sentinel int → "nan" | "category"
#   "nan"      → truly missing; map to NaN in decoded pass
#   "category" → encodes a real state; retain as-is
#
# Only affects missing_percent_decoded. Raw files are never modified.
# ----------------------------------------------------------------------
SENTINEL_RULES = {
    ("openml", "heloc"): {
        -9: "nan",
        -8: "nan",
        -7: "category",   # "condition not met" — NOT missingness
    },
}

# ----------------------------------------------------------------------
# Binarization rules (applied before balance measurement)
# ----------------------------------------------------------------------
BINARIZE_RULES = {
    ("tableshift", "assistments"):                                   lambda y: (y == 1).astype(int),
    ("uci",        "Diabetes 130-US Hospitals for Years 1999-2008"): lambda y: (y == "<30").astype(int),
}

# ----------------------------------------------------------------------
# Curation status — document provenance per dataset
# raw                  : pulled directly from original source, no transformation
# near-raw             : minor upstream processing but defects intact
# benchmark-processed  : upstream pipeline removed missingness / balanced classes
# ----------------------------------------------------------------------
CURATION_STATUS = {
    ("openml",     "phoneme"):                                       "raw",
    ("openml",     "bank-marketing"):                                "raw: V1-V16 anonymized columns restored from UCI; V12=duration is leakage variable",
    ("openml",     "MagicTelescope"):                                "raw",
    ("openml",     "churn"):                                         "raw",
    ("openml",     "heloc"):                                         "raw: full FICO original (10,459 rows), sentinels intact, 588 duplicate rows documented",
    ("openml",     "electricity"):                                   "raw",
    ("openml",     "jm1"):                                           "raw: published label-noise and data-quality defects documented in Shepperd et al. 2013",
    ("openml",     "road-safety"):                                   "raw: DfT GB 2015 accident records; target filtered to Male/Female (dropped 19,824 Unknown); single year only",
    ("openml",     "kick"):                                          "raw",
    ("uci",        "Default of Credit Card Clients"):                "raw",
    ("uci",        "Online Shoppers Purchasing Intention Dataset"):   "raw",
    ("uci",        "Diabetes 130-US Hospitals for Years 1999-2008"): "raw: UCI 296; target binarized <30=1 vs rest=0; weight 96.86% missing is a documented natural defect",
    ("folktables", "ACSIncome"):                                     "raw: state/year-stratified subsample applied to preserve natural shift structure",
    ("folktables", "ACSPublicCoverage"):                             "raw: state/year-stratified subsample applied to preserve natural shift structure",
    ("cleanml",    "Airbnb"):                                        "raw",
    ("cleanml",    "EEG"):                                           "raw",
    ("cleanml",    "Movie"):                                         "raw: source data only; ML target (Rating) constructed by CleanML pipeline, not present in raw file",
    ("cleanml",    "Restaurant"):                                    "raw: source data only; ML target (is_match) constructed by CleanML pipeline; entity-matching task excluded",
    ("tableshift", "assistments"):                                   "raw: target binarized (correct==1) per ASSISTments data dictionary",
}

# ----------------------------------------------------------------------
# Effective feature count — post-leakage-drop schema for modeling time
# Distinct from feature_count which reflects the raw file.
# ----------------------------------------------------------------------
EFFECTIVE_FEATURES = {
    ("openml", "bank-marketing"): 15,   # 16 raw minus V12 (duration, leakage)
}

# ----------------------------------------------------------------------
# Subsample protocols — required for datasets exceeding the row cap.
# Document seed and strategy so inclusion criteria remain reproducible.
# ----------------------------------------------------------------------
SUBSAMPLE_PROTOCOLS = {
    ("openml",     "road-safety"):       "random (seed=42), 50,000 rows from 343,419 filtered; year-stratification not applicable (single year: 2015)",
    ("openml",     "kick"):              "seeded (seed=42) random subsample to 50,000 rows",
    ("folktables", "ACSIncome"):         "state/year-stratified: CA-2018 as training domain; remaining states as shift targets; capped at 50,000 rows per domain",
    ("folktables", "ACSPublicCoverage"): "state/year-stratified: CA-2018 as training domain; remaining states as shift targets; capped at 50,000 rows per domain",
    ("tableshift", "assistments"):       "seeded (seed=42) random subsample to 25,000 rows; real-shift anchor status retained",
}

# ----------------------------------------------------------------------
# Selection decisions
# True  = passes all criteria, included in corpus
# False = fails a criterion (see reason inline)
# None  = pending (re-pull outstanding or decision not yet made)
#
# NOTE: AUC screening column is deliberately NOT pre-populated here.
# AUC gate runs AFTER this script on the finalized corpus.
# ----------------------------------------------------------------------
IS_SELECTED = {
    # --- confirmed selected (17) ---
    ("openml",     "phoneme"):                                       True,
    ("openml",     "bank-marketing"):                                True,
    ("openml",     "MagicTelescope"):                                True,
    ("openml",     "churn"):                                         True,
    ("openml",     "heloc"):                                         True,
    ("openml",     "electricity"):                                   True,
    ("openml",     "jm1"):                                           True,
    ("openml",     "road-safety"):                                   True,
    ("openml",     "kick"):                                          True,
    ("uci",        "Default of Credit Card Clients"):                True,
    ("uci",        "Online Shoppers Purchasing Intention Dataset"):   True,
    ("uci",        "Diabetes 130-US Hospitals for Years 1999-2008"): True,
    ("folktables", "ACSIncome"):                                     True,
    ("folktables", "ACSPublicCoverage"):                             True,
    ("cleanml",    "Airbnb"):                                        True,
    ("cleanml",    "EEG"):                                           True,
    ("tableshift", "assistments"):                                   True,

    # --- failed minority floor (< 10%) ---
    ("openml", "coil2000"):                        False,  # 5.97% minority
    ("openml", "seismic-bumps"):                   False,  # 6.58% minority
    ("uci",    "Taiwanese Bankruptcy Prediction"): False,  # 3.23% minority
    ("uci",    "Polish Companies Bankruptcy"):     False,  # 4.82% minority
    ("cleanml","Credit"):                          False,  # 6.68% minority

    # --- disqualified: structural / leakage / provenance failures ---
    ("openml",  "house_16H"):    False,  # non-domain-diverse; excluded
    ("cleanml", "Movie"):        False,  # target col 'Rating' absent from raw parquet; constructed by CleanML pipeline
    ("cleanml", "Restaurant"):   False,  # target col 'is_match' absent; entity-matching task, not standard classification
    # KDD Cup 1999: ~75% duplicate rows (Tavallaee et al. 2009), outcome-leakage
    # features (root_shell, su_attempted, num_compromised), synthetic network
    # generation, trivially separable classes (99%+ AUC by construction).
    # Not in metadata files — logged here for audit trail only.

    # --- deferred / pending explicit decision ---
    # compas-two-years, Insurance, rain-in-australia: skipped for now
}


# ----------------------------------------------------------------------
# Step 1: build unified catalog from per-source metadata files
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
# Step 2: path resolution (explicit override → fuzzy match)
# ----------------------------------------------------------------------
def _norm(s: str) -> str:
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_path(source, dataset_name) -> Path | None:
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


# ----------------------------------------------------------------------
# Step 3: per-row enrichment — balance, missingness, duplicates
# ----------------------------------------------------------------------
def compute_row_stats(row) -> dict:
    result = {
        "minority_percent":        pd.NA,
        "n_classes":               pd.NA,
        "majority_percent":        pd.NA,
        "balance_note":            pd.NA,
        "missing_cells_raw":       pd.NA,
        "missing_percent_raw":     pd.NA,
        "missing_cells_decoded":   pd.NA,
        "missing_percent_decoded": pd.NA,
        "sentinel_note":           "NO_SENTINELS",
        "duplicate_rows":          pd.NA,
    }

    target = row["target"]
    if pd.isna(target):
        result["balance_note"] = "NO_TARGET_IN_CATALOG"
        return result

    path = resolve_path(row["source"], row["dataset_name"])
    if path is None:
        result["balance_note"] = "FILE_NOT_FOUND"
        return result

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        result["balance_note"] = f"READ_ERROR:{type(e).__name__}"
        return result

    # --- duplicate rows (excluding target) --------------------------------
    feature_cols = [c for c in df.columns if c != target]
    try:
        result["duplicate_rows"] = int(df[feature_cols].duplicated().sum())
    except Exception:
        result["duplicate_rows"] = pd.NA

    # --- raw missingness --------------------------------------------------
    total_cells = df.shape[0] * df.shape[1]
    raw_missing = int(df.isna().sum().sum())
    result["missing_cells_raw"]   = raw_missing
    result["missing_percent_raw"] = round(100 * raw_missing / total_cells, 6) if total_cells else pd.NA

    # --- decoded missingness (sentinel-aware) -----------------------------
    key = (str(row["source"]), str(row["dataset_name"]))
    sentinel_rules = SENTINEL_RULES.get(key)

    if sentinel_rules:
        df_decoded = df.copy()
        nan_sentinels = [v for v, action in sentinel_rules.items() if action == "nan"]
        cat_sentinels = [v for v, action in sentinel_rules.items() if action == "category"]

        if nan_sentinels:
            num_cols = df_decoded.select_dtypes(include=[np.number]).columns
            df_decoded[num_cols] = df_decoded[num_cols].replace(nan_sentinels, np.nan)

        decoded_missing = int(df_decoded.isna().sum().sum())
        result["missing_cells_decoded"]   = decoded_missing
        result["missing_percent_decoded"] = round(100 * decoded_missing / total_cells, 6) if total_cells else pd.NA
        result["sentinel_note"] = (
            f"nan_sentinels={nan_sentinels}; "
            f"category_sentinels={cat_sentinels}; "
            f"delta_cells={decoded_missing - raw_missing}"
        )
    else:
        result["missing_cells_decoded"]   = raw_missing
        result["missing_percent_decoded"] = result["missing_percent_raw"]
        result["sentinel_note"]           = "NO_SENTINELS"

    # --- balance ----------------------------------------------------------
    if target not in df.columns:
        result["balance_note"] = "TARGET_COL_NOT_IN_FILE"
        return result

    y = df[target].dropna()
    if len(y) == 0:
        result["balance_note"] = "TARGET_ALL_NULL"
        return result

    note_suffix = ""
    if key in BINARIZE_RULES:
        y = BINARIZE_RULES[key](y)
        note_suffix = "_BINARIZED"

    counts    = y.value_counts()
    total     = counts.sum()
    minority  = round(100 * counts.min() / total, 2)
    majority  = round(100 * counts.max() / total, 2)
    n_classes = len(counts)

    result["minority_percent"] = minority
    result["n_classes"]        = n_classes
    result["majority_percent"] = majority
    result["balance_note"]     = (
        "OK" if n_classes == 2 else "MULTICLASS_NEEDS_COLLAPSE_RULE"
    ) + note_suffix

    return result


def enrich(catalog: pd.DataFrame) -> pd.DataFrame:
    stats = catalog.apply(compute_row_stats, axis=1, result_type="expand")
    catalog = pd.concat([catalog, stats], axis=1)

    catalog["minority_pass"] = catalog.apply(
        lambda r: pd.NA if pd.isna(r["minority_percent"]) or r["n_classes"] != 2
                  else r["minority_percent"] >= MINORITY_FLOOR,
        axis=1,
    )

    catalog["curation_status"] = catalog.apply(
        lambda r: CURATION_STATUS.get((str(r["source"]), str(r["dataset_name"])), "unspecified"),
        axis=1,
    )

    catalog["effective_features"] = catalog.apply(
        lambda r: EFFECTIVE_FEATURES.get(
            (str(r["source"]), str(r["dataset_name"])),
            r["feature_count"]
        ),
        axis=1,
    )

    catalog["subsample_protocol"] = catalog.apply(
        lambda r: SUBSAMPLE_PROTOCOLS.get(
            (str(r["source"]), str(r["dataset_name"])),
            "none_required"
        ),
        axis=1,
    )

    catalog["is_selected"] = catalog.apply(
        lambda r: IS_SELECTED.get(
            (str(r["source"]), str(r["dataset_name"])),
            None
        ),
        axis=1,
    )

    return catalog


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    catalog = build_catalog()
    catalog = enrich(catalog)

    catalog = catalog.drop(columns=["missing_percent"], errors="ignore")

    out_path = METADATA_DIR / "benchmark_catalog.csv"
    catalog.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"Rows  : {len(catalog)}")

    display_cols = [
        "source", "dataset_name",
        "missing_percent_raw", "missing_percent_decoded", "sentinel_note",
        "duplicate_rows",
        "minority_percent", "n_classes", "minority_pass", "balance_note",
        "curation_status", "effective_features", "subsample_protocol",
        "is_selected",
    ]
    print("\n--- Catalog summary ---")
    print(catalog[[c for c in display_cols if c in catalog.columns]].to_string(index=False))

    sentinel_rows = catalog[catalog["sentinel_note"] != "NO_SENTINELS"]
    if len(sentinel_rows):
        print(f"\n--- Sentinel-decoded datasets ({len(sentinel_rows)}) ---")
        print(sentinel_rows[[
            "dataset_name", "missing_percent_raw",
            "missing_percent_decoded", "sentinel_note", "duplicate_rows",
        ]].to_string(index=False))

    unresolved = catalog[~catalog["balance_note"].astype(str).str.startswith("OK", na=False)]
    if len(unresolved):
        print(f"\n--- {len(unresolved)} unresolved rows ---")
        print(unresolved[["source", "dataset_name", "balance_note"]].to_string(index=False))

    needs_subsample = catalog[catalog["subsample_protocol"] != "none_required"]
    if len(needs_subsample):
        print(f"\n--- {len(needs_subsample)} datasets require subsampling before modeling ---")
        print(needs_subsample[["source", "dataset_name", "subsample_protocol"]].to_string(index=False))

    selected = catalog[catalog["is_selected"] == True]
    excluded = catalog[catalog["is_selected"] == False]
    pending  = catalog[catalog["is_selected"].isna()]
    print(f"\n--- Selection summary ---")
    print(f"  Selected : {len(selected)}")
    print(f"  Excluded : {len(excluded)}")
    print(f"  Pending  : {len(pending)}")
    if len(pending):
        print("  Pending datasets:")
        print(pending[["source", "dataset_name"]].to_string(index=False))


if __name__ == "__main__":
    main()