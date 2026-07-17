"""
preprocess_to_processed.py

Raw -> Processed preprocessing for all 17 selected corpus datasets.

This script is the canonical, reproducible bridge between datasets/raw/ (untouched,
exactly as pulled from source) and datasets/processed/ (what D3's corruption engine
and all downstream Phase 1+ work actually read from).

Design principles (per D1 §12 reproducibility requirements and project-wide
canonical-source discipline):
  - datasets/raw/ is NEVER written to. Read-only, always.
  - Every transform applied here is one already DOCUMENTED in the Phase 0 catalog
    and in run_auc_screening.py's DATASETS registry — this script does not invent
    new preprocessing decisions, it operationalizes existing ones into a reusable file.
  - Every step is logged to a manifest (row counts before/after, columns dropped,
    transforms applied) so datasets/processed/ can always be regenerated and audited,
    not just trusted.
  - No label-encoding here. AUC screening needed numeric-only input for XGBoost;
    D3's corruption engine operates on domain-level values (categories, raw numerics),
    so categorical columns are left as-is. Label-encoding, if needed, is a D3-time
    concern, not a preprocessing-time one.
  - Fixed seed=42 throughout, matching run_auc_screening.py and D1 §3/§12.

For each dataset:
  1. Load raw parquet (read-only)
  2. Drop leakage / ID / temporal columns (per registry, same as AUC screening)
  3. Drop any remaining 100%-constant columns (safety net, logged)
  4. Filter target (e.g. road-safety: drop Unknown) and binarize/remap target
  5. Subsample if over row cap (seed=42)
  6. Write to datasets/processed/<source>/<name>.parquet
  7. Re-verify binary label status on the OUTPUT file (closes the loop on the
     original spot-check finding)

Output:
  datasets/processed/<source>/<name>.parquet   — one file per dataset
  datasets/metadata/preprocessing_manifest.csv — full audit log, one row per dataset
  datasets/metadata/preprocessing_run.log      — live progress log

Nothing is written to datasets/raw/. Read-only there, always.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT   = Path("/workspace")
RAW_DIR        = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR  = PROJECT_ROOT / "datasets" / "processed"
OUT_MANIFEST   = PROJECT_ROOT / "datasets" / "metadata" / "preprocessing_manifest.csv"
OUT_LOG        = PROJECT_ROOT / "datasets" / "metadata" / "preprocessing_run.log"

SEED = 42

# ----------------------------------------------------------------------
# Live logger — writes to both stdout and log file, flushed immediately
# ----------------------------------------------------------------------
_log_file = None

def log(msg="", end="\n"):
    print(msg, end=end, flush=True)
    if _log_file:
        _log_file.write(msg + end)
        _log_file.flush()

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ----------------------------------------------------------------------
# Dataset registry — identical source-of-truth to run_auc_screening.py's
# DATASETS list. Kept in sync deliberately: the same transforms that were
# used to produce the AUC screening numbers in the Phase 0 catalog must be
# the ones baked into datasets/processed/, or the two pipelines silently
# diverge and the catalog's documented AUCs stop being reproducible.
#
# label-encoding related fields from the AUC-screening registry are NOT
# used here — see module docstring.
# ----------------------------------------------------------------------
DATASETS = [
    # ------------------------------------------------------------------
    # OPENML
    # ------------------------------------------------------------------
    {
        "name":          "phoneme",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "phoneme.parquet",
        "target":        "Class",
        "target_map":    {1: 0, 2: 1},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "Target remap 1->0, 2->1 (minority=class 2, 29.35%).",
    },
    {
        "name":          "bank-marketing",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "bank_marketing.parquet",
        "target":        "Class",
        "target_map":    {1: 0, 2: 1},
        "binarize":      None,
        "drop_cols":     ["V12"],   # duration — post-call outcome, direct leakage
        "filter_target": None,
        "subsample":     None,
        "notes":         "Drop V12 (duration leakage). Target remap 1->0, 2->1.",
    },
    {
        "name":          "MagicTelescope",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "magic_telescope.parquet",
        "target":        "class:",
        "target_map":    {"g": 0, "h": 1},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "Target string g/h -> 0/1.",
    },
    {
        "name":          "churn",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "churn.parquet",
        "target":        "class",
        "target_map":    None,
        "binarize":      None,
        "drop_cols":     [
            "phone_number",
            "total_day_charge", "total_eve_charge",
            "total_night_charge", "total_intl_charge",
        ],
        "filter_target": None,
        "subsample":     None,
        "notes":         (
            "Drop phone_number (ID) and 4 charge cols (perfect linear duplicates "
            "of minutes cols, r=1.000, audit 2026-07-12)."
        ),
    },
    {
        "name":          "heloc",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "heloc.parquet",
        "target":        "RiskPerformance",
        "target_map":    {"Bad": 1, "Good": 0},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         (
            "Target Bad->1, Good->0. Sentinels -9/-8/-7 left intact (raw) — full "
            "sentinel decode is a D3-time concern, not preprocessing-time. "
            "588 documented duplicate rows retained."
        ),
    },
    {
        "name":          "electricity",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "electricity.parquet",
        "target":        "class",
        "target_map":    {"DOWN": 0, "UP": 1},
        "binarize":      None,
        "drop_cols":     ["date"],
        "filter_target": None,
        "subsample":     None,
        "notes":         "Drop date (temporal ID). Target DOWN->0, UP->1.",
    },
    {
        "name":          "jm1",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "jm1.parquet",
        "target":        "defects",
        "target_map":    {False: 0, True: 1},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "Target bool False->0, True->1. 2,061 duplicate rows retained (documented uniqueness defect).",
    },
    {
        "name":          "road-safety",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "road_safety.parquet",
        "target":        "Sex_of_Driver",
        "target_map":    {"1.0": 0, "2.0": 1},
        "binarize":      None,
        "drop_cols":     ["Accident_Index", "LSOA_of_Accident_Location"],
        "filter_target": ["1.0", "2.0"],   # drop 3.0=Unknown (19,824 rows)
        "subsample":     50_000,
        "notes":         (
            "Filter target to Male/Female (drops 19,824 Unknown/3.0 — resolves the "
            "spot-check FAIL). Subsample 343,419->50,000 (seed=42) AFTER filtering."
        ),
    },
    {
        "name":          "kick",
        "source":        "openml",
        "path":          RAW_DIR / "openml" / "kick.parquet",
        "target":        "IsBadBuy",
        "target_map":    None,
        "binarize":      None,
        "drop_cols":     [],   # PurchDate dropped at runtime if present
        "filter_target": None,
        "subsample":     50_000,
        "notes":         "Subsample 72,983->50,000 (seed=42). PurchDate dropped at runtime (temporal).",
    },

    # ------------------------------------------------------------------
    # UCI
    # ------------------------------------------------------------------
    {
        "name":          "default_credit",
        "source":        "uci",
        "path":          RAW_DIR / "uci" / "default_credit.parquet",
        "target":        "Y",
        "target_map":    None,
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "No transforms needed. Cleanest dataset in corpus.",
    },
    {
        "name":          "online_shoppers",
        "source":        "uci",
        "path":          RAW_DIR / "uci" / "online_shoppers.parquet",
        "target":        "Revenue",
        "target_map":    {False: 0, True: 1},
        "binarize":      None,
        "drop_cols":     ["PageValues"],
        "filter_target": None,
        "subsample":     None,
        "notes":         (
            "Drop PageValues (soft leakage: single-feature AUC=0.8633, audit 2026-07-12). "
            "Target bool False->0, True->1."
        ),
    },
    {
        "name":          "diabetes_readmission",
        "source":        "uci",
        "path":          RAW_DIR / "uci" / "diabetes_readmission.parquet",
        "target":        "readmitted",
        "target_map":    None,
        "binarize":      lambda y: (y == "<30").astype(int),
        "drop_cols":     [
            "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
            "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone",
        ],
        "filter_target": None,
        "subsample":     50_000,
        "notes":         (
            "Binarize: <30 days readmission=1, else=0 (resolves the spot-check FAIL — "
            "raw file has 3 label values NO/>30/<30). Drop 15 near-constant drug cols. "
            "Subsample 101,766->50,000 (seed=42) AFTER binarization."
        ),
    },

    # ------------------------------------------------------------------
    # FOLKTABLES
    # ------------------------------------------------------------------
    {
        "name":          "ACSIncome",
        "source":        "folktables",
        "path":          RAW_DIR / "folktables" / "ACSIncome.parquet",
        "target":        "target",
        "target_map":    {False: 0, True: 1},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     50_000,
        "notes":         "Target bool->int. Subsample 195,665->50,000 (seed=42).",
    },
    {
        "name":          "ACSPublicCoverage",
        "source":        "folktables",
        "path":          RAW_DIR / "folktables" / "ACSPublicCoverage.parquet",
        "target":        "target",
        "target_map":    {False: 0, True: 1},
        "binarize":      None,
        "drop_cols":     ["ST"],   # 100% constant — single state pulled
        "filter_target": None,
        "subsample":     50_000,
        "notes":         "Drop ST (constant). Target bool->int. Subsample 138,554->50,000 (seed=42).",
    },

    # ------------------------------------------------------------------
    # CLEANML
    # ------------------------------------------------------------------
    {
        "name":          "Airbnb",
        "source":        "cleanml",
        "path":          RAW_DIR / "cleanml" / "Airbnb_raw.parquet",
        "target":        "Rating",
        "target_map":    {"N": 0, "Y": 1},
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "Target Y->1 (majority), N->0 (minority 32.31%). 23,766 NaN cells retained.",
    },
    {
        "name":          "EEG",
        "source":        "cleanml",
        "path":          RAW_DIR / "cleanml" / "EEG_raw.parquet",
        "target":        "Eye",
        "target_map":    None,
        "binarize":      None,
        "drop_cols":     [],
        "filter_target": None,
        "subsample":     None,
        "notes":         "No transforms needed. Temporal signal — blocked CV required in Phase 1 (not this script's concern).",
    },

    # ------------------------------------------------------------------
    # TABLESHIFT
    # ------------------------------------------------------------------
    {
        "name":          "assistments",
        "source":        "tableshift",
        "path":          RAW_DIR / "tableshift" / "assistments.parquet",
        "target":        "correct",
        "target_map":    None,
        "binarize":      lambda y: (y == 1).astype(int),
        "drop_cols":     [
            "hint_count", "bottom_hint", "actions", "attempt_count",
            "ms_first_response", "overlap_time", "answer_id", "answer_text",
            "first_action", "end_time",
            "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
            "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
            "user_id", "problem_id", "template_id", "assistment_id",
            "assignment_id", "sequence_id", "base_sequence_id",
            "problem_log_id", "problemlogid",
            "start_time",
            "tutor_mode",
        ],
        "filter_target": None,
        "subsample":     25_000,   # real-shift anchor — 25k cap, not 50k
        "notes":         (
            "Binarize: correct==1->1, else->0 (resolves spot-check FAIL — raw file "
            "has 13 fractional partial-credit values). Drop 25 leakage/ID/constant "
            "cols, leaving 9 features. Subsample 6,123,270->25,000 (seed=42) AFTER "
            "binarization."
        ),
    },
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def drop_existing(df, cols):
    """Drop columns that exist in df — silently skip absent ones. Returns (df, dropped_list)."""
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=to_drop), to_drop


def apply_target(df, target, target_map, binarize, filter_target):
    """Filter rows, binarize/remap the target. Returns (df_with_target, rows_before, rows_after_filter)."""
    rows_before = len(df)

    if filter_target is not None:
        df = df[df[target].astype(str).isin([str(v) for v in filter_target])].copy()
    rows_after_filter = len(df)

    y = df[target].copy()

    if binarize is not None:
        y = binarize(y)
    elif target_map is not None:
        str_map = {str(k): v for k, v in target_map.items()}
        for k, v in list(str_map.items()):
            try:
                str_map[str(int(float(k)))] = v
                str_map[str(float(k))] = v
            except (ValueError, OverflowError):
                pass
        y = y.astype(str).map(str_map)

    y = y.astype(int)
    df = df.copy()
    df[target] = y
    return df, rows_before, rows_after_filter


def verify_binary(df, target):
    """Post-write check: confirm the target column in the OUTPUT file is binary."""
    n_classes = df[target].nunique(dropna=True)
    counts = df[target].value_counts().to_dict()
    return n_classes, counts


# ----------------------------------------------------------------------
# Per-dataset preprocessing
# ----------------------------------------------------------------------
def preprocess_dataset(ds, idx, total):
    name = ds["name"]

    log()
    log(f"{'─'*70}")
    log(f"[{ts()}] [{idx}/{total}] {name}  (source={ds['source']})")
    log(f"{'─'*70}")

    manifest_row = {
        "dataset":               name,
        "source":                ds["source"],
        "raw_path":              str(ds["path"]),
        "processed_path":        None,
        "raw_rows":              None,
        "raw_cols":              None,
        "leakage_cols_dropped":  None,
        "constant_cols_dropped": None,
        "rows_after_filter":     None,
        "rows_after_subsample":  None,
        "final_rows":            None,
        "final_cols":            None,
        "target_col":            ds["target"],
        "n_classes_output":      None,
        "class_counts_output":   None,
        "binary_check":          None,
        "subsample_applied":     ds["subsample"] is not None,
        "seed":                  SEED,
        "status":                None,
        "notes":                 ds["notes"],
    }

    raw_path = Path(ds["path"])
    if not raw_path.exists():
        log(f"  !! RAW FILE NOT FOUND: {raw_path}")
        manifest_row["status"] = "RAW_FILE_NOT_FOUND"
        return manifest_row

    # 1. Load raw (read-only)
    log(f"  [{ts()}] Loading raw {raw_path.name} ...", end=" ")
    try:
        df = pd.read_parquet(raw_path)
    except Exception as e:
        log(f"READ ERROR: {e}")
        manifest_row["status"] = f"READ_ERROR: {e}"
        return manifest_row
    manifest_row["raw_rows"] = len(df)
    manifest_row["raw_cols"] = df.shape[1]
    log(f"{df.shape[0]:,} rows x {df.shape[1]} cols")

    # 2. Drop leakage/ID/temporal columns per registry
    df, leakage_dropped = drop_existing(df, ds["drop_cols"])
    if name == "kick" and "PurchDate" in df.columns:
        df = df.drop(columns=["PurchDate"])
        leakage_dropped.append("PurchDate")
    if name == "churn" and "phone_number" in df.columns:
        df = df.drop(columns=["phone_number"])
        if "phone_number" not in leakage_dropped:
            leakage_dropped.append("phone_number")
    manifest_row["leakage_cols_dropped"] = ";".join(leakage_dropped) if leakage_dropped else ""
    if leakage_dropped:
        log(f"  [{ts()}] Dropped {len(leakage_dropped)} leakage/ID/temporal cols: {leakage_dropped}")

    # 3. Safety net: drop any remaining 100%-constant columns (logged, not silent)
    const_cols = [
        c for c in df.columns
        if c != ds["target"] and df[c].nunique(dropna=False) <= 1
    ]
    if const_cols:
        df = df.drop(columns=const_cols)
        log(f"  [{ts()}] Dropped {len(const_cols)} constant cols: {const_cols}")
    manifest_row["constant_cols_dropped"] = ";".join(const_cols) if const_cols else ""

    # 4. Filter + binarize/remap target
    log(f"  [{ts()}] Applying target filter/binarize/remap ...", end=" ")
    try:
        df, rows_before, rows_after_filter = apply_target(
            df, ds["target"], ds["target_map"], ds["binarize"], ds["filter_target"]
        )
    except Exception as e:
        log(f"TARGET ERROR: {e}")
        manifest_row["status"] = f"TARGET_ERROR: {e}"
        return manifest_row
    manifest_row["rows_after_filter"] = rows_after_filter
    log(f"done ({rows_before:,} -> {rows_after_filter:,} rows)")

    # Drop rows where target mapping produced NaN (unmapped values)
    valid = df[ds["target"]].notna()
    n_dropped_na_target = (~valid).sum()
    if n_dropped_na_target:
        log(f"  [{ts()}] Dropping {n_dropped_na_target} rows with unmapped/NaN target")
    df = df[valid].copy()

    # 5. Subsample (seed=42), AFTER filter/binarize per D1/D2-consistent ordering
    cap = ds["subsample"]
    if cap and len(df) > cap:
        log(f"  [{ts()}] Subsampling {len(df):,} -> {cap:,} rows (seed={SEED}) ...", end=" ")
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
        log("done")
    manifest_row["rows_after_subsample"] = len(df)

    manifest_row["final_rows"] = len(df)
    manifest_row["final_cols"] = df.shape[1]

    # 6. Write to datasets/processed/<source>/<name>.parquet
    out_dir = PROCESSED_DIR / ds["source"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        log(f"  [{ts()}] WRITE ERROR: {e}")
        manifest_row["status"] = f"WRITE_ERROR: {e}"
        return manifest_row
    manifest_row["processed_path"] = str(out_path)
    log(f"  [{ts()}] Written -> {out_path}  ({len(df):,} rows x {df.shape[1]} cols)")

    # 7. Re-verify binary status on the OUTPUT file — closes the loop on the
    #    original spot-check finding for this dataset.
    n_classes, counts = verify_binary(df, ds["target"])
    manifest_row["n_classes_output"] = n_classes
    manifest_row["class_counts_output"] = str(counts)
    manifest_row["binary_check"] = "PASS" if n_classes == 2 else f"FAIL ({n_classes} classes)"
    log(f"  [{ts()}] Binary check on output: {manifest_row['binary_check']}  classes={counts}")

    manifest_row["status"] = "OK" if n_classes == 2 else "OK_BUT_NOT_BINARY"
    return manifest_row


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global _log_file
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUT_LOG, "w", buffering=1)

    total = len(DATASETS)
    log(f"RAW -> PROCESSED PREPROCESSING -- {total} datasets")
    log(f"Seed        : {SEED}")
    log(f"Raw dir     : {RAW_DIR}  (read-only, never written to)")
    log(f"Processed   : {PROCESSED_DIR}")
    log(f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file    : {OUT_LOG}")
    log(f"Manifest    : {OUT_MANIFEST}")

    results = []
    for i, ds in enumerate(DATASETS, 1):
        r = preprocess_dataset(ds, i, total)
        results.append(r)
        # Write manifest after every dataset — progress never lost if interrupted
        pd.DataFrame(results).to_csv(OUT_MANIFEST, index=False)

    # Final summary
    log()
    log("=" * 70)
    log("PREPROCESSING SUMMARY")
    log("=" * 70)
    log(f"{'Dataset':<25} {'Rows(final)':>12} {'Cols':>6}  {'BinaryCheck':<20} Status")
    log("-" * 70)
    for r in results:
        rows = f"{r['final_rows']:,}" if r["final_rows"] is not None else "N/A"
        cols = str(r["final_cols"]) if r["final_cols"] is not None else "N/A"
        bc = r["binary_check"] or "N/A"
        log(f"{r['dataset']:<25} {rows:>12} {cols:>6}  {bc:<20} {r['status']}")

    ok      = [r for r in results if r["status"] == "OK"]
    not_bin = [r for r in results if r["status"] == "OK_BUT_NOT_BINARY"]
    errored = [r for r in results if r["status"] not in ("OK", "OK_BUT_NOT_BINARY")]

    log()
    log(f"OK (binary confirmed): {len(ok)}  |  OK_BUT_NOT_BINARY: {len(not_bin)}  |  ERROR: {len(errored)}")
    if not_bin:
        log("!! ACTION REQUIRED — these wrote successfully but are still not binary:")
        for r in not_bin:
            log(f"     - {r['dataset']}: {r['binary_check']}")
    if errored:
        log("!! ACTION REQUIRED — these failed to process:")
        for r in errored:
            log(f"     - {r['dataset']}: {r['status']}")
    log(f"Finished    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Manifest    -> {OUT_MANIFEST}")
    log(f"Log         -> {OUT_LOG}")

    _log_file.close()


if __name__ == "__main__":
    main()