"""
D1 §3 / §15 item 3 — Binary-label spot-check across the 17-dataset corpus.

Purpose:
    D1 assumes all 17 datasets are binary post-Phase-0-binarization. This is an
    UNVERIFIED assumption per D1 §3. This script verifies it directly against
    the raw/processed dataset files, per dataset, and reports:
        - detected label column
        - number of unique classes
        - class counts / balance
        - PASS (binary) / FAIL (not binary) / SKIP (couldn't locate label col)

    A FAIL here is not a bug to silently patch — it blocks D3's label-noise
    code and D4's severity grid for §6 (Label Noise) until resolved, per the
    project's own decision log (D1 §15 item 3).

Usage:
    python binary_label_spotcheck.py --root datasets/raw --catalog phase0_data_notes.md
    (catalog is optional; if omitted, the script infers label columns heuristically
    and you should visually confirm flagged datasets)

Output:
    - Console summary table
    - spotcheck_results.csv written to the working directory
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# 1. Dataset registry
#
# Fill in explicit (path, label_column) pairs here for the 17 datasets in the
# Phase 0 corpus. This is the authoritative source — heuristic detection below
# is only a fallback for datasets not listed here, and should not be trusted
# for the final record.
#
# Populate this from phase0_data_notes.md / build_benchmark_catalog.py output.
# Paths are relative to --root's parent, adjust as needed.
# ----------------------------------------------------------------------------
KNOWN_LABEL_COLUMNS = {
    # dataset_name -> label_column. Paths are NOT hardcoded here — discover_files()
    # below scans the actual directory tree and matches files to these names.
    "phoneme": "Class",
    "bank-marketing": "Class",
    "MagicTelescope": "class:",
    "churn": "class",
    "heloc": "RiskPerformance",
    "electricity": "class",
    "jm1": "defects",
    "kick": "IsBadBuy",
    "default_credit": "Y",
    "online_shoppers": "Revenue",
    "diabetes_130us": "readmitted",
    "acs_income": "target",
    "acs_public_coverage": "target",
    "airbnb": "Rating",
    "eeg": "Eye",
    "assistments": "correct",
    # FLAGGED — catalog fields contradict each other for this one:
    #   n_classes=3.0 / balance_note=MULTICLASS_NEEDS_COLLAPSE_RULE
    #   vs. curation_status text: "target filtered to Male/Female (dropped 19,824 Unknown)"
    "road_safety": "Sex_of_Driver",
}

# Fuzzy match tokens: dataset_name -> substrings to look for in filenames on disk,
# since actual filenames may not match dataset_name exactly (casing, separators, IDs).
NAME_MATCH_TOKENS = {
    "phoneme": ["phoneme"],
    "bank-marketing": ["bank-marketing", "bank_marketing", "bankmarketing"],
    "MagicTelescope": ["magictelescope", "magic-telescope", "magic_telescope"],
    "churn": ["churn"],
    "heloc": ["heloc", "46932"],
    "electricity": ["electricity"],
    "jm1": ["jm1"],
    "kick": ["kick"],
    "default_credit": ["default", "credit_card", "creditcard"],
    "online_shoppers": ["online_shoppers", "shoppers", "purchasing_intention"],
    "diabetes_130us": ["diabetes"],
    "acs_income": ["acsincome", "acs_income"],
    "acs_public_coverage": ["acspubliccoverage", "acs_public_coverage", "publiccoverage"],
    "airbnb": ["airbnb"],
    "eeg": ["eeg"],
    "assistments": ["assistments", "assistment"],
    "road_safety": ["road-safety", "road_safety", "roadsafety", "42803"],
}


def discover_files(root: Path) -> dict:
    """
    Search root recursively for all supported data files, and match each known
    dataset name to a file on disk via substring matching (case-insensitive).
    Returns {dataset_name: Path or None}.
    """
    all_files = find_candidate_files(root)
    matched = {}
    for name, tokens in NAME_MATCH_TOKENS.items():
        hit = None
        for f in all_files:
            fname_lower = f.name.lower()
            if any(tok.lower() in fname_lower for tok in tokens):
                hit = f
                break
        matched[name] = hit
    return matched

# Candidate label-column names to try if a dataset isn't in the registry above.
# Extend this list based on what you see in the fallback report.
LABEL_NAME_CANDIDATES = [
    "target", "label", "class", "y", "outcome", "Target", "Label", "Class",
    "Y", "Outcome", "readmitted", "Revenue", "RiskPerformance",
    "two_year_recid", "income", "default", "churn", "Churn",
]

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".tsv"}


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def detect_label_column(df: pd.DataFrame) -> str | None:
    """Fallback heuristic: try known candidate names, else the last column."""
    for cand in LABEL_NAME_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def spot_check_file(path: Path, label_col: str | None) -> dict:
    result = {
        "path": str(path),
        "label_column": label_col,
        "n_rows": None,
        "n_classes": None,
        "class_counts": None,
        "status": "SKIP",
        "note": "",
    }
    try:
        df = load_dataframe(path)
    except Exception as e:
        result["note"] = f"load error: {e}"
        return result

    result["n_rows"] = len(df)

    if label_col is None:
        label_col = detect_label_column(df)
        if label_col is None:
            result["note"] = "no label column found (registry + heuristics both failed)"
            return result
        result["note"] = f"label column auto-detected (verify manually): {label_col}"

    if label_col not in df.columns:
        result["note"] = f"registered label column '{label_col}' not found in file"
        return result

    result["label_column"] = label_col

    # Drop NaNs for class counting — missingness in the label itself is a
    # separate Phase 0 issue, not what this spot-check is verifying.
    counts = df[label_col].dropna().value_counts()
    result["n_classes"] = int(counts.shape[0])
    result["class_counts"] = counts.to_dict()

    if result["n_classes"] == 2:
        result["status"] = "PASS"
    elif result["n_classes"] == 1:
        result["status"] = "FAIL"
        result["note"] += " | only 1 class present — degenerate, check filtering upstream"
    else:
        result["status"] = "FAIL"
        result["note"] += f" | {result['n_classes']} classes — NOT binary, needs binarization rule or D1 §6.2 fallback"

    return result


def find_candidate_files(root: Path) -> list[Path]:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="D1 binary-label spot-check")
    parser.add_argument("--root", type=str, default="datasets/raw",
                         help="Root directory containing cleanml/folktables/openml/tableshift/uci subfolders")
    parser.add_argument("--registry", type=str, default=None,
                         help="Optional JSON file mapping dataset_name -> {label_column}, "
                              "overrides/extends KNOWN_LABEL_COLUMNS. Paths are always discovered "
                              "from --root, never taken from this file.")
    parser.add_argument("--out", type=str, default="spotcheck_results.csv")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path '{root}' does not exist. Run this on the machine/container "
              f"where the dataset corpus actually lives.", file=sys.stderr)
        sys.exit(1)

    label_cols = dict(KNOWN_LABEL_COLUMNS)
    if args.registry:
        with open(args.registry) as f:
            extra = json.load(f)
        for name, entry in extra.items():
            label_cols[name] = entry["label_column"]

    print(f"Searching {root} for dataset files...\n")
    matched_paths = discover_files(root)

    unmatched = [name for name, path in matched_paths.items() if path is None]
    if unmatched:
        print(f"WARNING: could not locate a file for {len(unmatched)} dataset(s) via name matching:")
        for name in unmatched:
            print(f"    - {name}  (tokens tried: {NAME_MATCH_TOKENS[name]})")
        print("  These will be marked SKIP below. Check NAME_MATCH_TOKENS or locate manually.\n")

    all_files = find_candidate_files(root)
    matched_file_set = {p for p in matched_paths.values() if p is not None}
    leftover = [f for f in all_files if f not in matched_file_set]
    if leftover:
        print(f"NOTE: {len(leftover)} file(s) under {root} were not matched to any of the 17 "
              f"known datasets (shown for awareness, not checked):")
        for f in leftover[:20]:
            print(f"    - {f}")
        if len(leftover) > 20:
            print(f"    ... and {len(leftover) - 20} more")
        print()

    results = []
    for name, path in matched_paths.items():
        label_col = label_cols.get(name)
        if path is None:
            results.append({
                "dataset_name": name, "path": None, "label_column": label_col,
                "n_rows": None, "n_classes": None, "class_counts": None,
                "status": "SKIP", "note": "file not found on disk via name matching",
            })
            continue
        r = spot_check_file(path, label_col)
        r["dataset_name"] = name
        results.append(r)

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.out, index=False)

    # Console summary
    print(f"\n{'='*90}")
    print(f"BINARY-LABEL SPOT-CHECK — {len(results)} dataset(s) checked")
    print(f"{'='*90}\n")

    for r in results:
        status_flag = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️ "}[r["status"]]
        print(f"{status_flag} {r['dataset_name']:<30} label='{r['label_column']}'  "
              f"classes={r['n_classes']}  rows={r['n_rows']}")
        if r["status"] != "PASS":
            print(f"     note: {r['note']}")
        if r["class_counts"]:
            print(f"     class_counts: {r['class_counts']}")
        print()

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")

    print(f"{'='*90}")
    print(f"SUMMARY: {n_pass} PASS | {n_fail} FAIL | {n_skip} SKIP  (results written to {args.out})")
    print(f"{'='*90}\n")

    if n_fail > 0 or n_skip > 0:
        print("ACTION REQUIRED: D1 §3's binary-corpus assumption does NOT hold for all datasets.")
        print("  - FAIL datasets need either a documented binarization rule (update Phase 0 notes)")
        print("    or explicit routing through D1 §6.2's multi-class fallback for label noise.")
        print("  - SKIP datasets need a registry entry (path + label column) before this check")
        print("    can be considered authoritative — do not treat SKIP as a silent pass.")
        sys.exit(2)
    else:
        print("All checked datasets confirmed binary. D1 §3 assumption holds — D3 label-noise")
        print("code and D4 severity grid for §6 are unblocked on this specific item.")


if __name__ == "__main__":
    main()