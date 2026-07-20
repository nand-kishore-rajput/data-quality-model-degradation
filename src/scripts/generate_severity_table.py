"""
generate_severity_table.py

Generates the actual 17-dataset x 14-sub-mechanism x 10-severity-level table
from D4's finalized calibration formulas, replacing the placeholder grid
D3's six runners have been using for smoke-testing (`[0.05, 0.10, 0.15, 0.20, 0.25]`).

D4's mechanisms split into two categories:

  CORPUS-WIDE FIXED alpha_max (11 of 14 sub-mechanisms): severity(l) =
    (l/10) * alpha_max, identical across all 17 datasets. No data pull
    needed -- these are the fixed constants D4 already decided (missingness
    0.0865, feature noise 0.30, quantization's implicit 1.0, distribution
    shift covariate 0.25 / concept 0.5, validity 0.1088, uniqueness 0.06).

  PER-DATASET alpha_max / extra parameters (3 of 14 sub-mechanisms):
    - label_noise (uniform, class_conditional): alpha_max is a Cleanlab-
      estimated overall label-noise rate per dataset (D4 Sec 2.3), clamped
      to <= 0.45. Requires a Cleanlab pass per dataset -- this script
      contains its own small, self-contained estimation function rather
      than modifying label_noise.py's internals, which already have their
      own tested behavior (estimating the ASYMMETRY RATIO, a different
      parameter) and shouldn't be touched for this.
    - distribution_shift/label: severity itself is corpus-wide (l/10,
      matching D1's alpha interpolation weight directly), but needs each
      dataset's natural class prior (directly computable, no Cleanlab) to
      set rho_target per D4 Sec 2.4's rule.

If Cleanlab is unavailable, label-noise rows are written with alpha_max
left blank and a note explaining why, rather than silently defaulting to
something -- consistent with label_noise.py's own no-silent-fallback
philosophy for the same dependency.

Usage:
    python -m src.scripts.generate_severity_table --all
    python -m src.scripts.generate_severity_table --datasets heloc jm1
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")  # for containerized runs; override with env var if needed
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "corruption_engine"
OUT_TABLE = OUT_DIR / "severity_table.csv"
OUT_LOG = OUT_DIR / "generate_severity_table.log"

TARGET_COLUMNS = {
    ("openml", "phoneme"):              "Class",
    ("openml", "bank-marketing"):       "Class",
    ("openml", "MagicTelescope"):       "class:",
    ("openml", "churn"):                "class",
    ("openml", "heloc"):                "RiskPerformance",
    ("openml", "electricity"):          "class",
    ("openml", "jm1"):                  "defects",
    ("openml", "road-safety"):          "Sex_of_Driver",
    ("openml", "kick"):                 "IsBadBuy",
    ("uci", "default_credit"):          "Y",
    ("uci", "online_shoppers"):         "Revenue",
    ("uci", "diabetes_readmission"):    "readmitted",
    ("folktables", "ACSIncome"):        "target",
    ("folktables", "ACSPublicCoverage"):"target",
    ("cleanml", "Airbnb"):              "Rating",
    ("cleanml", "EEG"):                 "Eye",
    ("tableshift", "assistments"):      "correct",
}

SEVERITY_LEVELS = list(range(1, 11))  # 1-10, per D4 Sec 1

# D4's finalized corpus-wide alpha_max values -- Decision Log entries 2, 3,
# 6, 8, 9, 10. Fixed constants, no data pull required.
ALPHA_MAX = {
    "missingness": 0.0865,           # D4 Sec 2.1, road-safety anchor
    "feature_noise_rate": 0.30,      # D4 Sec 2.2, gaussian + categorical_swap
    "feature_noise_quantization": 1.0,  # D4 Sec 2.2 -- severity IS the
                                      # fraction of the k_max-to-k_min range;
                                      # feature_noise.py computes k_max
                                      # internally per call, nothing dataset-
                                      # specific belongs in this table.
    "distribution_shift_covariate": 0.25,   # D4 Sec 2.4
    "distribution_shift_concept": 0.5,      # D4 Sec 2.4
    "validity_violations": 0.1088,   # D4 Sec 2.5, HELOC anchor
    "uniqueness": 0.06,              # D4 Sec 2.6, HELOC anchor (Q2)
}

LABEL_NOISE_MAX_CLAMP = 0.45  # D4 Sec 2.3
LABEL_SHIFT_MINORITY_FLOOR_FRACTION = 0.05  # D4 Sec 2.4, per David's review

# The 14 sub-mechanisms, tagged with which alpha_max category they use.
SUB_MECHANISMS = [
    ("missingness", "mcar", "corpus_wide", "missingness"),
    ("missingness", "mar", "corpus_wide", "missingness"),
    ("missingness", "mnar", "corpus_wide", "missingness"),
    ("feature_noise", "gaussian", "corpus_wide", "feature_noise_rate"),
    ("feature_noise", "categorical_swap", "corpus_wide", "feature_noise_rate"),
    ("feature_noise", "quantization", "corpus_wide", "feature_noise_quantization"),
    ("label_noise", "uniform", "per_dataset_cleanlab", None),
    ("label_noise", "class_conditional", "per_dataset_cleanlab", None),
    ("distribution_shift", "covariate", "corpus_wide", "distribution_shift_covariate"),
    ("distribution_shift", "label", "per_dataset_prior", None),
    ("distribution_shift", "concept", "corpus_wide", "distribution_shift_concept"),
    ("validity_violations", "validity_violations", "corpus_wide", "validity_violations"),
    ("uniqueness", "global", "corpus_wide", "uniqueness"),
    ("uniqueness", "slice_conditional", "corpus_wide", "uniqueness"),
]

_log_file = None


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)
    if _log_file:
        _log_file.write(msg + end)
        _log_file.flush()


def ts():
    return datetime.now().strftime("%H:%M:%S")


def discover_datasets():
    found = []
    for source_dir in sorted(PROCESSED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for f in sorted(source_dir.glob("*.parquet")):
            found.append((source_dir.name, f.stem, f))
    return found


def estimate_label_noise_alpha_max(df: pd.DataFrame, target_col: str, feature_cols: list) -> float:
    """
    Self-contained Cleanlab-based estimate of the OVERALL label-noise rate
    (not the asymmetry ratio label_noise.py's internals estimate) --
    intentionally a separate, small implementation rather than reusing or
    modifying label_noise.py's _estimate_asymmetry_ratio_cleanlab, which
    already has its own tested purpose (a different parameter) and
    shouldn't be touched for this.

    Estimated rate = (sum of confident-joint off-diagonal counts) / n --
    the fraction of labels Cleanlab's confident learning pass flags as
    likely mislabeled, directly derivable from the same confident_joint
    matrix used elsewhere for the ratio.
    """
    from cleanlab.count import compute_confident_joint
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].to_numpy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for the Cleanlab probe.")

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(clf, Xs, y, cv=3, method="predict_proba")
    cj = compute_confident_joint(labels=y.astype(int), pred_probs=pred_probs)

    off_diagonal_total = int(cj.sum()) - int(np.trace(cj))
    n = len(y)
    return off_diagonal_total / n if n else 0.0


def compute_natural_prior_and_target(df: pd.DataFrame, target_col: str):
    """
    Natural class prior rho = P(Y=1) (using whichever class is coded 1, per
    D3's binary-corpus convention), and rho_target per D4 Sec 2.4: push
    toward 0.9 if rho < 0.5, else 0.1, clamped so neither class falls below
    LABEL_SHIFT_MINORITY_FLOOR_FRACTION of the fold.
    """
    y = df[target_col].dropna()
    classes = sorted(y.unique())
    if len(classes) != 2:
        return None, None
    c1 = classes[1]
    rho = float((y == c1).mean())

    rho_target = 0.9 if rho < 0.5 else 0.1
    floor = LABEL_SHIFT_MINORITY_FLOOR_FRACTION
    rho_target = min(max(rho_target, floor), 1 - floor)
    return rho, rho_target


def build_rows_for_dataset(source, name, df, target_col, label_noise_alpha_max, natural_rho, rho_target):
    rows = []
    for mech_class, sub_mech, category, alpha_key in SUB_MECHANISMS:
        for level in SEVERITY_LEVELS:
            frac = level / 10.0
            row = {
                "source": source, "dataset": name,
                "mechanism_class": mech_class, "sub_mechanism": sub_mech,
                "severity_level": level,
            }
            if category == "corpus_wide":
                row["severity_value"] = round(frac * ALPHA_MAX[alpha_key], 6)
                row["alpha_max_used"] = ALPHA_MAX[alpha_key]
                row["alpha_max_source"] = "corpus_wide_fixed_D4"
                row["note"] = None
            elif category == "per_dataset_cleanlab":
                if label_noise_alpha_max is None:
                    row["severity_value"] = None
                    row["alpha_max_used"] = None
                    row["alpha_max_source"] = "UNAVAILABLE"
                    row["note"] = "Cleanlab estimation failed or unavailable for this dataset"
                else:
                    row["severity_value"] = round(frac * label_noise_alpha_max, 6)
                    row["alpha_max_used"] = round(label_noise_alpha_max, 6)
                    row["alpha_max_source"] = "cleanlab_estimated_per_dataset"
                    row["note"] = None
            elif category == "per_dataset_prior":
                row["severity_value"] = frac  # alpha itself, per D1's interpolation formula
                row["alpha_max_used"] = None
                row["alpha_max_source"] = "not_applicable_uses_rho_target_instead"
                row["note"] = (f"natural_rho={natural_rho:.4f}, rho_target={rho_target:.4f}"
                               if natural_rho is not None else "target column not binary, skipped")
            rows.append(row)
    return rows


def run_one_dataset(source, name, path, all_rows):
    key = (source, name)
    target_col = TARGET_COLUMNS.get(key)

    log()
    log(f"{'─'*70}")
    log(f"[{ts()}] {source}/{name}")
    log(f"{'─'*70}")

    if target_col is None:
        log(f"  !! No target-column mapping for {key} — skipping.")
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log(f"  !! READ ERROR: {e}")
        return

    if target_col not in df.columns:
        log(f"  !! target column '{target_col}' not found — skipping.")
        return

    feature_cols = [c for c in df.columns if c != target_col]

    # Label noise: per-dataset Cleanlab estimate.
    label_noise_alpha_max = None
    try:
        raw_estimate = estimate_label_noise_alpha_max(df, target_col, feature_cols)
        label_noise_alpha_max = min(raw_estimate, LABEL_NOISE_MAX_CLAMP)
        clamped_note = " (clamped)" if raw_estimate > LABEL_NOISE_MAX_CLAMP else ""
        log(f"  label_noise alpha_max: {label_noise_alpha_max:.4f}{clamped_note} (raw estimate: {raw_estimate:.4f})")
    except ImportError as e:
        log(f"  !! Cleanlab unavailable — label_noise severity rows will be blank for this dataset: {str(e)[:150]}")
    except Exception as e:
        log(f"  !! Label-noise estimation failed — rows will be blank: {str(e)[:150]}")

    # Distribution shift / label: natural prior + rho_target.
    natural_rho, rho_target = compute_natural_prior_and_target(df, target_col)
    if natural_rho is not None:
        log(f"  distribution_shift/label: natural_rho={natural_rho:.4f}, rho_target={rho_target:.4f}")
    else:
        log(f"  !! distribution_shift/label: target not binary — rows will note this.")

    rows = build_rows_for_dataset(source, name, df, target_col,
                                   label_noise_alpha_max, natural_rho, rho_target)
    all_rows.extend(rows)
    log(f"  {len(rows)} rows generated (14 sub-mechanisms x 10 severity levels)")


def main():
    parser = argparse.ArgumentParser(description="Generate the real severity table from D4's finalized formulas.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUT_LOG, "w", buffering=1)

    discovered = discover_datasets()
    if not discovered:
        log(f"No parquet files found under {PROCESSED_DIR}.")
        sys.exit(1)

    if args.all:
        selected = discovered
    elif args.datasets:
        selected = [d for d in discovered if d[1] in args.datasets]
        missing = set(args.datasets) - {d[1] for d in selected}
        if missing:
            log(f"Warning: requested datasets not found: {missing}")
    else:
        selected = discovered
        log("No --datasets or --all given; defaulting to all discovered datasets.")

    log(f"SEVERITY TABLE GENERATION — {len(selected)} dataset(s), "
        f"{len(SUB_MECHANISMS)} sub-mechanisms, {len(SEVERITY_LEVELS)} severity levels")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows = []
    for source, name, path in selected:
        run_one_dataset(source, name, path, all_rows)

    table = pd.DataFrame(all_rows)
    table.to_csv(OUT_TABLE, index=False)

    log()
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total rows generated: {len(table)}")
    if not table.empty:
        n_blank_label_noise = table[(table["mechanism_class"] == "label_noise") & (table["severity_value"].isna())].shape[0]
        log(f"Label-noise rows with no value (Cleanlab unavailable/failed): {n_blank_label_noise}")
        log()
        log("Sample (severity_level=5, one row per sub-mechanism, first dataset):")
        first_dataset = table["dataset"].iloc[0]
        sample = table[(table["dataset"] == first_dataset) & (table["severity_level"] == 5)]
        log(sample[["mechanism_class", "sub_mechanism", "severity_value", "note"]].to_string(index=False))
    log()
    log(f"Full table -> {OUT_TABLE}")
    log(f"Log        -> {OUT_LOG}")
    _log_file.close()


if __name__ == "__main__":
    main()