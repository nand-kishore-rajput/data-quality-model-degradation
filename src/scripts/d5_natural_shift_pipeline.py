"""
d5_natural_shift_pipeline.py

Implements D5 Sec 4's full methodology end to end:

  1. REAL natural-shift measurement. For each anchor dataset, split into
     Population A (train) and Population B (natural-shift eval) per D5's
     confirmed Decision 3:
       - road-safety: A=Urban (Urban_or_Rural_Area==1), B=Rural (==2)
       - diabetes_readmission: A=Emergency (admission_source_id==7),
         B=Non-emergency (all other defined codes)
     Train each of the 4 model families on A's train partition, evaluate on
     A's held-out test (M_clean) and on B in full (M_shifted). Compute the
     real Relative dM = (M_clean - M_shifted) / M_clean per family.

     LEAKAGE GUARD: the split-defining column itself (Urban_or_Rural_Area /
     admission_source_id) is excluded from the feature set during this
     measurement -- otherwise the model could trivially detect which
     population a row belongs to, which isn't a fair test of whether the
     feature-label RELATIONSHIP actually differs between populations.

  2. SYNTHETIC concept-shift severity sweep, for the same two datasets,
     using the SAME standard 70/30 split D4's severity grid was itself
     computed against (all features included, no leakage guard needed --
     this is the standard D3 pipeline, unmodified). Train each family once
     on the clean train fold, evaluate clean AUC, then apply
     distribution_shift.py's concept-shift corruption at each of D4's 10
     severity levels (3 seeds each, averaged) and evaluate corrupted AUC.
     Compute synthetic Relative dM per severity level.

  3. COMPARISON: report which severity level's average synthetic Relative
     dM most closely matches the real natural-shift Relative dM, per
     dataset, and flag if no level is a good match (a D4 finding, not
     silently ignored).

Model families: XGBoost, Random Forest, Logistic Regression, MLP -- matching
the project's locked four-family set. Reasonable, simple default
hyperparameters throughout -- flagged as a D5 implementation decision, not
independently tuned or validated.

Metric M: ROC-AUC, matching this project's Phase 0 XGBoost-AUC screening
convention. Flagged as an assumption -- confirm this matches D11's literal
metric choice if that document is available to check against directly.

Usage:
    python -m src.scripts.d5_natural_shift_pipeline
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.corruption_engine.distribution_shift import corrupt
from src.corruption_engine.base import derive_call_seed

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/workspace")
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
SEVERITY_TABLE_PATH = PROJECT_ROOT / "results" / "corruption_engine" / "severity_table.csv"
OUT_PATH = PROJECT_ROOT / "results" / "d5_natural_shift_report.txt"

SPLIT_RATIO = 0.7
SPLIT_SEED = 42
SEEDS = [1, 2, 3]

DATASETS = {
    "road-safety": {
        "path": PROCESSED_DIR / "openml" / "road-safety.parquet",
        "target_col": "Sex_of_Driver",
        "split_col": "Urban_or_Rural_Area",
        "pop_a_value": 1.0,  # Urban -- train
        "pop_b_value": 2.0,  # Rural -- natural-shift eval
    },
    "diabetes_readmission": {
        "path": PROCESSED_DIR / "uci" / "diabetes_readmission.parquet",
        "target_col": "readmitted",
        "split_col": "admission_source_id",
        "pop_a_value": "EMERGENCY_CODE_7",  # handled specially below
        "pop_b_value": "NON_EMERGENCY",     # handled specially below
    },
}
AMBIGUOUS_ADMISSION_CODES = {9, 15, 17, 20, 21}

_log_lines = []


def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(msg)


def get_model_families(seed):
    families = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }
    if HAS_XGBOOST:
        families["xgboost"] = XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=seed
        )
    else:
        log("  !! xgboost not installed -- skipping this family for all measurements.")
    return families


def encode_features(train_df, other_dfs, feature_cols):
    """
    Simple, uniform preprocessing for all 4 families: one-hot encode
    object/categorical columns (categories fit on the TRAIN fold only, per
    D2 Decision 4), median-impute numeric NaN (train-fold median),
    standardize numeric columns (fit on train only) for the scale-sensitive
    families (logistic regression, MLP) -- tree-based families are
    unaffected by scaling, so this is a strict improvement, not a tradeoff.
    Extra categories present in an eval frame but not seen in train are
    dropped; categories missing from an eval frame are filled with zero.

    Fixed 2026-07-21: two real bugs found via the first real-corpus run.
    (1) StandardScaler was imported but never applied -- real data showed
    an ~8x spread in Relative_dM between families on the same measurement
    (road-safety: MLP 0.0514 vs XGBoost 0.0061), with MLP's clean AUC
    barely above chance (0.5607) -- a clear signature of unscaled features
    hurting scale-sensitive models. (2) One-hot-encoded column names
    containing brackets (e.g. diabetes_readmission's age column, binned as
    string ranges like "[70-80)") crashed XGBoost entirely with
    "feature_names must be string, and may not contain [, ] or <" --
    column names are now sanitized to alphanumeric+underscore for every
    family, not just XGBoost, for robustness.
    """
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    medians = train_df[numeric_cols].median() if numeric_cols else pd.Series(dtype=float)
    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(train_df[numeric_cols].fillna(medians))

    def transform(df):
        df = df[feature_cols].copy()
        for c in numeric_cols:
            df[c] = df[c].fillna(medians[c])
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
        return df

    train_enc = transform(train_df)
    others_enc = [transform(d) for d in other_dfs]
    others_enc = [o.reindex(columns=train_enc.columns, fill_value=0) for o in others_enc]

    # Sanitize column names for every family (not just XGBoost) -- brackets,
    # parens, and other special characters in dummy-encoded column names
    # (e.g. "age_[70-80)") are a real, not merely cosmetic, compatibility
    # issue across ML libraries, not only XGBoost's strict checker.
    def sanitize(cols):
        return [str(c).replace("[", "_").replace("]", "_").replace("<", "lt")
                     .replace(">", "gt").replace("(", "_").replace(")", "_")
                     .replace(" ", "_").replace(",", "_") for c in cols]

    sanitized_cols = sanitize(train_enc.columns)
    train_enc.columns = sanitized_cols
    for o in others_enc:
        o.columns = sanitized_cols

    return train_enc, others_enc


def stratified_split(df, target_col, ratio, seed, dataset_name):
    call_seed = derive_call_seed(seed, dataset_name, "d5_split", 0.0)
    rng = np.random.default_rng(call_seed)
    train_idx_parts, test_idx_parts = [], []
    for cls, group in df.groupby(target_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(len(idx) * ratio)
        train_idx_parts.append(idx[:cut])
        test_idx_parts.append(idx[cut:])
    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def natural_shift_measurement(name, cfg):
    log(f"\n{'='*70}\n{name}: REAL natural-shift measurement\n{'='*70}")
    df = pd.read_parquet(cfg["path"])
    target_col, split_col = cfg["target_col"], cfg["split_col"]

    if name == "diabetes_readmission":
        emergency_mask = df[split_col] == 7
        ambiguous_mask = df[split_col].isin(AMBIGUOUS_ADMISSION_CODES)
        pop_a = df[emergency_mask].reset_index(drop=True)      # Emergency (larger)
        pop_b = df[~emergency_mask & ~ambiguous_mask].reset_index(drop=True)  # Non-emergency
    else:
        pop_a = df[df[split_col] == cfg["pop_a_value"]].reset_index(drop=True)
        pop_b = df[df[split_col] == cfg["pop_b_value"]].reset_index(drop=True)

    log(f"  Population A (train): {len(pop_a)} rows | Population B (eval): {len(pop_b)} rows")

    a_train, a_test = stratified_split(pop_a, target_col, SPLIT_RATIO, SPLIT_SEED, f"{name}_natural_A")

    feature_cols = [c for c in df.columns if c not in (target_col, split_col)]
    train_enc, (test_enc, popb_enc) = encode_features(a_train, [a_test, pop_b], feature_cols)

    y_train = a_train[target_col].to_numpy()
    y_test = a_test[target_col].to_numpy()
    y_popb = pop_b[target_col].to_numpy()

    results = {}
    for fam_name, model in get_model_families(seed=1).items():
        try:
            model.fit(train_enc, y_train)
            m_clean = roc_auc_score(y_test, model.predict_proba(test_enc)[:, 1])
            m_shifted = roc_auc_score(y_popb, model.predict_proba(popb_enc)[:, 1])
            rel_dm = (m_clean - m_shifted) / m_clean if m_clean != 0 else float("nan")
            results[fam_name] = {"m_clean": m_clean, "m_shifted": m_shifted, "relative_dm": rel_dm}
            log(f"  {fam_name}: M_clean={m_clean:.4f}, M_shifted={m_shifted:.4f}, Relative_dM={rel_dm:.4f}")
        except Exception as e:
            log(f"  !! {fam_name} failed: {type(e).__name__}: {str(e)[:200]}")
            results[fam_name] = None
    return results


def concept_shift_sweep(name, cfg, severity_table):
    log(f"\n{'='*70}\n{name}: SYNTHETIC concept-shift severity sweep\n{'='*70}")
    df = pd.read_parquet(cfg["path"])
    target_col = cfg["target_col"]

    train_fold, test_fold = stratified_split(df, target_col, SPLIT_RATIO, SPLIT_SEED, name)
    feature_cols = [c for c in df.columns if c != target_col]
    train_enc, (test_enc,) = encode_features(train_fold, [test_fold], feature_cols)
    y_train = train_fold[target_col].to_numpy()
    y_test = test_fold[target_col].to_numpy()

    levels = (severity_table[(severity_table["dataset"] == name) &
                              (severity_table["sub_mechanism"] == "concept")]
              .sort_values("severity_level"))
    if levels.empty:
        log(f"  !! No concept-shift severity rows found for '{name}' in severity_table.csv.")
        return {}

    results_by_family = {}
    for fam_name, model in get_model_families(seed=1).items():
        try:
            model.fit(train_enc, y_train)
            m_clean = roc_auc_score(y_test, model.predict_proba(test_enc)[:, 1])
        except Exception as e:
            log(f"  !! {fam_name} clean fit failed: {str(e)[:150]}")
            continue

        family_results = {}
        for _, row in levels.iterrows():
            severity = row["severity_value"]
            rel_dms = []
            for seed in SEEDS:
                try:
                    corrupted_test, _ = corrupt(test_fold, "concept", severity, seed,
                                                train_fold_reference=train_fold,
                                                target_col=target_col, dataset_name=name)
                    _, (corrupted_enc,) = encode_features(train_fold, [corrupted_test], feature_cols)
                    y_corrupted = corrupted_test[target_col].to_numpy()
                    m_shifted = roc_auc_score(y_corrupted, model.predict_proba(corrupted_enc)[:, 1])
                    rel_dm = (m_clean - m_shifted) / m_clean if m_clean != 0 else float("nan")
                    rel_dms.append(rel_dm)
                except Exception as e:
                    log(f"    !! {fam_name} sev={severity} seed={seed} failed: {str(e)[:150]}")
            if rel_dms:
                family_results[severity] = float(np.mean(rel_dms))
        results_by_family[fam_name] = family_results
        log(f"  {fam_name}: " + ", ".join(f"{s:.3f}:{v:+.4f}" for s, v in family_results.items()))

    return results_by_family


def compare(name, natural_results, synthetic_results):
    log(f"\n{'='*70}\n{name}: COMPARISON -- real natural-shift vs synthetic concept-shift\n{'='*70}")

    valid_natural = {k: v["relative_dm"] for k, v in natural_results.items() if v is not None}
    if not valid_natural:
        log("  !! No valid natural-shift measurements to compare against.")
        return
    real_avg = float(np.mean(list(valid_natural.values())))
    log(f"  Real natural-shift Relative_dM (averaged across families): {real_avg:+.4f}")

    all_severities = sorted({s for fam in synthetic_results.values() for s in fam.keys()})
    if not all_severities:
        log("  !! No synthetic severity results to compare against.")
        return

    best_severity, best_gap = None, float("inf")
    for sev in all_severities:
        vals = [fam[sev] for fam in synthetic_results.values() if sev in fam]
        if not vals:
            continue
        avg = float(np.mean(vals))
        gap = abs(avg - real_avg)
        marker = ""
        if gap < best_gap:
            best_gap, best_severity = gap, sev
            marker = "  <-- closest so far"
        log(f"    severity={sev:.3f}: synthetic avg Relative_dM={avg:+.4f}, gap={gap:.4f}{marker}")

    log(f"\n  BEST MATCH: severity={best_severity:.3f} (gap={best_gap:.4f})")
    if best_gap > 0.10:
        log(f"  !! FINDING: even the closest severity level's gap ({best_gap:.4f}) is large. "
            f"D4's concept-shift calibration for '{name}' may need revisiting -- "
            f"flag back to D4, don't silently accept this as a good anchor.")
    else:
        log(f"  Reasonable match -- supports D4's current concept-shift calibration for '{name}'.")


def main():
    if not SEVERITY_TABLE_PATH.exists():
        log(f"!! severity_table.csv not found at {SEVERITY_TABLE_PATH}. Run generate_severity_table.py first.")
        return
    severity_table = pd.read_csv(SEVERITY_TABLE_PATH)

    for name, cfg in DATASETS.items():
        if not cfg["path"].exists():
            log(f"!! {name} not found at {cfg['path']} -- skipping.")
            continue
        natural_results = natural_shift_measurement(name, cfg)
        synthetic_results = concept_shift_sweep(name, cfg, severity_table)
        compare(name, natural_results, synthetic_results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(_log_lines))
    log(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()