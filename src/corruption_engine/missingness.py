"""
missingness.py

Implements D1 Section 4 — Missingness Mechanisms (MCAR, MAR, MNAR).
First module built per the agreed D3 sequencing: MCAR/MAR/MNAR share the
same conditioning interface, so implementing them together lets D8's
validation checks be written once (Phase 1 Requirements session).

Governing constraints this module must respect (does not re-derive them,
just implements them):

  D2 Decision 2 — this module has no opinion on which fold it's given.
    Callers are responsible for passing only the test fold for corruption;
    train stays clean. `fold` here means "whatever fold you handed me."

  D2 Decision 4 — every corruption parameter (conditioning feature choice,
    z-score reference mean/std) is estimated from `train_fold_reference`,
    passed in separately, NEVER from `fold` itself and NEVER from the full
    corpus.

  D1 Section 3 — severity `alpha` is in [0, alpha_max), never approaching 1
    (a fully-missing feature is degenerate, not "high severity").

  D1 Section 4.3 — the MNAR mechanism-fidelity check (missingness
    significantly associated with the feature's own pre-corruption value)
    is BLOCKING, not advisory. corrupt_mnar raises rather than silently
    returning MCAR-equivalent behavior when this check fails.

  D1 Section 11 — every call validates both Corruption Fidelity (achieved
    rate matches target) and, for MAR/MNAR, Mechanism Fidelity (missingness
    is actually associated with the right thing, not just present).
"""

from typing import Optional, Literal, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

from .base import (
    CorruptionMetadata, sigmoid, reference_zscore, solve_logit_offset_for_target_rate,
    rate_tolerance, reference_variance_issue, nan_adjusted_target, derive_call_seed,
)

MechanismName = Literal["mcar", "mar", "mnar"]

# Fixed steepness constant for MAR/MNAR's sigmoid — controls how sharply
# missingness probability concentrates as |z| grows. This is a shape choice,
# not part of D1's frozen severity axis (severity is calibrated via the
# offset in solve_logit_offset_for_target_rate, not via this constant).
# Flagged here as a single named constant rather than a magic number buried
# in the formula, so it's a one-line change if D8's validation suggests a
# different value.
ALPHA_SCALE = 3.0

SIGNIFICANCE_ALPHA = 0.05  # for mechanism-fidelity association tests (D1 Sec 11.2)
MIN_MISSING_FOR_ASSOCIATION_TEST = 10  # below this count, a point-biserial
    # test has no statistical power regardless of whether the mechanism is
    # correctly implemented — most commonly hit at severity approaching 0,
    # a legitimate boundary case, not a fidelity failure. Columns below this
    # threshold are excluded from the blocking check rather than auto-failed.

# rate_tolerance, reference_variance_issue, and nan_adjusted_target moved to
# base.py (2026-07-19) — feature_noise.py needs the same rate-tolerance,
# variance-check, and NaN-fraction-adjustment logic, and duplicating it
# risked the two copies drifling apart, the same single-source-of-truth
# concern already raised for TARGET_COLUMNS in the runner script.


def _safe_association_test(r_indicator: np.ndarray, reference_values: pd.Series) -> dict:
    """
    Point-biserial association test between a missingness indicator and a
    reference value series, excluding rows where reference_values is itself
    NaN (pre-existing missingness) — scipy's pointbiserialr does not handle
    NaN internally, and passing it through would silently produce NaN
    correlation/pvalue for any column with real pre-existing missing values
    (surfaced by testing MNAR against Airbnb, which has genuine pre-existing
    missing data by design — a documented Phase 0 realism anchor).
    """
    valid = ~reference_values.isna().to_numpy()
    r_valid = r_indicator[valid]
    ref_valid = reference_values.to_numpy()[valid]
    n_missing = int(r_valid.sum())
    if n_missing < MIN_MISSING_FOR_ASSOCIATION_TEST:
        return {"corr": float("nan"), "pval": float("nan"), "significant": True,
                "skipped_insufficient_data": True, "n_missing": n_missing}
    if pd.Series(r_valid).nunique() > 1 and pd.Series(ref_valid).nunique() > 1:
        corr, pval = stats.pointbiserialr(r_valid, ref_valid)
        return {"corr": float(corr), "pval": float(pval), "significant": pval < SIGNIFICANCE_ALPHA}
    return {"corr": float("nan"), "pval": float("nan"), "significant": False}


def _select_conditioning_feature(
    train_fold_reference: pd.DataFrame, target_col: str
) -> str:
    """
    D1 Section 4.2 — MAR's conditioning feature k: the numeric feature with
    highest |correlation| to the target column, selected once, estimated on
    the train fold only (D2 Decision 4).
    """
    numeric_cols = [
        c for c in train_fold_reference.columns
        if c != target_col and pd.api.types.is_numeric_dtype(train_fold_reference[c])
    ]
    if not numeric_cols:
        raise ValueError(
            f"MAR requires at least one numeric conditioning candidate besides "
            f"target column '{target_col}'; none found in train_fold_reference."
        )
    corrs = {
        c: abs(train_fold_reference[c].corr(train_fold_reference[target_col]))
        for c in numeric_cols
    }
    corrs = {c: v for c, v in corrs.items() if not np.isnan(v)}
    if not corrs:
        raise ValueError(
            f"All candidate conditioning features for '{target_col}' had "
            f"undefined correlation (likely constant columns)."
        )
    return max(corrs, key=corrs.get)


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1 — a fully-missing feature is degenerate, not "
            f"high severity."
        )


def corrupt_mcar(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 4.1 — MCAR: P(R_ij = 0 | X, Y) = pi_j(alpha), independent
    Bernoulli masking per cell. train_fold_reference is accepted for
    interface consistency with MAR/MNAR but unused here — MCAR has no
    conditioning structure to estimate.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "mcar", severity)
    rng = np.random.default_rng(call_seed)
    corrupted = fold.copy()
    achieved_rate = {}

    for col in columns:
        n = len(corrupted)
        mask = rng.random(n) < severity
        corrupted.loc[mask, col] = np.nan
        achieved_rate[col] = float(mask.mean())

    max_deviation = max((abs(r - severity) for r in achieved_rate.values()), default=0.0)
    tolerance = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="mcar",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_rate,
        validation_passed=max_deviation < tolerance,
        validation_details={"max_rate_deviation": max_deviation, "tolerance_used": tolerance},
    )
    return corrupted, metadata


def corrupt_mar(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    conditioning_feature: Optional[str] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 4.2 — MAR: P(R_ij = 0) = sigmoid(alpha * z(X_ik)), k != j,
    a fixed conditioning feature selected once per (dataset, target column).
    z(X_ik) is computed with train-fold reference mean/std (D2 Decision 4),
    applied to `fold`'s (possibly corrupted-elsewhere) values of k.

    Per-column fidelity granularity (decided 2026-07-19, replacing an
    earlier all-or-nothing design): the mechanism-fidelity check (D1 Sec
    11.2) is evaluated per column, BEFORE that column's NaN mask is applied.
    A column that fails is excluded from this call's output (left
    uncorrupted, listed in metadata.excluded_columns) rather than
    invalidating every other column requested in the same call. Surfaced by
    testing against EEG, where one channel (AF3) showed a real but weak
    self-dependency that intermittently failed significance while four
    other channels in the same call consistently passed — the all-or-nothing
    design was discarding valid corruption on those four channels every time
    AF3's borderline effect happened to miss. Raises only if EVERY requested
    column fails — i.e. nothing in this call could be validly corrupted.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "mar", severity)
    rng = np.random.default_rng(call_seed)
    corrupted = fold.copy()
    achieved_rate = {}
    conditioning_used = {}
    assoc_checks = {}
    excluded_columns = {}

    for col in columns:
        k = conditioning_feature or _select_conditioning_feature(train_fold_reference, col)

        variance_issue = reference_variance_issue(train_fold_reference, k)
        if variance_issue is not None:
            excluded_columns[col] = f"conditioning feature '{k}': {variance_issue}"
            continue

        ref_mean = train_fold_reference[k].mean()
        ref_std = train_fold_reference[k].std()
        z = reference_zscore(fold[k], ref_mean, ref_std).values

        adjusted_target = nan_adjusted_target(severity, fold[k])
        offset = solve_logit_offset_for_target_rate(z, adjusted_target, ALPHA_SCALE)
        p_missing = sigmoid(offset + ALPHA_SCALE * z)
        mask = rng.random(len(corrupted)) < p_missing

        # Mechanism fidelity (D1 Sec 11.2), checked BEFORE applying the mask.
        r_indicator = mask.astype(int)
        check = _safe_association_test(r_indicator, fold[k])
        assoc_checks[col] = check

        if not check["significant"]:
            excluded_columns[col] = (
                f"mechanism-fidelity check failed against conditioning feature "
                f"'{k}': corr={check['corr']:.4f}, pval={check['pval']:.4f}"
            )
            continue

        conditioning_used[col] = k
        corrupted.loc[mask, col] = np.nan
        achieved_rate[col] = float(mask.mean())

    if not achieved_rate:
        raise RuntimeError(
            f"MAR: every requested column failed the mechanism-fidelity check "
            f"for dataset='{dataset_name}', severity={severity}, seed={seed}. "
            f"No column in this call could be validly corrupted. "
            f"Excluded: {excluded_columns}"
        )

    max_deviation = max((abs(r - severity) for r in achieved_rate.values()), default=0.0)
    tolerance = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="mar",
        severity=severity,
        seed=seed,
        conditioning_feature=";".join(f"{c}<-{k}" for c, k in conditioning_used.items()),
        achieved_rate=achieved_rate,
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=max_deviation < tolerance,
        validation_details={
            "max_rate_deviation": max_deviation,
            "tolerance_used": tolerance,
            "conditioning_association": assoc_checks,
        },
    )
    return corrupted, metadata


def corrupt_mnar(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 4.3 — MNAR: P(R_ij = 0) = sigmoid(alpha * z(X_ij)), depending
    on the column's OWN pre-corruption value. z(X_ij) uses train-fold
    reference mean/std of column j (D2 Decision 4).

    Per-column fidelity granularity (decided 2026-07-19, replacing an
    earlier all-or-nothing design) — see corrupt_mar's docstring for the
    full rationale; identical reasoning applies here. A column failing the
    mechanism-fidelity check (D1 Sec 11.2) is excluded from this call's
    output (metadata.excluded_columns) rather than invalidating every other
    column requested in the same call. Raises only if EVERY requested
    column fails.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "mnar", severity)
    rng = np.random.default_rng(call_seed)
    corrupted = fold.copy()
    achieved_rate = {}
    assoc_checks = {}
    excluded_columns = {}

    for col in columns:
        variance_issue = reference_variance_issue(train_fold_reference, col)
        if variance_issue is not None:
            excluded_columns[col] = variance_issue
            continue

        ref_mean = train_fold_reference[col].mean()
        ref_std = train_fold_reference[col].std()
        z = reference_zscore(fold[col], ref_mean, ref_std).values

        adjusted_target = nan_adjusted_target(severity, fold[col])
        offset = solve_logit_offset_for_target_rate(z, adjusted_target, ALPHA_SCALE)
        p_missing = sigmoid(offset + ALPHA_SCALE * z)

        pre_corruption_values = fold[col].copy()
        mask = rng.random(len(corrupted)) < p_missing

        # Mechanism fidelity (D1 Sec 11.2), checked BEFORE applying the mask.
        r_indicator = mask.astype(int)
        check = _safe_association_test(r_indicator, pre_corruption_values)
        assoc_checks[col] = check

        if not check["significant"]:
            excluded_columns[col] = (
                f"mechanism-fidelity check failed against own pre-corruption "
                f"value: corr={check['corr']:.4f}, pval={check['pval']:.4f}"
            )
            continue

        corrupted.loc[mask, col] = np.nan
        achieved_rate[col] = float(mask.mean())

    if not achieved_rate:
        raise RuntimeError(
            f"MNAR: every requested column failed the mechanism-fidelity check "
            f"for dataset='{dataset_name}', severity={severity}, seed={seed}. "
            f"No column in this call could be validly corrupted. "
            f"Excluded: {excluded_columns}"
        )

    max_deviation = max((abs(r - severity) for r in achieved_rate.values()), default=0.0)
    tolerance = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="mnar",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_rate,
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=max_deviation < tolerance,
        validation_details={
            "max_rate_deviation": max_deviation,
            "tolerance_used": tolerance,
            "self_association": assoc_checks,
        },
    )
    return corrupted, metadata


def corrupt(
    fold: pd.DataFrame,
    mechanism: MechanismName,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    columns: Optional[List[str]] = None,
    conditioning_feature: Optional[str] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    Unified missingness entry point, matching the D3 interface contract:
        corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)
    (parameter named `fold` here rather than `dataset` — this module always
    receives a single fold, per D2 Decision 2's split-then-corrupt design.)

    Parameters
    ----------
    fold : the fold to corrupt. Per D2 Decision 2, callers should pass the
        TEST fold only — this function has no opinion on which fold it's given.
    mechanism : "mcar" | "mar" | "mnar"
    severity : alpha in [0, alpha_max) — the resolved numeric severity from
        D4's grid, not a raw guess.
    seed : corruption-injection random seed (D1 Sec 3 / Sec 12).
    train_fold_reference : the CLEAN train fold for this dataset/split — used
        only for parameter estimation (D2 Decision 4); never itself corrupted
        or read from for anything but reference statistics.
    columns : eligible feature columns. Defaults to all numeric columns in
        `fold` if not supplied.
    conditioning_feature : optional override for MAR's conditioning feature
        (skips auto-selection via correlation).
    dataset_name : for metadata/logging only.
    """
    if columns is None:
        columns = [c for c in fold.columns if pd.api.types.is_numeric_dtype(fold[c])]

    if mechanism == "mcar":
        return corrupt_mcar(fold, columns, severity, seed, train_fold_reference, dataset_name)
    elif mechanism == "mar":
        return corrupt_mar(fold, columns, severity, seed, train_fold_reference,
                            conditioning_feature, dataset_name)
    elif mechanism == "mnar":
        return corrupt_mnar(fold, columns, severity, seed, train_fold_reference, dataset_name)
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism!r}. Expected 'mcar', 'mar', or 'mnar'.")