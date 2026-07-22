"""
distribution_shift.py

Implements D1 Section 7 — Distribution Shift Mechanisms:
  7.1 Covariate Shift          P'(X) != P(X),  P'(Y|X) = P(Y|X)
  7.2 Label Shift              P'(Y) != P(Y),  P'(X|Y) = P(X|Y)
  7.3 Slice-Conditional Concept Shift   P'(Y|X in S) != P(Y|X in S), X unchanged

Fourth and final D3 mechanism module, and the most intricate: each
sub-mechanism changes a DIFFERENT part of the joint distribution while
holding the others fixed, and the validation for each is about proving the
right thing moved and the right things stayed put.

Governing constraints (implemented, not re-derived):

  D2 Decision 2 — corrupts only the fold it's given (the test fold).

  D2 Decision 4 — the PC1 direction (7.1), the class base rates and
    per-class feature blocks (7.2), and the frozen slice bounds (7.3) are all
    estimated on train_fold_reference only.

  D1 Section 7.1 — PC1 is the FROZEN, SOLE covariate-shift generator. It is
    deliberately the max-variance direction, NOT the max-Y-predictive one, so
    that drift magnitude and degradation magnitude do not always move
    together (a benchmark where they did would be trivially solvable). This
    module does not substitute a supervised direction.

  D1 Section 7.3 — the slice S is chosen once and frozen, and is the SAME
    slice uniqueness.py (9.2) will reuse. This module gets S from base.py's
    shared select_frozen_slice_bounds, never computing its own — that shared
    source is what guarantees the two mechanisms align.

  D1 Section 3 / 7.2 — label-shift's target prior must not push a fold's
    minority count below what downstream needs; this module exposes the
    achieved prior so D4's constraint can be checked, and refuses a
    target_prior that would empty a class.
"""

from typing import Optional, Literal, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

from .base import (
    CorruptionMetadata,
    select_frozen_slice_bounds,
    derive_call_seed,
    reference_variance_issue,
)

MechanismName = Literal["covariate", "label", "concept"]

PSI_BINS = 10  # bins for the PSI validation check (D1 Sec 7.1 requires PSI to
    # rise with severity). Matches D10's PSI binning convention (quantile bins,
    # 10) so the validation here is measured the same way the meta-model will
    # later measure drift — not a second, inconsistent PSI definition.
CONCEPT_INVARIANCE_ALPHA = 0.05  # significance level for the P(X) / P(X|Y)
    # invariance checks (a shift that was supposed to leave P(X) alone should
    # NOT show a significant KS difference on the untouched features).
G_CLIP_Z = 6.0  # covariate shift's PC1 score, clipped to this many of its OWN
    # standard deviations before exponential weighting (see corrupt_covariate).
    # NOT specified by D1 -- a D3 implementation decision addressing a real
    # degeneracy found on jm1 in the real corpus, not a reinterpretation of
    # D1's P(select) ~ exp(alpha*g) formula. 6 is chosen to be generous (barely
    # touches well-behaved, roughly-normal g distributions) while still ruling
    # out the single-point domination that collapsed jm1's effective sample
    # size to ~1. Worth revisiting against D4's full severity grid.
MIN_EFFECTIVE_SAMPLE_FRACTION = 0.10  # below this (effective_sample_size / n),
    # the resample is flagged as degenerate in validation_details even after
    # clipping -- a diagnostic floor, not a hard block, so D8 can see it
    # rather than have it silently pass.


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1."
        )


def _numeric_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    return [c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])]


def _psi(reference: np.ndarray, target: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index, quantile-binned on the reference — same
    convention as D10 Decision 4 (including the zero-count-bin floor)."""
    reference = reference[~np.isnan(reference)]
    target = target[~np.isnan(target)]
    if len(reference) < bins or len(target) == 0:
        return float("nan")
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    ref_hist = np.histogram(reference, bins=edges)[0] / len(reference)
    tgt_hist = np.histogram(target, bins=edges)[0] / len(target)
    floor = 1.0 / (2 * len(target))
    tgt_hist = np.where(tgt_hist == 0, floor, tgt_hist)
    ref_hist = np.where(ref_hist == 0, floor, ref_hist)
    return float(np.sum((tgt_hist - ref_hist) * np.log(tgt_hist / ref_hist)))


# ---------------------------------------------------------------------------
# 7.1 Covariate Shift
# ---------------------------------------------------------------------------

def corrupt_covariate(
    fold: pd.DataFrame,
    target_col: str,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 7.1 — Covariate Shift via importance-weighted RESAMPLING along
    PC1:
        g(x) = PC1(x)  (train-fold-standardized numeric features)
        P(select) proportional to exp(alpha * g(x))

    Implemented as weighted resampling WITH replacement of the test fold's own
    rows: this shifts P(X) (rows with high PC1 score become over-represented)
    while leaving P(Y|X) exactly intact — because each resampled row keeps its
    own original (x, y) pairing, the conditional label distribution given x is
    unchanged by construction. That's the cleanest way to satisfy D1 7.1's
    "P'(Y|X) = P(Y|X)" requirement: never touch any individual row's label,
    only change which rows appear and how often.

    PC1 (direction + standardization center/scale) is fit on the TRAIN fold
    (D2 Decision 4) and applied to the test fold.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "covariate", severity)
    rng = np.random.default_rng(call_seed)
    candidate_cols = _numeric_feature_cols(train_fold_reference, target_col)

    # Fixed 2026-07-21: exclude near-constant train-fold columns from PC1's
    # feature set entirely, rather than only guarding against EXACTLY zero
    # variance. Real-corpus testing found EEG's covariate shift totally
    # degenerate (effective_sample_fraction=0.000) at EVERY severity level,
    # immediately -- worse than jm1's earlier residual case, which degraded
    # gradually with severity. Root cause, confirmed by reproduction: a
    # channel (almost certainly AF3, already flagged elsewhere in this
    # project for borderline signal properties) with a train-fold std small
    # enough that a handful of genuine EEG artifact spikes in the test fold
    # produce enormous z-scores, dominating PC1's loading. G_CLIP_Z clipping
    # doesn't help here because it clips to a multiple of g_centered's OWN
    # std, which is itself inflated by the same extreme points -- a non-
    # robust statistic chasing the outliers it's meant to bound, rather than
    # anchoring to the bulk of well-behaved data. Excluding near-constant
    # reference columns upstream, before PC1 is ever computed, addresses the
    # root cause directly rather than patching the symptom further downstream.
    excluded_features = {}
    feature_cols = []
    for col in candidate_cols:
        issue = reference_variance_issue(train_fold_reference, col)
        if issue is not None:
            excluded_features[col] = issue
        else:
            feature_cols.append(col)

    if len(feature_cols) < 1:
        raise RuntimeError(
            f"Covariate shift needs at least one numeric feature for PC1 with "
            f"non-degenerate train-fold variance; dataset='{dataset_name}' has "
            f"none. Excluded: {excluded_features}"
        )

    # Fit standardization + PC1 on the train fold.
    ref_X = train_fold_reference[feature_cols].to_numpy(dtype=float)
    ref_mean = np.nanmean(ref_X, axis=0)
    ref_std = np.nanstd(ref_X, axis=0)
    ref_std[ref_std == 0] = 1.0  # belt-and-suspenders: reference_variance_issue
        # already excludes near-zero-variance columns above; this guard only
        # matters if MIN_REFERENCE_STD itself is ever set to exactly 0.

    ref_Xs = (ref_X - ref_mean) / ref_std
    ref_Xs = np.nan_to_num(ref_Xs, nan=0.0)  # pre-existing NaN -> 0 (mean) for PCA fit
    # PC1 via SVD on the centered, standardized train matrix.
    _, _, Vt = np.linalg.svd(ref_Xs - ref_Xs.mean(axis=0), full_matrices=False)
    pc1 = Vt[0]

    # Apply to the test fold using train-fold center/scale + PC1 direction.
    X = fold[feature_cols].to_numpy(dtype=float)
    Xs = (X - ref_mean) / ref_std
    Xs = np.nan_to_num(Xs, nan=0.0)
    g = Xs @ pc1  # PC1 score per test row

    # Selection weights proportional to exp(alpha * g), normalized. Center g
    # first to keep exp() numerically stable regardless of g's scale.
    g_centered = g - g.mean()
    # Clip before exponentiating (fix, 2026-07-20): raw exp(severity*g_centered)
    # is numerically unbounded. When PC1 is dominated by even one or two
    # extreme outliers -- realistic for correlated, skewed real features like
    # software-complexity metrics (loc, v(g), ev(g) are all derived from the
    # same underlying code properties) -- a single point's g_centered can be
    # 30-40x larger than every other row's, and the exponential doesn't just
    # up-weight it, it can drive the effective sample size to ~1 out of
    # thousands even at moderate severity: not covariate shift, degenerate
    # duplication of one row. Surfaced by testing against jm1 in the real
    # corpus, whose PSI hit 7.9 and then plateaued identically across two
    # different severities -- the exact signature of a resample that had
    # already fully collapsed. Clipping g_centered to a fixed number of its
    # own standard deviations keeps the mechanism's INTENT (select more
    # strongly toward one tail as severity rises) while preventing any single
    # point from capturing the entire probability mass.
    #
    # Fixed 2026-07-21: g_std (plain standard deviation) is itself NOT robust
    # to the same outliers it's meant to bound against -- surfaced by EEG,
    # whose covariate shift collapsed totally (effective_sample_fraction
    # exactly 0.000) at EVERY severity level, immediately, unlike jm1's
    # gradual degradation. Root cause: a channel with disproportionately
    # tiny scale relative to the others (not degenerate/constant -- excluding
    # near-constant columns doesn't catch this) meant a handful of genuine
    # small artifacts, once z-scored, produced g values so extreme that they
    # inflated g_centered.std() itself, so the clip bound scaled up right
    # along with the outliers instead of bounding relative to the well-
    # behaved bulk of the distribution -- a classic non-robust-statistic-
    # chasing-its-own-outliers failure. Replaced with a MAD-based robust
    # scale estimate, which is far less sensitive to a small number of
    # extreme points, so the clip bound stays anchored to the bulk of the
    # data regardless of how extreme the outliers get.
    median_g = np.median(g_centered)
    mad = np.median(np.abs(g_centered - median_g))
    g_scale = 1.4826 * mad  # scaled so MAD approximates std under normality
    if g_scale > 0:
        g_clip = G_CLIP_Z * g_scale
        g_centered = np.clip(g_centered, -g_clip, g_clip)
    weights = np.exp(severity * g_centered)
    weights = weights / weights.sum()
    effective_sample_size = float(1.0 / np.sum(weights ** 2))

    n = len(fold)
    selected_idx = rng.choice(n, size=n, replace=True, p=weights)
    corrupted = fold.iloc[selected_idx].reset_index(drop=True)

    # Validation (D1 Sec 7.1): PSI should rise with severity; P(Y|X) unchanged
    # (holds by construction — labels never touched — so we report a
    # structural confirmation, plus a measurable PSI on PC1 score and the
    # highest-loading feature).
    top_feature = feature_cols[int(np.argmax(np.abs(pc1)))]
    psi_top = _psi(train_fold_reference[top_feature].to_numpy(dtype=float),
                   corrupted[top_feature].to_numpy(dtype=float))
    psi_pc1 = _psi(g, (Xs[selected_idx] @ pc1))
    ess_fraction = effective_sample_size / n
    degenerate_resample = ess_fraction < MIN_EFFECTIVE_SAMPLE_FRACTION

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="covariate_shift",
        severity=severity,
        seed=seed,
        conditioning_feature=f"PC1(top_loading={top_feature})",
        achieved_rate={"psi_pc1_score": psi_pc1, "psi_top_feature": psi_top},
        excluded_columns=excluded_features if excluded_features else None,
        validation_passed=(psi_pc1 >= 0) and not degenerate_resample,
        validation_details={
            "pc1_top_loading_feature": top_feature,
            "psi_pc1_score": psi_pc1,
            "psi_top_feature": psi_top,
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": ess_fraction,
            "degenerate_resample": degenerate_resample,
            "excluded_from_pc1": excluded_features if excluded_features else None,
            "conditional_label_preserved_by_construction": True,
            "note": "P(Y|X) unchanged because resampling preserves each row's own (x,y) pairing.",
        },
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# 7.2 Label Shift
# ---------------------------------------------------------------------------

def corrupt_label(
    fold: pd.DataFrame,
    target_col: str,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    target_prior: Optional[float] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 7.2 — Label Shift:
        P'(Y) != P(Y),  P'(X|Y) = P(X|Y)
        rho'(alpha) = (1-alpha)*rho + alpha*rho_target

    Implemented as class-stratified resampling: draw rows so the new class
    prior is rho'(alpha), sampling WITHIN each class with replacement from
    that class's own rows — which leaves P(X|Y) exactly intact (a row's
    features are only ever drawn alongside their true class). rho is the
    train-fold class prior (D2 Decision 4); rho_target is the maximum
    permitted shifted prior (D1 7.2 freezes the CONSTRAINT, D4 supplies the
    numeric value — passed here as target_prior).

    Refuses a target_prior that would drive either class's drawn count to
    zero (D1 Sec 3 / 7.2 minority-floor constraint): an empty class is
    degenerate, not high severity.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "label", severity)
    rng = np.random.default_rng(call_seed)

    classes = sorted(fold[target_col].dropna().unique())
    if len(classes) != 2:
        raise NotImplementedError(
            f"Label shift currently implements the binary case only "
            f"(corpus is binary per D1 Sec 3). Found {len(classes)} classes."
        )
    c0, c1 = classes

    # Natural prior rho = P(Y=c1) on the train fold.
    rho = float((train_fold_reference[target_col] == c1).mean())
    # Default rho_target: push toward the nearer extreme, capped away from 0/1
    # so no class empties. D4 will supply a real value; this default keeps the
    # mechanism runnable for smoke-testing without one.
    if target_prior is None:
        target_prior = 0.9 if rho < 0.5 else 0.1
    rho_shifted = (1 - severity) * rho + severity * target_prior
    rho_shifted = min(max(rho_shifted, 1e-6), 1 - 1e-6)

    idx_c0 = fold.index[fold[target_col] == c0].to_numpy()
    idx_c1 = fold.index[fold[target_col] == c1].to_numpy()
    n = len(fold)
    n_c1 = int(round(rho_shifted * n))
    n_c0 = n - n_c1

    if n_c0 <= 0 or n_c1 <= 0:
        raise ValueError(
            f"Label shift target_prior={target_prior} at severity={severity} "
            f"would drive a class to zero rows (n_c0={n_c0}, n_c1={n_c1}) for "
            f"dataset='{dataset_name}'. D1 Sec 7.2's minority-floor constraint "
            f"forbids this — pick a target_prior that keeps both classes "
            f"non-empty at this severity."
        )

    draw_c0 = rng.choice(idx_c0, size=n_c0, replace=True)
    draw_c1 = rng.choice(idx_c1, size=n_c1, replace=True)
    corrupted = fold.loc[np.concatenate([draw_c0, draw_c1])].sample(
        frac=1, random_state=seed).reset_index(drop=True)

    achieved_prior = float((corrupted[target_col] == c1).mean())

    # Validation (D1 Sec 7.2): achieved prior matches target; P(X|Y) preserved.
    # P(X|Y) is preserved by construction (within-class resampling), confirmed
    # structurally; we also spot-check one feature's within-class distribution
    # via KS between original and resampled, expecting NO significant change.
    feature_cols = _numeric_feature_cols(fold, target_col)
    pxy_checks = {}
    if feature_cols:
        f = feature_cols[0]
        for cls in (c0, c1):
            orig = fold.loc[fold[target_col] == cls, f].dropna().to_numpy()
            new = corrupted.loc[corrupted[target_col] == cls, f].dropna().to_numpy()
            if len(orig) > 1 and len(new) > 1:
                ks, pval = stats.ks_2samp(orig, new)
                pxy_checks[f"{f}|Y={cls}"] = {
                    "ks": float(ks), "pval": float(pval),
                    "preserved": pval > CONCEPT_INVARIANCE_ALPHA,
                }

    prior_tol = 0.02
    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="label_shift",
        severity=severity,
        seed=seed,
        achieved_rate={"achieved_class_prior": achieved_prior, "target_class_prior": rho_shifted},
        validation_passed=abs(achieved_prior - rho_shifted) < prior_tol,
        validation_details={
            "natural_prior_rho": rho,
            "target_prior_extreme": target_prior,
            "shifted_prior_rho_alpha": rho_shifted,
            "achieved_prior": achieved_prior,
            "conditional_feature_dist_preserved_by_construction": True,
            "pxy_spot_check": pxy_checks,
        },
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# 7.3 Slice-Conditional Concept Shift
# ---------------------------------------------------------------------------

def corrupt_concept(
    fold: pd.DataFrame,
    target_col: str,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 7.3 — Slice-Conditional Concept Shift:
        P'(Y | X in S) != P(Y | X in S),  X unchanged, S fixed.

    Within the frozen slice S, flip a proportion `severity` of labels; leave
    everything outside S untouched, and never modify any feature (only Y
    changes, and only inside S). Because X is never touched, P(X) invariance
    holds by construction — the validation confirms slice selection was
    clean, per D1 Sec 7.3's note.

    S comes from base.py's shared select_frozen_slice_bounds — the SAME slice
    uniqueness.py (9.2) will reuse. This module never computes its own slice;
    that shared source is the structural guarantee the two mechanisms align.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "concept", severity)
    rng = np.random.default_rng(call_seed)

    classes = sorted(fold[target_col].dropna().unique())
    if len(classes) != 2:
        raise NotImplementedError(
            f"Concept shift currently implements the binary case only. "
            f"Found {len(classes)} classes."
        )
    c0, c1 = classes

    # Get the frozen slice definition from the shared source (train-fold
    # estimated bounds, D2 Decision 4), then apply it to THIS fold.
    bounds = select_frozen_slice_bounds(train_fold_reference, seed, dataset_name)
    sf = bounds["slice_feature"]
    if sf not in fold.columns:
        raise RuntimeError(
            f"Slice feature '{sf}' (frozen on the train fold) is absent from "
            f"the fold being corrupted for dataset='{dataset_name}'."
        )
    in_slice = fold[sf].between(bounds["lo_val"], bounds["hi_val"]).to_numpy()

    y = fold[target_col].to_numpy()
    eligible = in_slice & (~pd.isna(fold[target_col]).to_numpy())
    flip_mask = eligible & (rng.random(len(y)) < severity)

    new_y = y.copy()
    new_y[flip_mask & (y == c0)] = c1
    new_y[flip_mask & (y == c1)] = c0
    corrupted = fold.copy()
    corrupted[target_col] = new_y

    n_in_slice = int(in_slice.sum())
    n_flipped = int(flip_mask.sum())
    achieved_within_slice_flip = n_flipped / n_in_slice if n_in_slice else 0.0

    # Validation (D1 Sec 7.3): X untouched (P(X) invariant by construction —
    # only Y changed); confirm no feature column differs from the original.
    features_untouched = all(
        fold[c].equals(corrupted[c]) for c in fold.columns if c != target_col
    )
    # Confirm the label change is concentrated INSIDE S (outside S, Y is
    # identical).
    outside = ~in_slice
    outside_unchanged = bool(np.array_equal(y[outside], new_y[outside]))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="concept_shift",
        severity=severity,
        seed=seed,
        conditioning_feature=f"slice_S({sf} in [{bounds['lo_val']:.4g}, {bounds['hi_val']:.4g}])",
        achieved_rate={"within_slice_flip_rate": achieved_within_slice_flip},
        validation_passed=(features_untouched and outside_unchanged),
        validation_details={
            "slice_feature": sf,
            "slice_bounds": [bounds["lo_val"], bounds["hi_val"]],
            "slice_fraction_target": bounds["slice_fraction"],
            "n_rows_in_slice": n_in_slice,
            "n_labels_flipped": n_flipped,
            "within_slice_flip_rate": achieved_within_slice_flip,
            "features_untouched": features_untouched,
            "labels_outside_slice_unchanged": outside_unchanged,
        },
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def corrupt(
    fold: pd.DataFrame,
    mechanism: MechanismName,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    target_col: Optional[str] = None,
    target_prior: Optional[float] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    Unified distribution-shift entry point:
        corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)

    mechanism : "covariate" (7.1) | "label" (7.2) | "concept" (7.3)
    target_col : required — all three sub-mechanisms need to know the label
        column (covariate shift to exclude it from PC1, label/concept shift to
        operate on it).
    target_prior : label-shift only — the rho_target extreme (D4 supplies the
        real value; a runnable default is used if omitted).

    All three return a corrupted fold of the SAME logical schema as the input.
    Note covariate (7.1) and label (7.2) shift change the ROW MULTISET (via
    resampling) while concept shift (7.3) preserves rows and changes only some
    labels — callers/D7 should not assume row identity is preserved across all
    three.
    """
    if target_col is None:
        raise ValueError("distribution_shift.corrupt requires target_col.")

    if mechanism == "covariate":
        return corrupt_covariate(fold, target_col, severity, seed, train_fold_reference, dataset_name)
    elif mechanism == "label":
        return corrupt_label(fold, target_col, severity, seed, train_fold_reference,
                             target_prior=target_prior, dataset_name=dataset_name)
    elif mechanism == "concept":
        return corrupt_concept(fold, target_col, severity, seed, train_fold_reference, dataset_name)
    else:
        raise ValueError(
            f"Unknown distribution shift mechanism: {mechanism!r}. Expected "
            f"'covariate', 'label', or 'concept'."
        )