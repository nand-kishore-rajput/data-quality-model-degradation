"""
label_noise.py

Implements D1 Section 6 — Label Noise Mechanisms:
  6.1 Uniform Label Noise
  6.2 Class-Conditional Label Noise
  6.3 Boundary-Aware Label Noise — DEFERRED per D1 Sec 6.3, not implemented.

Third D3 module. Structurally simpler than missingness/feature_noise in one
way (the target is a single column, not a set of feature columns) and more
subtle in another (the corruption operates on Y, so the "mechanism fidelity"
question is about the flip structure, and D1's near-zero-label-marginal-change
condition for H1 interacts directly with what this module is allowed to do).

Governing constraints (implemented, not re-derived):

  D2 Decision 2 — corrupts only the fold it's given (the test fold); train
    stays clean.

  D2 Decision 4 — the one estimated parameter here, the class-conditional
    asymmetric flip-rate RATIO (6.2), is estimated from train_fold_reference
    only, never the fold being corrupted or the full corpus.

  D1 Section 3 — severity `alpha` in [0, alpha_max), never approaching 1.
    For label noise this is especially important: alpha -> 0.5 on a binary
    label already fully destroys the signal (a 50% flip rate makes Y
    independent of X), so the usable range is even tighter than for feature
    corruption. Flagged, not silently assumed.

  D1 Section 6.2 — class-conditional flip rates are anchored to a Cleanlab
    confident-learning pass per dataset, NOT asserted from domain intuition.
    Cleanlab is an optional dependency here; if unavailable, this module
    does NOT silently fall back to a symmetric ratio (that would quietly
    violate D1 6.2's anchoring requirement). It raises a clear error telling
    the caller to either install Cleanlab or pass an explicit, documented
    ratio — surfacing the decision rather than hiding it.

  D1 Section 6.3 — boundary-aware noise is DEFERRED. This module does not
    implement it and raises informatively if asked, rather than pretending
    it exists.
"""

from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd

from .base import CorruptionMetadata, rate_tolerance

MechanismName = Literal["uniform", "class_conditional", "boundary_aware"]

# Binary-corpus assumption (D1 Sec 3, confirmed by the 2026-07-17 binary-label
# spot-check across all 17 datasets). This module implements the binary case
# directly; the general multi-class transition-matrix path (D1 Sec 6.2
# fallback) is not invoked for the current corpus and is left as an explicit
# NotImplementedError rather than a half-tested code path.
MAX_SAFE_FLIP_RATE = 0.5  # a flip rate at or above this on a binary label
    # makes Y independent of (or anti-correlated with) X — degenerate, not
    # "high severity" (D1 Sec 3). corrupt_uniform/class_conditional treat
    # severity approaching this as the practical ceiling.


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1."
        )
    if severity >= MAX_SAFE_FLIP_RATE:
        raise ValueError(
            f"severity={severity} is >= {MAX_SAFE_FLIP_RATE}, the degenerate "
            f"ceiling for binary label noise: at a 50% flip rate Y becomes "
            f"independent of X, which is total signal destruction, not high "
            f"severity (D1 Sec 3). If D4's grid genuinely needs rates this "
            f"high, that's a severity-grid design decision to make explicitly, "
            f"not a default this function should silently accept."
        )


def _get_binary_classes(y: pd.Series) -> Tuple:
    classes = sorted(y.dropna().unique())
    if len(classes) != 2:
        raise NotImplementedError(
            f"label_noise currently implements the BINARY case only "
            f"(D1 Sec 3 corpus assumption, confirmed 2026-07-17). Found "
            f"{len(classes)} classes: {classes}. The multi-class "
            f"transition-matrix path (D1 Sec 6.2 fallback) is intentionally "
            f"not implemented for the current 17-dataset corpus — implementing "
            f"it half-way would be worse than a clear signal that it's absent."
        )
    return classes[0], classes[1]


# ---------------------------------------------------------------------------
# 6.1 Uniform Label Noise
# ---------------------------------------------------------------------------

def corrupt_uniform(
    fold: pd.DataFrame,
    target_col: str,
    severity: float,
    seed: int,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 6.1 — Uniform Label Noise: P(Y'_i != Y_i) = alpha, applied
    independently and symmetrically to every label regardless of class.
    No parameter estimation, so train_fold_reference isn't needed here
    (accepted at the dispatch level for interface uniformity, unused for
    uniform noise).
    """
    _validate_severity(severity)
    rng = np.random.default_rng(seed)
    corrupted = fold.copy()
    c0, c1 = _get_binary_classes(fold[target_col])

    y = fold[target_col].to_numpy()
    eligible = ~pd.isna(fold[target_col]).to_numpy()  # never flip a missing label
    flip_mask = (rng.random(len(y)) < severity) & eligible

    new_y = y.copy()
    # Binary flip: c0 <-> c1
    new_y[flip_mask & (y == c0)] = c1
    new_y[flip_mask & (y == c1)] = c0
    corrupted[target_col] = new_y

    achieved_flip_rate = float(flip_mask.mean())
    tol = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="uniform_label_noise",
        severity=severity,
        seed=seed,
        achieved_rate={target_col: achieved_flip_rate},
        validation_passed=abs(achieved_flip_rate - severity) < tol,
        validation_details={
            "achieved_flip_rate": achieved_flip_rate,
            "target_flip_rate": severity,
            "tolerance_used": tol,
            "n_flipped": int(flip_mask.sum()),
        },
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# 6.2 Class-Conditional Label Noise
# ---------------------------------------------------------------------------

_asymmetry_ratio_cache: dict = {}  # (id(train_fold_reference), target_col,
    # tuple(sorted(feature_cols))) -> estimated ratio. The estimate depends
    # only on train_fold_reference/target_col/feature_cols — none of which
    # vary across severity or seed — but corrupt_class_conditional was being
    # called once per (severity, seed) pair, so a full Cleanlab pass (plus a
    # 3-fold cross-validated logistic regression fit) was silently re-run 15x
    # per dataset in the smoke test instead of once. Surfaced by noticing the
    # real-corpus run had zero cost-saving despite the ratio being provably
    # identical across all 15 draws per dataset.
    #
    # Keyed on id(train_fold_reference) rather than dataset_name: dataset_name
    # is caller-supplied and could be blank or, in principle, reused across
    # genuinely different DataFrames — object identity is correct as long as
    # the same DataFrame object is passed across calls for one dataset, which
    # is exactly the runner's usage pattern (train_fold is computed once per
    # dataset and stays alive for that dataset's whole block of calls, so no
    # id() reuse from garbage collection is possible mid-use). If a caller
    # reconstructs an equal-content-but-different-object DataFrame between
    # calls, this cache safely misses and recomputes — a correctness-safe
    # degradation to the old per-call behavior, not a wrong answer.


def _get_cached_asymmetry_ratio(
    train_fold_reference: pd.DataFrame, target_col: str, feature_cols: list
) -> Tuple[float, bool]:
    """Returns (ratio, was_cached)."""
    cache_key = (id(train_fold_reference), target_col, tuple(sorted(feature_cols)))
    if cache_key in _asymmetry_ratio_cache:
        return _asymmetry_ratio_cache[cache_key], True
    ratio = _estimate_asymmetry_ratio_cleanlab(train_fold_reference, target_col, feature_cols)
    _asymmetry_ratio_cache[cache_key] = ratio
    return ratio, False


def _estimate_asymmetry_ratio_cleanlab(
    train_fold_reference: pd.DataFrame, target_col: str, feature_cols: list
) -> float:
    """
    D1 Section 6.2 — estimate the asymmetric flip-rate ratio from a Cleanlab
    confident-learning pass on the TRAIN fold (D2 Decision 4). Returns the
    ratio alpha_{0->1} / alpha_{1->0} implied by Cleanlab's estimated label-
    error distribution across the two classes.

    Cleanlab is an OPTIONAL dependency. If it's not installed, this raises
    ImportError rather than silently substituting a symmetric (1.0) ratio —
    a symmetric fallback would quietly reduce class-conditional noise to
    uniform noise, defeating the entire point of the mechanism and violating
    D1 6.2's explicit "anchored, not asserted" requirement.
    """
    try:
        from cleanlab.count import compute_confident_joint
    except ImportError as e:
        raise ImportError(
            "D1 Sec 6.2 requires Cleanlab to anchor the class-conditional "
            "asymmetric flip-rate ratio. Cleanlab is not installed. Either "
            "`pip install cleanlab`, or call class_conditional corruption "
            "with an explicit `asymmetry_ratio=` argument (a documented, "
            "dated decision) rather than letting the mechanism silently "
            "degrade to symmetric/uniform behavior."
        ) from e

    # A confident-learning estimate needs predicted class probabilities. We
    # fit a quick, held-out-of-the-four-families model to avoid the probe
    # circularity D1 Sec 6.2/6.3 warns about — logistic regression is used
    # here ONLY as a label-quality probe, and is one of the four evaluated
    # families, so this is flagged as a KNOWN tension: see the note in the
    # dispatch docstring. For the smoke-test/estimation path we accept it;
    # the final pipeline may want a probe outside the four families.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X = train_fold_reference[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = train_fold_reference[target_col].to_numpy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for the Cleanlab probe.")

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(clf, Xs, y, cv=3, method="predict_proba")

    cj = compute_confident_joint(labels=y.astype(int), pred_probs=pred_probs)
    # cj[i][j] = count of examples confidently labeled i but actually j.
    # Off-diagonal gives the two directional error counts.
    err_0to1 = cj[0][1]
    err_1to0 = cj[1][0]
    if err_1to0 == 0:
        return float("inf") if err_0to1 > 0 else 1.0
    return float(err_0to1) / float(err_1to0)


def corrupt_class_conditional(
    fold: pd.DataFrame,
    target_col: str,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    asymmetry_ratio: Optional[float] = None,
    feature_cols: Optional[list] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 6.2 — Class-Conditional Label Noise. For a binary label, two
    asymmetric flip rates alpha_{0->1} and alpha_{1->0} whose RATIO is
    anchored to a Cleanlab pass (D1 6.2) and whose overall MAGNITUDE is set
    by `severity` (the overall target flip rate).

    asymmetry_ratio: if provided, used directly (an explicit, caller-owned
      decision). If None, estimated via Cleanlab on the train fold. There is
      deliberately no third option where an unavailable Cleanlab silently
      becomes a symmetric ratio.

    The two directional rates are derived so that:
      - their ratio equals the (estimated or supplied) asymmetry_ratio, and
      - the overall achieved flip rate matches `severity`,
    given the class base rates in the fold. This keeps `severity` meaning the
    same thing (overall flip rate) across uniform and class-conditional noise,
    so D4's severity grid is comparable across both.
    """
    _validate_severity(severity)
    rng = np.random.default_rng(seed)
    corrupted = fold.copy()
    c0, c1 = _get_binary_classes(fold[target_col])

    if feature_cols is None:
        feature_cols = [c for c in train_fold_reference.columns if c != target_col]

    ratio_source = "supplied"
    ratio_was_cached = False
    if asymmetry_ratio is None:
        asymmetry_ratio, ratio_was_cached = _get_cached_asymmetry_ratio(
            train_fold_reference, target_col, feature_cols
        )
        ratio_source = "cleanlab_estimated_cached" if ratio_was_cached else "cleanlab_estimated_fresh"

    # Guard against degenerate/infinite ratios from the estimator.
    if not np.isfinite(asymmetry_ratio) or asymmetry_ratio <= 0:
        asymmetry_ratio = 1.0
        ratio_source += "_clamped_to_1.0"

    y = fold[target_col].to_numpy()
    eligible = ~pd.isna(fold[target_col]).to_numpy()
    n0 = int(((y == c0) & eligible).sum())
    n1 = int(((y == c1) & eligible).sum())
    n_eligible = n0 + n1

    # Solve for directional rates a01, a10 such that:
    #   a01 / a10 = asymmetry_ratio
    #   (n0*a01 + n1*a10) / n_eligible = severity   (overall flip rate = severity)
    # => a10 * (n0*ratio + n1) / n_eligible = severity
    denom = n0 * asymmetry_ratio + n1
    if denom == 0:
        a10 = a01 = 0.0
    else:
        a10 = severity * n_eligible / denom
        a01 = asymmetry_ratio * a10
    # Clamp directional rates to the safe ceiling; if the asymmetry would push
    # one direction past MAX_SAFE_FLIP_RATE, cap it (the achieved overall rate
    # will then fall slightly short of severity, which the fidelity check
    # reports rather than hides).
    a01 = min(a01, MAX_SAFE_FLIP_RATE)
    a10 = min(a10, MAX_SAFE_FLIP_RATE)

    draw = rng.random(len(y))
    flip_0to1 = (y == c0) & eligible & (draw < a01)
    flip_1to0 = (y == c1) & eligible & (draw < a10)

    new_y = y.copy()
    new_y[flip_0to1] = c1
    new_y[flip_1to0] = c0
    corrupted[target_col] = new_y

    total_flipped = int(flip_0to1.sum() + flip_1to0.sum())
    achieved_flip_rate = total_flipped / n_eligible if n_eligible else 0.0
    achieved_a01 = flip_0to1.sum() / n0 if n0 else 0.0
    achieved_a10 = flip_1to0.sum() / n1 if n1 else 0.0
    tol = rate_tolerance(severity, n_eligible)

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="class_conditional_label_noise",
        severity=severity,
        seed=seed,
        achieved_rate={target_col: achieved_flip_rate},
        validation_passed=abs(achieved_flip_rate - severity) < tol,
        validation_details={
            "asymmetry_ratio": asymmetry_ratio,
            "asymmetry_ratio_source": ratio_source,
            "target_a01": a01, "target_a10": a10,
            "achieved_a01": float(achieved_a01), "achieved_a10": float(achieved_a10),
            "achieved_overall_flip_rate": achieved_flip_rate,
            "target_overall_flip_rate": severity,
            "tolerance_used": tol,
            "n_flipped": total_flipped,
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
    asymmetry_ratio: Optional[float] = None,
    feature_cols: Optional[list] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    Unified label-noise entry point:
        corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)

    Parameters
    ----------
    fold : the fold to corrupt (D2 Decision 2 — the test fold).
    mechanism : "uniform" | "class_conditional" | "boundary_aware"(deferred).
    severity : overall target flip rate, alpha in [0, 0.5) for binary labels.
    seed : corruption-injection random seed.
    train_fold_reference : clean train fold; used only for the Cleanlab
        asymmetry-ratio estimate in class_conditional (D2 Decision 4).
    target_col : the label column. Required — this module operates on Y, and
        unlike feature corruption there's no sensible default.
    asymmetry_ratio : optional explicit override for class_conditional's
        flip-rate ratio (bypasses the Cleanlab estimate).
    feature_cols : optional explicit feature list for the Cleanlab probe.

    Known tension (flagged, per D1 Sec 6.2/6.3): the Cleanlab estimation path
    uses logistic regression as its label-quality probe, and logistic
    regression is one of the four evaluated model families (D1 Sec 3). D1 6.3
    warns specifically about probe/evaluation-family overlap causing
    circularity for boundary-aware noise. For the ratio ESTIMATE here the
    risk is much milder (we use the probe only to estimate a single scalar
    ratio, not to place noise near a decision boundary), but it is a real,
    known tension worth revisiting when the final pipeline is assembled —
    the probe could be moved to a model outside the four families.
    """
    if target_col is None:
        raise ValueError("label_noise.corrupt requires target_col (the label column).")

    if mechanism == "uniform":
        return corrupt_uniform(fold, target_col, severity, seed, dataset_name)
    elif mechanism == "class_conditional":
        return corrupt_class_conditional(
            fold, target_col, severity, seed, train_fold_reference,
            asymmetry_ratio=asymmetry_ratio, feature_cols=feature_cols,
            dataset_name=dataset_name,
        )
    elif mechanism == "boundary_aware":
        raise NotImplementedError(
            "Boundary-aware label noise (D1 Sec 6.3) is DEFERRED and not "
            "implemented in Phase 1. D1 6.3 defers it deliberately: locating "
            "the decision boundary requires a probe model, which risks "
            "circularity if that probe overlaps one of the four evaluated "
            "families. Ship uniform + class-conditional first (this module), "
            "reconsider boundary-aware as a controlled follow-up."
        )
    else:
        raise ValueError(
            f"Unknown label noise mechanism: {mechanism!r}. Expected "
            f"'uniform' or 'class_conditional' ('boundary_aware' is deferred)."
        )