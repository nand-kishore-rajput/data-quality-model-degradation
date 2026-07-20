"""
feature_noise.py

Implements D1 Section 5 — Feature Noise Mechanisms (Additive Gaussian Noise,
Categorical Swap, Quantization/Bin Collapse). Second D3 module, following
missingness.py's interface and testing discipline.

Interface: corrupt(fold, mechanism, severity, seed, train_fold_reference,
                    columns=None, dataset_name="") -> (corrupted_dataset, metadata)

Governing constraints (same as missingness.py — not re-derived, just
implemented):

  D2 Decision 2 — this module has no opinion on which fold it's given.
    Callers pass only the fold to corrupt; train stays clean elsewhere.

  D2 Decision 4 — every corruption parameter (train-fold SD for Gaussian
    noise, empirical category marginal for categorical swap, bin edges for
    quantization) is estimated from train_fold_reference only.

  D1 Section 3 — severity `alpha` is in [0, alpha_max), never approaching 1.

A structural difference from missingness.py, worth stating up front:
Additive Gaussian Noise and Categorical Swap are per-cell mechanisms (a
proportion alpha of CELLS is affected, same "selection rate" framing as
MCAR/MAR/MNAR). Quantization is NOT — D1's formula
(k = k_max - alpha*(k_max - k_min)) defines a bin COUNT applied uniformly to
every value in the column; there is no per-cell selection concept for
quantization, alpha controls coarseness, not coverage. This module's
metadata reflects that difference rather than forcing quantization into an
achieved_rate framing it doesn't have.
"""

from typing import Optional, Literal, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

from .base import (
    CorruptionMetadata, reference_zscore,
    rate_tolerance, reference_variance_issue, nan_adjusted_target,
)

MechanismName = Literal["gaussian", "categorical_swap", "quantization"]

MIN_UNIQUE_FOR_MARGINAL_TEST = 2  # a categorical column needs at least 2
    # distinct values in the reference marginal for "swap to a different
    # category" to even be meaningful.
MARGINAL_TEST_ALPHA = 0.05  # significance level for the chi-square
    # goodness-of-fit check that the post-swap marginal still resembles the
    # reference marginal (D1 Sec 5, "marginal distribution approximately
    # preserved").

QUANTIZATION_K_MIN = 3  # frozen by D1 Sec 5.3 directly: "k_min >= 3".
QUANTIZATION_K_MAX_CAP = 20  # NOT specified in D1 — D1 defines k as a
    # function of k_max and k_min but never pins down k_max itself. Flagged
    # explicitly as a D3 implementation decision: k_max is set per-column as
    # min(number of distinct values in the train-fold reference,
    # QUANTIZATION_K_MAX_CAP), so "zero severity" means "closest to the
    # feature's natural granularity, capped at a computationally reasonable
    # ceiling" rather than an arbitrarily large or unbounded bin count.


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1 — total information destruction is "
            f"degenerate, not high severity."
        )


# ---------------------------------------------------------------------------
# 5.1 Additive Gaussian Noise
# ---------------------------------------------------------------------------

def corrupt_gaussian(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 5.1 — Additive Gaussian Noise:
        X'_ij = X_ij + eps_ij,  eps_ij ~ N(0, (alpha * s_j)^2)
        s_j = train-fold SD of feature j

    Applied only to continuous features (D1 Sec 5.1's explicit restriction —
    caller's responsibility to pass only continuous columns; this function
    does not attempt to detect feature type itself, same as missingness.py's
    columns contract).

    Per-column exclusion (matching missingness.py's 2026-07-19 granularity
    decision, applied here from the start rather than retrofitted): a column
    with near-zero train-fold reference variance is excluded from this call
    (adding N(0, 0) noise is a well-defined no-op, not a crash, but silently
    achieving nothing is worse than an explicit, visible exclusion).

    Pre-existing NaN in a column requires no special handling here, unlike
    missingness.py's MAR/MNAR: NaN + eps = NaN in floating-point arithmetic,
    so already-missing cells are automatically left as NaN rather than
    getting a spurious noise value — the correct behavior falls out of
    ordinary numpy semantics, not a workaround.
    """
    _validate_severity(severity)
    rng = np.random.default_rng(seed)
    corrupted = fold.copy()
    achieved_variance = {}
    excluded_columns = {}

    for col in columns:
        variance_issue = reference_variance_issue(train_fold_reference, col)
        if variance_issue is not None:
            excluded_columns[col] = variance_issue
            continue

        s_j = train_fold_reference[col].std()
        target_variance = (severity * s_j) ** 2
        n = len(corrupted)

        eps = rng.normal(0, severity * s_j, n)
        corrupted[col] = fold[col] + eps  # NaN + eps = NaN automatically

        # Achieved variance measured only over cells that actually received
        # noise (i.e. were not pre-existing NaN) — matching the "eligible
        # subset" framing used throughout missingness.py for the same reason.
        added_noise = (corrupted[col] - fold[col]).dropna()
        achieved_variance[col] = float(added_noise.var()) if len(added_noise) > 1 else float("nan")

    if not achieved_variance:
        raise RuntimeError(
            f"Gaussian noise: every requested column was excluded (zero "
            f"reference variance) for dataset='{dataset_name}', severity="
            f"{severity}, seed={seed}. Excluded: {excluded_columns}"
        )

    # Corruption fidelity (D1 Sec 11.1): achieved noise variance vs target,
    # using a chi-square-based tolerance on sample variance rather than an
    # arbitrary percentage — same z-based philosophy as rate_tolerance,
    # applied to variance instead of a proportion. For n iid N(0,sigma^2)
    # draws, SE(sample variance) ~= sigma^2 * sqrt(2/(n-1)) for large n.
    deviations = {}
    for col, achieved_var in achieved_variance.items():
        s_j = train_fold_reference[col].std()
        target_var = (severity * s_j) ** 2
        n_eff = int((corrupted[col] - fold[col]).notna().sum())
        se = target_var * (2 / max(n_eff - 1, 1)) ** 0.5
        tol = max(3.5 * se, 1e-10)
        deviations[col] = {"target": target_var, "achieved": achieved_var,
                            "deviation": abs(achieved_var - target_var), "tolerance": tol,
                            "within_tolerance": abs(achieved_var - target_var) < tol}
    max_relative_issue = any(not d["within_tolerance"] for d in deviations.values())

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="gaussian",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_variance,  # reusing achieved_rate as the
            # generic "achieved corruption magnitude" slot — here it holds
            # variance, not a proportion; validation_details spells this out.
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=not max_relative_issue,
        validation_details={"achieved_rate_is_variance_not_proportion": True,
                             "per_column_variance_check": deviations},
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# 5.2 Categorical Swap
# ---------------------------------------------------------------------------

def corrupt_categorical_swap(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 5.2 — Categorical Swap: for a proportion `severity` of values,
    replace the category with another category sampled from the empirical
    marginal (of the train-fold reference — D2 Decision 4), excluding the
    cell's own current value, so every selected cell genuinely changes.

    Pre-existing NaN cells are not eligible for swapping (no "own value" to
    exclude, and swapping into a missing cell would fabricate data rather
    than corrupt it) — same philosophy as missingness.py's NaN handling,
    and uses the same nan_adjusted_target helper so the achieved swap rate,
    measured over ALL rows, still lands on the requested severity despite
    some rows being permanently ineligible.
    """
    _validate_severity(severity)
    rng = np.random.default_rng(seed)
    corrupted = fold.copy()
    achieved_rate = {}
    marginal_checks = {}
    excluded_columns = {}

    for col in columns:
        marginal = train_fold_reference[col].value_counts(normalize=True, dropna=True)
        if len(marginal) < MIN_UNIQUE_FOR_MARGINAL_TEST:
            excluded_columns[col] = (
                f"train-fold reference has {len(marginal)} distinct category/ies "
                f"— need at least {MIN_UNIQUE_FOR_MARGINAL_TEST} for a swap to "
                f"a genuinely different category to be possible"
            )
            continue

        adjusted_target = nan_adjusted_target(severity, fold[col])
        eligible = fold[col].notna().to_numpy()
        n = len(corrupted)
        select_mask = (rng.random(n) < adjusted_target) & eligible

        categories = marginal.index.to_numpy()
        probs = marginal.to_numpy()
        new_values = fold[col].copy()
        selected_idx = np.where(select_mask)[0]

        for i in selected_idx:
            own_value = fold[col].iloc[i]
            # Resample from the marginal, excluding the cell's own current
            # value, so a selected cell is guaranteed to actually change —
            # a no-op "swap" back to the same category wouldn't be corruption.
            candidate_mask = categories != own_value
            if not candidate_mask.any():
                continue  # only possible if len(marginal) == 1, already excluded above
            cand_categories = categories[candidate_mask]
            cand_probs = probs[candidate_mask]
            cand_probs = cand_probs / cand_probs.sum()
            new_values.iloc[i] = rng.choice(cand_categories, p=cand_probs)

        corrupted[col] = new_values
        achieved_rate[col] = float(select_mask.mean())

        # Marginal-preservation check (D1 Sec 5.2): chi-square goodness-of-fit
        # between the reference marginal and the post-corruption marginal,
        # restricted to non-NaN cells.
        #
        # Bug fixed 2026-07-20: the original version DROPPED bins with
        # expected < 5 (the standard chi-square rule of thumb) via boolean
        # indexing -- but dropping bins breaks scipy.stats.chisquare's
        # required invariant that sum(f_obs) == sum(f_exp), since the
        # dropped observed and dropped expected totals don't generally
        # cancel. This crashed on every real dataset with at least one
        # low-frequency category in its marginal -- 150/765 draws in the
        # real-corpus run, deterministically (every dataset with a rare
        # category failed 15/15; every dataset without one passed 15/15).
        # The correct fix is to POOL low-expected-count categories into a
        # single combined bin rather than excluding them -- this keeps both
        # totals exactly equal while still satisfying the expected>=5 rule.
        post_counts = corrupted[col].dropna().value_counts()
        post_counts = post_counts.reindex(categories, fill_value=0)
        expected = probs * post_counts.sum()
        low_bins = expected < 5
        n_low = int(low_bins.sum())

        if n_low == 0:
            # No pooling needed -- all categories individually clear the threshold.
            f_obs = post_counts.to_numpy()
            f_exp = expected
        elif n_low == len(categories):
            # Every category is low-frequency -- pooling collapses to one bin,
            # which chi-square can't test against (needs >=2). Skip, matching
            # the existing insufficient-data fallback below.
            f_obs, f_exp = None, None
        else:
            # Pool the low-expected-count categories into a single combined
            # "other" bin; keep the remaining categories individually.
            f_obs = np.concatenate([
                post_counts.to_numpy()[~low_bins],
                [post_counts.to_numpy()[low_bins].sum()],
            ])
            f_exp = np.concatenate([
                expected[~low_bins],
                [expected[low_bins].sum()],
            ])

        if f_obs is not None and len(f_obs) >= 2:
            chi2, pval = stats.chisquare(f_obs, f_exp=f_exp)
            marginal_checks[col] = {"chi2": float(chi2), "pval": float(pval),
                                     "marginal_preserved": pval > MARGINAL_TEST_ALPHA,
                                     "n_categories_pooled": n_low if 0 < n_low < len(categories) else 0}
        else:
            marginal_checks[col] = {"chi2": float("nan"), "pval": float("nan"),
                                     "marginal_preserved": True, "skipped_too_few_bins": True}

    if not achieved_rate:
        raise RuntimeError(
            f"Categorical swap: every requested column was excluded for "
            f"dataset='{dataset_name}', severity={severity}, seed={seed}. "
            f"Excluded: {excluded_columns}"
        )

    max_deviation = max((abs(achieved_rate[c] - severity) for c in achieved_rate), default=0.0)
    tol = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="categorical_swap",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_rate,
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=(max_deviation < tol),
        validation_details={"max_rate_deviation": max_deviation, "tolerance_used": tol,
                             "marginal_preservation": marginal_checks},
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# 5.3 Quantization / Bin Collapse
# ---------------------------------------------------------------------------

def corrupt_quantization(
    fold: pd.DataFrame,
    columns: List[str],
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 5.3 — Quantization / Bin Collapse:
        k = k_max - alpha*(k_max - k_min),  k_min >= 3
        Equal-frequency binning on the train fold.

    Structurally different from every other mechanism in this codebase so
    far: alpha controls the COARSENESS applied to every value in the column,
    not a per-cell selection rate. There is no "achieved_rate" in the
    proportion-of-cells sense — the validation question is "did we land on
    the intended number of bins," not "what fraction of cells were touched."
    seed is accepted for interface consistency but unused: bin edges are
    deterministic given the train-fold reference and k, and every eligible
    cell is transformed, so there's no random draw to seed.
    """
    _validate_severity(severity)
    corrupted = fold.copy()
    achieved_bin_count = {}
    excluded_columns = {}
    bin_details = {}

    for col in columns:
        variance_issue = reference_variance_issue(train_fold_reference, col)
        if variance_issue is not None:
            excluded_columns[col] = variance_issue
            continue

        n_unique_train = train_fold_reference[col].nunique()
        k_max = max(min(n_unique_train, QUANTIZATION_K_MAX_CAP), QUANTIZATION_K_MIN)
        k_target = round(k_max - severity * (k_max - QUANTIZATION_K_MIN))
        k_target = max(k_target, QUANTIZATION_K_MIN)

        if k_target >= n_unique_train:
            # Column already has fewer (or equal) distinct values than the
            # target bin count — quantizing would either do nothing or
            # increase resolution, neither of which is "collapse." Excluded
            # rather than silently no-op'd.
            excluded_columns[col] = (
                f"target bin count ({k_target}) >= train-fold distinct value "
                f"count ({n_unique_train}) — quantization would not coarsen "
                f"this column"
            )
            continue

        try:
            bin_edges = pd.qcut(train_fold_reference[col], q=k_target, retbins=True, duplicates="drop")[1]
        except ValueError as e:
            excluded_columns[col] = f"equal-frequency binning failed: {e}"
            continue

        if len(bin_edges) < 3:  # fewer than 2 actual bins after dropping duplicate edges
            excluded_columns[col] = (
                f"equal-frequency binning on the train-fold reference collapsed "
                f"to fewer than {QUANTIZATION_K_MIN} bins after dropping "
                f"duplicate edges (heavily tied/skewed distribution) — cannot "
                f"reach the target bin count for this column"
            )
            continue

        bin_edges = bin_edges.copy()
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_indices = pd.cut(fold[col], bins=bin_edges, labels=False, include_lowest=True)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_midpoints[0] = bin_edges[1]  # -inf edge has no finite midpoint; use the
        bin_midpoints[-1] = bin_edges[-2]  # inner edge instead, same convention both tails.

        quantized = bin_indices.map(lambda i: bin_midpoints[int(i)] if pd.notna(i) else np.nan)
        corrupted[col] = quantized

        achieved_k = int(quantized.dropna().nunique())
        achieved_bin_count[col] = achieved_k
        bin_details[col] = {"target_k": k_target, "achieved_k": achieved_k,
                             "k_max_used": k_max, "n_unique_train_fold": n_unique_train}

    if not achieved_bin_count:
        raise RuntimeError(
            f"Quantization: every requested column was excluded for "
            f"dataset='{dataset_name}', severity={severity}. "
            f"Excluded: {excluded_columns}"
        )

    # Corruption fidelity (D1 Sec 11.1) for quantization means achieved bin
    # count close to target — exact match isn't always possible (ties in
    # equal-frequency binning can merge intended bin boundaries), so allow a
    # small integer tolerance rather than requiring exact equality.
    BIN_COUNT_TOLERANCE = 1
    all_within_tolerance = all(
        abs(d["achieved_k"] - d["target_k"]) <= BIN_COUNT_TOLERANCE for d in bin_details.values()
    )

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="quantization",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_bin_count,  # holds bin counts, not proportions —
            # see validation_details flag, same pattern as corrupt_gaussian.
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=all_within_tolerance,
        validation_details={"achieved_rate_is_bin_count_not_proportion": True,
                             "bin_count_tolerance": BIN_COUNT_TOLERANCE,
                             "per_column_bin_detail": bin_details},
    )
    return corrupted, metadata


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def corrupt(
    fold: pd.DataFrame,
    mechanism: MechanismName,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    columns: Optional[List[str]] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    Unified feature-noise entry point, matching the D3 interface contract:
        corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)

    Parameters
    ----------
    fold : the fold to corrupt (D2 Decision 2 — callers pass the test fold).
    mechanism : "gaussian" | "categorical_swap" | "quantization"
    severity : alpha in [0, alpha_max) — resolved numeric severity from D4's
        grid. For "quantization", controls bin coarseness, not a selection
        rate — see corrupt_quantization's docstring.
    seed : corruption-injection random seed. Unused by "quantization", which
        is deterministic given the reference and k (accepted anyway for
        interface consistency across all three mechanisms).
    train_fold_reference : the CLEAN train fold — used only for parameter
        estimation (D2 Decision 4): reference SD for gaussian, empirical
        marginal for categorical_swap, bin edges for quantization.
    columns : eligible columns. No default type-inference is performed —
        gaussian and quantization expect continuous columns, categorical_swap
        expects categorical columns; passing the wrong type produces either
        an early exclusion (e.g. reference_variance_issue on a constant-
        looking categorical code) or a runtime error from the underlying
        pandas/numpy operation, not a silent misapplication.
    dataset_name : for metadata/logging only.
    """
    if columns is None:
        raise ValueError(
            "feature_noise.corrupt requires explicit columns — unlike "
            "missingness.py, there is no safe default (gaussian/quantization "
            "need continuous columns, categorical_swap needs categorical "
            "columns, and this module does not infer which is which)."
        )

    if mechanism == "gaussian":
        return corrupt_gaussian(fold, columns, severity, seed, train_fold_reference, dataset_name)
    elif mechanism == "categorical_swap":
        return corrupt_categorical_swap(fold, columns, severity, seed, train_fold_reference, dataset_name)
    elif mechanism == "quantization":
        return corrupt_quantization(fold, columns, severity, seed, train_fold_reference, dataset_name)
    else:
        raise ValueError(
            f"Unknown feature noise mechanism: {mechanism!r}. Expected "
            f"'gaussian', 'categorical_swap', or 'quantization'."
        )