"""
base.py

Shared infrastructure for D3's corruption engine modules
(missingness.py, feature_noise.py, label_noise.py, distribution_shift.py).

Per the D3 spec (Phase 1 Requirements session):
    Output:    src/corruption_engine/{missingness,feature_noise,label_noise,distribution_shift,base}.py
    Interface: corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)

This module holds only what's genuinely shared across mechanism classes:
the metadata schema every module logs into (feeds D7's manifest), and the
two math primitives (sigmoid, reference-anchored z-score) that MAR/MNAR
(missingness) and covariate/label shift (distribution_shift) both need.
Mechanism-specific logic stays in its own module.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class CorruptionMetadata:
    """
    Logged per D1 Section 12 (Reproducibility Requirements) and consumed by
    D7's manifest. One instance per (dataset, fold, mechanism, severity, seed)
    corruption call.
    """
    dataset: str
    mechanism: str
    severity: float
    seed: int
    conditioning_feature: Optional[str] = None
    achieved_rate: Optional[dict] = None
    excluded_columns: Optional[dict] = None  # {column: reason} for columns
        # excluded from this call due to a failed mechanism-fidelity check
        # (per-column granularity, added 2026-07-19) — these columns were
        # NOT corrupted in this call, as opposed to columns present in
        # achieved_rate, which were.
    validation_passed: Optional[bool] = None
    validation_details: dict = field(default_factory=dict)

    def to_manifest_row(self) -> dict:
        """Flatten to a single dict suitable for D7's manifest — one row per call."""
        return {
            "dataset": self.dataset,
            "mechanism": self.mechanism,
            "severity": self.severity,
            "seed": self.seed,
            "conditioning_feature": self.conditioning_feature,
            "achieved_rate": self.achieved_rate,
            "excluded_columns": self.excluded_columns,
            "validation_passed": self.validation_passed,
            "validation_details": self.validation_details,
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def reference_zscore(series: pd.Series, ref_mean: float, ref_std: float) -> pd.Series:
    """
    Z-score a series using a REFERENCE mean/std (the clean train fold),
    never the series' own mean/std.

    Required by D2 Decision 4: all corruption parameters — including any
    normalization statistics used inside a mechanism's functional form —
    are estimated from the current fold's training data only, never from
    the fold being corrupted and never from the full corpus.
    """
    if ref_std == 0 or np.isnan(ref_std):
        # Constant reference feature: z is undefined: return all-zero so the
        # mechanism degenerates to its alpha=0 behavior rather than raising,
        # since a constant conditioning feature is a data property, not a
        # corruption-engine bug.
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - ref_mean) / ref_std


def solve_logit_offset_for_target_rate(
    z: np.ndarray,
    target_rate: float,
    slope: float,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Bisection search for offset b such that mean(sigmoid(b + slope*z)) ~= target_rate.

    Why the offset exists at all: D1's MAR/MNAR formulas are written as
    P = sigmoid(alpha * z), with no intercept term. Taken completely
    literally, that gives P = sigmoid(0) = 0.5 at z = 0 regardless of alpha —
    meaning even alpha -> 0 would produce ~50% missingness for average-valued
    rows, not the near-zero rate a "severity" parameter should give at low
    severity. D1 Section 11.1 requires the ACHIEVED rate to match the target
    severity, which the raw formula can't satisfy without this calibration
    step — a D3 implementation decision filling a gap D1 Section 10 scopes
    to D3/D4, not a reinterpretation of D1's math. The slope (alpha's
    effective steepness) is a separate, fixed shape constant — see
    ALPHA_SCALE in missingness.py — distinct from this offset.

    Handles pre-existing NaN in z (from features that already have real
    missing values, e.g. Airbnb): plain .mean() on an array containing NaN
    returns NaN, and every comparison against NaN is False in numpy, which
    made the bisection always take the same branch and collapse to its -50
    floor regardless of target_rate — surfaced by testing against real data
    with genuine pre-existing missingness, where this produced an achieved
    rate of exactly 0.0 at every requested severity. Fixed by calibrating
    against only the non-NaN z values; rows with NaN z (pre-existing missing
    conditioning/self value) get p_missing = NaN downstream, which correctly
    excludes them from additional injected missingness rather than assigning
    them an undefined probability.
    """
    z = z[~np.isnan(z)]
    if len(z) == 0:
        raise ValueError(
            "All z-values are NaN — every row has a pre-existing missing "
            "conditioning/self value in this fold. Cannot calibrate a "
            "missingness rate with no valid rows to condition on."
        )
    if target_rate <= 0:
        return -50.0
    if target_rate >= 1:
        return 50.0
    lo, hi = -50.0, 50.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + slope * z).mean()
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Shared validation/calibration helpers — originally written for missingness.py
# (MCAR/MAR/MNAR), moved here since feature_noise.py (additive Gaussian noise,
# categorical swap) needs the same rate-tolerance, reference-variance, and
# NaN-fraction-adjustment logic. Not mechanism-specific despite the "rate"
# framing: any mechanism that selects a proportion of cells via a Bernoulli
# draw and needs that proportion to match a requested severity uses these.
# ---------------------------------------------------------------------------

RATE_TOLERANCE_Z = 3.5  # multiplier on the binomial standard error for
    # "achieved rate matches target" checks (D1 Sec 11.1). A flat percentage-
    # point tolerance was tried first and found miscalibrated — it produced a
    # severity-dependent false-flag rate on real data (see missingness.py's
    # amendment history) because true sampling noise scales with
    # sqrt(rate*(1-rate)/n), not with a fixed number.
MIN_ABS_TOLERANCE = 1e-6  # floor for the tolerance below: the SE formula
    # degenerates to exactly 0 at rate=0 (and near 1), which would make even
    # a perfect 0.0 deviation fail a strict less-than check against a 0.0
    # tolerance — a real edge case (severity=0), not just theoretical.

MIN_REFERENCE_STD = 1e-8  # below this, a column is treated as effectively
    # constant in the train-fold reference — no variance for a mechanism
    # whose corruption depends on that variance (a z-score, a noise scale
    # proportional to it, etc.) to operate meaningfully on.


def rate_tolerance(target_rate: float, n: int) -> float:
    """Binomial standard-error-based tolerance for 'achieved proportion of
    cells matches target' checks, floored to handle the target_rate=0 (or
    near-1) edge case where the raw SE formula is exactly/near 0."""
    se = (target_rate * (1 - target_rate) / max(n, 1)) ** 0.5
    return max(RATE_TOLERANCE_Z * se, MIN_ABS_TOLERANCE)


def reference_variance_issue(train_fold_reference: pd.DataFrame, col: str) -> Optional[str]:
    """Returns a reason string if col's reference std is below
    MIN_REFERENCE_STD (effectively constant), or None if it's fine. Used for
    per-column exclusion (not a raise) by any mechanism whose corruption
    magnitude/probability depends on that column's reference variance."""
    ref_std = train_fold_reference[col].std()
    if ref_std < MIN_REFERENCE_STD or np.isnan(ref_std):
        return (f"train-fold reference std is {ref_std!r} (< {MIN_REFERENCE_STD}), "
                f"effectively constant — no variance for this mechanism to "
                f"operate on")
    return None


def nan_adjusted_target(severity: float, reference_values: pd.Series) -> float:
    """
    Rows/cells with pre-existing NaN in the relevant column can never be
    selected for additional corruption (no valid value to condition on or
    modify). Left uncorrected, an achieved rate measured as a fraction of
    ALL rows — including these permanently-ineligible ones — systematically
    undershoots the requested severity by roughly nan_fraction * severity.
    Surfaced by testing missingness.py against Airbnb, which has substantial
    genuine pre-existing missingness (a documented Phase 0 realism anchor).

    Scales the target passed to whatever calibration/selection step follows
    so the eligible subset's rate, once diluted by the permanently-ineligible
    fraction, lands back on the originally requested severity.
    """
    nan_fraction = reference_values.isna().mean()
    if nan_fraction >= 1.0:
        return severity
    return min(severity / (1 - nan_fraction), 1 - 1e-6)


# ---------------------------------------------------------------------------
# Shared frozen slice S — used by BOTH distribution_shift.py (D1 Sec 7.3,
# slice-conditional concept shift) and, later, uniqueness.py (D1 Sec 9.2,
# slice-conditional duplication). D1 requires these two mechanisms to operate
# on the SAME slice, so D11's stratified evaluation can ask whether a given
# region degrades under concept shift AND duplication, or only one. Defining
# S in one shared, deterministic function — rather than computing it
# independently in each module — is what makes "same slice" a structural
# guarantee instead of a convention two modules have to remember to honor.
# ---------------------------------------------------------------------------

SLICE_FRACTION = 0.30  # target fraction of rows the slice S covers. NOT
    # specified by D1 — D1 says S is "chosen once, frozen, documented per
    # dataset" but doesn't fix its size. Flagged as a D3 implementation
    # decision: 0.30 is large enough that within-slice corruption (concept
    # shift, duplication) has a measurable effect, small enough that S is a
    # genuine sub-region rather than "most of the data." Worth confirming
    # against D4/D5 when concept-shift severity is anchored.

_slice_cache: dict = {}  # (dataset, seed) -> boolean mask over train-fold rows.
    # In-memory only, per-run. Deliberately NOT persisted to disk: S is fully
    # determined by (train_fold_reference contents, seed), so recomputing it
    # is safe and cheap, whereas a stale on-disk slice surviving a code or
    # data change would silently violate the "frozen from THIS fold" guarantee
    # — the exact kind of undocumented cross-run state the project's audit
    # philosophy warns against.


def select_frozen_slice(
    train_fold_reference: pd.DataFrame,
    seed: int,
    dataset_name: str = "",
    slice_fraction: float = SLICE_FRACTION,
) -> np.ndarray:
    """
    Deterministically select the frozen slice S as a boolean mask over the
    train-fold reference's rows. Same (train_fold_reference, seed, dataset)
    always yields the same S — by construction, not by luck.

    Definition of S (a D3 implementation choice filling D1's "chosen once,
    frozen" without pinning the selection rule): S is a contiguous region in
    the space of the highest-variance numeric feature — specifically, the
    rows whose value of that feature falls within a randomly-placed (seeded)
    window covering `slice_fraction` of the rows. A feature-space region,
    rather than a purely random row sample, is chosen deliberately: concept
    shift "within a slice" (D1 Sec 7.3) is only a meaningful, interpretable
    manipulation if the slice corresponds to an actual region of feature
    space (e.g. "high-income applicants", "older vehicles") rather than an
    arbitrary scattering of unrelated rows. The seed places the window so S
    isn't always the same tail of the distribution across seeds.

    Returned as a mask over the TRAIN fold. Callers applying corruption to a
    TEST fold re-derive membership on the test fold using the same feature
    and threshold bounds (which are what actually get frozen and returned in
    the companion metadata via select_frozen_slice_bounds) — see
    distribution_shift.py for how the bounds transfer across folds.
    """
    cache_key = (dataset_name, seed, round(slice_fraction, 6))
    if cache_key in _slice_cache:
        return _slice_cache[cache_key]

    numeric_cols = [c for c in train_fold_reference.columns
                    if pd.api.types.is_numeric_dtype(train_fold_reference[c])]
    if not numeric_cols:
        raise ValueError(
            f"select_frozen_slice needs at least one numeric feature to define "
            f"a feature-space region for dataset='{dataset_name}'; none found."
        )
    # Highest-variance numeric feature, measured on the train fold (D2 Dec 4).
    variances = {c: train_fold_reference[c].var() for c in numeric_cols}
    variances = {c: v for c, v in variances.items() if pd.notna(v) and v > 0}
    if not variances:
        raise ValueError(
            f"All numeric features are constant/NaN-variance for "
            f"dataset='{dataset_name}'; cannot define a feature-space slice."
        )
    slice_feature = max(variances, key=variances.get)

    vals = train_fold_reference[slice_feature]
    lo_q, hi_q = _placed_window(seed, slice_fraction)
    lo_val = vals.quantile(lo_q)
    hi_val = vals.quantile(hi_q)
    mask = ((vals >= lo_val) & (vals <= hi_val)).to_numpy()

    _slice_cache[cache_key] = mask
    # Stash the transferable bounds so a test-fold caller can reconstruct the
    # SAME region without access to the train fold's quantiles.
    _slice_cache[(cache_key, "bounds")] = {
        "slice_feature": slice_feature,
        "lo_val": float(lo_val),
        "hi_val": float(hi_val),
        "slice_fraction": slice_fraction,
    }
    return mask


def select_frozen_slice_bounds(
    train_fold_reference: pd.DataFrame,
    seed: int,
    dataset_name: str = "",
    slice_fraction: float = SLICE_FRACTION,
) -> dict:
    """
    Return the frozen slice's transferable definition:
        {slice_feature, lo_val, hi_val, slice_fraction}
    computed on the train fold. A caller corrupting a TEST fold uses these
    bounds to select the same feature-space region on the test fold
    (test_fold[slice_feature].between(lo_val, hi_val)), guaranteeing
    distribution_shift.py's §7.3 and uniqueness.py's §9.2 operate on the
    identical region definition, transferred via train-fold-estimated bounds
    (D2 Decision 4) rather than recomputed per fold.
    """
    # Ensure the slice (and thus its cached bounds) has been computed.
    select_frozen_slice(train_fold_reference, seed, dataset_name, slice_fraction)
    cache_key = (dataset_name, seed, round(slice_fraction, 6))
    return _slice_cache[(cache_key, "bounds")]


def _placed_window(seed: int, width: float) -> tuple:
    """Deterministically place a quantile window of the given width in [0,1]
    using the seed, so the slice isn't always the same tail across seeds."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0, 1 - width)
    return lo, lo + width