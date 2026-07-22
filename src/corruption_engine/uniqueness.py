"""
uniqueness.py

Implements D1 Section 9 — Uniqueness Violations. Sixth and final D3
mechanism module.

    9.1 Global Duplication
    9.2 Slice-Conditional Duplication

Duplicating a row is equivalent, in an empirical-risk-minimization sense, to
increasing its weight in the training objective — this is the causal
mechanism by which uniqueness violations degrade performance, distinct from
the noise/removal mechanisms in every other D3 module.

Structural note, unlike every other module: duplication APPENDS rows rather
than transforming existing ones. The corrupted output is LARGER than the
input fold (n_original + n_appended), not the same size. Callers/D7 should
not assume row-count invariance for this mechanism the way it's safe to for
missingness, feature_noise, or label_noise.

Governing constraints (implemented, not re-derived):

  D2 Decision 2 — corrupts only the fold it's given (test fold).
  D1 Section 9.1 — sampled WITH replacement, appended as EXACT duplicates
    (no jitter — jitter would conflate this with feature_noise's Gaussian
    mechanism, corrupting the same underlying signal two different mechanism
    modules both claim to own).
  D1 Section 9.2 — the slice S is the SAME slice distribution_shift.py's
    concept shift (Sec 7.3) uses — this module gets it from base.py's shared
    select_frozen_slice_bounds, never computing its own, which is the
    structural guarantee (not just a convention) that the two mechanisms
    align, verified directly when that shared function was built.
  D1 Section 9 (both variants) — severity is anchored to naturally occurring
    duplicate rows already documented in Phase 0 (e.g. HELOC's 588 duplicate
    rows); this module takes severity as an already-resolved input (D4's
    job), same as every other mechanism module.
  D1 Section 9.1's explicit design note — global duplication is retained as
    a deliberately weak/near-null-effect regime: a defect present but
    producing ~zero measured degradation in most datasets is a hard-negative
    case for H1, not a flaw to engineer away.
"""

from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd

from .base import CorruptionMetadata, select_frozen_slice_bounds, derive_call_seed

MechanismName = Literal["global", "slice_conditional"]


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1 (inherited here as the standing D1 convention "
            f"— duplication doesn't have the same sharp 'total signal "
            f"destruction' ceiling label noise or missingness does, but the "
            f"bound is kept consistent across every mechanism module rather "
            f"than special-cased)."
        )


def corrupt_global(
    fold: pd.DataFrame,
    severity: float,
    seed: int,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 9.1 — Global Duplication. k = round(severity * n) records
    sampled uniformly WITH replacement from the whole fold, appended as
    exact duplicates (byte-identical, no jitter). Output row order is
    shuffled so the appended duplicates aren't trivially identifiable by
    their position at the end of the DataFrame.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "global", severity)
    rng = np.random.default_rng(call_seed)
    n = len(fold)
    k = round(severity * n)

    if k == 0:
        corrupted = fold.reset_index(drop=True)
        sampled_idx = np.array([], dtype=int)
    else:
        sampled_idx = rng.choice(n, size=k, replace=True)
        duplicates = fold.iloc[sampled_idx].reset_index(drop=True)
        corrupted = pd.concat([fold, duplicates], ignore_index=True)
        corrupted = corrupted.sample(frac=1, random_state=seed).reset_index(drop=True)

    achieved_rate = k / n if n else 0.0

    # Baseline check (not part of the corruption itself, purely diagnostic):
    # how many duplicate rows already existed in the ORIGINAL fold, before
    # any injection. Several corpus datasets (jm1, HELOC) have documented
    # natural duplicates from Phase 0 — reporting this separately from the
    # injected count keeps the two from being conflated, since generic
    # post-hoc duplicate detection on the corrupted output would count both
    # together and overstate what THIS mechanism actually did.
    pre_existing_duplicate_rate = float(fold.duplicated().mean()) if n else 0.0

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="global_duplication",
        severity=severity,
        seed=seed,
        achieved_rate={"duplication_rate": achieved_rate},
        validation_passed=(k == round(severity * n)),  # exact by construction
            # — k is directly computed, not achieved via independent
            # per-row Bernoulli draws the way rate-based mechanisms
            # elsewhere in this codebase are, so there is no sampling
            # noise to tolerate here.
        validation_details={
            "n_original": n,
            "n_appended": int(k),
            "n_final": len(corrupted),
            "achieved_duplication_rate": achieved_rate,
            "target_duplication_rate": severity,
            "pre_existing_duplicate_rate_in_original_fold": pre_existing_duplicate_rate,
            "uniform_sampling_by_construction": True,
            "note": "Output row count exceeds input (n_original + n_appended) — this "
                    "mechanism appends rows, it does not transform in place.",
        },
    )
    return corrupted, metadata


def corrupt_slice_conditional(
    fold: pd.DataFrame,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 9.2 — Slice-Conditional Duplication. Same mechanics as global
    duplication, restricted to rows within the frozen slice S — the SAME
    slice distribution_shift.py's concept shift (7.3) uses, obtained from
    base.py's shared select_frozen_slice_bounds rather than computed here.
    Sampling (with replacement) and appending both draw only from the
    in-slice subset; duplicates remain in S by construction, since nothing
    outside S is ever touched or sampled from.
    """
    _validate_severity(severity)
    call_seed = derive_call_seed(seed, dataset_name, "slice_conditional", severity)
    rng = np.random.default_rng(call_seed)

    bounds = select_frozen_slice_bounds(train_fold_reference, seed, dataset_name)
    sf = bounds["slice_feature"]
    if sf not in fold.columns:
        raise RuntimeError(
            f"Slice feature '{sf}' (frozen on the train fold) is absent from "
            f"the fold being corrupted for dataset='{dataset_name}'."
        )
    in_slice_mask = fold[sf].between(bounds["lo_val"], bounds["hi_val"]).to_numpy()
    in_slice_idx = np.where(in_slice_mask)[0]
    n_in_slice = len(in_slice_idx)

    if n_in_slice == 0:
        raise RuntimeError(
            f"No rows fall within the frozen slice S (feature='{sf}', "
            f"bounds=[{bounds['lo_val']:.4g}, {bounds['hi_val']:.4g}]) in this "
            f"fold for dataset='{dataset_name}' — cannot duplicate within an "
            f"empty slice."
        )

    k = round(severity * n_in_slice)

    if k == 0:
        corrupted = fold.reset_index(drop=True)
        sampled_idx = np.array([], dtype=int)
    else:
        sampled_idx = rng.choice(in_slice_idx, size=k, replace=True)
        duplicates = fold.iloc[sampled_idx].reset_index(drop=True)
        corrupted = pd.concat([fold, duplicates], ignore_index=True)
        corrupted = corrupted.sample(frac=1, random_state=seed).reset_index(drop=True)

    achieved_rate = k / n_in_slice if n_in_slice else 0.0

    # Confirm concentration: every appended duplicate's slice-feature value
    # falls within [lo_val, hi_val] by construction (sampled_idx only ever
    # drawn from in_slice_idx) -- verified explicitly here rather than only
    # asserted, since this is exactly the property D1 9.2's validation asks
    # for ("duplicates remain concentrated within S").
    if k > 0:
        dup_vals = fold[sf].to_numpy()[sampled_idx]
        duplicates_in_slice = bool(np.all((dup_vals >= bounds["lo_val"]) & (dup_vals <= bounds["hi_val"])))
    else:
        duplicates_in_slice = True

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="slice_conditional_duplication",
        severity=severity,
        seed=seed,
        conditioning_feature=f"slice_S({sf} in [{bounds['lo_val']:.4g}, {bounds['hi_val']:.4g}])",
        achieved_rate={"within_slice_duplication_rate": achieved_rate},
        validation_passed=(k == round(severity * n_in_slice)) and duplicates_in_slice,
        validation_details={
            "slice_feature": sf,
            "slice_bounds": [bounds["lo_val"], bounds["hi_val"]],
            "n_rows_in_slice": n_in_slice,
            "n_appended": int(k),
            "n_final": len(corrupted),
            "achieved_within_slice_duplication_rate": achieved_rate,
            "target_within_slice_duplication_rate": severity,
            "duplicates_confirmed_in_slice": duplicates_in_slice,
            "note": "Output row count exceeds input (n_original + n_appended).",
        },
    )
    return corrupted, metadata


def corrupt(
    fold: pd.DataFrame,
    mechanism: MechanismName,
    severity: float,
    seed: int,
    train_fold_reference: Optional[pd.DataFrame] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    Unified uniqueness-violations entry point:
        corrupt(dataset, mechanism, severity, seed) -> (corrupted_dataset, metadata)

    mechanism : "global" (9.1) | "slice_conditional" (9.2)
    train_fold_reference : required for "slice_conditional" (needed to derive
        the shared frozen slice S, D2 Decision 4); unused for "global",
        accepted anyway for interface consistency with every other module.

    Note for callers/D7: the corrupted output from EITHER mechanism has more
    rows than the input fold — this is the one D3 module where row count is
    not preserved.
    """
    if mechanism == "global":
        return corrupt_global(fold, severity, seed, dataset_name)
    elif mechanism == "slice_conditional":
        if train_fold_reference is None:
            raise ValueError(
                "slice_conditional duplication requires train_fold_reference "
                "to derive the shared frozen slice S (D2 Decision 4)."
            )
        return corrupt_slice_conditional(fold, severity, seed, train_fold_reference, dataset_name)
    else:
        raise ValueError(
            f"Unknown uniqueness mechanism: {mechanism!r}. Expected 'global' "
            f"or 'slice_conditional'."
        )