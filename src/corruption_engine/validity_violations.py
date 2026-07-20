"""
validity_violations.py

Implements D1 Section 8 — Validity Violations. Fifth D3 mechanism module.

    X_ij -> v_invalid for proportion `severity` of selected cells.
    v_invalid chosen per-feature from the one-time domain constraint
    annotation (Kishore, confirmed 2026-07-20, validity_constraints_annotated.csv):
        no_negative     -> negate the value (a directly-violating sign flip)
        range:[a,b]     -> push beyond whichever bound (a or b), chosen per cell
        none            -> value beyond a train-fold IQR-based bound (D1 8's
                            explicit fallback for undocumented constraints)
        categorical     -> EXCLUDED. D1 8 operates on numeric magnitude
                            constraints; a categorical code has no "negative"
                            or "range" to violate. The annotation file itself
                            marks these "not handled by this mechanism."

Governing constraints (implemented, not re-derived):

  D2 Decision 2 — corrupts only the fold it's given (test fold).
  D2 Decision 4 — the IQR bound for `none`-constraint columns is estimated
    on train_fold_reference only.
  D1 Section 3 — severity in [0, alpha_max), never approaching 1.
  D1 Section 8 — severity is anchored to naturally occurring validity
    violations already documented in Phase 0 (e.g. HELOC's sentinel values),
    not asserted; this module takes severity as an already-resolved input
    (D4's job), same as every other mechanism module.
  D1 Decision Log item 6 — the annotation pass is now confirmed complete
    (all 17 datasets, 369 rows, zero blank confirmations) — this module
    would have raised loudly on load if it weren't.
"""

from typing import Optional, Literal, Tuple, List, Dict
import re
import numpy as np
import pandas as pd

from .base import CorruptionMetadata, rate_tolerance, nan_adjusted_target

ConstraintType = Literal["no_negative", "range", "none", "categorical"]

IQR_MULTIPLIER = 3.0  # NOT specified by D1 -- D1 says "value beyond train-fold
    # IQR-based bound" but doesn't fix the multiplier. Flagged as a D3
    # implementation decision: 3.0 (vs. the common mild-outlier convention of
    # 1.5) is chosen deliberately more extreme, since this needs to produce a
    # value that reads as clearly INVALID, not just a mild statistical
    # outlier — a value 1.5*IQR beyond the box could still be a legitimate
    # extreme observation; 3*IQR is closer to "implausible."
RANGE_OVERSHOOT_FRACTION = 0.10  # how far beyond a range bound to push, as a
    # fraction of the range's width -- e.g. for range:[0,100], severity
    # pushes to 100 + 0.10*100 = 110 or 0 - 10 = -10. NOT specified by D1;
    # flagged as a D3 implementation decision. Chosen to be unambiguously
    # outside the valid range without being an absurd magnitude that would
    # be trivially caught by any downstream sanity filter.


_RANGE_PATTERN = re.compile(r"range:\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


def parse_constraint(constraint_str: str) -> Tuple[ConstraintType, Optional[Tuple[float, float]]]:
    """Parse a constraint_type_CONFIRMED value from the annotation file into
    (type, bounds). bounds is (a, b) for 'range', else None."""
    constraint_str = str(constraint_str).strip()
    if constraint_str == "no_negative":
        return "no_negative", None
    if constraint_str == "categorical":
        return "categorical", None
    if constraint_str == "none":
        return "none", None
    m = _RANGE_PATTERN.match(constraint_str)
    if m:
        return "range", (float(m.group(1)), float(m.group(2)))
    raise ValueError(
        f"Unrecognized constraint_type_CONFIRMED value: {constraint_str!r}. "
        f"Expected 'no_negative', 'categorical', 'none', or 'range:[a,b]'."
    )


def load_constraints(csv_path: str, dataset_name: str) -> Dict[str, str]:
    """
    Load the confirmed constraint type per column for one dataset from the
    annotation file (validity_constraints_annotated.csv). Returns
    {column_name: constraint_type_CONFIRMED_string}, using the CONFIRMED
    column specifically — never the SUGGESTED column, which is the
    unreviewed auto-guess (per D1 Sec 8's ownership requirement, this
    mechanism must run on Kishore's confirmed annotations, not the
    heuristic pre-fill).
    """
    df = pd.read_csv(csv_path)
    subset = df[df["dataset"] == dataset_name]
    if subset.empty:
        raise ValueError(
            f"No annotation rows found for dataset='{dataset_name}' in {csv_path}. "
            f"Available datasets: {sorted(df['dataset'].unique())}"
        )
    if subset["constraint_type_CONFIRMED"].isna().any():
        unconfirmed = subset[subset["constraint_type_CONFIRMED"].isna()]["column"].tolist()
        raise ValueError(
            f"dataset='{dataset_name}' has unconfirmed (blank) constraint "
            f"annotations for columns: {unconfirmed}. D1 Sec 8 requires the "
            f"CONFIRMED annotation, not the auto-suggested one — these columns "
            f"need review before this mechanism can run on them."
        )
    return dict(zip(subset["column"], subset["constraint_type_CONFIRMED"]))


def _validate_severity(severity: float) -> None:
    if not (0 <= severity < 1):
        raise ValueError(
            f"severity must be in [0, 1); got {severity}. Per D1 Sec 3, alpha "
            f"never approaches 1."
        )


def corrupt(
    fold: pd.DataFrame,
    severity: float,
    seed: int,
    train_fold_reference: pd.DataFrame,
    constraints: Dict[str, str],
    columns: Optional[List[str]] = None,
    dataset_name: str = "",
) -> Tuple[pd.DataFrame, CorruptionMetadata]:
    """
    D1 Section 8 entry point. Unlike the other four modules there is only one
    mechanism (no MechanismName dispatch needed) — the constraint TYPE, not a
    selectable mechanism, determines how each column is corrupted.

    Parameters
    ----------
    fold : the fold to corrupt (D2 Decision 2 — the test fold).
    severity : alpha in [0, alpha_max) — resolved numeric severity from D4's
        grid, anchored per D1 Sec 8 to naturally occurring validity defects.
    seed : corruption-injection random seed.
    train_fold_reference : clean train fold — used for the IQR bound on
        `none`-constraint columns (D2 Decision 4).
    constraints : {column: constraint_type_CONFIRMED string}, from
        load_constraints() or an equivalent caller-supplied mapping. Required
        — this mechanism cannot run without the confirmed annotation.
    columns : which columns to corrupt. Defaults to all columns present in
        `constraints`. Columns whose confirmed type is 'categorical' are
        excluded automatically regardless of what's requested here.
    dataset_name : for metadata/logging only.
    """
    _validate_severity(severity)
    rng = np.random.default_rng(seed)
    corrupted = fold.copy()

    if columns is None:
        columns = list(constraints.keys())

    achieved_rate = {}
    excluded_columns = {}
    per_column_detail = {}

    for col in columns:
        if col not in constraints:
            excluded_columns[col] = (
                f"no confirmed constraint annotation found for this column — "
                f"cannot determine v_invalid without it"
            )
            continue
        if col not in fold.columns:
            excluded_columns[col] = "column not present in the fold being corrupted"
            continue

        ctype, bounds = parse_constraint(constraints[col])
        if ctype == "categorical":
            excluded_columns[col] = (
                "constraint type is 'categorical' — D1 Sec 8 operates on "
                "numeric magnitude constraints only; this column has no "
                "'negative' or 'range' to violate"
            )
            continue
        if not pd.api.types.is_numeric_dtype(fold[col]):
            excluded_columns[col] = (
                f"constraint type '{ctype}' expects a numeric column, but "
                f"'{col}' is not numeric dtype"
            )
            continue

        adjusted_target = nan_adjusted_target(severity, fold[col])
        eligible = fold[col].notna().to_numpy()
        n = len(corrupted)
        select_mask = (rng.random(n) < adjusted_target) & eligible
        selected_idx = np.where(select_mask)[0]

        if ctype == "no_negative":
            new_vals = corrupted[col].to_numpy(dtype=float).copy()
            new_vals[selected_idx] = -fold[col].to_numpy(dtype=float)[selected_idx]
            # Guard: negating a value that's already <= 0 (shouldn't happen
            # for a genuinely no_negative-constrained column with clean data,
            # but a pre-existing negative would produce a "violation" that's
            # actually valid, i.e. non-negative) -- push those explicitly
            # positive-then-negated to guarantee an actual violation.
            already_nonpositive = new_vals[selected_idx] >= 0
            if already_nonpositive.any():
                bad_idx = selected_idx[already_nonpositive]
                ref_scale = train_fold_reference[col].abs().median()
                ref_scale = ref_scale if ref_scale > 0 else 1.0
                new_vals[bad_idx] = -(np.abs(fold[col].to_numpy(dtype=float)[bad_idx]) + ref_scale)
            corrupted[col] = new_vals

        elif ctype == "range":
            a, b = bounds
            width = b - a
            overshoot = max(width * RANGE_OVERSHOOT_FRACTION, 1e-6)
            new_vals = corrupted[col].to_numpy(dtype=float).copy()
            push_high = rng.random(len(selected_idx)) < 0.5
            new_vals[selected_idx[push_high]] = b + overshoot
            new_vals[selected_idx[~push_high]] = a - overshoot
            corrupted[col] = new_vals

        elif ctype == "none":
            ref = train_fold_reference[col]
            q1, q3 = ref.quantile(0.25), ref.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                excluded_columns[col] = (
                    f"constraint type 'none' needs a train-fold IQR-based bound, "
                    f"but this column's train-fold IQR is {iqr!r} (degenerate/"
                    f"near-constant) — cannot define 'beyond IQR' meaningfully"
                )
                continue
            low_bound = q1 - IQR_MULTIPLIER * iqr
            high_bound = q3 + IQR_MULTIPLIER * iqr
            new_vals = corrupted[col].to_numpy(dtype=float).copy()
            push_high = rng.random(len(selected_idx)) < 0.5
            new_vals[selected_idx[push_high]] = high_bound
            new_vals[selected_idx[~push_high]] = low_bound
            corrupted[col] = new_vals

        else:
            excluded_columns[col] = f"unhandled constraint type '{ctype}'"
            continue

        achieved_rate[col] = float(select_mask.mean())
        per_column_detail[col] = {"constraint_type": constraints[col], "n_corrupted": int(select_mask.sum())}

    if not achieved_rate:
        raise RuntimeError(
            f"Validity violations: every requested column was excluded for "
            f"dataset='{dataset_name}', severity={severity}, seed={seed}. "
            f"Excluded: {excluded_columns}"
        )

    max_deviation = max((abs(achieved_rate[c] - severity) for c in achieved_rate), default=0.0)
    tol = rate_tolerance(severity, len(fold))

    metadata = CorruptionMetadata(
        dataset=dataset_name,
        mechanism="validity_violations",
        severity=severity,
        seed=seed,
        achieved_rate=achieved_rate,
        excluded_columns=excluded_columns if excluded_columns else None,
        validation_passed=(max_deviation < tol),
        validation_details={
            "max_rate_deviation": max_deviation,
            "tolerance_used": tol,
            "per_column_constraint": per_column_detail,
            "note": "Injected values violate the predefined validity rule by construction (D1 Sec 8).",
        },
    )
    return corrupted, metadata