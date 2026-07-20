"""
validity_violations.py — D1 §8 Validity Violations

Two modes, matching the two halves of §8:

  --mode scaffold
      Scans datasets/processed/*.parquet and auto-generates a per-column
      constraint annotation TEMPLATE — dtype, observed range, an auto-suggested
      constraint guess — so Kishore's ~half-day annotation pass (D1 §15 item 6)
      is CONFIRMING/CORRECTING suggestions rather than starting from a blank
      sheet for ~17 datasets x dozens of columns each.

      Output: datasets/metadata/validity_constraints_template.csv
      Kishore edits this file directly (fixes wrong guesses, fills `notes`),
      then it becomes the authoritative annotation D1 §8 requires.

  --mode inject
      Implements D1 §8's actual corruption mechanism:
          X_ij -> v_invalid for proportion alpha of selected cells
      v_invalid chosen per D1 §8:
        - known sign/range constraint  -> negation or large offset
        - no documented constraint     -> value beyond train-fold IQR-based bound
      Reads the COMPLETED annotation (after Kishore's pass), applies to a
      given dataset at a given severity, fold-locally (train-fold statistics
      only, per D2 Decision 4).

Constraint types (what Kishore assigns per column in the template):
  "no_negative"   -> known non-negative quantity (age, counts, amounts).
                     Invalid value = negation, or -1 * |large offset| if 0.
  "range"         -> known bounded range (e.g. percentage 0-100, month 1-12).
                     Invalid value = min_bound - large_offset or max_bound + large_offset.
  "none"          -> no documented real-world constraint.
                     Invalid value = beyond train-fold IQR-based bound (D1 §8 fallback).
  "categorical"   -> not applicable to this mechanism (validity violations target
                     numeric sign/range constraints per D1 §8's formula); categorical
                     out-of-domain injection is a separate concern, flagged not
                     auto-handled — do not annotate categorical columns here.

Usage:
  python validity_violations.py --mode scaffold --processed-dir datasets/processed
  python validity_violations.py --mode inject --dataset heloc \
      --annotation datasets/metadata/validity_constraints_annotated.csv \
      --alpha 0.05 --seed 42 --fold train
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

SEED_DEFAULT = 42
LARGE_OFFSET_MULTIPLIER = 3.0   # multiplier on train-fold IQR/scale for offset-based invalid values
IQR_BOUND_MULTIPLIER = 3.0      # how far beyond IQR counts as "invalid" for unconstrained columns


# ------------------------------------------------------------------------
# MODE 1: Annotation scaffold generator
# ------------------------------------------------------------------------

def guess_constraint(series: pd.Series, col_name: str) -> tuple[str, str, str]:
    """
    Heuristic first guess at a constraint type, for Kishore to confirm/correct —
    NOT authoritative. Purely to speed up the manual pass.
    Returns (constraint_type_guess, reasoning_note, sentinel_values_guess).
    """
    name_lower = col_name.lower()
    non_null = series.dropna()

    if non_null.empty:
        return "none", "column entirely null in sample — cannot infer", ""

    if not pd.api.types.is_numeric_dtype(series):
        return "categorical", "non-numeric — not handled by this mechanism, do not annotate as no_negative/range", ""

    min_val, max_val = non_null.min(), non_null.max()

    # Sentinel detection: common sentinel codes (round negative numbers, 999-style
    # codes) that appear suspiciously often relative to the rest of the distribution —
    # flag for Kishore, do NOT auto-confirm as sentinel_documented (needs domain
    # knowledge, e.g. does -9 mean "no bureau record" vs. a real -9 value).
    common_sentinel_candidates = [-9, -8, -7, -1, 99, 999, -999, 9999, -9999]
    present_sentinels = [s for s in common_sentinel_candidates if (non_null == s).sum() > 0]
    if present_sentinels:
        counts = {s: int((non_null == s).sum()) for s in present_sentinels}
        return ("none",
                f"POSSIBLE SENTINELS DETECTED {counts} — verify against dataset documentation; "
                f"if confirmed, set constraint_type_CONFIRMED='sentinel_documented' and fill "
                f"sentinel_values with the confirmed codes (e.g. '-9;-8;-7')",
                ";".join(str(s) for s in present_sentinels))

    # Name-based heuristics for non-negative quantities
    non_negative_tokens = [
        "age", "count", "num_", "n_", "amount", "balance", "income",
        "price", "cost", "duration", "length", "quantity", "total_",
        "size", "weight", "height", "distance", "score", "rate",
    ]
    if any(tok in name_lower for tok in non_negative_tokens) and min_val >= 0:
        return "no_negative", f"name suggests non-negative quantity; observed min={min_val:.3g} >= 0", ""

    # Range heuristics: proportions/percentages, bounded codes
    if 0 <= min_val and max_val <= 1 and ("rate" in name_lower or "pct" in name_lower or "proportion" in name_lower):
        return "range", f"looks like a proportion in [0,1]; observed range=[{min_val:.3g}, {max_val:.3g}]", ""
    if 0 <= min_val and max_val <= 100 and ("pct" in name_lower or "percent" in name_lower):
        return "range", f"looks like a percentage in [0,100]; observed range=[{min_val:.3g}, {max_val:.3g}]", ""

    if min_val < -1:
        return "none", f"negative values present (min={min_val:.3g}) — verify: real negative quantity, or undocumented sentinel?", ""

    if min_val >= 0:
        return "no_negative", f"observed min={min_val:.3g} >= 0, no clear name signal — weak guess, please confirm", ""

    return "none", f"no constraint signal found; observed range=[{min_val:.3g}, {max_val:.3g}]", ""


def generate_scaffold(processed_dir: Path, out_path: Path):
    rows = []
    files = sorted(processed_dir.rglob("*.parquet"))
    if not files:
        print(f"No .parquet files found under {processed_dir}. Run preprocess_to_processed.py first.")
        return

    for f in files:
        dataset_name = f.stem
        source = f.parent.name
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  !! Could not read {f}: {e}")
            continue

        for col in df.columns:
            guess, note, sentinel_guess = guess_constraint(df[col], col)
            rows.append({
                "dataset": dataset_name,
                "source": source,
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].nunique(dropna=True),
                "min_observed": df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else "",
                "max_observed": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else "",
                "constraint_type_SUGGESTED": guess,
                "constraint_type_CONFIRMED": "",   # Kishore fills this in — blank = not yet reviewed
                "sentinel_values_SUGGESTED": sentinel_guess,
                "sentinel_values": "",             # Kishore fills this in if CONFIRMED=sentinel_documented
                "auto_note": note,
                "kishore_note": "",                # free text for Kishore
            })
        print(f"  Scaffolded {dataset_name}: {df.shape[1]} columns")

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nScaffold written -> {out_path}")
    print(f"Total: {len(files)} datasets, {len(out_df)} columns")
    print(f"\nNEXT STEP: Kishore reviews 'constraint_type_SUGGESTED' per row, fills in "
          f"'constraint_type_CONFIRMED' (no_negative / range / none / categorical), "
          f"and 'kishore_note' where the guess needs context. This closes D1 §15 item 6.")


# ------------------------------------------------------------------------
# MODE 2: Injection mechanism (D1 §8)
# ------------------------------------------------------------------------

import re

KNOWN_CONSTRAINT_TYPES = {"no_negative", "range", "none", "categorical", "sentinel_documented"}
RANGE_PATTERN = re.compile(r'^range:\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]$')


def compute_invalid_value(train_series: pd.Series, constraint_type: str, rng: np.random.Generator,
                           sentinel_values: list = None):
    """
    Per D1 §8: v_invalid chosen per-feature from the domain constraint annotation.
      no_negative         -> negation, or large negative offset if the value is 0
      range:[low,high]    -> beyond the EXPLICIT documented bound (Kishore-supplied,
                              e.g. 'range:[0,110]' for age, 'range:[1990,2015]' for a
                              model year) — this is the preferred, most accurate form
                              of D1 §8's "beyond the documented range bound" rule.
      range                -> (bare, no explicit bounds given) IQR-based offset beyond
                              observed max — fallback only, less accurate than above.
      sentinel_documented  -> one of the ACTUAL documented sentinel codes.
      none                 -> beyond train-fold IQR-based bound (D1 §8 fallback).

    Returns either a single float or (for sentinel_documented) a value to be drawn
    from the sentinel set by the caller.
    """
    range_match = RANGE_PATTERN.match(constraint_type.strip()) if isinstance(constraint_type, str) else None

    if constraint_type not in KNOWN_CONSTRAINT_TYPES and range_match is None:
        raise ValueError(
            f"Unrecognized constraint_type_CONFIRMED value: '{constraint_type}'. "
            f"Known types: {sorted(KNOWN_CONSTRAINT_TYPES)}, or 'range:[low,high]' "
            f"with explicit numeric bounds (e.g. 'range:[0,110]'). "
            f"If this is a genuinely new constraint category, it must be added "
            f"explicitly before use — silently falling back to generic logic "
            f"would inject the wrong kind of value and pass validation checks anyway."
        )

    q1, q3 = train_series.quantile(0.25), train_series.quantile(0.75)
    iqr = q3 - q1
    scale = iqr if iqr > 0 else (train_series.std() if train_series.std() > 0 else 1.0)

    if constraint_type == "no_negative":
        return -abs(scale) * LARGE_OFFSET_MULTIPLIER
    elif range_match is not None:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        offset = abs(scale) * LARGE_OFFSET_MULTIPLIER if scale > 0 else abs(high - low) * 0.1 + 1.0
        # Push beyond whichever bound — randomized per column so the mechanism doesn't
        # always violate the same direction across the whole dataset.
        if rng.random() < 0.5:
            return low - offset
        else:
            return high + offset
    elif constraint_type == "range":
        # Bare "range" with no explicit bounds supplied — fallback only.
        return train_series.max() + scale * LARGE_OFFSET_MULTIPLIER
    elif constraint_type == "sentinel_documented":
        if not sentinel_values:
            raise ValueError(
                f"constraint_type='sentinel_documented' requires a non-empty "
                f"'sentinel_values' annotation (e.g. '-9;-8;-7') — none was provided. "
                f"Kishore must fill in the actual documented sentinel codes for this "
                f"column, not just the constraint type."
            )
        return rng.choice(sentinel_values)
    else:  # "none" fallback per D1 §8
        return train_series.max() + scale * IQR_BOUND_MULTIPLIER


def inject_validity_violations(
    df: pd.DataFrame,
    annotation: pd.DataFrame,
    dataset_name: str,
    alpha: float,
    seed: int = SEED_DEFAULT,
    train_index: pd.Index = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Applies D1 §8 validity-violation corruption to `df`.

    df            : the fold's data to corrupt (test/held-out side, per D2 Decision 2)
    annotation    : completed constraint annotation (constraint_type_CONFIRMED filled in)
    dataset_name  : used to filter the annotation to this dataset's columns
    alpha         : severity in [0, alpha_max) — proportion of selected cells corrupted
    train_index   : row index of the TRAIN fold, used to compute fold-local statistics
                    (D2 Decision 4) — if None, uses df itself (only valid if df IS the
                    train fold; for test-side injection this must be passed explicitly)

    Returns (corrupted_df, metadata_dict) — metadata_dict is what D1 §12 requires logging
    (achieved rate, frozen constants) into the D7 manifest.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    ds_annotation = annotation[annotation["dataset"] == dataset_name]
    if ds_annotation.empty:
        raise ValueError(f"No annotation rows found for dataset '{dataset_name}' — "
                          f"check the annotation file has been completed for this dataset.")

    unreviewed = ds_annotation[ds_annotation["constraint_type_CONFIRMED"].isin(["", None]) |
                                ds_annotation["constraint_type_CONFIRMED"].isna()]
    if not unreviewed.empty:
        raise ValueError(
            f"{len(unreviewed)} column(s) in '{dataset_name}' have no CONFIRMED constraint "
            f"type — Kishore's annotation pass is incomplete for this dataset. "
            f"Columns: {unreviewed['column'].tolist()}"
        )

    metadata = {"dataset": dataset_name, "alpha": alpha, "seed": seed, "per_column": {}}

    for _, row in ds_annotation.iterrows():
        col = row["column"]
        constraint_type = row["constraint_type_CONFIRMED"]

        if constraint_type == "categorical":
            # D1 §8's formula targets numeric sign/range constraints — skip explicitly,
            # not silently, so this doesn't look like an oversight later.
            continue
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        train_series = df[col] if train_index is None else df.loc[df.index.isin(train_index), col]
        train_series = train_series.dropna()
        if train_series.empty:
            continue

        sentinel_values = None
        if constraint_type == "sentinel_documented":
            raw = row.get("sentinel_values", "")
            if not (pd.isna(raw) or str(raw).strip() == ""):
                sentinel_values = [float(v.strip()) for v in str(raw).split(";") if v.strip() != ""]

        n_cells = len(df)
        n_corrupt = int(round(alpha * n_cells))
        if n_corrupt == 0:
            metadata["per_column"][col] = {
                "constraint_type": constraint_type, "v_invalid": None,
                "target_rate": alpha, "achieved_rate": 0.0, "n_corrupted": 0,
            }
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float64")

        # Determine the invalid value(s) up front, THEN exclude any cell that already
        # holds that value from the candidate pool. This matters for ANY constraint
        # type, not just sentinel_documented: a computed offset can coincidentally
        # collide with a real pre-existing value (e.g. HELOC's -9/-8/-7 sentinels
        # appear across many columns, and small-magnitude no_negative offsets on
        # low-scale integer columns can land exactly on -9, -8, -6, -3, etc.).
        # Corrupting an already-matching cell produces no detectable change and
        # silently understates the true injected rate — so it must be excluded
        # from selection, not just tolerated by the validator.
        if constraint_type == "sentinel_documented":
            if not sentinel_values:
                raise ValueError(
                    f"constraint_type='sentinel_documented' requires a non-empty "
                    f"'sentinel_values' annotation for column '{col}' — none was provided."
                )
            eligible_idx = df.index[~df[col].isin(sentinel_values)]
            v_invalid_record = sentinel_values
        else:
            v_invalid = compute_invalid_value(train_series, constraint_type, rng)
            eligible_idx = df.index[df[col] != v_invalid]
            v_invalid_record = float(v_invalid)

        n_eligible = len(eligible_idx)
        if n_corrupt > n_eligible:
            print(f"    !! {col}: requested {n_corrupt} cells but only {n_eligible} "
                  f"eligible (rest already hold the invalid value) — capping.")
            n_corrupt = n_eligible

        if n_corrupt == 0:
            metadata["per_column"][col] = {
                "constraint_type": constraint_type, "v_invalid": v_invalid_record,
                "target_rate": alpha, "achieved_rate": 0.0, "n_corrupted": 0,
                "dtype_upcast_to_float64": bool(pd.api.types.is_integer_dtype(train_series.dtype)),
            }
            continue

        selected_idx = rng.choice(eligible_idx, size=n_corrupt, replace=False)

        if constraint_type == "sentinel_documented":
            injected_values = rng.choice(sentinel_values, size=n_corrupt)
            df.loc[selected_idx, col] = injected_values
        else:
            df.loc[selected_idx, col] = v_invalid_record

        metadata["per_column"][col] = {
            "constraint_type": constraint_type,
            "v_invalid": v_invalid_record,
            "target_rate": alpha,
            "achieved_rate": n_corrupt / n_cells,
            "n_corrupted": n_corrupt,
            "dtype_upcast_to_float64": bool(pd.api.types.is_integer_dtype(train_series.dtype)),
        }

    return df, metadata


def validate_injection(original_df: pd.DataFrame, corrupted_df: pd.DataFrame, metadata: dict) -> dict:
    """
    D1 §11 validation: Corruption Fidelity + Mechanism Fidelity checks for this mechanism.
    Returns a dict of per-column pass/fail — flag failures, do not silently accept (§11).
    """
    checks = {}
    for col, meta in metadata["per_column"].items():
        if meta["n_corrupted"] == 0:
            checks[col] = {"corruption_fidelity": "SKIP (alpha too small for any cells)",
                            "mechanism_fidelity": "SKIP"}
            continue
        rate_ok = abs(meta["achieved_rate"] - meta["target_rate"]) < 0.02  # 2pp tolerance
        # NaN-safe comparison: pandas/numpy treat NaN != NaN as True by default,
        # which would flag every pre-existing missing cell as "changed" even when
        # untouched — inflating n_changed and causing false FAILs on any column
        # with real missing values (e.g. jm1's documented NaN columns).
        both_nan = original_df[col].isna() & corrupted_df[col].isna()
        changed_mask = (original_df[col] != corrupted_df[col]) & ~both_nan
        n_changed = changed_mask.sum()
        mechanism_ok = n_changed == meta["n_corrupted"]

        if n_changed > 0:
            if isinstance(meta["v_invalid"], list):
                # sentinel_documented: every changed cell must be one of the documented codes
                values_correct = corrupted_df.loc[changed_mask, col].isin(meta["v_invalid"]).all()
            else:
                values_correct = (corrupted_df.loc[changed_mask, col] == meta["v_invalid"]).all()
        else:
            values_correct = True

        checks[col] = {
            "corruption_fidelity": "PASS" if rate_ok else "FAIL",
            "mechanism_fidelity": "PASS" if (mechanism_ok and values_correct) else "FAIL",
        }
    return checks


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D1 §8 Validity Violations — scaffold + injection")
    parser.add_argument("--mode", choices=["scaffold", "inject"], required=True)
    parser.add_argument("--processed-dir", type=str, default="datasets/processed")
    parser.add_argument("--out", type=str, default="datasets/metadata/validity_constraints_template.csv")
    parser.add_argument("--dataset", type=str, help="[inject mode] dataset name to corrupt")
    parser.add_argument("--annotation", type=str, help="[inject mode] path to COMPLETED annotation CSV")
    parser.add_argument("--alpha", type=float, help="[inject mode] severity, e.g. 0.05")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    if args.mode == "scaffold":
        generate_scaffold(Path(args.processed_dir), Path(args.out))

    elif args.mode == "inject":
        if not all([args.dataset, args.annotation, args.alpha is not None]):
            parser.error("--inject mode requires --dataset, --annotation, and --alpha")

        annotation = pd.read_csv(args.annotation)
        # Locate the processed file for this dataset
        candidates = list(Path(args.processed_dir).rglob(f"{args.dataset}.parquet"))
        if not candidates:
            parser.error(f"Could not find {args.dataset}.parquet under {args.processed_dir}")
        df = pd.read_parquet(candidates[0])

        corrupted_df, metadata = inject_validity_violations(
            df, annotation, args.dataset, args.alpha, seed=args.seed
        )
        checks = validate_injection(df, corrupted_df, metadata)

        print(f"\nValidity violation injection — {args.dataset}  alpha={args.alpha}  seed={args.seed}")
        print("-" * 70)
        for col, meta in metadata["per_column"].items():
            check = checks.get(col, {})
            v_display = (f"sentinels{meta['v_invalid']}" if isinstance(meta["v_invalid"], list)
                         else f"{meta['v_invalid']:.3f}" if meta["v_invalid"] is not None else "N/A")
            print(f"  {col:<30} type={meta['constraint_type']:<20} "
                  f"v_invalid={v_display}  "
                  f"achieved={meta['achieved_rate']:.3f}  "
                  f"[{check.get('corruption_fidelity')}/{check.get('mechanism_fidelity')}]")

        any_fail = any(c["corruption_fidelity"] == "FAIL" or c["mechanism_fidelity"] == "FAIL"
                        for c in checks.values())
        if any_fail:
            print("\n!! D1 §11 VALIDATION FAILURE — do not accept this corruption run silently.")
        else:
            print("\nAll columns passed §11 Corruption + Mechanism Fidelity checks.")


if __name__ == "__main__":
    main()