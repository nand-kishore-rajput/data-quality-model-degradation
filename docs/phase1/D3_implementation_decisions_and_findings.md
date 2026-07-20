# D3 — Implementation Decisions and Findings

**Status:** Companion record to the D3 Corruption Engine codebase, not a design-gate document. Written after all six mechanism modules were built and validated against the real 17-dataset corpus (2026-07-20). Purpose: consolidate everything that was decided, discovered, or fixed during implementation so D4 (Severity Grid) and D7 (Phase-2-Ready Manifest) can be built against explicit information rather than requiring a re-read of the build history.
**Covers:** `base.py`, `missingness.py`, `feature_noise.py`, `label_noise.py`, `distribution_shift.py`, `validity_violations.py`, `uniqueness.py` — all six mechanism classes (14 sub-mechanisms), all real-data-confirmed.

---

## 1. Uncanonicalized Constants — Direct D4 Inputs

These are D3 implementation decisions that fill gaps D1 deliberately left open (D1 §10: "this document defines calibration methodology only"). None of them are reinterpretations of D1's math — each is flagged in its own module's code comments at the point of use — but D4 needs to either confirm them or explicitly override them, since several directly shape what a given severity value actually does.

| Constant                           | Value                             | Module                   | Controls                                                                         | D4 relevance                                                                                      |
| ---------------------------------- | --------------------------------- | ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ALPHA_SCALE`                      | 3.0                               | `missingness.py`         | MAR/MNAR sigmoid steepness                                                       | Fixed shape constant, not part of the severity axis itself — low priority to revisit              |
| `RATE_TOLERANCE_Z`                 | 3.5                               | `base.py` (shared)       | Corruption-fidelity tolerance (achieved rate vs. target)                         | Affects what counts as a validation pass in D8, not severity itself                               |
| `MIN_REFERENCE_STD`                | 1e-8                              | `base.py` (shared)       | Threshold for treating a column as constant                                      | Low priority                                                                                      |
| `QUANTIZATION_K_MAX_CAP`           | 20                                | `feature_noise.py`       | Ceiling on quantization bin count at severity=0                                  | **D4 should record this explicitly** — recommended: keep as-is, document in D4 rather than change |
| `G_CLIP_Z`                         | 6.0                               | `distribution_shift.py`  | Clips PC1 score before exponential weighting (covariate shift)                   | **Directly relevant** — see jm1 finding below                                                     |
| `MIN_EFFECTIVE_SAMPLE_FRACTION`    | 0.10                              | `distribution_shift.py`  | Diagnostic floor flagging degenerate covariate-shift resamples                   | **D4 input** — see jm1 finding                                                                    |
| `SLICE_FRACTION`                   | 0.30                              | `base.py` (shared)       | Size of the frozen slice S (used by both concept shift §7.3 and uniqueness §9.2) | **D4 should confirm** — 30% of rows is a D3 default, not a D1-specified value                     |
| Label-shift default `target_prior` | 0.9 / 0.1 (toward nearer extreme) | `distribution_shift.py`  | Placeholder `rho_target` when D4 hasn't supplied one                             | **D4 must supply the real value** — this is explicitly a runnable placeholder, not a decision     |
| `IQR_MULTIPLIER`                   | 3.0                               | `validity_violations.py` | How far beyond train-fold IQR a `none`-constraint violation is pushed            | Low priority                                                                                      |
| `RANGE_OVERSHOOT_FRACTION`         | 0.10                              | `validity_violations.py` | How far beyond a `range:[a,b]` bound a violation is pushed                       | Low priority                                                                                      |
| `MARGINAL_TEST_ALPHA`              | 0.05                              | `feature_noise.py`       | Significance level for categorical-swap marginal-preservation check              | Standard convention, low priority                                                                 |
| `MIN_MISSING_FOR_ASSOCIATION_TEST` | 10                                | `missingness.py`         | Statistical-power floor for MAR/MNAR fidelity checks                             | Low priority                                                                                      |

---

## 2. Real Dataset-Specific Findings

These came directly out of validating D3 against the actual corpus, not out of any design document. Worth carrying into D4's per-dataset severity calibration rather than treating every dataset identically.

### 2.1 EEG's channel structure (missingness, MAR/MNAR)

Of EEG's 5 auto-selected channels, `F7`, `F3`, `FC5`, `T7` show strong, consistently significant self/cross-dependency; `AF3` shows a real but weak, often-borderline self-dependency that intermittently fails MNAR's significance check (correctly excluded per-column, not blocking the other four). For MAR specifically, cross-channel correlations among most EEG channel pairs are close to zero — only `F7` reliably passes as a conditioning-feature target. Consistent with EEG channels being largely independent raw signals at the amplitude level. No action needed; documented for awareness if EEG's missingness behavior looks asymmetric across mechanisms in later results.

### 2.2 jm1's covariate-shift behavior (distribution_shift, §7.1)

jm1's software-complexity features (`loc`, `v(g)`, `ev(g)`, `iv(g)`, `n`) are correlated and outlier-heavy (a few large/complex files among mostly small ones). Before the `G_CLIP_Z` fix, this caused PC1-weighted resampling to collapse to duplicates of a single row by severity 0.15 (PSI=7.9, effective sample size ~0.05%). After the fix, jm1 behaves like the rest of the corpus at low-to-moderate severity, but **still trips the effective-sample-size floor at severity=0.25 specifically (effective_sample_fraction=0.046, all 3 seeds, fully reproducible)**. PSI at that point is a reasonable 1.19 (not degenerate), but real diversity in the resampled fold is reduced.
**D4 action needed:** either cap jm1's covariate-shift severity below 0.25, or accept 4.6% effective sample size as a known, disclosed property of this dataset at high severity rather than a defect.

### 2.3 Natural pre-existing duplicate rates (all 17 datasets, uniqueness §9)

Measured directly from the actual processed test folds (70/30 split, seed=42):

| Dataset                                                          | Pre-existing duplicate rate |
| ---------------------------------------------------------------- | --------------------------- |
| assistments                                                      | 13.68%                      |
| jm1                                                              | 11.39%                      |
| ACSPublicCoverage                                                | 6.41%                       |
| heloc                                                            | 5.96%                       |
| Airbnb                                                           | 4.20%                       |
| ACSIncome                                                        | 0.33%                       |
| online_shoppers                                                  | 0.43%                       |
| MagicTelescope                                                   | 0.19%                       |
| phoneme                                                          | 0.18%                       |
| default_credit                                                   | 0.02%                       |
| bank-marketing                                                   | 0.01%                       |
| EEG, churn, electricity, kick, road-safety, diabetes_readmission | ~0.00%                      |

**heloc's 5.96%** directly confirms D1 §9's own severity anchor ("HELOC's 588 duplicate rows") is still accurate on the current processed data, not just a historical Phase 0 number.
**assistments (13.68%) and jm1 (11.39%) are new, higher findings** not referenced in D1's anchor. Plausibly explainable — assistments is tutoring-interaction log data where a student repeating an identical problem produces genuinely identical feature rows; jm1's small, near-identical trivial functions could produce a similar effect, and may be related to the same outlier/skew structure behind §2.2.
**D4 action needed:** decide whether to anchor each dataset's `α_max` to its own natural rate (more principled, matches D1 §9's stated anchoring philosophy) or keep a single corpus-wide anchor (simpler, less faithful to individual datasets). Also worth a decision on whether assistments' and jm1's rates are citable data-quality findings (in the spirit of HELOC's sentinel values) or background notes only.

---

## 3. Bugs Found and Fixed — Audit Trail

Every one of these was invisible in synthetic testing and only surfaced against the real corpus. Kept here as a record, not because the code still has them — all are fixed and re-verified against real data.

| Module                  | Bug                                                                                                                                                                                                                               | Fix                                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `missingness.py`        | Flat rate tolerance (`0.02`) produced a severity-dependent false-positive rate (~11% on real data)                                                                                                                                | Replaced with binomial-SE-based tolerance (`RATE_TOLERANCE_Z`), re-verified at ~1% false-positive rate                                         |
| `missingness.py`        | Pre-existing NaN in conditioning/target columns propagated through the calibration bisection, collapsing achieved rate to exactly 0.0 regardless of severity                                                                      | Bisection now calibrates against only non-NaN z-values; NaN-fraction-adjusted target rate added                                                |
| `missingness.py`        | `scipy.stats.pointbiserialr` silently mishandled pre-existing NaN in the association test                                                                                                                                         | NaN-safe wrapper excluding those rows from the test                                                                                            |
| `missingness.py`        | All-or-nothing blocking check invalidated an entire multi-column call if even one column (e.g. EEG's `AF3`) failed fidelity                                                                                                       | Redesigned to per-column exclusion — failing columns excluded, passing columns still corrupted                                                 |
| `feature_noise.py`      | Chi-square marginal-preservation check dropped low-expected-count bins via boolean indexing, breaking scipy's required observed/expected sum equality — crashed on every dataset with a rare category (150/765 real-corpus draws) | Low-count bins pooled into a combined "other" bin instead of dropped                                                                           |
| `label_noise.py`        | Cleanlab import referenced a function name that doesn't exist in the installed version, always falling into the "Cleanlab unavailable" path even when it was installed                                                            | Fixed to use the actual installed API (`compute_confident_joint`)                                                                              |
| `label_noise.py`        | Cleanlab's confident-learning estimate (expensive: 3-fold CV logistic regression) was recomputed on every one of 15 severity/seed calls per dataset instead of once                                                               | Cached per `(train_fold_reference identity, target_col, feature_cols)`, verified ~15x speedup on real data                                     |
| `distribution_shift.py` | Unbounded `exp(severity * PC1_score)` importance weighting collapsed to near-total mass on a single outlier row for correlated/skewed real features (jm1), producing PSI=7.9 and an effective sample size of ~1 row               | PC1 score clipped to a bounded number of its own standard deviations before exponentiating; effective-sample-size diagnostic added to metadata |

---

## 4. Structural Notes for D4 and D7

1. **Uniqueness's output is larger than its input.** All five other mechanisms preserve row count; `uniqueness.py` (§9.1, §9.2) appends duplicate rows. D7's manifest schema must record actual corrupted-fold size explicitly rather than assuming it equals the original test-fold size.
2. **The frozen slice S is structurally shared, not just conventionally aligned.** `distribution_shift.py` (§7.3) and `uniqueness.py` (§9.2) both call `base.py`'s `select_frozen_slice_bounds` and are verified to receive byte-identical bounds given the same `(train_fold_reference, seed, dataset_name)`. No further action needed — this is confirmed working, noted here so it isn't accidentally "fixed" into two independent implementations later.
3. **D5's target definition needs updating for consistency.** Carried forward from D11: D5 (Natural-Shift Validation) was originally scoped around "ΔAUC" curves; should use Relative ΔM instead, per D11 Decision 1/12. Not a D3 finding, but adjacent enough to note here since it's still unresolved.
4. **Every constant in §1 is currently a Python-level default, not a per-dataset override mechanism.** If D4 wants different values per dataset (e.g., a lower `G_CLIP_Z` ceiling for jm1 specifically), the modules currently don't have a clean per-dataset override path — worth deciding whether D4's grid should specify per-dataset overrides for these constants, or whether corpus-wide defaults are acceptable.
