# Phase 0 — Dataset Notes, Special-Value Semantics & Disqualification Log

**Project:** _When Does Data Quality Predict Model Failure? A Benchmark for Knowing Before It Breaks_
**Phase:** 0 — Corpus Assembly (~85% complete)
**Status of this document:** Working notes. Items marked `[OPEN]` block Phase 0 exit; items marked `[VERIFY]` require external confirmation before being cited in the paper.
**Last updated:** 2026-07-09

---

## 1. HELOC — Special-Value (Sentinel) Semantics

**Source:** FICO Explainable ML Challenge (OpenML id 45023, n = 10,000)
**Headline finding:** The raw manifest reports `missing_cells = 0`. This is a **measurement artifact**, not a property of the data. HELOC encodes missingness and structural inapplicability via negative sentinel values. Reporting 0% missingness in a data-quality paper would be self-refuting.

### 1.1 Observed special values

| Sentinel | Observed pattern                                                         | Interpretation (per FICO data dictionary) `[VERIFY]`                                                                                                   |
| -------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **−9**   | ~5.57% of rows, consistent across many features                          | No bureau record / no valid investigation. **Row-level missingness** — thin-file applicants with no usable credit history.                             |
| **−8**   | Feature-dependent occurrence                                             | No usable/valid trades or inquiries for that feature. **Feature-level missingness.**                                                                   |
| **−7**   | `MSinceMostRecentDelq` → 45.11%; `MSinceMostRecentInqexcl7days` → 17.70% | **Condition not met** — e.g., the applicant has _never been delinquent_. This is **not missing data**; it is a highly informative structural category. |

### 1.2 Why this matters (three distinct missingness mechanisms in one dataset)

HELOC is a natural case study in why naive missingness metrics fail:

1. **−9** ≈ unit non-response (MNAR-adjacent: thin-file status correlates with the outcome).
2. **−8** ≈ item non-response.
3. **−7** ≈ _structural inapplicability_ — a value that looks like missingness to a scanner but carries strong predictive signal.

This distinction (MNAR vs. structurally inapplicable) is directly citable in the methodology section and is evidence for the paper's core thesis: **off-the-shelf quality metrics mis-measure real datasets unless provenance-aware decoding is applied.** This is a free, concrete illustration of the provenance-weighted quality feature contribution (Novelty Bet #1).

### 1.3 Prescribed treatment (do NOT decode all three as NaN)

| Sentinel | Treatment                                                                                                                                                                                                                                             |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| −9       | → `NaN` (missing)                                                                                                                                                                                                                                     |
| −8       | → `NaN` (missing)                                                                                                                                                                                                                                     |
| −7       | → **retain as structural category**: add binary indicator column (e.g., `never_delinquent`) + impute the base feature with a sentinel-neutral value, **or** treat as explicit category for tree models. Decision to be fixed before Phase 1. `[OPEN]` |

> **Anti-pattern warning:** Converting −7 to NaN would destroy signal and replace one measurement error (false 0% missingness) with another (false ~45% missingness on `MSinceMostRecentDelq`). Both errors are first-order for this paper.

### 1.4 Manifest implication — dual missingness columns

The manifest gains two columns for **all** datasets:

- `missing_percent_raw` — as scanned, no decoding (HELOC: 0.0)
- `missing_percent_decoded` — after sentinel decoding (HELOC: to be computed from −9/−8 only)

The **delta between these columns is itself a data-quality finding** and should be reported in the paper.

### 1.5 Open items

- [ ] `[VERIFY]` Confirm sentinel semantics against official FICO HELOC data dictionary / challenge documentation; record citation.
- [ ] `[OPEN]` Fix −7 treatment (indicator + imputation vs. explicit category).
- [ ] `[OPEN]` Compute `missing_percent_decoded`; regenerate HELOC manifest row.
- [ ] No transformations applied yet — raw file is untouched (correct; keep raw immutable, decode in the loader).

---

## 2. Disqualification Log

Datasets removed from the corpus during Phase 0, with rationale. All removals are documented for the paper's appendix and for mentor communication.

**Minority-floor convention:** decisions below assume a **10% minimum positive-class rate**. Entries tagged `[FLOOR-SENSITIVE]` re-qualify if the floor is relaxed to 5%. **The floor decision itself is still open — see §2.7.** `[OPEN]`

### 2.1 coil2000 `[FLOOR-SENSITIVE]`

- **Source:** OpenML id 298 (n = 9,822; 86 columns)
- **Metric:** minority class (`CARAVAN = 1`) = **5.97%**
- **Reason:** Below the 10% minority-class floor. Imbalance this severe entangles corruption effects (especially injected label noise) with baseline sampling variance, undermining clean Δ-performance measurement.
- **Disposition:** Re-qualifies only if floor is set to 5% (would then enter the imbalance stress-test cell, §2.7).

### 2.2 seismic-bumps — ⚠️ CONTRADICTION TO RESOLVE `[OPEN]`

Two conflicting records currently exist:

| Record                       | Claim                                                                                                                                                                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **This log (earlier entry)** | Pulled file (intended OpenML id 1500) was actually **UCI Seeds** — 210 rows, 7 features, 3 classes at exactly 70/70/70. Unmistakable provenance failure, likely off-by-one in the fetch ID. Removed; fails on size (n < 2,000) and task (3-class). |
| **Current manifest**         | Row exists: OpenML id **46956**, n = 2,584, target `HighEnergySeismicBump`, minority = 6.58%, `minority_pass = False`.                                                                                                                             |

**Most plausible reconciliation:** the dataset was **re-pulled under a corrected ID (46956)** after the provenance failure; the log entry was never updated. If so:

- Update this entry to: _"Original pull (id 1500 path) failed provenance check — file was UCI Seeds. Re-pulled successfully as id 46956. Now subject only to the minority-floor decision (6.58% < 10%; passes at 5%)."_
- seismic-bumps becomes `[FLOOR-SENSITIVE]`, and is the strongest candidate for the imbalance stress-test cell (only geophysics domain in the corpus).

**Regardless of disposition, retain the Seeds incident** in the paper as the motivating example of _silent data-provenance error_ — a real, self-experienced instance of the failure class the benchmark addresses.

- [ ] `[OPEN]` Reconcile log vs. manifest; make one document authoritative.
- [ ] Eyeball-verify id 46956 feature names (`genergy`, `gpuls`, `nbumps`, …) to confirm the re-pull is genuine seismic-bumps.

### 2.3 house_16H

- **Source:** OpenML id 821 (n = 22,784)
- **Metric:** quasi-anonymized column names (`P1`, `P5p1`, `H2p2`, …)
- **Reason:** Not in the locked corpus. Opaque header codes conflict with the SHAP feature-attribution contribution — same hard-exclusion rule applied to jannis / rl / albert / Higgs. Target is a binarized regression variable with weak semantic grounding.
- **Disposition:** Permanently excluded.

### 2.4 taiwanese_bankruptcy

- **Source:** UCI id 572 (n = 6,819; 96 columns)
- **Metric:** minority class (`Bankrupt? = 1`) = **3.23%**
- **Reason:** Below the floor at both 10% **and** 5% — fails decisively regardless of threshold. Also contains near-constant flag columns (e.g., `Net Income Flag`) adding little signal.
- **Disposition:** Permanently excluded.

### 2.5 polish_bankruptcy

- **Source:** UCI id 365 (n = 43,405; 66 columns)
- **Metric:** minority class = **4.82%**
- **Reason:** Below the floor at both thresholds. Additionally pools five separate forecasting horizons (years 1–5) into a single pseudo-task via a `year` column, conflating distinct prediction problems.
- **Disposition:** Permanently excluded. (Also note: with taiwanese_bankruptcy, would have been a near-duplicate task — the corpus should carry at most one bankruptcy dataset in any case.)

### 2.6 Credit (CleanML / Give Me Some Credit)

- **Source:** CleanML; n = 150,000
- **Metrics:** rows exceed the 50k cap; target `SeriousDlqin2yrs` ≈ 6–7% positive
- **Reason (two independent grounds):**
  1. **Row cap:** 150k rows exceed the 50k cap and the dataset falls outside the exception protocol (iid-track, not a designated shift anchor).
  2. **Domain concentration:** would be the third credit-risk dataset alongside HELOC and default_credit — over-weighting a single domain and mirroring the exact source-concentration weakness this project critiques in PAPE.
- Expected minority rate also likely fails the 10% floor (`[VERIFY]` with the corrected target inspector if ever revisited).
- **Disposition:** Permanently excluded.

### 2.7 The minority-floor decision `[OPEN]` — decide once, propagate everywhere

Two defensible options; **having no decision is the only indefensible state**:

| Option                                         | Consequence                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Hard 10% floor**                          | coil2000 and seismic-bumps both excluded. Corpus loses its only geophysics domain. No imbalance stress-test cell. Simplest to defend.                                                                                                                                                                   |
| **B. 5% floor + PR-AUC override band (5–10%)** | seismic-bumps (6.58%) and coil2000 (5.97%) enter a deliberate **imbalance stress-test cell**, screened with PR-AUC instead of accuracy/AUC. Adds practical relevance (degradation prediction under imbalance is where practitioners actually hurt) at the cost of one more experimental cell to defend. |

**Supervisor recommendation:** Option B, capped at **one to two** datasets in the cell (seismic-bumps preferred for domain diversity). Whatever is chosen: record it in the corpus design note, tag the manifest with the applied rule, and never revisit ad hoc.

### 2.8 Finance-concentration audit (related pruning pressure)

Post-disqualification finance/credit count: bank-marketing, HELOC, default_credit, kick (auto finance) = **4 of ~20**. Acceptable but at the ceiling. Do not admit further finance datasets; if the corpus must shrink, kick vs. default_credit is the first tie to break.

---

## 3. assistments (TableShift)

**Source:** ASSISTments 2012–2013 school year, open-source snapshot
**Path:** `datasets/raw/tableshift/assistments.parquet`
**Column reference:** ASSISTments data dictionary — https://sites.google.com/site/assistmentsdata/how-to-interpret

### 3.1 Basic stats (raw, pre-subsample)

| Property      | Value              |
| ------------- | ------------------ |
| Rows          | 6,123,270          |
| Columns       | 35                 |
| Missing cells | 13,426,647 (6.26%) |

### 3.2 Target definition

- **Raw column:** `correct`
- **Raw type:** _not_ binary — 13 distinct values (1 = correct on first attempt; decimals = partial credit from hints/attempts; 0 = incorrect / answer revealed / credit exhausted).
- **Binarization rule (per official documentation):** `1 = correct`, `< 1 = incorrect`.
- **Binarized minority % not yet measured** — the raw multiclass read returned a meaningless 0.00%. `[OPEN]` The balance check must be re-run on the binarized target before this dataset can pass/fail the screen. **assistments cannot be counted in the surviving corpus until this lands.**

### 3.3 Leakage-column exclusion list

**Hard exclusions — features reflecting the solving process, not pre-attempt state:**

```
hint_count, bottom_hint, actions, attempt_count, ms_first_response,
overlap_time, answer_id, answer_text, first_action, end_time
```

**Proxy-leakage exclusions — affect detectors computed on the same interaction being labeled:**

```
Average_confidence(FRUSTRATED), Average_confidence(CONFUSED),
Average_confidence(CONCENTRATING), Average_confidence(BORED)
```

**Decision:** drop the affect columns outright. A keep-with-caveat treatment is textbook proxy leakage and invites reviewer objections for negligible upside.

### 3.4 Temporally safe features (known at/before problem presentation)

```
skill, problem_id, user_id, assignment_id, assistment_id, start_time,
problem_type, original, tutor_mode, sequence_id, student_class_id,
position, type, base_sequence_id, skill_id, teacher_id, school_id,
template_id
```

> **⚠️ Second-order threat — ID memorization.** "Temporally safe" ≠ "modeling safe." `user_id`, `problem_id`, `template_id`, etc. are high-cardinality identifiers. XGBoost will memorize them, and corruption experiments would then measure _memorization decay_, not quality effects. `[OPEN]` Decide one of:
>
> 1. Fold-safe target encoding of high-cardinality IDs, or
> 2. **Drop raw IDs; retain only aggregable grouping variables** (`school_id`, `teacher_id`) — which are required anyway for group-aware conformal prediction and blocked CV. _(Supervisor recommendation: option 2 — simpler, safer, and aligned with the conformal contribution.)_

### 3.5 Subsampling strategy

- 6.1M rows → far over cap; **blocked** subsample mandatory (random subsampling destroys the shift phenomenon).
- **Recommended block axis: school × time.** School is the natural conformal group; time is the shift axis. Blocking at student-level _within_ school-level risks leaving too few groups for group-aware conformal calibration.

### 3.6 Open items

- [ ] `[OPEN]` Binarize `correct`; re-run minority-class check; record pass/fail.
- [ ] Apply both leakage exclusion lists in the loader (never mutate the raw parquet).
- [ ] `[OPEN]` Fix ID-handling decision (§3.4).
- [ ] `[OPEN]` Implement school × time blocked subsample; document block sizes.

---

## 4. Cross-Cutting Manifest Requirements (additions from this review)

New manifest columns to add so the corpus state is auditable from one file:

| Column                                            | Purpose                                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `missing_percent_raw` / `missing_percent_decoded` | Sentinel-aware missingness (HELOC delta is a finding in itself)                                         |
| `headers_semantic` (bool)                         | Verifies column-name restoration (bank_marketing, default_credit, phoneme) — protects SHAP contribution |
| `floor_rule_applied`                              | Records which minority-floor option (§2.7) was applied                                                  |
| `binarization_rule`                               | For non-binary raw targets (assistments), the exact rule, not just an `OK_BINARIZED` flag               |
| `subsample_strategy`                              | `none` / `stratified_50k` / `blocked:<axis>`                                                            |
| `cv_scheme`                                       | `stratified` / `temporal` / `blocked:<group>` (EEG, electricity → temporal)                             |

---

## 5. Phase 0 Exit Checklist

Phase 0 closes when **all** of the following are committed:

- [ ] seismic-bumps contradiction reconciled (§2.2); one authoritative record
- [ ] Minority-floor decision made and propagated (§2.7)
- [ ] HELOC sentinel decoding implemented with three-way semantics (§1.3); dual missingness columns populated
- [ ] assistments binarized + balance-checked + ID decision made (§3)
- [ ] **Four re-pulls completed: KDD, diabetes-readmission, compas-two-years, rain-in-australia** — compas and diabetes-readmission are load-bearing for the group-aware conformal contribution (Novelty Bet #2)
- [ ] Repeated stratified 5-fold XGBoost screen executed on the final surviving corpus; `screening_results.csv` (pass/fail per dataset) committed alongside the manifest
- [ ] Manifest extended with §4 columns and regenerated

**Single blocking action:** the screening run — but it must run _last_, after the fixes above, or its results are invalid on 3–4 datasets.
