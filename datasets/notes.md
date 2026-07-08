Dataset: HELOC

Special values observed:

-7
-8
-9

Occurrences:

-7:
MSinceMostRecentDelq -> 45.11%
MSinceMostRecentInqexcl7days -> 17.70%

-8:
...

-9:
Appears consistently in ~5.57% of rows across many features.

Status:
Awaiting verification against official FICO documentation.

No transformations applied.

## Disqualified Datasets

Datasets removed from the corpus during Phase 0, with rationale.
Minority-floor decisions assume a 10% minimum positive-class rate
(see corpus design note). Entries marked [FLOOR-SENSITIVE] would
re-qualify if the floor is relaxed to 5%.

### coil2000 [FLOOR-SENSITIVE]

- Source: OpenML id 298
- Metric: minority class (CARAVAN=1) = 5.97%
- Reason: Below the 10% minority-class floor. Class imbalance this
  severe entangles corruption effects (esp. injected label noise)
  with baseline sampling variance, undermining clean Δ-performance
  measurement. Re-qualifies only if the floor is set to 5%.

### seismic-bumps (file was actually UCI Seeds)

- Source: intended OpenML id 1500; pulled file does not match
- Metric: 210 rows, 7 features, 3 classes at exactly 70/70/70
- Reason: Provenance failure — the file on disk is the UCI Seeds
  (wheat kernel) dataset, not seismic-bumps (~2,584 rows, binary).
  Signature is unmistakable (Seeds is 210 rows / 3 balanced classes).
  Likely off-by-one in the OpenML fetch ID. Not in the locked corpus;
  also fails on size (n<2,000) and task (3-class). Removed; retained
  as the motivating example of silent data-provenance error.

### house_16H

- Source: OpenML id 821
- Metric: quasi-anonymized column names (P1, P5p1, H2p2, ...)
- Reason: Not in the locked corpus. Column headers are opaque codes,
  which conflicts with the SHAP feature-attribution contribution
  (same exclusion rule applied to jannis/rl/albert/Higgs). Target is
  a binarized regression variable with weak semantic grounding.

### taiwanese_bankruptcy

- Source: UCI id 572
- Metric: minority class (Bankrupt?=1) = 3.23%
- Reason: Below the minority-class floor at both 10% and 5%. Also
  contains near-constant flag columns (e.g. Net Income Flag), adding
  little signal. Fails floor decisively regardless of threshold.

### polish_bankruptcy

- Source: UCI id 365
- Metric: minority class = 4.82%
- Reason: Below the minority-class floor at both 10% and 5%.
  Additionally pools five separate forecasting horizons (year 1-5)
  into a single pseudo-task via a `year` column, conflating distinct
  prediction problems. Fails floor decisively regardless of threshold.

### Credit (CleanML)

- Source: CleanML (Give Me Some Credit); n = 150,000
- Metric: rows exceed cap; target SeriousDlqin2yrs ~6-7% positive
- Reason: Two independent grounds. (1) 150k rows exceed the 50k row
  cap and fall outside the exception protocol (iid-track, not a
  designated shift anchor). (2) Domain concentration — would be the
  third credit-risk dataset alongside HELOC and default_credit,
  over-weighting a single domain and mirroring the source-concentration
  weakness this project critiques in PAPE. Expected minority rate also
  likely fails the 10% floor (verify with corrected target inspector).

# Dataset: assistments (TableShift)

**Source:** ASSISTments 2012–2013 school year, open-source data snapshot
**Path:** `datasets/raw/tableshift/assistments.parquet`
**Column reference:** ASSISTments data dictionary — https://sites.google.com/site/assistmentsdata/how-to-interpret

## Basic stats

- Rows: 6,123,270
- Columns: 35
- Missing cells: 13,426,647 (6.26%)

## Target

- Raw column: `correct`
- Raw type: **not binary** — 13 distinct values (1 = correct on first attempt; decimal values = partial credit based on hints/attempts; 0 = incorrect/answer revealed/exhausted credit)
- **Binarization rule (per official documentation):** `1 = correct`, `<1 = incorrect`
- Binarized minority % **not yet measured** — raw multiclass read returned a meaningless 0.00%; must re-run balance check on the binarized target before this dataset can pass/fail the screen

## Known leakage risk (features that reflect the solving process, not pre-attempt state)

Flagged for exclusion during preprocessing:
`hint_count`, `bottom_hint`, `actions`, `attempt_count`, `ms_first_response`, `overlap_time`, `answer_id`, `answer_text`, `first_action`, `end_time`

Also flagged (likely derived from the same interaction being labeled — proxy leakage risk):
`Average_confidence(FRUSTRATED)`, `Average_confidence(CONFUSED)`, `Average_confidence(CONCENTRATING)`, `Average_confidence(BORED)`

## Likely safe features (known at/before problem presentation)

`skill`, `problem_id`, `user_id`, `assignment_id`, `assistment_id`, `start_time`, `problem_type`, `original`, `tutor_mode`, `sequence_id`, `student_class_id`, `position`, `type`, `base_sequence_id`, `skill_id`, `teacher_id`, `school_id`, `template_id`

## Open items (deferred to preprocessing phase)

- [ ] Binarize `correct` and re-run minority-class check
- [ ] Confirm/apply leakage-column exclusion list
- [ ] Define subsample strategy: 6.1M rows → cap, must be **blocked** (student/school/time axis), not random
- [ ] Decide treatment of the two affect-confidence proxy columns (drop vs. keep-with-caveat)
