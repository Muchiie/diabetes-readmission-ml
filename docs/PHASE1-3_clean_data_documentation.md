# Phases 1-3: Inspecting, Cleaning and Preparing the Data

**Script:** `clean_data.py`
**Input:** `diabetic_data.csv` (raw UCI data)
**Outputs:** `diabetic_cleaned.csv` (end of Phase 2), `diabetic_features.csv` (end of Phase 3)
**Run with:** Python 3.11 (`py -3.11 clean_data.py`), pandas 2.2.2
**Logging:** none. Results print to the terminal only.

This document records what each step did, the numbers it produced, and why the step was taken.

---

## Decisions that apply to these phases

| Decision | Reason |
| --- | --- |
| Binary target: readmitted within 30 days (`<30`) = 1, everything else = 0 | Follows the group presentation. |
| Keep the 2,423 death/hospice-discharge rows | Group decision. Stated as a limitation in the report (see Phase 3). |
| Group diagnosis codes into 10 categories | Each diagnosis column has 700+ unique codes, too many for a model to learn from. |
| Use original columns only, with no engineered features | Stays consistent with the presentation. |
| Keep `patient_nbr` only as a split key, never as a feature | Phase 4 splits by patient. Using it as a feature would let the model identify patients. |
| F1 will be the selection score (used in Phase 5) | Group decision. Accuracy would reward predicting "not readmitted" for everyone. |

---

## Phase 1: Load and inspect the raw data

**Purpose:** find out exactly what we start with before changing anything. Nothing is modified or saved in this phase.

### 1a. Size and target

- **Raw data:** 101,766 rows, 50 columns.
- **Target column `readmitted`** has three values:

| Value | Rows | Share |
| --- | --- | --- |
| NO | 54,864 | 53.91% |
| >30 | 35,545 | 34.93% |
| <30 | 11,357 | 11.16% |

- **Binary view** (`<30` = 1, else 0): 11,357 positive vs 90,409 negative.
- **Baseline:** predicting "not readmitted" for everyone would score 88.84% accuracy.

**Why this matters:** the classes are heavily imbalanced. A model could look accurate while missing every early readmission, which is why we select models on F1 and use class weighting later.

### 1b. Missing values (two kinds)

The dataset hides most missing values as the text `?`, so an ordinary null check would wrongly report none. The script counts both kinds.

| Column | Kind | Missing | Share |
| --- | --- | --- | --- |
| weight | `?` | 98,569 | 96.86% |
| max_glu_serum | NaN | 96,420 | 94.75% |
| A1Cresult | NaN | 84,748 | 83.28% |
| medical_specialty | `?` | 49,949 | 49.08% |
| payer_code | `?` | 40,256 | 39.56% |
| race | `?` | 2,273 | 2.23% |
| diag_3 | `?` | 1,423 | 1.40% |
| diag_2 | `?` | 358 | 0.35% |
| diag_1 | `?` | 21 | 0.02% |

### 1c. Columns with a single value

`examide` and `citoglipton` contain one value in every row, so they carry no information.

### 1d. Invalid categories

Gender: Female 54,708, Male 47,055, **Unknown/Invalid 3**.

### 1e. Repeated patients

- Unique patients: 71,518.
- Repeat encounters (rows beyond a patient's first visit): 30,248.
- Exact duplicate rows: 0.

**Why this matters:** the same patient can appear many times. If one visit lands in training and another in testing, the model is tested on a patient it has already learned from, which inflates scores. This is the reason Phase 4 splits by patient.

### 1f. Diagnosis code cardinality

| Column | Unique codes |
| --- | --- |
| diag_1 | 717 |
| diag_2 | 749 |
| diag_3 | 790 |

**Why this matters:** most codes appear only a handful of times, so a model cannot learn from them directly. They are grouped in Phase 3.

---

## Phase 2: Clean the data

**Purpose:** remove or label everything that would confuse a model, and record each change so it can be explained. The raw table is copied first so the original stays untouched for comparison.

**Start:** 101,766 rows, 50 columns.

### Step 1: Drop 8 columns

| Column | Reason |
| --- | --- |
| encounter_id | An ID only, not medical information |
| weight | 96.86% missing |
| payer_code | 39.56% missing |
| medical_specialty | 49.08% missing |
| examide | One value in the whole column |
| citoglipton | One value in the whole column |
| max_glu_serum | 94.75% missing |
| A1Cresult | 83.28% missing |

Result: 42 columns. `patient_nbr` is kept only as the split key.

**Why drop instead of fill:** with this much missing data, any filled-in values would be mostly invented. The two single-value columns cannot help a model.

### Step 2: Remove invalid gender rows

3 rows with gender `Unknown/Invalid` were removed, leaving **101,763 rows**.

**Why:** only 3 rows, so removing them costs almost nothing and avoids a meaningless third category.

### Step 3: Race `?` becomes "Unknown"

2,271 values (2.23%) were relabelled `Unknown`. No rows were lost.

**Why:** the missing race values are a small share, and a separate "Unknown" category keeps those patients in the data.

### Step 4: Diagnosis `?` becomes "Missing"

| Column | Values replaced | Share |
| --- | --- | --- |
| diag_1 | 21 | 0.02% |
| diag_2 | 358 | 0.35% |
| diag_3 | 1,423 | 1.40% |

**Why:** same reasoning as race. A "Missing" category keeps the rows and keeps the fact that no diagnosis was recorded.

### Checks after cleaning

- Remaining `?` values: 0
- Remaining NaN values: 0
- Exact duplicate rows: 0
- Single-value columns: none
- Final shape: **101,763 rows x 42 columns** = 40 features + `patient_nbr` + target
- Target check: `<30` = 11,357 (11.16%), unchanged. The 3 removed rows were all in the negative class (90,409 became 90,406).

**Saved:** `diabetic_cleaned.csv`

---

## Phase 3: Build the target and the features

**Purpose:** give the models a numeric 0/1 target and features in a usable form. Encoding and scaling are not done here. They are fitted on training data only in Phase 5, so nothing from the test set leaks in.

### Step 1: Binary target `readmitted_binary`

- `<30` becomes 1 (readmitted within 30 days).
- `NO` and `>30` become 0.
- Class 0: 90,406 (88.84%). Class 1: 11,357 (11.16%).
- The original `readmitted` column is removed.

**Why remove it:** it is the answer. Keeping it as a feature would leak the target.

### Step 2: Group diagnosis codes into 10 categories

Each of `diag_1`, `diag_2`, `diag_3` (717, 749 and 790 codes) becomes a category column with **10 categories**: Circulatory, Respiratory, Digestive, Genitourinary, Musculoskeletal, Injury, Neoplasms, Diabetes, Other and Missing. The raw code columns are then removed.

Counts for `diag_1_category`:

| Category | Rows | Share |
| --- | --- | --- |
| Circulatory | 30,436 | 29.91% |
| Other | 18,172 | 17.86% |
| Respiratory | 14,423 | 14.17% |
| Digestive | 9,475 | 9.31% |
| Diabetes | 8,757 | 8.61% |
| Injury | 6,972 | 6.85% |
| Genitourinary | 5,117 | 5.03% |
| Musculoskeletal | 4,957 | 4.87% |
| Neoplasms | 3,433 | 3.37% |
| Missing | 21 | 0.02% |

**Why:** the groups follow standard ICD-9 chapter ranges, as used in the published study of this dataset (Strack et al., 2014). Hundreds of rarely seen codes become a few meaningful groups.

### Step 3: Define the features

**40 features in total:**

- **8 numeric:** `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.
- **32 categorical:** `race`, `gender`, `age` (kept as its original 10 age bands), 3 ID-like columns, the 3 diagnosis categories, 21 medication columns, `change` and `diabetesMed`.

**Why the 3 ID-like columns are categories:** `admission_type_id`, `discharge_disposition_id` and `admission_source_id` hold codes, not amounts. Code 6 is not "twice" code 3, so treating them as numbers would mislead the model.

**Check:** every column is accounted for, and the target and `patient_nbr` are not features.

### Information check: death and hospice discharges (no change made)

| Item | Value |
| --- | --- |
| Rows with death/hospice discharge codes (11, 13, 14, 19, 20, 21) | 2,423 (2.38%) |
| Early-readmission rate among them | 1.77% |
| Overall early-readmission rate | 11.16% |

**Decision:** these rows were kept. A patient who has died cannot be readmitted, so these rows are a known limitation and are stated as one in the report. The low 1.77% rate suggests the model can learn the pattern from `discharge_disposition_id`.

**Final Phase 3 table:** 101,763 rows x 42 columns (`patient_nbr` + 40 features + target).

**Saved:** `diabetic_features.csv`

---

## Summary of the data's journey

| Stage | Rows | Columns |
| --- | --- | --- |
| Raw | 101,766 | 50 |
| After Phase 2 (cleaned) | 101,763 | 42 |
| After Phase 3 (features) | 101,763 | 42 (`patient_nbr` + 40 features + target) |
