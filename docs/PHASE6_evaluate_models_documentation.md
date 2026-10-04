# Phase 6: Evaluating the Tuned Models on the Test Set

**Script:** `evaluate_models.py`
**Inputs:** `best_models.joblib`, `train_split.csv`, `test_split.csv` (from `train_and_tuning.py`)
**Outputs:** `evaluation_results.csv`, `feature_importance_random_forest.csv`, `confusion_matrices.png`, `precision_recall_curve.png`, `threshold_analysis.png`, `top_predictors.png`
**Run with:** Python 3.11 (`py -3.11 evaluate_models.py`)
**Logging:** none. Results print to the terminal only.

This document records what each step did, the numbers it produced, and why the step was taken.

---

## Decisions that apply to this phase

| Decision | Reason |
| --- | --- |
| The test set is used once, here, to report results only | Nothing is chosen or tuned on it, so its results are an honest estimate on patients the models have never seen. |
| Models are scored with their default rule (flag a patient when the predicted probability is above 0.5) | The threshold was not changed on the test set. Class weighting already makes 0.5 sensitive to the minority class. |
| Accuracy is not the headline metric | The data is 89% "not readmitted", so a model that flags nobody scores high accuracy while catching no one. |
| Threshold analysis uses the training data only (out-of-fold predictions) | It lets us study thresholds without touching the test set. |
| No PR-AUC number is reported | Group decision. The precision-recall curve is a picture only. |
| Random Forest is the final model | Group decision, made before this phase. |
| `threshold_analysis_train.csv` and `final_model_random_forest.joblib` are not saved | Nothing reads them. The threshold table prints on screen and its picture is still drawn. The Random Forest is already stored inside `best_models.joblib`. |

---

## Step 1: Load the models and the data

- Models loaded: Logistic Regression, Decision Tree, Random Forest (each already tuned and refitted on the whole training set in Phase 5).
- Test set: **20,237 rows, 2,190 early readmissions (10.82%)**.

**Why:** loading the saved models means the slow tuning is never repeated, and the test set is scored by exactly the models that were chosen.

---

## Step 2: Metrics on the test set (default decision rule)

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.673 | 0.176 | 0.551 | 0.267 |
| Decision Tree | 0.651 | 0.167 | 0.556 | 0.256 |
| Random Forest | 0.675 | 0.179 | 0.558 | 0.271 |

**Reference points**
- A "model" that predicts "not readmitted" for every patient would score Accuracy **0.892** but Recall **0.000**. The models' lower accuracy is the price of actually catching early readmissions.
- Random guessing would give Precision of about **0.108** (the share of positives). All three models are well above that.

**Reading the metrics**
- **Recall** is the share of early readmissions the model catches. **Precision** is the share of flagged patients who really were readmitted early. **F1** balances the two.
- The three models are close. Random Forest is highest on F1, precision and accuracy, and Decision Tree is lowest on all three, but the gap between the best and worst F1 is only 0.015.

**Test scores compared with cross-validation scores (Phase 5)**

| Model | CV F1 (validation folds) | Test F1 | Difference |
| --- | --- | --- | --- |
| Logistic Regression | 0.273 | 0.267 | -0.006 |
| Decision Tree | 0.264 | 0.256 | -0.008 |
| Random Forest | 0.276 | 0.271 | -0.005 |

The test scores are slightly lower than the CV scores for every model. The differences (0.005 to 0.008) are about the size of the variation between folds (0.003 to 0.010), so there is no sign that the tuning overfitted the validation folds. The test set also has a slightly lower positive rate than the training set (10.82% against 11.24%), which can lower precision and F1 a little.

**Saved:** `evaluation_results.csv` (3 rows x 9 columns: the four metrics plus the confusion-matrix counts).

---

## Step 3: Confusion matrices (test set)

**How to read them:** rows are what really happened and columns are what the model predicted. TN = correctly said "not readmitted early". FP = false alarm. FN = a missed early readmission (the costliest error). TP = correctly flagged.

| Model | TN | FP | FN | TP | Caught | Missed | Patients flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 12,408 | 5,639 | 983 | 1,207 | 1,207 of 2,190 (55.1%) | 983 | 6,846 |
| Decision Tree | 11,963 | 6,084 | 973 | 1,217 | 1,217 of 2,190 (55.6%) | 973 | 7,301 |
| Random Forest | 12,442 | 5,605 | 969 | 1,221 | 1,221 of 2,190 (55.8%) | 969 | 6,826 |

**What this shows (Random Forest as the example):** it flags 6,826 patients and catches 1,221 of the 2,190 early readmissions, but 5,605 of the flagged patients are false alarms (precision 17.9%), and 969 early readmissions are missed.

**Why this view:** a single number hides the two kinds of error. The matrix shows exactly how many patients are caught, missed or wrongly flagged.

**Saved:** `confusion_matrices.png` (one matrix per model, with row percentages).

---

## Step 4: Threshold analysis (training data only)

**Method:** each training patient is scored by a model that never saw them (the same patient-grouped 5-fold split as Phase 5, seed 42). Recall, precision and F1 are then computed for thresholds from 0.20 to 0.80. The test set is not used.

**Why:** a model flags a patient when its probability passes a threshold. Moving the threshold trades false alarms against missed patients. The table shows whether a threshold other than 0.5 would have worked better.

Selected rows (full tables are in the terminal output):

| Model | Threshold | Patients flagged | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.45 | 48.2% | 0.689 | 0.161 | 0.260 |
| Logistic Regression | **0.50 (default)** | 34.0% | 0.549 | 0.182 | 0.273 |
| Logistic Regression | **0.55 (best F1)** | 23.0% | 0.420 | 0.205 | 0.276 |
| Decision Tree | **0.50 (default)** | 36.6% | 0.562 | 0.173 | 0.264 |
| Decision Tree | **0.55 (best F1)** | 26.2% | 0.447 | 0.192 | 0.268 |
| Random Forest | 0.45 | 54.2% | 0.751 | 0.156 | 0.258 |
| Random Forest | **0.50 (default and best F1)** | 33.9% | 0.554 | 0.184 | 0.276 |
| Random Forest | 0.55 | 16.0% | 0.322 | 0.227 | 0.266 |

**What this shows**
- Lowering the threshold flags more patients, so recall rises and precision falls. Raising it does the opposite.
- The best F1 threshold is 0.55 for Logistic Regression and Decision Tree and 0.50 for Random Forest. For the first two, the gain over the default is only 0.003 to 0.004. So the default 0.5 is at or very close to the best F1 point for all three models, and there is no reason to change it for the final model.
- The Random Forest rarely gives probabilities above 0.65. At 0.75 it flags almost no one (0.0% of patients, none of them correct), and at 0.80 it flags no one, which is why that row of the table shows dashes.
- Which point to use in practice is a hospital decision about how many false alarms it can afford. For example, a lower threshold would catch more early readmissions at the cost of many more false alarms.

**Saved:** `threshold_analysis.png` (recall, precision and F1 against the threshold, one chart per model).

---

## Step 5: Precision-recall curve (test set)

**Why:** the curve shows, for every possible threshold, how much precision must be given up to gain recall. A dot on each curve marks the model's default rule. The dashed line is random guessing (precision 0.108): the further a curve is above it, the better. It is a picture only, and nothing is chosen from it.

**Saved:** `precision_recall_curve.png`.

---

## Step 6: Top predictors of the final model (Random Forest)

**Method (permutation importance):** each factor is shuffled in the test set, five times, so it carries no real information, and the fall in the model's F1 is measured. A big fall means the model relies on that factor. Nothing is chosen from this. It only explains the model.

| Rank | Factor | Fall in F1 | +/- |
| --- | --- | --- | --- |
| 1 | number_inpatient | 0.0448 | 0.0016 |
| 2 | discharge_disposition_id | 0.0382 | 0.0027 |
| 3 | diag_1_category | 0.0047 | 0.0011 |
| 4 | diabetesMed | 0.0027 | 0.0023 |
| 5 | number_emergency | 0.0025 | 0.0007 |
| 6 | num_procedures | 0.0024 | 0.0007 |
| 7 | num_lab_procedures | 0.0021 | 0.0011 |
| 8 | admission_source_id | 0.0019 | 0.0018 |
| 9 | admission_type_id | 0.0018 | 0.0013 |
| 10 | age | 0.0016 | 0.0004 |

**What this shows**
- Two factors dominate: the number of previous inpatient visits and the discharge disposition. Shuffling either one lowers F1 by about 0.04, while the third-ranked factor lowers it by only 0.005.
- **23 of the 40 factors** have a fall no bigger than their own spread, so the model gets little or nothing from them.
- Discharge disposition includes the death and hospice codes of the 2,423 rows that were kept in Phase 3 (early-readmission rate 1.77%). Part of this factor's importance may come from those rows. The output does not show how much.

**Cautions**
- This shows what the model relies on, not what causes readmission.
- Related factors (for example the medication columns) share credit, so each one can look smaller than the group really is.

**Saved:** `feature_importance_random_forest.csv` (all 40 factors) and `top_predictors.png` (top 15).

---

## Step 7: The final model in use

The final model is the tuned Random Forest, stored inside `best_models.joblib`. It is a pipeline: a raw patient record (40 fields) goes through scaling and one-hot encoding, then the Random Forest, which gives the probability of early readmission. The patient is flagged if that is above 0.5.

**Example:** 5 patients picked at random from the test set (fixed seed 7, not hand-picked):

| Age | Main diagnosis | Prior inpatient visits | Risk | Flagged | Really readmitted <30 |
| --- | --- | --- | --- | --- | --- |
| [50-60) | Other | 4 | 0.56 | yes | no |
| [50-60) | Respiratory | 8 | 0.63 | yes | no |
| [70-80) | Genitourinary | 0 | 0.54 | yes | no |
| [70-80) | Circulatory | 1 | 0.55 | yes | no |
| [70-80) | Circulatory | 1 | 0.49 | no | no |

Four of these five were flagged and none was readmitted early. This is only a random draw of five, so it is not representative. It does illustrate the false-alarm problem: about 4 in 5 flagged patients would not be readmitted early (precision 17.9%).

To reuse the model later: `model = joblib.load('best_models.joblib')['Random Forest']`, then `model.predict_proba(new_patients)[:, 1]` with the same 40 columns.

**It is a screening aid, not a diagnosis.**

---

## Summary

| Model | Test F1 | Recall | Precision | Early readmissions caught |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.267 | 0.551 | 0.176 | 1,207 of 2,190 |
| Decision Tree | 0.256 | 0.556 | 0.167 | 1,217 of 2,190 |
| Random Forest (final) | 0.271 | 0.558 | 0.179 | 1,221 of 2,190 |

## Limitations

- F1 scores are modest (about 0.26 to 0.27). Predicting 30-day readmission from these columns alone is hard.
- The models catch about 55% of early readmissions and miss about 45%, while roughly 4 in 5 flagged patients are false alarms.
- The three models are close, and the Random Forest's lead is small, so the choice should not be over-interpreted.
- The 2,423 death and hospice rows were kept and may influence the results, especially through discharge disposition.
- Results come from a single 80/20 split. No confidence interval was calculated for the test scores.
- Permutation importance shows what the model uses, not what causes readmission.
