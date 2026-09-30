# Diabetes 30-Day Readmission Prediction

A machine learning pipeline that predicts whether a diabetic patient will be readmitted to hospital within 30 days, built from the UCI "Diabetes 130-US hospitals" dataset (`diabetic_data.csv`).

This is a University of Limpopo group project (SCOA032). The code follows the approach set out in the group presentation, rebuilt from the raw data one phase at a time so every step can be traced and explained.

## Repository contents

| File | What it does | Why it is a separate file |
| --- | --- | --- |
| `clean_data.py` | Phases 1-3: inspects the raw data, cleans it, builds the target and the feature set. Writes `diabetic_features.csv`. | Data preparation is done once. Keeping it apart means modelling can be re-run without repeating it. |
| `train_and_tuning.py` | Phases 4-5: splits the data, then trains and tunes three models with cross-validation. Writes `train_split.csv`, `test_split.csv` and `best_models.joblib`. | Training is the slowest step, so it is isolated from evaluation. |
| `evaluate_models.py` | Phase 6: evaluates the tuned models on the held-out test set, produces the charts and saves the final model. | Evaluation can be re-run quickly without retraining. |
| `.gitignore` | Keeps the dataset and generated files out of the repository. | See "Files not included" below. |

## How to run

Run the scripts in this order, from one folder:

1. **Get the data.** Download `diabetic_data.csv` from the UCI Machine Learning Repository ("Diabetes 130-US hospitals for years 1999-2008") and place it in the same folder as the scripts.
2. `python clean_data.py`
3. `python train_and_tuning.py`
4. `python evaluate_models.py`

**Requirements:** Python 3 with `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `joblib`.

The order matters because each script reads the file the previous one wrote.

## Method and reasons

**Cleaning (Phases 1-2).** The cleaned data has 101,763 rows and 42 columns. The 2,423 rows where the patient died or went to hospice were kept rather than removed. This is a known limitation: these patients cannot be readmitted, so they may distort the results.

**Target and features (Phase 3).** The target is readmission within 30 days. The model uses 40 features. The diagnosis codes (`diag_1` to `diag_3`) are grouped into about 10 clinical categories, because the raw codes have hundreds of values and would be too sparse to learn from. Only the original columns are used, with no engineered features, to stay consistent with the presentation.

**Splitting (Phase 4).** The data is split 80% train and 20% test, with seed 42 for reproducibility. The split is stratified by the target, so both parts keep the same readmission rate, and grouped by patient, so one patient's hospital visits never appear in both parts. Without grouping, the model could memorise a patient it saw in training and inflate the test scores.

**Models and tuning (Phase 5).** Three models were used, as in the presentation: Logistic Regression, Decision Tree and Random Forest.

- Class weighting is set to `balanced`, because readmissions are the minority class.
- Hyperparameters are tuned with grouped, stratified 5-fold cross-validation on the training portion only, so the test set is never used for tuning. 5 folds is a standard balance between reliable estimates and run time.
- Models are chosen on F1-score, which balances precision and recall and suits an imbalanced target. Accuracy alone would reward predicting "not readmitted" for everyone.

Chosen settings:

| Model | Settings | CV F1 |
| --- | --- | --- |
| Logistic Regression | C = 1.0 | 0.271 |
| Decision Tree | max_depth = 6, min_samples_leaf = 50 | 0.270 |
| Random Forest | max_depth = 12, min_samples_leaf = 20 | 0.275 |

**Evaluation (Phase 6).** Test-set results at the default threshold:

| Model | F1 | Recall | Precision |
| --- | --- | --- | --- |
| Logistic Regression | 0.275 | 0.552 | 0.183 |
| Decision Tree | 0.266 | 0.557 | 0.175 |
| Random Forest | 0.280 | 0.576 | 0.185 |

Random Forest was chosen as the final model. Its margin over the other two is small, so the choice should not be over-interpreted. The evaluation also includes confusion matrices, a precision-recall curve, a threshold analysis and a top-predictors chart based on permutation importance.

**Main finding.** `number_inpatient` and `discharge_disposition_id` dominate the importance ranking, and all other features matter far less. The discharge feature may partly reflect the death and hospice rows that were kept.

## Files not included

The repository contains code only. These are excluded by `.gitignore` because they are large and are recreated by running the scripts:

- `diabetic_data.csv` (raw dataset, download it from UCI)
- `diabetic_cleaned.csv`, `diabetic_features.csv`, `train_split.csv`, `test_split.csv`
- `*.joblib` (saved models)

## Limitations

- Death and hospice-discharge rows were kept, as described above.
- F1 scores are modest (about 0.27-0.28), which reflects how hard 30-day readmission is to predict from these columns alone.
- Differences between the three models are small.
- The choice of 5 folds is a convention and was not compared against other values.
