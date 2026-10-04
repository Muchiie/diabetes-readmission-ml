# Phases 4-5: Splitting the Data, Then Training and Tuning the Models

**Script:** `train_and_tuning.py`
**Input:** `diabetic_features.csv` (from `clean_data.py`)
**Outputs:** `train_split.csv`, `test_split.csv`, `cv_results_phase5.csv`, `cv_fold_scores_phase5.csv`, `cv_folds_structure_phase5.png`, `cv_f1_by_fold_phase5.png`, `best_models.joblib`
**Run with:** Python 3.11 (`py -3.11 train_and_tuning.py`)
**Logging:** none. Results print to the terminal only.

This document records what each step did, the numbers it produced, and why the step was taken.

---

## Decisions that apply to these phases

| Decision | Reason |
| --- | --- |
| 80/20 train/test split, by patient and stratified, seed 42 | Test patients must be unseen, and both sets need a similar share of early readmissions. A fixed seed makes the split repeatable. |
| Same three models as the presentation: Logistic Regression, Decision Tree, Random Forest | Follows the presentation. |
| `class_weight='balanced'` for all three | Only 11.16% of rows are positive. |
| Stratified, patient-grouped 5-fold cross-validation on the training set only | The test set must stay untouched so its final result is honest. |
| F1 as the score for choosing settings | Balances precision and recall for an imbalanced target. Recall and precision are also recorded. |
| Light tuning: one setting per model, the others fixed | Group decision. Keeps the search small and easy to explain (30 fits). |
| Show the cross-validation as two pictures (fold structure, F1 on each fold) | Makes the 5-fold process and its stability easy to present. F1 only, since F1 is the selection score. |
| Keep `best_models.joblib` | The evaluation script loads it, so it does not retrain, and it scores exactly the models that were tuned. |

---

## Phase 4: Split into train and test

**Input:** 101,763 rows x 42 columns, 40 features, 11,357 positives (11.16%).

### Why a plain random split is not enough

- **Repeat patients:** 30,248 rows are repeat visits by patients who appear more than once. A row-level split could put one visit in training and another in testing, so the model would be tested on a patient it had already learned from. That is leakage and it inflates scores.
- **Class imbalance:** only 11.16% of rows are positive, so a chance split could leave the two sets with different positive rates.

**Solution:** a stratified group split. All visits of one patient stay together, and the positive rate stays similar in both sets.

### How the split was made

`StratifiedGroupKFold` with 5 folds divides the patients into 5 groups of roughly equal size and positive rate. The first group (about 20%) is the test set, and the other four (about 80%) are the training set. `random_state=42` produces the same split every run.

### Results

| Set | Rows | Share | Patients | Positives | Positive rate |
| --- | --- | --- | --- | --- | --- |
| Overall | 101,763 | 100.0% | 71,515 | 11,357 | 11.16% |
| Train | 81,526 | 80.1% | 57,175 | 9,167 | 11.24% |
| Test | 20,237 | 19.9% | 14,340 | 2,190 | 10.82% |

### Checks

- Patients in both train and test: **0** (must be 0).
- Train rows + test rows = 101,763 = all rows (nothing lost or duplicated).
- Difference in positive rate between train and test: **0.42 percentage points**.

**Saved:** `train_split.csv` and `test_split.csv`. `patient_nbr` is kept in both because cross-validation in Phase 5 must also keep each patient's visits together.

---

## Phase 5: Train and tune the three models

**Why tune:** each model has settings (hyperparameters) that change how it behaves. Settings are chosen by cross-validation on the **training set only**, so the test set is not used anywhere in this phase.

**Training data:** 81,526 rows, 40 features, 9,167 positives (11.24%).

### Step 1: Preprocessing (inside the model pipeline)

| Feature type | Treatment | Reason |
| --- | --- | --- |
| 8 numeric | `StandardScaler` | Puts counts and durations on the same scale. Logistic Regression needs it, and it is harmless for the trees. |
| 32 categorical | `OneHotEncoder` (unseen categories ignored) | Models need numbers. Each category becomes a 0/1 column. |

Both are fitted on the training part of each cross-validation round only, so nothing from the held-out part leaks in. That is why they live inside the pipeline.

### Step 2: Class weighting

`class_weight='balanced'` for all three models. Without weights, a model can score well by almost always predicting "not readmitted" and miss the patients we care about. The weights make a missed early readmission cost about 8 times more than a false alarm, which reflects the 88.84% to 11.16% class ratio.

### Step 3: Cross-validation

- 5 folds, stratified and grouped by patient (the same idea as the Phase 4 split).
- Each fold keeps about 11% positives, and each patient's visits stay in one fold.
- Averaging over 5 folds gives a steadier score than a single validation split, which could be lucky or unlucky.

**Fold sizes:** the 5 folds run on the 80% training set, so in each round one fold (20% of 80%, about **16% of the full data**) is used for validation and the other four (about 64% of the full data) for training. The 16% is the size of each validation fold, not a separate setting.

**Why 5 folds:** a standard balance between reliable estimates and run time. Each training round still uses a large share of the data, and each validation fold is large enough to hold a stable number of positives. 5 is a convention and was not compared against other values.

### Step 4: Settings tried (light tuning, 30 fits)

One setting is tuned per model. The others are fixed.

| Model | Tuned | Fixed | Reason |
| --- | --- | --- | --- |
| Logistic Regression | `C` = 0.1, 1.0 | `max_iter` = 1000 | `C` is the regularisation strength (smaller means a simpler model). Two values a factor of 10 apart show whether a simpler or freer model works better. |
| Decision Tree | `max_depth` = 6, 8 | `min_samples_leaf` = 50 | Deep trees memorise the training data, so depth is the main setting. Shallow depths also keep the tree explainable to doctors. |
| Random Forest | `max_depth` = 8, 12 | `min_samples_leaf` = 20, `n_estimators` = 100 | Same reason for depth. 100 trees is a standard size, since more adds time for little gain. |

**Note on the fixed values:** `min_samples_leaf` = 50 (tree) and 20 (forest) were the winning values in an earlier full-grid run. That run also used cross-validation on the training set only, so the test set was not involved, but these two values were not chosen blind.

### Step 5: Results (mean over 5 folds)

**Logistic Regression** (10 fits, 30 seconds)

| Settings | F1 | +/- | Recall | Precision |
| --- | --- | --- | --- | --- |
| C=0.1 | 0.273 | 0.005 | 0.549 | 0.182 |
| C=1.0 | 0.273 | 0.005 | 0.550 | 0.181 |

**Chosen: C=0.1** (F1 0.273). The two settings tie to three decimals, so the choice makes no practical difference. C=0.1 is listed first and is also the simpler model.

**Decision Tree** (10 fits, 25 seconds)

| Settings | F1 | +/- | Recall | Precision |
| --- | --- | --- | --- | --- |
| max_depth=6 | 0.264 | 0.005 | 0.562 | 0.173 |
| max_depth=8 | 0.260 | 0.010 | 0.583 | 0.167 |

**Chosen: max_depth=6** (F1 0.264). Depth 8 has higher recall but lower precision and lower F1.

**Random Forest** (10 fits, 77 seconds)

| Settings | F1 | +/- | Recall | Precision |
| --- | --- | --- | --- | --- |
| max_depth=12 | 0.276 | 0.003 | 0.554 | 0.184 |
| max_depth=8 | 0.272 | 0.003 | 0.594 | 0.177 |

**Chosen: max_depth=12** (F1 0.276).

### Summary of chosen settings

| Model | Chosen settings | CV F1 |
| --- | --- | --- |
| Logistic Regression | C=0.1 | 0.273 |
| Decision Tree | max_depth=6, min_samples_leaf=50 | 0.264 |
| Random Forest | max_depth=12, min_samples_leaf=20, 100 trees | 0.276 |

### Reading the results

- The three F1 scores are close (0.264 to 0.276), and the fold-to-fold spread is +/- 0.003 to 0.010. Differences between models and between settings are small and should not be over-interpreted.
- Each chosen setting was then refitted on the whole training set.
- Recall and precision trade off against each other, and the direction is not the same for every model (for example, the deeper tree has higher recall, while the deeper forest has lower recall). F1 balances the two and decided each choice.

### Step 6: How the 5 folds were built (actual sizes)

The folds are rebuilt with the same settings and seed as the tuning, so they are the same folds. Picture: `cv_folds_structure_phase5.png`.

| Round | Validation rows | % of all data | Training rows | % of all data | Validation positives | Positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16,309 | 16.0% | 65,217 | 64.1% | 1,917 | 11.75% |
| 2 | 16,287 | 16.0% | 65,239 | 64.1% | 1,836 | 11.27% |
| 3 | 16,219 | 15.9% | 65,307 | 64.2% | 1,834 | 11.31% |
| 4 | 16,415 | 16.1% | 65,111 | 64.0% | 1,800 | 10.97% |
| 5 | 16,296 | 16.0% | 65,230 | 64.1% | 1,780 | 10.92% |

- On average, each round validates on **16.0% of all the data** and trains on **64.1%**. The remaining 19.9% (the test set) is not touched.
- The folds are close to equal in size but not identical, because the split keeps each patient's visits together and patients have different numbers of visits.
- Each fold keeps a positive rate of about 11% (10.92% to 11.75%), which is the point of stratifying.
- Check: no patient appears in both the training and validation part of any round.

**Why this picture:** it turns "5-fold cross-validation on the 80% training set" into numbers. 16% is not a setting. It is the size of each validation fold (a fifth of the 80%).

### Step 7: F1 on each validation fold

The scores were recorded during the tuning, so this needed no extra fits. Picture: `cv_f1_by_fold_phase5.png`. Data: `cv_fold_scores_phase5.csv` (30 rows: 6 settings x 5 folds, with F1, recall and precision).

| Model | Settings | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | **C=0.1 (chosen)** | 0.276 | 0.270 | 0.279 | 0.266 | 0.273 | 0.273 |
| Logistic Regression | C=1.0 | 0.277 | 0.268 | 0.280 | 0.267 | 0.272 | 0.273 |
| Decision Tree | **max_depth=6 (chosen)** | 0.272 | 0.267 | 0.261 | 0.266 | 0.256 | 0.264 |
| Decision Tree | max_depth=8 | 0.275 | 0.260 | 0.263 | 0.258 | 0.244 | 0.260 |
| Random Forest | **max_depth=12 (chosen)** | 0.280 | 0.275 | 0.277 | 0.277 | 0.271 | 0.276 |
| Random Forest | max_depth=8 | 0.275 | 0.270 | 0.275 | 0.274 | 0.267 | 0.272 |

Check: the 5 fold scores of every setting average to its Phase 5 mean F1.

**What the per-fold view shows**
- **Logistic Regression:** the two settings differ by 0.002 or less on every fold, which supports calling the choice a tie.
- **Decision Tree:** depth 6 beats depth 8 on 3 of the 5 folds, and depth 8 is slightly ahead on the other 2. Depth 6 wins on average, with less variation between folds.
- **Random Forest:** depth 12 beats depth 8 on all 5 folds, so that choice is consistent rather than the result of one lucky fold.
- **Across folds:** fold 5 is the lowest fold for both Decision Tree settings and both Random Forest settings, while fold 4 is the lowest for both Logistic Regression settings. Within one setting the scores vary by about 0.01 to 0.03 across folds, which is larger than the gap between most settings, so small differences between settings and models should not be over-interpreted.

**Why this picture:** the averages in Step 5 hide how the score moves between folds. This shows how stable each model is and that each chosen setting is not just one lucky fold.

### Files saved

| File | Contents |
| --- | --- |
| `cv_results_phase5.csv` | Every setting tried with its cross-validation scores (6 rows) |
| `cv_fold_scores_phase5.csv` | F1, recall and precision of every setting on every validation fold (30 rows) |
| `cv_folds_structure_phase5.png` | Picture of how the 5 folds are cut from the 80% training set |
| `cv_f1_by_fold_phase5.png` | F1 on each validation fold, one chart per model |
| `best_models.joblib` | The three tuned models, loaded by the evaluation script |

**The test set has not been touched.** It is used once, in the evaluation phase (Phase 6).
