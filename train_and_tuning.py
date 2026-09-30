"""
TRAIN_AND_TUNING.PY - PHASES 4 TO 5: split, then train and tune the three models
================================================================================
Models (as in the presentation): Logistic Regression, Decision Tree, Random Forest.

HOW TO RUN
  1. Run clean_data.py first (it creates diabetic_features.csv)
  2. Keep this file in the SAME folder, then: python train_and_tuning.py
  Outputs (created next to this file):
       train_split.csv        - Phase 4 training set (about 80% of patients)
       test_split.csv         - Phase 4 test set (about 20% of patients, never used for tuning)
       cv_results_phase5.csv  - every setting tried in Phase 5 with its cross-validation scores
       best_models.joblib     - the three tuned models (used by the evaluation phase)
       train_log.txt          - everything printed, kept as proof for the report

SECTIONS
  PHASE 4: Split into train and test (by patient, stratified)
  PHASE 5: Train and tune the three models (grouped cross-validation, F1)
  (the evaluation phase is added here once approved)
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logging: everything printed also goes to train_log.txt (traceability)
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(os.path.join(HERE, "train_log.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ===========================================================================
# LOAD THE OUTPUT OF PHASE 3 (created by clean_data.py)
# ===========================================================================
header("LOAD THE PHASE 3 OUTPUT")

FEATURES_PATH = os.path.join(HERE, "diabetic_features.csv")
if not os.path.exists(FEATURES_PATH):
    print("ERROR: diabetic_features.csv not found. Run clean_data.py first, in this same folder:")
    print(f"       {HERE}")
    sys.exit(1)

df = pd.read_csv(FEATURES_PATH)
print(f"Reading: {FEATURES_PATH}")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# Same feature lists as Phase 3 in clean_data.py (numeric names must match it)
numeric_features = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
feature_cols = [c for c in df.columns if c not in ("patient_nbr", "readmitted_binary")]
categorical_features = [c for c in feature_cols if c not in numeric_features]
assert len(feature_cols) == 40 and len(numeric_features) == 8, "unexpected feature count"
print(f"Features: {len(feature_cols)} = {len(numeric_features)} numeric + {len(categorical_features)} categorical")
print(f"Target 'readmitted_binary': {int(df['readmitted_binary'].sum()):,} positives "
      f"({df['readmitted_binary'].mean() * 100:.2f}%)")

# ===========================================================================
# PHASE 4: SPLIT INTO TRAIN AND TEST
# WHY: the test set must contain patients the model has never seen, and it
# must have the same share of early readmissions as the training set, so that
# the test results are a fair estimate of real-world performance.
# ===========================================================================
header("PHASE 4: SPLIT INTO TRAIN AND TEST (BY PATIENT, STRATIFIED)")

from sklearn.model_selection import StratifiedGroupKFold

X = df[["patient_nbr"] + feature_cols]
y = df["readmitted_binary"]
groups = df["patient_nbr"]

# ---- Step 1: why a plain random split is not enough ------------------------
print("\nStep 1: the two problems a plain random row split would have")
print(f"    - {len(df) - df['patient_nbr'].nunique():,} rows are repeat visits by patients who appear more than once.")
print("      A row split can put one visit in train and another in test, so the model would")
print("      be tested on a patient it already learned from (leakage, inflated scores).")
print("    - The positive class is only 11.16%, so a chance split can leave the two sets")
print("      with different positive rates.")
print("    Solution: split by PATIENT (all visits of a patient stay together) AND keep the")
print("    positive rate similar in both sets (stratified group split).")

# ---- Step 2: stratified group split ----------------------------------------
# StratifiedGroupKFold(n_splits=5) cuts the patients into 5 groups of about equal
# size and positive rate. We take one group (about 20%) as the test set and the
# other four (about 80%) as the training set.
print("\nStep 2: stratified group split (5 folds, first fold = test, about 20%)")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()
print("    random_state=42 (fixed, so the same split is produced every run)")

# ---- Step 3: checks --------------------------------------------------------
print("\nStep 3: checks")
overlap = set(train_df["patient_nbr"]) & set(test_df["patient_nbr"])
assert len(overlap) == 0, f"{len(overlap)} patients appear in both sets!"
print(f"    patients in BOTH train and test: {len(overlap)}  (must be 0)")
assert len(train_df) + len(test_df) == len(df), "rows lost in split"
print(f"    train + test rows = {len(train_df) + len(test_df):,} = all {len(df):,} rows (nothing lost or duplicated)")

print("\n    Class proportions:")
print(f"    {'set':<8}{'rows':>9}{'share':>8}{'patients':>10}{'positives':>11}{'positive rate':>15}")
for name, part in [("overall", df), ("train", train_df), ("test", test_df)]:
    pos = int(part["readmitted_binary"].sum())
    print(f"    {name:<8}{len(part):>9,}{len(part) / len(df) * 100:>7.1f}%"
          f"{part['patient_nbr'].nunique():>10,}{pos:>11,}{pos / len(part) * 100:>14.2f}%")
gap = abs(train_df["readmitted_binary"].mean() - test_df["readmitted_binary"].mean()) * 100
print(f"    difference in positive rate, train vs test: {gap:.2f} percentage points")

TRAIN_PATH = os.path.join(HERE, "train_split.csv")
TEST_PATH = os.path.join(HERE, "test_split.csv")
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"\nSaved: {TRAIN_PATH}")
print(f"Saved: {TEST_PATH}")
print("patient_nbr is kept in both files: Phase 5 cross-validation must also keep patients together.")


# ===========================================================================
# PHASE 5: TRAIN AND TUNE THE THREE MODELS
# WHY: each model has settings (hyperparameters) that change how it behaves.
# We pick them with cross-validation on the TRAINING set only, so the test set
# stays untouched and its result at the end is honest.
# ===========================================================================
header("PHASE 5: TRAIN AND TUNE THE THREE MODELS")

import time
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train = train_df[feature_cols]
y_train = train_df["readmitted_binary"]
groups_train = train_df["patient_nbr"]

# ---- Step 1: what goes in ---------------------------------------------------
print("\nStep 1: training data (the test set is NOT used anywhere in this phase)")
print(f"    rows: {len(X_train):,} | features: {X_train.shape[1]} | positives: {int(y_train.sum()):,} "
      f"({y_train.mean() * 100:.2f}%)")

# ---- Step 2: preprocessing --------------------------------------------------
print("\nStep 2: preprocessing (inside the model pipeline)")
print("    numeric features     -> StandardScaler: puts counts and durations on the same scale;")
print("                            needed by Logistic Regression, harmless for the trees")
print("    categorical features -> OneHotEncoder: turns each category into a 0/1 column, because")
print("                            models need numbers; unseen categories are ignored, not crashed on")
print("    It is fitted on the training part of each cross-validation round only, so nothing")
print("    from the held-out part leaks in. That is why it lives INSIDE the pipeline.")


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return Pipeline([("prep", prep), ("model", model)])


# ---- Step 3: class weighting ------------------------------------------------
print("\nStep 3: class weighting")
print("    class_weight='balanced' for all three models.")
print("    Reason: only 11.16% of rows are positive. Without weights a model can score well by")
print("    almost always predicting 'not readmitted' and miss the patients we care about.")
print("    The weights make a missed early readmission cost about 8 times more than a false alarm.")

# ---- Step 4: cross-validation ----------------------------------------------
print("\nStep 4: cross-validation setup")
print("    5-fold, StratifiedGroupKFold: same idea as the Phase 4 split. Each patient's visits stay in")
print("    one fold, and each fold keeps about 11% positives.")
print("    Reason: one validation split can be lucky or unlucky; averaging over 5 gives a steadier")
print("    score for choosing settings.")
print("    Score used to choose the best settings: F1 (your decision). Recall and Precision are")
print("    also recorded so the trade-off is visible.")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"f1": "f1", "recall": "recall", "precision": "precision"}

# ---- Step 5: the settings tried and why ------------------------------------
models = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        {"model__C": [0.01, 0.1, 1.0]},
        "C = regularisation strength (smaller = simpler model). Three values a factor of 10 apart "
        "show whether a simpler or a freer model works better.",
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [20, 50, 100]},
        "max_depth limits how many questions the tree can ask (deep trees memorise the training data); "
        "min_samples_leaf stops it making rules for tiny groups. Shallow depths also keep the tree "
        "explainable to doctors, which is the reason it is on your slides.",
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=1),
        {"model__max_depth": [8, 12], "model__min_samples_leaf": [20, 50]},
        "100 trees is a standard size (more adds time, little gain). max_depth and min_samples_leaf "
        "control overfitting in the same way as for the single tree.",
    ),
}

best_models = {}
all_cv_rows = []
for name, (clf, grid, reason) in models.items():
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print("\n" + "-" * 72)
    print(f"Step 5: tune {name}  ({n_combos} settings x 5 folds = {n_combos * 5} fits)")
    print(f"    why these settings: {reason}")
    t0 = time.time()
    search = GridSearchCV(make_pipeline(clf), grid, scoring=scoring, refit="f1",
                          cv=cv, n_jobs=-1, return_train_score=False)
    search.fit(X_train, y_train, groups=groups_train)
    secs = time.time() - t0

    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1")
    print(f"    finished in {secs:.0f} seconds. All settings, best first (mean over 5 folds):")
    print(f"    {'settings':<44}{'F1':>8}{'(+/-)':>8}{'Recall':>9}{'Precision':>11}")
    for _, r in res.iterrows():
        label = ", ".join(f"{k.replace('model__', '')}={v}" for k, v in r["params"].items())
        print(f"    {label:<44}{r['mean_test_f1']:>8.3f}{r['std_test_f1']:>8.3f}"
              f"{r['mean_test_recall']:>9.3f}{r['mean_test_precision']:>11.3f}")
        all_cv_rows.append({"model": name, "settings": label, "rank": int(r["rank_test_f1"]),
                            "mean_f1": r["mean_test_f1"], "std_f1": r["std_test_f1"],
                            "mean_recall": r["mean_test_recall"],
                            "mean_precision": r["mean_test_precision"]})
    best_label = ", ".join(f"{k.replace('model__', '')}={v}" for k, v in search.best_params_.items())
    print(f"    CHOSEN: {best_label}  (highest mean F1 = {search.best_score_:.3f})")
    print("    The chosen settings were then refitted on the whole training set.")
    best_models[name] = search.best_estimator_

pd.DataFrame(all_cv_rows).to_csv(os.path.join(HERE, "cv_results_phase5.csv"), index=False)
joblib.dump(best_models, os.path.join(HERE, "best_models.joblib"))
print("\nSaved: cv_results_phase5.csv (every setting tried) and best_models.joblib (the 3 tuned models)")
print("The test set has still not been touched. It is used once, in the evaluation phase.")

header("DONE: phases 4-5 complete. Log saved to train_log.txt")
sys.stdout = sys.__stdout__  # stop copying to the log before closing it
log_file.close()
