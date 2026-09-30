"""
EVALUATE_MODELS.PY - PHASE 6: test the tuned models once on the unseen test set
==============================================================================
Reads what train_and_tuning.py saved, so the (slow) tuning is never repeated.

HOW TO RUN
  1. Run clean_data.py, then train_and_tuning.py first
  2. Keep this file in the SAME folder, then: python evaluate_models.py
  Outputs (created next to this file):
       evaluation_results.csv            - test-set metrics and confusion-matrix counts
       threshold_analysis_train.csv      - threshold table (from the TRAINING data only)
       confusion_matrices.png            - one confusion matrix per model
       precision_recall_curve.png        - precision-recall curve per model (test set)
       threshold_analysis.png            - recall / precision / F1 against the threshold
       top_predictors.png                - the factors the final model relies on most
       feature_importance_random_forest.csv - importance of all 40 factors
       final_model_random_forest.joblib  - the final model (Random Forest) as one file
       eval_log.txt                      - everything printed, kept as proof for the report

SECTIONS
  Step 1: load models and data          Step 4: threshold analysis (training data)
  Step 2: metrics on the test set       Step 5: precision-recall curve (test set)
  Step 3: confusion matrices            Step 6: top predictors of the final model
                                        Step 7: the final model in use
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logging: everything printed also goes to eval_log.txt (traceability)
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


log_file = open(os.path.join(HERE, "eval_log.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


header("PHASE 6: EVALUATE THE TUNED MODELS ON THE TEST SET")

import joblib
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

try:
    import matplotlib
    matplotlib.use("Agg")  # save pictures to files; no window needed
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except ImportError:
    HAVE_PLOT = False
    print("NOTE: matplotlib is not installed, so the pictures will be skipped.")
    print("      Install it with:  python -m pip install matplotlib   (numbers are still produced)")

# ===========================================================================
# STEP 1: LOAD THE TUNED MODELS AND THE DATA
# ===========================================================================
print("\nStep 1: load the tuned models and the data")
needed = ["best_models.joblib", "train_split.csv", "test_split.csv"]
missing = [f for f in needed if not os.path.exists(os.path.join(HERE, f))]
if missing:
    print(f"ERROR: missing {missing}. Run clean_data.py and then train_and_tuning.py first,")
    print(f"       in this same folder: {HERE}")
    sys.exit(1)

models = joblib.load(os.path.join(HERE, "best_models.joblib"))
train_df = pd.read_csv(os.path.join(HERE, "train_split.csv"))
test_df = pd.read_csv(os.path.join(HERE, "test_split.csv"))

feature_cols = [c for c in train_df.columns if c not in ("patient_nbr", "readmitted_binary")]
X_train, y_train = train_df[feature_cols], train_df["readmitted_binary"]
groups_train = train_df["patient_nbr"]
X_test, y_test = test_df[feature_cols], test_df["readmitted_binary"].values

print(f"    models loaded: {list(models)}")
print(f"    test set: {len(X_test):,} rows, {int(y_test.sum()):,} early readmissions "
      f"({y_test.mean() * 100:.2f}%)")
print("    The test set has never been used for training or tuning. It is used once, here.")

# probabilities on the test set, from each tuned model
test_proba = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
test_pred = {name: m.predict(X_test) for name, m in models.items()}

# ===========================================================================
# STEP 2: METRICS ON THE TEST SET (as on the slides)
# ===========================================================================
header("STEP 2: METRICS ON THE TEST SET (default decision rule)")
print("Default rule: each model's own prediction (flag a patient when the model's probability")
print("of early readmission is above 0.5; class weighting already makes 0.5 sensitive to the")
print("minority class).\n")

rows = []
for name in models:
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred[name]).ravel()
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, test_pred[name]),
        "Precision": precision_score(y_test, test_pred[name], zero_division=0),
        "Recall": recall_score(y_test, test_pred[name]),
        "F1": f1_score(y_test, test_pred[name]),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    })
results = pd.DataFrame(rows)
print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
for _, r in results.iterrows():
    print(f"{r['Model']:<22}{r['Accuracy']:>10.3f}{r['Precision']:>11.3f}{r['Recall']:>9.3f}{r['F1']:>8.3f}")

base_acc = 1 - y_test.mean()
print(f"\nWhy Accuracy is NOT the headline metric: a 'model' that predicts 'not readmitted' for")
print(f"every patient scores Accuracy {base_acc:.3f} but Recall 0.000 (it finds no one).")
print(f"So the models' lower Accuracy is the price of actually catching early readmissions.")
print(f"Random guessing would give Precision of about {y_test.mean():.3f} (the share of positives).")
results.to_csv(os.path.join(HERE, "evaluation_results.csv"), index=False)
print("\nSaved: evaluation_results.csv")

# ===========================================================================
# STEP 3: CONFUSION MATRICES
# ===========================================================================
header("STEP 3: CONFUSION MATRICES (test set)")
print("Rows = what really happened, columns = what the model predicted.")
print("  TN = correctly said 'not readmitted early'   FP = false alarm (flagged, but was fine)")
print("  FN = MISSED an early readmission (costliest)  TP = correctly flagged an early readmission")
for _, r in results.iterrows():
    tn, fp, fn, tp = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
    print(f"\n{r['Model']}")
    print(f"    {'':<26}{'predicted: not readm.':>23}{'predicted: readm. <30':>23}")
    print(f"    {'actual: not readmitted':<26}{'TN = ' + format(tn, ','):>23}{'FP = ' + format(fp, ','):>23}")
    print(f"    {'actual: readmitted <30':<26}{'FN = ' + format(fn, ','):>23}{'TP = ' + format(tp, ','):>23}")
    print(f"    Caught {tp:,} of {tp + fn:,} early readmissions (Recall {tp / (tp + fn) * 100:.1f}%), "
          f"missed {fn:,}.")
    print(f"    Flagged {tp + fp:,} patients; {fp:,} of them were false alarms "
          f"(Precision {tp / (tp + fp) * 100:.1f}%).")

if HAVE_PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (_, r) in zip(axes, results.iterrows()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=int)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}% of row)", ha="center", va="center",
                        color="white" if pct[i, j] > 55 else "black", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not readmitted", "Readmitted <30"])
        ax.set_yticklabels(["Not readmitted", "Readmitted <30"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{r['Model']}\nRecall {r['Recall'] * 100:.1f}%  |  Precision {r['Precision'] * 100:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "confusion_matrices.png"), dpi=200)
    plt.close()
    print("\nSaved: confusion_matrices.png")

# ===========================================================================
# STEP 4: THRESHOLD ANALYSIS (uses the TRAINING data only)
# WHY: a model flags a patient when its probability passes a threshold. Moving the
# threshold trades false alarms against missed patients. To study this without
# touching the test set, we use out-of-fold predictions: each training patient is
# scored by a model that never saw them (same grouped 5-fold as Phase 5).
# ===========================================================================
header("STEP 4: THRESHOLD ANALYSIS (from the training data, not the test set)")
print("Getting out-of-fold predictions for the training set (this takes a minute or two)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
grid = np.round(np.arange(0.20, 0.801, 0.05), 2)
fine = np.round(np.arange(0.05, 0.951, 0.01), 2)
oof = {}
thr_rows = []
for name, m in models.items():
    oof[name] = cross_val_predict(clone(m), X_train, y_train, groups=groups_train, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    p = oof[name]
    best_f1, best_t = -1, None
    table = []
    for t in grid:
        pred = p > t   # 'above t', so 0.50 matches the models' default rule
        rec = recall_score(y_train, pred)
        prec = precision_score(y_train, pred, zero_division=0)
        f1 = f1_score(y_train, pred, zero_division=0)
        table.append((t, pred.mean(), rec, prec, f1))
        thr_rows.append({"model": name, "threshold": t, "flagged_share": pred.mean(),
                         "recall": rec, "precision": prec, "f1": f1})
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\n{name}")
    print(f"    {'threshold':>9}{'patients flagged':>18}{'Recall':>9}{'Precision':>11}{'F1':>8}")
    for t, flagged, rec, prec, f1 in table:
        note = "  <- default (0.50)" if abs(t - 0.5) < 1e-9 else ""
        note += "  <- best F1" if t == best_t else ""
        if flagged == 0:
            print(f"    {t:>9.2f}{flagged * 100:>17.1f}%{'-':>9}{'-':>11}{'-':>8}{note}")
        else:
            print(f"    {t:>9.2f}{flagged * 100:>17.1f}%{rec:>9.3f}{prec:>11.3f}{f1:>8.3f}{note}")

pd.DataFrame(thr_rows).to_csv(os.path.join(HERE, "threshold_analysis_train.csv"), index=False)
print("\nHow to read this: lowering the threshold flags more patients, so Recall goes up and")
print("Precision goes down (more false alarms). Raising it does the opposite. Which point to use")
print("is a decision about how many false alarms the hospital can afford. No threshold has been")
print("changed on the test set: the results in Steps 2 and 3 use the default rule only.")
print("\nSaved: threshold_analysis_train.csv")

if HAVE_PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, name in zip(axes, models):
        p = oof[name]
        rec = [recall_score(y_train, p > t) for t in fine]
        prec = [precision_score(y_train, p > t, zero_division=0) for t in fine]
        f1 = [f1_score(y_train, p > t, zero_division=0) for t in fine]
        ax.plot(fine, rec, label="Recall"); ax.plot(fine, prec, label="Precision"); ax.plot(fine, f1, label="F1")
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="default 0.50")
        ax.set_title(name); ax.set_xlabel("Threshold"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score (training data, out-of-fold)")
    axes[0].legend(loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "threshold_analysis.png"), dpi=200)
    plt.close()
    print("Saved: threshold_analysis.png")

# ===========================================================================
# STEP 5: PRECISION-RECALL CURVE (test set)
# WHY: shows, for every possible threshold, how much Precision must be given up to
# gain Recall. It is drawn from the test set but nothing is chosen from it. No
# PR-AUC number is reported (your decision): the curve is a picture only.
# ===========================================================================
header("STEP 5: PRECISION-RECALL CURVE (test set)")
if HAVE_PLOT:
    plt.figure(figsize=(8, 6))
    for name in models:
        prec_c, rec_c, _ = precision_recall_curve(y_test, test_proba[name])
        line, = plt.plot(rec_c, prec_c, label=name, linewidth=2)
        r = results.loc[results["Model"] == name].iloc[0]
        plt.scatter([r["Recall"]], [r["Precision"]], color=line.get_color(), s=70, zorder=5,
                    edgecolor="black")
    plt.axhline(y_test.mean(), color="black", linestyle="--", linewidth=1,
                label=f"Random guessing ({y_test.mean():.3f})")
    plt.xlabel("Recall (share of early readmissions caught)")
    plt.ylabel("Precision (share of flagged patients who really were readmitted)")
    plt.title("Precision-recall curve, test set\n(dots = each model's default decision rule)")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "precision_recall_curve.png"), dpi=200)
    plt.close()
    print("Saved: precision_recall_curve.png")
else:
    print("Skipped (matplotlib not installed).")
print("Reading it: a curve close to the dashed line means the model is only slightly better than")
print("random guessing at that Recall level; the further above it, the better.")


# ===========================================================================
# STEP 6: WHICH FACTORS DOES THE FINAL MODEL RELY ON? (permutation importance)
# The group chose the Random Forest as the final model.
# METHOD: shuffle ONE factor at a time in the test set, so it carries no real
# information, and measure how much the model's F1 falls. A big fall means the
# model relies on that factor. Nothing is chosen from this: it only explains.
# ===========================================================================
header("STEP 6: TOP PREDICTORS OF THE FINAL MODEL (Random Forest)")
from sklearn.inspection import permutation_importance

rf_model = models["Random Forest"]
print("Final model chosen by the group: Random Forest (tuned in Phase 5).")
print("Method: shuffle one factor at a time in the test set and measure the fall in F1.")
print("Each factor is shuffled 5 times; the average fall and its spread are shown.\n")
pi = permutation_importance(rf_model, X_test, y_test, scoring="f1", n_repeats=5,
                            random_state=42, n_jobs=-1)
imp = pd.DataFrame({"factor": feature_cols, "f1_drop": pi.importances_mean,
                    "spread": pi.importances_std}).sort_values("f1_drop", ascending=False)
imp.to_csv(os.path.join(HERE, "feature_importance_random_forest.csv"), index=False)

print(f"{'rank':>4}  {'factor':<26}{'fall in F1':>11}{'(+/-)':>9}")
for rank, (_, r) in enumerate(imp.head(15).iterrows(), start=1):
    print(f"{rank:>4}  {r['factor']:<26}{r['f1_drop']:>11.4f}{r['spread']:>9.4f}")
n_zero = int((imp["f1_drop"] <= imp["spread"]).sum())
print(f"\n{n_zero} of the 40 factors have a fall in F1 no bigger than their own spread, so the")
print("model gets little or nothing from them.")
print("Caution: this shows what the MODEL relies on, not what CAUSES readmission. Related factors")
print("(for example the medication columns) share credit, so each one can look smaller than the")
print("group really is.")
print("\nSaved: feature_importance_random_forest.csv (all 40 factors)")

if HAVE_PLOT:
    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["factor"], top["f1_drop"], xerr=top["spread"], color="#2b7bba", capsize=3)
    plt.xlabel("Fall in F1 when the factor is shuffled (bigger = model relies on it more)")
    plt.title("Top 15 predictors of early readmission\nFinal model: Random Forest (test set)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "top_predictors.png"), dpi=200)
    plt.close()
    print("Saved: top_predictors.png")

# ===========================================================================
# STEP 7: THE FINAL MODEL IN USE
# ===========================================================================
header("STEP 7: THE FINAL MODEL IN USE")
final_path = os.path.join(HERE, "final_model_random_forest.joblib")
joblib.dump(rf_model, final_path)
print(f"Saved the final model as one file: {final_path}")
print("It is a pipeline: raw patient record (40 fields) -> scale numbers and one-hot encode")
print("categories -> Random Forest -> probability of early readmission -> flag if above 0.5.\n")
print("Example: 5 patients picked at random from the test set (fixed seed 7, not hand-picked)")
sample = X_test.sample(5, random_state=7)
p5 = rf_model.predict_proba(sample)[:, 1]
print(f"    {'age':<10}{'main diagnosis':<17}{'prior inpatient':>16}{'risk':>8}{'flagged':>9}{'really readmitted <30':>23}")
for (_, row), prob, actual in zip(sample.iterrows(), p5, y_test[sample.index]):
    print(f"    {row['age']:<10}{row['diag_1_category']:<17}{int(row['number_inpatient']):>16}"
          f"{prob:>8.2f}{'yes' if prob > 0.5 else 'no':>9}{'yes' if actual == 1 else 'no':>23}")
print("\nTo use it later:  model = joblib.load('final_model_random_forest.joblib')")
print("                  model.predict_proba(new_patients)[:, 1]   (new_patients = same 40 columns)")
print("It is a screening aid, not a diagnosis: about 4 in 5 flagged patients would not be")
print("readmitted early.")

header("DONE: phase 6 complete. Log saved to eval_log.txt")
sys.stdout = sys.__stdout__  # stop copying to the log before closing it
log_file.close()
