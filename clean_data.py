"""
CLEAN_DATA.PY - PHASES 1 TO 3: inspect, clean, build target and features
=========================================================================
Follows the group presentation: binary target (<30 days = 1, else 0).

HOW TO RUN
  1. Put this file in the SAME folder as diabetic_data.csv
  2. python clean_data.py
  3. Then run train_and_tuning.py (it reads the file this script creates)
  Outputs (created next to this file):
       diabetic_cleaned.csv   - result of Phase 2
       diabetic_features.csv  - result of Phase 3 (target + 40 features, ready to split)

SECTIONS
  PHASE 1: Load and inspect the raw data
  PHASE 2: Clean the data
  PHASE 3: Build the target and the features
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def find_raw_csv():
    candidates = [
        os.path.join(HERE, "diabetic_data.csv"),
        os.path.join(HERE, "data", "diabetic_data.csv"),
        os.path.join(os.getcwd(), "diabetic_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    print("ERROR: diabetic_data.csv not found. Put it in the same folder as this script:")
    print(f"       {HERE}")
    sys.exit(1)


# ===========================================================================
# PHASE 1: LOAD AND INSPECT THE RAW DATA (nothing is changed here)
# WHY: we must know what we start with before deciding any cleaning step.
# ===========================================================================
header("PHASE 1: LOAD AND INSPECT THE RAW DATA")

RAW_PATH = find_raw_csv()
print(f"Reading: {RAW_PATH}")
raw = pd.read_csv(RAW_PATH)
print(f"Rows: {raw.shape[0]:,} | Columns: {raw.shape[1]}")

print("\n1a. Target column 'readmitted' (original 3 values)")
for k, v in raw["readmitted"].value_counts().items():
    print(f"    {k:>4}: {v:>7,} ({v / len(raw) * 100:5.2f}%)")
pos_raw = (raw["readmitted"] == "<30").sum()
print(f"    Binary view (<30 = 1, else 0): {pos_raw:,} positive vs {len(raw) - pos_raw:,} negative")
print(f"    Positive rate {pos_raw / len(raw) * 100:.2f}% -> predicting 'not readmitted' "
      f"for everyone would give {100 - pos_raw / len(raw) * 100:.2f}% accuracy")

print("\n1b. Missing values - two kinds")
print("    '?' placeholders:")
q = (raw == "?").sum()
for col, n in q[q > 0].sort_values(ascending=False).items():
    print(f"      {col:<20} {n:>7,} ({n / len(raw) * 100:5.2f}%)")
print("    true NaN:")
na = raw.isna().sum()
for col, n in na[na > 0].sort_values(ascending=False).items():
    print(f"      {col:<20} {n:>7,} ({n / len(raw) * 100:5.2f}%)")

print("\n1c. Columns with only one value (no information)")
print("    ", [c for c in raw.columns if raw[c].nunique() <= 1])

print("\n1d. Invalid categories")
print("    gender:", raw["gender"].value_counts().to_dict())

print("\n1e. Repeated patients (matters for the train/test split)")
uniq = raw["patient_nbr"].nunique()
print(f"    Unique patients: {uniq:,} | Repeat encounters: {len(raw) - uniq:,}")
print(f"    Exact duplicate rows: {raw.duplicated().sum()}")

print("\n1f. Diagnosis code cardinality (why they get grouped into categories)")
for c in ["diag_1", "diag_2", "diag_3"]:
    print(f"    {c}: {raw[c].nunique()} unique codes")


# ===========================================================================
# PHASE 2: CLEAN THE DATA
# WHY: remove or label everything that would confuse a model, and log each
# change so it can be explained.
# ===========================================================================
header("PHASE 2: CLEAN THE DATA")

df = raw.copy()  # the raw table stays untouched for comparison
print(f"START: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ---- Step 1: drop columns listed on slide 3 --------------------------------
# patient_nbr is KEPT for now: the train/test split (later phase) uses it so
# that all visits of one patient stay on the same side. It is NOT a feature.
drop_cols = {
    "encounter_id":      "ID only, not medical information",
    "weight":            "96.86% missing",
    "payer_code":        "39.56% missing",
    "medical_specialty": "49.08% missing",
    "examide":           "only one value in the whole column",
    "citoglipton":       "only one value in the whole column",
    "max_glu_serum":     "94.75% missing",
    "A1Cresult":         "83.28% missing",
}
print("\nStep 1: drop columns")
for c, why in drop_cols.items():
    print(f"    dropped {c:<18} reason: {why}")
df = df.drop(columns=list(drop_cols))
print(f"    -> now {df.shape[1]} columns (patient_nbr kept only as the split key)")

# ---- Step 2: remove invalid gender rows ------------------------------------
print("\nStep 2: remove invalid gender rows")
bad = df["gender"] == "Unknown/Invalid"
print(f"    rows with gender = 'Unknown/Invalid': {bad.sum()}")
df = df[~bad].copy()
print(f"    -> {len(df):,} rows remain")

# ---- Step 3: '?' in race -> 'Unknown' --------------------------------------
print("\nStep 3: race '?' -> 'Unknown' category")
n = (df["race"] == "?").sum()
df["race"] = df["race"].replace("?", "Unknown")
print(f"    replaced {n:,} values ({n / len(df) * 100:.2f}%) - kept as a category, no rows lost")

# ---- Step 4: '?' in diagnosis columns -> 'Missing' -------------------------
print("\nStep 4: diag_1/2/3 '?' -> 'Missing' category")
for c in ["diag_1", "diag_2", "diag_3"]:
    n = (df[c] == "?").sum()
    df[c] = df[c].replace("?", "Missing")
    print(f"    {c}: replaced {n:,} values ({n / len(df) * 100:.2f}%)")

# ---- Checks ----------------------------------------------------------------
print("\nChecks")
print(f"    remaining '?' values anywhere : {int((df == '?').sum().sum())}")
print(f"    remaining NaN values anywhere : {int(df.isna().sum().sum())}")
print(f"    exact duplicate rows          : {int(df.duplicated().sum())}")
print(f"    single-value columns          : {[c for c in df.columns if df[c].nunique() <= 1]}")
print(f"    final shape: {df.shape[0]:,} rows x {df.shape[1]} columns = "
      f"{df.shape[1] - 2} features + patient_nbr (split key) + readmitted (target)")
pos = (df["readmitted"] == "<30").sum()
print(f"    target check: <30 = {pos:,} ({pos / len(df) * 100:.2f}%)")

CLEAN_PATH = os.path.join(HERE, "diabetic_cleaned.csv")
df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved cleaned data: {CLEAN_PATH}")


# ===========================================================================
# PHASE 3: BUILD THE TARGET AND THE FEATURES
# WHY: models need a numeric 0/1 target and features in a usable form.
# Encoding/scaling is NOT done here: it must be fitted on the training data
# only (later phase), otherwise information from the test set leaks in.
# ===========================================================================
header("PHASE 3: BUILD THE TARGET AND THE FEATURES")

# ---- Step 1: binary target (slide 2) ---------------------------------------
print("\nStep 1: binary target 'readmitted_binary'")
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
print("    <30 -> 1 (readmitted within 30 days)")
print("    NO and >30 -> 0 (not readmitted, or readmitted after 30 days)")
vc = df["readmitted_binary"].value_counts().sort_index()
print(f"    class 0: {vc[0]:,} ({vc[0] / len(df) * 100:.2f}%) | class 1: {vc[1]:,} ({vc[1] / len(df) * 100:.2f}%)")
df = df.drop(columns=["readmitted"])
print("    original 'readmitted' column removed (it IS the answer; keeping it would leak the target)")


# ---- Step 2: group ICD-9 diagnosis codes into ~10 clinical categories ------
# WHY: each diag column has 700+ unique codes; most appear a handful of times,
# so a model cannot learn from them. Standard ICD-9 chapter ranges (as used in
# the published study of this dataset, Strack et al. 2014) give meaningful groups.
def icd9_category(code):
    if code == "Missing":
        return "Missing"
    if str(code).upper().startswith(("V", "E")):   # supplementary codes
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if int(v) == 250:
        return "Diabetes"
    if 390 <= v < 460 or int(v) == 785:
        return "Circulatory"
    if 460 <= v < 520 or int(v) == 786:
        return "Respiratory"
    if 520 <= v < 580 or int(v) == 787:
        return "Digestive"
    if 580 <= v < 630 or int(v) == 788:
        return "Genitourinary"
    if 710 <= v < 740:
        return "Musculoskeletal"
    if 800 <= v < 1000:
        return "Injury"
    if 140 <= v < 240:
        return "Neoplasms"
    return "Other"


print("\nStep 2: group diag_1/2/3 into clinical categories")
for c in ["diag_1", "diag_2", "diag_3"]:
    df[c + "_category"] = df[c].apply(icd9_category)
    print(f"    {c}: {df[c].nunique()} codes -> {df[c + '_category'].nunique()} categories")
print("    diag_1_category counts:")
for k, v in df["diag_1_category"].value_counts().items():
    print(f"      {k:<16} {v:>7,} ({v / len(df) * 100:5.2f}%)")
df = df.drop(columns=["diag_1", "diag_2", "diag_3"])
print("    raw diagnosis code columns removed (replaced by their categories)")


# ---- Step 3: define the feature list ---------------------------------------
print("\nStep 3: separate features from the target and the split key")
numeric_features = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
id_like = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
med_columns = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
categorical_features = (
    ["race", "gender", "age"] + id_like
    + ["diag_1_category", "diag_2_category", "diag_3_category"]
    + med_columns + ["change", "diabetesMed"]
)
feature_cols = numeric_features + categorical_features
print(f"    numeric features    : {len(numeric_features)} (counts and durations)")
print(f"    categorical features: {len(categorical_features)}")
print(f"        - age is kept as its 10 bands, the original column")
print(f"        - {len(id_like)} ID-like columns are treated as CATEGORIES: their numbers are labels,")
print(f"          not amounts (code 6 is not 'twice' code 3)")
print(f"        - 21 medication columns + change + diabetesMed")
print(f"    TOTAL features      : {len(feature_cols)}")

# sanity checks: nothing lost, nothing extra, no leakage columns
assert set(feature_cols) | {"patient_nbr", "readmitted_binary"} == set(df.columns), "column mismatch"
assert "readmitted" not in df.columns and "patient_nbr" not in feature_cols
print("    check: every column is accounted for; target and patient_nbr are NOT features")

df = df[["patient_nbr"] + feature_cols + ["readmitted_binary"]]

# ---- Information check (no change made): discharge disposition -------------
# Some discharge codes mean the patient died or went to hospice, so a
# readmission is impossible. This is NOT on the slides, so nothing is removed
# here; the numbers are printed so the group can decide.
print("\nInformation check (no change made): discharge codes for death / hospice")
end_of_life = [11, 13, 14, 19, 20, 21]
mask = df["discharge_disposition_id"].isin(end_of_life)
print(f"    rows with these discharge codes: {mask.sum():,} ({mask.mean() * 100:.2f}%)")
print(f"    early-readmission rate among them: {df.loc[mask, 'readmitted_binary'].mean() * 100:.2f}% "
      f"(overall: {df['readmitted_binary'].mean() * 100:.2f}%)")

print(f"\nFinal Phase 3 table: {df.shape[0]:,} rows x {df.shape[1]} columns "
      f"= patient_nbr + {len(feature_cols)} features + target")
FEATURES_PATH = os.path.join(HERE, "diabetic_features.csv")
df.to_csv(FEATURES_PATH, index=False)
print(f"Saved: {FEATURES_PATH}")

header("DONE: phases 1-3 complete. Next: run train_and_tuning.py")
