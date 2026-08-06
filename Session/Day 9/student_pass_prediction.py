"""
END-TO-END ML PROJECT (MANUAL VERSION — NO Pipeline / ColumnTransformer)
Goal: Predict whether a student PASSES (1) or FAILS (0)

Every step below is done "by hand" with pandas / numpy so you can see
exactly what a Pipeline/ColumnTransformer would normally hide from you.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

pd.set_option("display.width", 120)

# ============================================================
# STEP 0 — RAW DATA
# ============================================================
df = pd.DataFrame({
    "Study_Hours":[5,8,np.nan,2,9,7,6,4,10,3,8,6],
    "Attendance":[90,95,85,np.nan,98,92,88,75,99,80,94,87],
    "Gender":["Male","Female","Female","Male","Female",
              "Male","Male","Female","Female","Male","Female","Male"],
    "City":["Dhaka","Dhaka","Chittagong","Khulna","Dhaka",
            "Khulna","Rajshahi","Dhaka","Rajshahi",
            "Chittagong","Dhaka","Khulna"],
    "Family_Income":[50000,70000,65000,40000,np.nan,
                     80000,55000,48000,90000,45000,72000,60000],
    "Previous_GPA":[3.2,3.8,3.5,2.7,3.9,3.4,3.1,np.nan,4.0,2.9,3.7,3.3],
    "Pass":[1,1,1,0,1,1,1,0,1,0,1,1]
})

print("="*70)
print("STEP 1 — EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)
print("\nShape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nNumeric summary:\n", df.describe())
print("\nTarget balance:\n", df["Pass"].value_counts())

# ============================================================
# STEP 2 — TRAIN / TEST SPLIT (BEFORE any preprocessing!)
# ============================================================
# CRITICAL RULE: split first, then learn imputation values / scaling
# stats / encoding categories ONLY from the training set. This avoids
# "data leakage" from test set into training decisions.
print("\n" + "="*70)
print("STEP 2 — TRAIN/TEST SPLIT (done BEFORE preprocessing to avoid leakage)")
print("="*70)

X = df.drop(columns=["Pass"])
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
# work on copies so we never touch the originals by accident
X_train = X_train.copy()
X_test = X_test.copy()

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# ============================================================
# STEP 3 — MISSING VALUE IMPUTATION (manual, fit on TRAIN only)
# ============================================================
print("\n" + "="*70)
print("STEP 3 — MISSING VALUE IMPUTATION")
print("="*70)

numeric_cols = ["Study_Hours", "Attendance", "Family_Income", "Previous_GPA"]

# "Fit": learn the median of each numeric column FROM TRAINING DATA ONLY
impute_values = {col: X_train[col].median() for col in numeric_cols}
print("Median values learned from TRAIN set (used to fill NaNs):")
for k, v in impute_values.items():
    print(f"  {k}: {v}")

# "Transform": apply the SAME learned values to both train and test
for col in numeric_cols:
    X_train[col] = X_train[col].fillna(impute_values[col])
    X_test[col] = X_test[col].fillna(impute_values[col])

print("\nMissing values after imputation (train):\n", X_train[numeric_cols].isnull().sum())
print("Missing values after imputation (test):\n", X_test[numeric_cols].isnull().sum())

# ============================================================
# STEP 4 — ENCODING CATEGORICAL VARIABLES (manual)
# ============================================================
print("\n" + "="*70)
print("STEP 4 — CATEGORICAL ENCODING")
print("="*70)

# --- Gender: binary column -> simple manual mapping (Label Encoding) ---
gender_map = {"Male": 0, "Female": 1}
X_train["Gender"] = X_train["Gender"].map(gender_map)
X_test["Gender"] = X_test["Gender"].map(gender_map)
print("Gender encoded with mapping:", gender_map)

# --- City: nominal, >2 categories -> manual One-Hot Encoding ---
# "Fit": learn the set of known categories from TRAIN only
city_categories = sorted(X_train["City"].unique())
print("City categories learned from TRAIN:", city_categories)

def one_hot_encode(df_in, column, categories):
    """Manually one-hot encode `column` using a FIXED category list
    (learned from training data), so train/test always end up with
    the exact same set of dummy columns in the same order."""
    out = df_in.copy()
    for cat in categories:
        out[f"{column}_{cat}"] = (out[column] == cat).astype(int)
    out = out.drop(columns=[column])
    return out

X_train = one_hot_encode(X_train, "City", city_categories)
X_test = one_hot_encode(X_test, "City", city_categories)

# Guard: if test set had an unseen category, one_hot_encode would just
# produce all-zero dummy columns for that row -> which is correct behavior.

print("\nColumns after encoding:\n", X_train.columns.tolist())

# ============================================================
# STEP 5 — FEATURE SCALING (manual standardization, fit on TRAIN only)
# ============================================================
print("\n" + "="*70)
print("STEP 5 — FEATURE SCALING (Standardization: z = (x - mean) / std)")
print("="*70)

scale_cols = ["Study_Hours", "Attendance", "Family_Income", "Previous_GPA"]

# "Fit": learn mean & std from TRAIN only
scale_params = {
    col: {"mean": X_train[col].mean(), "std": X_train[col].std()}
    for col in scale_cols
}
print("Mean/Std learned from TRAIN set:")
for k, v in scale_params.items():
    print(f"  {k}: mean={v['mean']:.2f}, std={v['std']:.2f}")

# "Transform": apply to both train and test using TRAIN's mean/std
for col in scale_cols:
    mean, std = scale_params[col]["mean"], scale_params[col]["std"]
    X_train[col] = (X_train[col] - mean) / std
    X_test[col] = (X_test[col] - mean) / std

print("\nTrain features after scaling (head):\n", X_train.head())

# ============================================================
# STEP 6 — MODEL TRAINING
# ============================================================
print("\n" + "="*70)
print("STEP 6 — MODEL TRAINING (Logistic Regression)")
print("="*70)

# Make sure train/test have identical column order
X_test = X_test[X_train.columns]

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model trained.")
print("Feature order used:", X_train.columns.tolist())
print("Learned coefficients:", np.round(model.coef_[0], 3))
print("Intercept:", round(model.intercept_[0], 3))

# ============================================================
# STEP 7 — EVALUATION
# ============================================================
print("\n" + "="*70)
print("STEP 7 — EVALUATION ON TEST SET")
print("="*70)

y_pred = model.predict(X_test)

print("Predictions: ", y_pred.tolist())
print("Actual:      ", y_test.tolist())

print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1-score :", round(f1_score(y_test, y_pred, zero_division=0), 3))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# STEP 8 — PREDICT ON A NEW / UNSEEN STUDENT (full manual pipeline)
# ============================================================
print("\n" + "="*70)
print("STEP 8 — INFERENCE ON A NEW, UNSEEN STUDENT")
print("="*70)

new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [np.nan],       # missing on purpose, to test imputation
    "Gender": ["Female"],
    "City": ["Sylhet"],           # unseen category on purpose
    "Family_Income": [58000],
    "Previous_GPA": [3.4],
})

# 1) impute using the SAME learned train medians
for col in numeric_cols:
    new_student[col] = new_student[col].fillna(impute_values[col])

# 2) encode using the SAME learned mappings
new_student["Gender"] = new_student["Gender"].map(gender_map)
new_student = one_hot_encode(new_student, "City", city_categories)

# 3) scale using the SAME learned train mean/std
for col in scale_cols:
    mean, std = scale_params[col]["mean"], scale_params[col]["std"]
    new_student[col] = (new_student[col] - mean) / std

# 4) align columns exactly like training data
new_student = new_student.reindex(columns=X_train.columns, fill_value=0)

pred = model.predict(new_student)[0]
prob = model.predict_proba(new_student)[0][1]

print("Processed new student row:\n", new_student)
print(f"\nPrediction: {'PASS' if pred==1 else 'FAIL'} (probability of passing = {prob:.2f})")
