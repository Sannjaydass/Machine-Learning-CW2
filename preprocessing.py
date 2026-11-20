
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# 6001CMD MACHINE LEARNING GROUP PROJECT
# Bank Term Deposit Subscription Prediction
#
# STEP 4: DATA PREPROCESSING
# =====================================================

# -----------------------------------------------------
# 1. SET UP PROJECT PATHS
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "bank-additional-full.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("6001CMD MACHINE LEARNING GROUP PROJECT")
print("BANK MARKETING - DATA PREPROCESSING")
print("=" * 60)


# -----------------------------------------------------
# 2. LOAD THE ORIGINAL DATASET
# -----------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

print("\n1. ORIGINAL DATASET INFORMATION")
print("-" * 45)

print("Original dataset shape:", df.shape)
print("Total records:", len(df))
print("Total columns:", len(df.columns))


# -----------------------------------------------------
# 3. CHECK DATA QUALITY
# -----------------------------------------------------

print("\n2. DATA QUALITY CHECK")
print("-" * 45)

missing_values = df.isnull().sum().sum()
unknown_values = (df == "unknown").sum().sum()
original_duplicates = df.duplicated().sum()

print("Total missing values:", missing_values)
print("Total unknown entries:", unknown_values)
print("Exact duplicate rows:", original_duplicates)

# Unknown values are retained as separate categories.
# They may contain useful information for prediction.

print("\nUnknown categories will be retained.")

# Do not remove duplicates after dropping duration.
# Different calls can otherwise appear identical.


# -----------------------------------------------------
# 4. ENCODE THE TARGET VARIABLE
# -----------------------------------------------------

print("\n3. TARGET VARIABLE ENCODING")
print("-" * 45)

# Original target:
# no  = Customer did not subscribe
# yes = Customer subscribed

y = df["y"].map({
    "no": 0,
    "yes": 1
})

if y.isnull().any():
    raise ValueError("Unexpected target values found.")

print("Target encoding:")
print("no  = 0")
print("yes = 1")

print("\nOriginal target distribution:")
print(y.value_counts())

print("\nOriginal target percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# -----------------------------------------------------
# 5. REMOVE TARGET AND LEAKAGE FEATURE
# -----------------------------------------------------

print("\n4. FEATURE SELECTION")
print("-" * 45)

# Duration is only known after the phone call.
# Excluding it makes the model suitable for
# prediction before the call.

X = df.drop(columns=["y", "duration"]).copy()

print("Removed target column: y")
print("Removed leakage feature: duration")

print("\nRemaining original features:")
print(X.columns.tolist())


# -----------------------------------------------------
# 6. HANDLE PDAYS FEATURE
# -----------------------------------------------------

print("\n5. PDAYS FEATURE ENGINEERING")
print("-" * 45)

# pdays = 999 means the customer was not
# previously contacted.

# Create a new binary indicator.
X["previously_contacted"] = (
    X["pdays"] != 999
).astype(int)

# Replace sentinel value with NaN.
# Missing values will be imputed later
# using training-only preprocessing pipelines.

X["pdays"] = X["pdays"].replace(999, np.nan)

print("Created feature: previously_contacted")
print("999 in pdays replaced with NaN")

print("\nPreviously contacted distribution:")
print(X["previously_contacted"].value_counts())


# -----------------------------------------------------
# 7. STRATIFIED TRAIN-TEST SPLIT
# -----------------------------------------------------

print("\n6. TRAIN-TEST SPLIT")
print("-" * 45)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Split method: Stratified")
print("Training percentage: 80%")
print("Testing percentage: 20%")
print("Random state: 42")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------------------------
# 8. RETAIN ALL TRAINING OBSERVATIONS
# -----------------------------------------------------

print("\n7. DUPLICATE HANDLING")
print("-" * 45)

# The original dataset contains a small number
# of exact duplicates.
#
# However, removing duration can make additional
# distinct records appear identical.
#
# To avoid discarding potentially valid customer
# observations, all records are retained.

print("Original exact duplicates:", original_duplicates)

print("Duplicate removal: Skipped")

print("Training records retained:", len(X_train))
print("Testing records retained:", len(X_test))


# -----------------------------------------------------
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# -----------------------------------------------------

print("\n8. FEATURE TYPE IDENTIFICATION")
print("-" * 45)

numerical_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of numerical features:",
      len(numerical_features))

print("Number of categorical features:",
      len(categorical_features))

# One-hot encoding, imputation and scaling
# will be fitted on training data only
# during model development.


# -----------------------------------------------------
# 10. CHECK CLASS DISTRIBUTION
# -----------------------------------------------------

print("\n9. TRAINING TARGET DISTRIBUTION")
print("-" * 45)

print(y_train.value_counts())

print("\nTraining percentages:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)

print("\n10. TESTING TARGET DISTRIBUTION")
print("-" * 45)

print(y_test.value_counts())

print("\nTesting percentages:")
print(
    (y_test.value_counts(normalize=True) * 100)
    .round(2)
)


# -----------------------------------------------------
# 11. SAVE TRAINING AND TESTING DATA
# -----------------------------------------------------

print("\n11. SAVING PROCESSED DATA")
print("-" * 45)

X_train.to_csv(
    DATA_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    DATA_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    DATA_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    DATA_DIR / "y_test.csv",
    index=False
)

print("Saved: X_train.csv")
print("Saved: X_test.csv")
print("Saved: y_train.csv")
print("Saved: y_test.csv")


# -----------------------------------------------------
# 12. FINAL VERIFICATION
# -----------------------------------------------------

print("\n12. FINAL DATASET VERIFICATION")
print("-" * 45)

print("Original dataset:", df.shape)

print("Final training shape:", X_train.shape)
print("Final testing shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("Total records after splitting:",
      len(X_train) + len(X_test))

assert len(X_train) + len(X_test) == len(df)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

assert not set(X_train.index).intersection(
    set(X_test.index)
), "Train and test indices overlap!"

print("\nAll verification checks passed.")

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("All four CSV files saved in the data folder.")
print("Dataset ready for machine learning models.")
print("=" * 60)
