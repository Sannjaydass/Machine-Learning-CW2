
import pandas as pd
import time
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================================
# 6001CMD MACHINE LEARNING GROUP PROJECT
# BANK TERM DEPOSIT SUBSCRIPTION PREDICTION
#
# STEP 5: THREE BASE MACHINE LEARNING MODELS
# =====================================================

# -----------------------------------------------------
# 1. SET UP PROJECT PATHS
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("6001CMD - THREE BASE MODELS")
print("=" * 60)


# -----------------------------------------------------
# 2. LOAD PREPROCESSED TRAINING DATA
# -----------------------------------------------------

X = pd.read_csv(DATA_DIR / "X_train.csv")
y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]

print("\n1. DATASET INFORMATION")
print("-" * 45)

print("Total training dataset records:", len(X))
print("Total features:", X.shape[1])


# -----------------------------------------------------
# 3. CREATE INTERNAL VALIDATION SPLIT
# -----------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n2. TRAINING AND VALIDATION SPLIT")
print("-" * 45)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print("\nFinal test dataset remains untouched.")


# -----------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# -----------------------------------------------------

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\n3. FEATURE INFORMATION")
print("-" * 45)

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# -----------------------------------------------------
# 5. CREATE PREPROCESSING PIPELINE
# -----------------------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer,
     categorical_features)
])


# -----------------------------------------------------
# 6. DEFINE THREE BASE MODELS
# -----------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42
    )
}

results = []


# -----------------------------------------------------
# 7. TRAIN, EVALUATE AND SAVE MODELS
# -----------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    start_time = time.perf_counter()

    # Train model
    pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    # Save trained model
    model_filename = (
        name.lower().replace(" ", "_") + "_base.joblib"
    )

    model_path = MODEL_DIR / model_filename

    joblib.dump(pipeline, model_path)

    print("Saved model:", model_filename)

    # Make predictions
    predictions = pipeline.predict(X_val)

    probabilities = pipeline.predict_proba(
        X_val
    )[:, 1]

    # Calculate performance metrics
    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    # Display results
    print(f"\nAccuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "Training Time (s)": training_time
    })


# -----------------------------------------------------
# 8. CREATE MODEL COMPARISON TABLE
# -----------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("BASE MODEL VALIDATION COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# -----------------------------------------------------
# 9. SAVE MODEL COMPARISON RESULTS
# -----------------------------------------------------

results_path = MODEL_DIR / "base_model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nComparison results saved:")
print(results_path)


# -----------------------------------------------------
# 10. VERIFY SAVED MODEL FILES
# -----------------------------------------------------

print("\n" + "=" * 60)
print("SAVED MACHINE LEARNING MODELS")
print("=" * 60)

expected_files = [
    "logistic_regression_base.joblib",
    "random_forest_base.joblib",
    "gradient_boosting_base.joblib"
]

for filename in expected_files:

    file_path = MODEL_DIR / filename

    if file_path.exists():
        print("[OK]", filename)
    else:
        print("[MISSING]", filename)


# -----------------------------------------------------
# 11. COMPLETION MESSAGE
# -----------------------------------------------------

print("\n" + "=" * 60)
print("ALL THREE BASE MODELS COMPLETED SUCCESSFULLY!")
print("Trained models saved in the models folder.")
print("Model comparison results saved successfully.")
print("=" * 60)
