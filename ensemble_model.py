
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# 6001CMD MACHINE LEARNING GROUP PROJECT
# STEP 7: ENSEMBLE LEARNING
#
# Models combined:
# 1. Tuned Random Forest
# 2. Tuned Gradient Boosting
#
# Ensemble method: Soft Voting
# =====================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("6001CMD - SOFT VOTING ENSEMBLE")
print("=" * 60)

# -----------------------------------------------------
# 1. LOAD TRAINING DATA
# -----------------------------------------------------

X = pd.read_csv(DATA_DIR / "X_train.csv")
y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]

# Recreate the same validation split used
# in base_models.py and hyperparameter_tuning.py.

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n1. DATASET INFORMATION")
print("-" * 45)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Final test data: Not used")

# -----------------------------------------------------
# 2. LOAD TUNED MODELS
# -----------------------------------------------------

print("\n2. LOADING TUNED MODELS")
print("-" * 45)

rf_path = MODEL_DIR / "random_forest_tuned.joblib"
gb_path = MODEL_DIR / "gradient_boosting_tuned.joblib"

if not rf_path.exists():
    raise FileNotFoundError(
        "Missing tuned Random Forest model. "
        "Run hyperparameter_tuning.py first."
    )

if not gb_path.exists():
    raise FileNotFoundError(
        "Missing tuned Gradient Boosting model. "
        "Run hyperparameter_tuning.py first."
    )

random_forest = joblib.load(rf_path)
gradient_boosting = joblib.load(gb_path)

print("[OK] Tuned Random Forest loaded")
print("[OK] Tuned Gradient Boosting loaded")

# These fitted pipelines were trained on X_train
# during hyperparameter optimisation.
# We reuse them without refitting.

# -----------------------------------------------------
# 3. GENERATE PROBABILITIES
# -----------------------------------------------------

print("\n3. GENERATING MODEL PREDICTIONS")
print("-" * 45)

start_time = time.perf_counter()

rf_probabilities = random_forest.predict_proba(
    X_val
)[:, 1]

gb_probabilities = gradient_boosting.predict_proba(
    X_val
)[:, 1]

# Equal-weight soft voting
ensemble_probabilities = (
    rf_probabilities + gb_probabilities
) / 2

# Default decision threshold: 0.50
ensemble_predictions = (
    ensemble_probabilities >= 0.50
).astype(int)

prediction_time = time.perf_counter() - start_time

print("Random Forest predictions completed")
print("Gradient Boosting predictions completed")
print("Soft voting combination completed")

# -----------------------------------------------------
# 4. DEFINE EVALUATION FUNCTION
# -----------------------------------------------------

def evaluate_model(name, predictions, probabilities):

    return {
        "Model": name,
        "Accuracy": accuracy_score(
            y_val, predictions
        ),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_val, probabilities
        ),
        "Average Precision": average_precision_score(
            y_val, probabilities
        )
    }

# -----------------------------------------------------
# 5. EVALUATE INDIVIDUAL AND ENSEMBLE MODELS
# -----------------------------------------------------

print("\n4. MODEL EVALUATION")
print("-" * 45)

rf_predictions = random_forest.predict(X_val)
gb_predictions = gradient_boosting.predict(X_val)

results = [
    evaluate_model(
        "Random Forest",
        rf_predictions,
        rf_probabilities
    ),
    evaluate_model(
        "Gradient Boosting",
        gb_predictions,
        gb_probabilities
    ),
    evaluate_model(
        "Soft Voting Ensemble",
        ensemble_predictions,
        ensemble_probabilities
    )
]

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ENSEMBLE VALIDATION COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# -----------------------------------------------------
# 6. CONFUSION MATRIX
# -----------------------------------------------------

print("\n5. ENSEMBLE CONFUSION MATRIX")
print("-" * 45)

cm = confusion_matrix(
    y_val,
    ensemble_predictions,
    labels=[0, 1]
)

print(cm)

print("\nClassification report:")
print(
    classification_report(
        y_val,
        ensemble_predictions,
        target_names=["No", "Yes"],
        zero_division=0
    )
)

# -----------------------------------------------------
# 7. SAVE ENSEMBLE MODEL
# -----------------------------------------------------

print("\n6. SAVING ENSEMBLE")
print("-" * 45)

# Store both fitted pipelines and voting weights.
# This bundle can be loaded for future predictions.

ensemble_bundle = {
    "method": "soft_voting",
    "models": {
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting
    },
    "weights": [0.5, 0.5],
    "threshold": 0.5,
    "random_state": RANDOM_STATE
}

ensemble_path = (
    MODEL_DIR / "soft_voting_ensemble.joblib"
)

joblib.dump(
    ensemble_bundle,
    ensemble_path
)

print("Saved:", ensemble_path.name)

# -----------------------------------------------------
# 8. SAVE RESULTS
# -----------------------------------------------------

results_df.to_csv(
    MODEL_DIR / "ensemble_validation_results.csv",
    index=False
)

# Save validation probabilities for analysis.
predictions_df = pd.DataFrame({
    "Actual": y_val.to_numpy(),
    "RF_Probability": rf_probabilities,
    "GB_Probability": gb_probabilities,
    "Ensemble_Probability": ensemble_probabilities,
    "Ensemble_Prediction": ensemble_predictions
})

predictions_df.to_csv(
    MODEL_DIR / "ensemble_validation_predictions.csv",
    index=False
)

print("Saved: ensemble_validation_results.csv")
print("Saved: ensemble_validation_predictions.csv")

print(
    f"\nEnsemble prediction time: "
    f"{prediction_time:.4f} seconds"
)

# -----------------------------------------------------
# 9. FINAL VERIFICATION
# -----------------------------------------------------

print("\n7. FILE VERIFICATION")
print("-" * 45)

required_files = [
    "soft_voting_ensemble.joblib",
    "ensemble_validation_results.csv",
    "ensemble_validation_predictions.csv"
]

for filename in required_files:
    path = MODEL_DIR / filename

    if path.exists():
        print("[OK]", filename)
    else:
        print("[MISSING]", filename)

print("\n" + "=" * 60)
print("ENSEMBLE MODEL COMPLETED SUCCESSFULLY!")
print("Two tuned models combined using soft voting.")
print("Final test dataset remains untouched.")
print("=" * 60)
