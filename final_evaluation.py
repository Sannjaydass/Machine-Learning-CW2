
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# =====================================================
# 6001CMD MACHINE LEARNING GROUP PROJECT
# STEP 8: FINAL MODEL EVALUATION
#
# Four models:
# 1. Logistic Regression
# 2. Random Forest
# 3. Gradient Boosting
# 4. Soft Voting Ensemble
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIGURE_DIR = BASE_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("6001CMD - FINAL MODEL EVALUATION")
print("=" * 65)

# -----------------------------------------------------
# 1. LOAD TRAINING AND TEST DATA
# -----------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).iloc[:, 0]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

assert X_train.columns.tolist() == X_test.columns.tolist()

# -----------------------------------------------------
# 2. LOAD TUNED MODEL CONFIGURATIONS
# -----------------------------------------------------

model_names = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting"
}

models = {}

for display_name, filename in model_names.items():

    model_path = MODEL_DIR / f"{filename}_tuned.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing tuned model: {model_path}"
        )

    # Load the tuned configuration.
    # Clone removes its previous fitted state.
    tuned_pipeline = joblib.load(model_path)

    models[display_name] = clone(tuned_pipeline)

print("\nAll three tuned configurations loaded.")

# -----------------------------------------------------
# 3. RETRAIN MODELS ON FULL TRAINING DATA
# -----------------------------------------------------

training_times = {}

for name, model in models.items():

    print("\nTraining final model:", name)

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_times[name] = time.perf_counter() - start

    filename = (
        name.lower().replace(" ", "_") + "_final.joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

    print(f"Training time: {training_times[name]:.2f}s")
    print("Saved:", filename)

# -----------------------------------------------------
# 4. BUILD FINAL ENSEMBLE
# -----------------------------------------------------

# Both constituent models are now fitted
# on the complete training partition.

rf_model = models["Random Forest"]
gb_model = models["Gradient Boosting"]

ensemble_bundle = {
    "method": "soft_voting",
    "models": {
        "random_forest": rf_model,
        "gradient_boosting": gb_model
    },
    "weights": [0.5, 0.5],
    "threshold": 0.5
}

joblib.dump(
    ensemble_bundle,
    MODEL_DIR / "soft_voting_ensemble_final.joblib"
)

print("\nFinal ensemble saved.")

# -----------------------------------------------------
# 5. GENERATE FINAL TEST PREDICTIONS
# -----------------------------------------------------

predictions = {}
probabilities = {}
prediction_times = {}

for name, model in models.items():

    start = time.perf_counter()

    probabilities[name] = model.predict_proba(
        X_test
    )[:, 1]

    predictions[name] = (
        probabilities[name] >= 0.5
    ).astype(int)

    prediction_times[name] = time.perf_counter() - start

# Equal-weight soft voting
start = time.perf_counter()

probabilities["Soft Voting Ensemble"] = (
    probabilities["Random Forest"]
    + probabilities["Gradient Boosting"]
) / 2

predictions["Soft Voting Ensemble"] = (
    probabilities["Soft Voting Ensemble"] >= 0.5
).astype(int)

prediction_times["Soft Voting Ensemble"] = (
    prediction_times["Random Forest"]
    + prediction_times["Gradient Boosting"]
    + time.perf_counter() - start
)

# The ensemble reuses its fitted constituent models.
# Its fitting cost is approximately the combined
# fitting cost of Random Forest and Gradient Boosting.

training_times["Soft Voting Ensemble"] = (
    training_times["Random Forest"]
    + training_times["Gradient Boosting"]
)

# -----------------------------------------------------
# 6. CALCULATE EVALUATION METRICS
# -----------------------------------------------------

results = []

for name in predictions:

    pred = predictions[name]
    prob = probabilities[name]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Average Precision": average_precision_score(
            y_test, prob
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, pred
        ),
        "Training Time (s)": training_times[name],
        "Prediction Time (s)": prediction_times[name]
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Average Precision",
    ascending=False
)

print("\n" + "=" * 65)
print("FINAL FOUR-MODEL TEST COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

results_df.to_csv(
    MODEL_DIR / "final_model_comparison.csv",
    index=False
)

# -----------------------------------------------------
# 7. SAVE CLASSIFICATION REPORTS
# -----------------------------------------------------

for name, pred in predictions.items():

    safe_name = name.lower().replace(" ", "_")

    report = classification_report(
        y_test,
        pred,
        target_names=["No", "Yes"],
        zero_division=0
    )

    print(f"\nClassification Report: {name}")
    print(report)

    with open(
        MODEL_DIR / f"{safe_name}_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

# -----------------------------------------------------
# 8. CONFUSION MATRICES
# -----------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, pred) in zip(
    axes.flat,
    predictions.items()
):

    cm = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_final_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------
# 9. ROC CURVES
# -----------------------------------------------------

plt.figure(figsize=(9, 7))

for name, prob in probabilities.items():

    fpr, tpr, _ = roc_curve(y_test, prob)

    auc = roc_auc_score(y_test, prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_final_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------
# 10. PRECISION-RECALL CURVES
# -----------------------------------------------------

plt.figure(figsize=(9, 7))

for name, prob in probabilities.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        prob
    )

    ap = average_precision_score(y_test, prob)

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP={ap:.3f})"
    )

plt.axhline(
    y=y_test.mean(),
    color="gray",
    linestyle="--",
    label="Class prevalence"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_final_precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------
# 11. MODEL COMPARISON CHART
# -----------------------------------------------------

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "Average Precision"
]

plot_df = results_df.set_index("Model")[
    metric_columns
]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("Final Machine Learning Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)

plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "10_final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------
# 12. FINAL COMPLETION
# -----------------------------------------------------

print("\n" + "=" * 65)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nSaved results:")
print("models/final_model_comparison.csv")

print("\nSaved figures:")
print("figures/07_final_confusion_matrices.png")
print("figures/08_final_roc_curves.png")
print("figures/09_final_precision_recall_curves.png")
print("figures/10_final_model_comparison.png")

print("\nFour final models saved in models folder.")
print("Final test evaluation completed.")
