
import time
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    roc_auc_score,
    average_precision_score
)

# =====================================================
# 6001CMD MACHINE LEARNING GROUP PROJECT
# STEP 6: HYPERPARAMETER OPTIMISATION
# =====================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("6001CMD - HYPERPARAMETER OPTIMISATION")
print("=" * 60)

# 1. LOAD TRAINING DATA
X = pd.read_csv(DATA_DIR / "X_train.csv")
y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]

# Same validation split as the base models
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Final test data: Not used")

# 2. IDENTIFY FEATURES
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# 3. PREPROCESSING PIPELINE
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
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

# 4. MODELS AND HYPERPARAMETER SEARCH SPACES
configurations = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "Gradient Boosting": {
        "model": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_iter": [100, 150, 200],
            "classifier__max_leaf_nodes": [15, 31],
            "classifier__l2_regularization": [0.0, 0.1, 1.0]
        }
    }
}

# 5. CROSS-VALIDATION
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

# 6. TUNE EACH MODEL
for name, config in configurations.items():

    print("\n" + "=" * 55)
    print("Optimising:", name)
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", config["model"])
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    start_time = time.perf_counter()

    search.fit(X_train, y_train)

    tuning_time = time.perf_counter() - start_time

    best_model = search.best_estimator_

    predictions = best_model.predict(X_val)
    probabilities = best_model.predict_proba(X_val)[:, 1]

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
        "Average Precision": average_precision_score(
            y_val, probabilities
        ),
        "Best CV Average Precision": search.best_score_,
        "Tuning Time (s)": tuning_time
    }

    results.append(scores)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest cross-validation AP:",
          round(search.best_score_, 4))

    print("\nValidation results:")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

    safe_name = name.lower().replace(" ", "_")

    # Save tuned model
    joblib.dump(
        best_model,
        MODEL_DIR / f"{safe_name}_tuned.joblib"
    )

    # Save best parameters
    with open(
        MODEL_DIR / f"{safe_name}_best_params.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(search.best_params_, file, indent=4)

    # Save all cross-validation results
    pd.DataFrame(search.cv_results_).to_csv(
        MODEL_DIR / f"{safe_name}_cv_results.csv",
        index=False
    )

    print("\nTuned model and search results saved.")

# 7. FINAL COMPARISON
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Average Precision",
    ascending=False
)

results_df.to_csv(
    MODEL_DIR / "tuning_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL HYPERPARAMETER TUNING COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("ALL THREE MODELS OPTIMISED SUCCESSFULLY!")
print("Results saved in the models folder.")
print("=" * 60)
