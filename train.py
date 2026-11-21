"""Train and validate a Titanic survival classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def train_model(data_dir: str = "data", models_dir: str = "models") -> None:
    df = pd.read_csv(Path(data_dir) / "train.csv")
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked"]
    X = df[numeric + categorical]
    y = df["Survived"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", RandomForestClassifier(random_state=42)),
    ])
    search = GridSearchCV(
        pipeline,
        {"model__n_estimators": [50, 100],
         "model__max_depth": [None, 5, 10],
         "model__min_samples_split": [2, 5]},
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_valid)
    metrics = {
        "validation_rows": len(y_valid),
        "random_state": 42,
        "best_cv_accuracy": search.best_score_,
        "best_parameters": search.best_params_,
        "accuracy": accuracy_score(y_valid, predictions),
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "f1": f1_score(y_valid, predictions, zero_division=0),
    }
    output = Path(models_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output / "titanic_pipeline.joblib")
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and validation metrics in {output}")


if __name__ == "__main__":
    train_model()
