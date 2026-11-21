"""Generate competition predictions using the saved training pipeline."""

from pathlib import Path

import joblib
import pandas as pd


def predict_test(data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs") -> None:
    model_path = Path(models_dir) / "titanic_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No trained pipeline found. Run python train.py first.")
    pipeline = joblib.load(model_path)
    passengers = pd.read_csv(Path(data_dir) / "test.csv")
    predictions = pipeline.predict(passengers)
    output = Path(outputs_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({
        "PassengerId": passengers["PassengerId"],
        "Survived": predictions,
    })
    result.to_csv(output / "predictions.csv", index=False)
    print(f"Wrote {len(result)} predictions to {output / 'predictions.csv'}")
    print("Test labels are not supplied; see models/validation_metrics.json for held-out evaluation.")


if __name__ == "__main__":
    predict_test()
