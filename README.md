# Titanic Survival Pipeline

A reproducible machine-learning pipeline that predicts passenger survival from the Titanic dataset using a scikit-learn preprocessing pipeline and a tuned random forest.

**Python · pandas · scikit-learn · Random Forest · GitHub Actions**

## Workflow

```text
Passenger records → missing-value handling → categorical encoding
                  → cross-validated model search → held-out evaluation
                  → saved pipeline → prediction CSV
```

- Median imputation for numerical features and most-frequent imputation for categorical features.
- One-hot encoding that tolerates previously unseen categories.
- Five-fold grid search over forest size, depth, and minimum split size.
- A stratified 20% validation split kept outside hyperparameter search.
- A single serialized pipeline containing both preprocessing and the classifier.

## Run locally

Use Python 3.11. The passenger CSVs are included in `data/`.

```sh
git clone https://github.com/MohZak-AI/titanic-survival-pipeline.git
cd titanic-survival-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python test.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Outputs

| Artifact | Contents |
| --- | --- |
| `models/titanic_pipeline.joblib` | Preprocessing and fitted classifier |
| `models/validation_metrics.json` | Held-out accuracy, precision, recall, F1, and search settings |
| `outputs/predictions.csv` | `PassengerId` and predicted `Survived` for every test passenger |

The competition test set does not include ground-truth survival labels. `test.py` generates predictions; reported evaluation comes from the held-out portion of the labeled training set. The competition's sample submission is not used as ground truth.

## Measured baseline

A local run with the pinned dependencies produced **81.0% validation accuracy** and **0.730 F1** on 179 held-out passengers (seed 42). This is one reproducible split, not a competition leaderboard score. Full settings and metrics are in [`docs/baseline.json`](docs/baseline.json).

## Code map

- [`preprocessing.py`](preprocessing.py): numerical and categorical transformations.
- [`train.py`](train.py): split, search, evaluation, and model serialization.
- [`test.py`](test.py): load the saved model and produce a submission CSV.
- [CI workflow](.github/workflows/ci.yml): train, predict, and check output IDs and classes.

The saved model is a reproducible baseline, not a deployed prediction service. Load only trusted joblib files.
