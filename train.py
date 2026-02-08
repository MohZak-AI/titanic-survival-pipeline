import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

# bring in the helper tool for cleaning data
from src.data.preprocessing import build_preprocessor

def train_model(data_dir: str = "data", models_dir: str = "models") -> None:
    # find where the training file is
    train_csv = os.path.join(data_dir, "train.csv")
    print(f"[TRAIN] Loading {train_csv} ...")
    
    # read the data and keep the answer column separate
    df = pd.read_csv(train_csv)
    y = df["Survived"].copy()
    
    # remove columns that don't help with the math
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

    # pick which columns are numbers and which are words
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]
    
    # set up the data cleaner and the model
    pre = build_preprocessor(numeric_features, categorical_features)
    model = RandomForestClassifier(random_state=42)
    
    # put everything into one single package
    pipe = Pipeline([("preprocess", pre), ("model", model)])

    # try different settings to see which works best
    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }

    # run the test to find the best settings
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
    grid.fit(X, y)
    
    # show the results
    print(f"[TRAIN] Best params: {grid.best_params_}")
    print(f"[TRAIN] Best CV accuracy: {grid.best_score_:.4f}")

    # make a folder for the model and save it
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "titanic_pipeline.joblib")
    joblib.dump(grid.best_estimator_, out_path)
    print(f"[TRAIN] Saved pipeline to {out_path}")
