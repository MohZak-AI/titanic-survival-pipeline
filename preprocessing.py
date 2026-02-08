from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# this function sets up how we clean the data
def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # fill in any missing numbers with the middle value
    numeric = SimpleImputer(strategy="median")
    
    # for columns with words, fill missing spots and turn them into numbers
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # combine both into one single tool
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])
