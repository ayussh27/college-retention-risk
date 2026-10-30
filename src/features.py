"""Preprocessing pipelines. Fitted inside cross-validation, never on held-out rows."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config as C


def load_cohort(include_extreme: bool = False) -> pd.DataFrame:
    df = pd.read_csv(C.COHORT_CSV)
    if not include_extreme:
        df = df[df["split"] != "excluded_extreme"]
    return df.reset_index(drop=True)


def make_preprocessor(features=None, scale: bool = True) -> ColumnTransformer:
    """Median imputation with missing-value indicators, log of skewed columns,
    optional standardization, and one-hot encoding of categoricals.

    scale=False is meant for tree models, which need neither logs nor scaling.
    """
    features = C.FEATURES if features is None else list(features)
    numeric = [c for c in features if c in C.NUMERIC]
    skewed = [c for c in numeric if c in C.SKEWED] if scale else []
    plain = [c for c in numeric if c not in skewed]
    categorical = [c for c in features if c in C.CATEGORICAL]

    def numeric_pipe(log: bool) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if log:
            steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if plain:
        transformers.append(("num", numeric_pipe(log=False), plain))
    if skewed:
        transformers.append(("log", numeric_pipe(log=True), skewed))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)
