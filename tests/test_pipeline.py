"""Checks on the processed cohort and the preprocessing pipeline.

Run from the project root:  python -m pytest
These use data/processed/cohort.csv (committed), so they run without the raw download.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src import config as C
from src.features import load_cohort, make_preprocessor


@pytest.fixture(scope="module")
def cohort():
    return load_cohort(include_extreme=True)


def test_cohort_counts(cohort):
    assert len(cohort) == 1805
    assert cohort["extreme_retention"].sum() == 68
    assert (cohort["split"] == "train").sum() == 1389
    assert (cohort["split"] == "test").sum() == 348


def test_unique_institutions(cohort):
    assert cohort["UNITID"].is_unique


def test_target_in_range(cohort):
    main = cohort[~cohort["extreme_retention"]]
    assert main[C.TARGET].gt(0).all() and main[C.TARGET].lt(1).all()


def test_no_outcome_columns_among_features():
    assert not [c for c in C.FEATURES if c.startswith(C.LEAKAGE_PREFIXES)]
    assert C.TARGET not in C.FEATURES


def test_feature_groups_cover_features_once():
    flat = [c for cols in C.FEATURE_GROUPS.values() for c in cols]
    assert len(flat) == len(set(flat)) == len(C.FEATURES)


def test_at_risk_cutoff_comes_from_training_split(cohort):
    train = cohort[cohort["split"] == "train"]
    cutoff = train[C.TARGET].quantile(C.AT_RISK_QUANTILE)
    assert (cohort["at_risk"] == (cohort[C.TARGET] <= cutoff)).all()
    assert abs(train["at_risk"].mean() - 0.25) < 0.01


def test_split_is_stratified(cohort):
    main = cohort[~cohort["extreme_retention"]]
    by_split = main.groupby("split")[C.TARGET].mean()
    assert abs(by_split["train"] - by_split["test"]) < 0.02


@pytest.mark.parametrize("scale", [True, False])
def test_preprocessor_handles_missing_and_unseen_values(scale):
    train = load_cohort()
    train = train[train["split"] == "train"]
    pipe = Pipeline([("prep", make_preprocessor(scale=scale)), ("model", Ridge())])
    pipe.fit(train[C.FEATURES], train[C.TARGET])

    row = pd.DataFrame(np.nan, index=[0], columns=C.FEATURES)  # every feature missing
    row["REGION"] = 99                  # category never seen in training
    pred = pipe.predict(row)
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_preprocessor_output_has_no_nans():
    train = load_cohort()
    X = make_preprocessor().fit_transform(train[C.FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else X
    assert not np.isnan(X).any()
