"""Reference performance before any tuning.

Run from the project root:  python -m src.baselines

Uses the training split only, with repeated 5-fold cross-validation (5 x 3).
The test split is not touched here; it is used once, for the final model.
"""
import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config as C
from src.features import load_cohort, make_preprocessor

N_SPLITS, N_REPEATS = 5, 3

REGRESSORS = {
    "Mean (dummy)": (DummyRegressor(strategy="mean"), True),
    "Ridge (alpha=1)": (Ridge(alpha=1.0), True),
    "Random forest (untuned)": (RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=C.SEED), False),
}
CLASSIFIERS = {
    "Majority / prior (dummy)": (DummyClassifier(strategy="prior"), True),
    "Logistic regression (C=1)": (LogisticRegression(max_iter=2000), True),
    "Random forest (untuned)": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=C.SEED), False),
}
REG_SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
CLF_SCORING = {"ROC-AUC": "roc_auc", "Avg precision": "average_precision",
               "F1": "f1", "Precision": "precision", "Recall": "recall"}


def evaluate(models, scoring, X, y, cv, task):
    rows = []
    for name, (model, scale) in models.items():
        pipe = Pipeline([("prep", make_preprocessor(scale=scale)), ("model", model)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        row = {"task": task, "model": name}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            if metric in ("MAE", "RMSE"):
                vals = -vals
            row[f"{metric} mean"] = vals.mean()
            row[f"{metric} std"] = vals.std()
        rows.append(row)
        print(f"  {name:28s}" + "  ".join(f"{m} {row[f'{m} mean']:.3f}+/-{row[f'{m} std']:.3f}" for m in scoring))
    return rows


def main() -> None:
    train = load_cohort()
    train = train[train["split"] == "train"]
    X = train[C.FEATURES]
    print(f"training rows: {len(train):,}; features: {len(C.FEATURES)}; "
          f"at-risk share: {train['at_risk'].mean():.3f}")

    print("\nRegression: RET_FT4")
    rows = evaluate(REGRESSORS, REG_SCORING, X, train[C.TARGET],
                    RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=C.SEED), "regression")
    print("\nClassification: at_risk (bottom quartile)")
    # Precision/F1 are undefined for the dummy (it never predicts at-risk); sklearn reports 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        rows += evaluate(CLASSIFIERS, CLF_SCORING, X, train["at_risk"],
                         RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=C.SEED),
                         "classification")

    C.RESULTS.mkdir(exist_ok=True)
    out = C.RESULTS / "baselines.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
