"""Build the analysis cohort from the raw College Scorecard file.

Run from the project root:  python -m src.clean

Steps
1. Keep predominantly bachelor's-degree institutions (PREDDEG == 3) that report RET_FT4.
2. Flag institutions with retention exactly 0 or 1 (excluded from the main analysis,
   kept in the file for the E4 robustness check).
3. Keep only ID columns, the allow-listed features and the target; convert
   "PrivacySuppressed"/"NULL" strings to NaN; derive LOCALE_TYPE.
4. Assign a stratified 80/20 train/test split (stratified by retention quartile) and
   the at-risk label, whose bottom-quartile cut-off is computed on the training set only.

No imputation, scaling or encoding happens here: those are fitted inside the
scikit-learn pipelines (src/features.py) so nothing is learned from held-out data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV, low_memory=False)


def to_numeric(s: pd.Series) -> pd.Series:
    # Scorecard uses "PrivacySuppressed" and "NULL" for withheld values.
    return pd.to_numeric(s.replace({"PrivacySuppressed": np.nan, "NULL": np.nan}), errors="coerce")


def build_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["PREDDEG"] == C.PREDDEG_BACHELORS].copy()
    df[C.TARGET] = to_numeric(df[C.TARGET])
    df = df[df[C.TARGET].notna()]

    # LOCALE codes are two digits: 11-13 city, 21-23 suburb, 31-33 town, 41-43 rural.
    df["LOCALE_TYPE"] = (to_numeric(df["LOCALE"]) // 10).where(lambda s: s.between(1, 4))

    raw_features = [c for c in C.FEATURES if c != "LOCALE_TYPE"]
    out = df[C.ID_COLS + raw_features + ["LOCALE_TYPE", C.TARGET]].copy()
    for c in raw_features:
        out[c] = to_numeric(out[c])

    out["extreme_retention"] = out[C.TARGET].isin([0.0, 1.0])
    return out.reset_index(drop=True)


def assign_split(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = cohort.copy()
    main = cohort[~cohort["extreme_retention"]]
    quartile = pd.qcut(main[C.TARGET], 4, labels=False)
    train_idx, test_idx = train_test_split(
        main.index, test_size=C.TEST_SIZE, stratify=quartile, random_state=C.SEED)

    cohort["split"] = "excluded_extreme"
    cohort.loc[train_idx, "split"] = "train"
    cohort.loc[test_idx, "split"] = "test"

    # The at-risk cut-off is a statistic of the labels, so it comes from training data only.
    cutoff = cohort.loc[train_idx, C.TARGET].quantile(C.AT_RISK_QUANTILE)
    cohort["at_risk"] = (cohort[C.TARGET] <= cutoff).astype(int)
    cohort.attrs["at_risk_cutoff"] = cutoff
    return cohort


def check(cohort: pd.DataFrame) -> None:
    model_inputs = set(C.FEATURES)
    leaks = [c for c in model_inputs if c.startswith(C.LEAKAGE_PREFIXES)]
    assert not leaks, f"outcome columns among features: {leaks}"
    assert cohort["UNITID"].is_unique, "duplicate institutions"
    assert cohort[C.TARGET].between(0, 1).all(), "retention outside [0, 1]"
    not_shares = {"SAT_AVG", "COSTT4_A", "TUITIONFEE_IN", "TUITFTE", "INEXPFTE", "AVGFACSAL", "UGDS"}
    for c in [c for c in C.NUMERIC if c not in not_shares]:
        s = cohort[c].dropna()
        assert s.between(0, 1).all(), f"{c} has values outside [0, 1]"
    assert set(cohort.loc[~cohort["extreme_retention"], "split"]) == {"train", "test"}


def main() -> None:
    raw = load_raw()
    cohort = assign_split(build_cohort(raw))
    check(cohort)

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(C.COHORT_CSV, index=False)

    main_rows = cohort[cohort["split"] != "excluded_extreme"]
    print(f"raw file:                 {raw.shape[0]:,} institutions x {raw.shape[1]:,} columns")
    print(f"PREDDEG == 3:             {(raw['PREDDEG'] == C.PREDDEG_BACHELORS).sum():,}")
    print(f"  with RET_FT4:           {len(cohort):,}")
    print(f"  excluded 0%/100%:       {cohort['extreme_retention'].sum():,}")
    print(f"analysis cohort:          {len(main_rows):,}  "
          f"(train {(cohort['split'] == 'train').sum():,}, test {(cohort['split'] == 'test').sum():,})")
    print(f"mean retention:           {main_rows[C.TARGET].mean():.3f}")
    print(f"at-risk cut-off (train):  RET_FT4 <= {cohort.attrs['at_risk_cutoff']:.4f}; "
          f"at-risk share {main_rows['at_risk'].mean():.3f}")
    print(f"wrote {C.COHORT_CSV.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
