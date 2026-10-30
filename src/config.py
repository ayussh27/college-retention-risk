"""Paths, cohort rules and feature groups shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Dataset" / "Most-Recent-Cohorts-Institution.csv"
DICT_XLSX = ROOT / "docs" / "reference" / "CollegeScorecardDataDictionary.xlsx"
PROCESSED = ROOT / "data" / "processed"
COHORT_CSV = PROCESSED / "cohort.csv"
RESULTS = ROOT / "results"

SEED = 42
TEST_SIZE = 0.20
AT_RISK_QUANTILE = 0.25  # bottom quartile of retention = "at risk"

TARGET = "RET_FT4"  # first-time, full-time retention at four-year institutions
PREDDEG_BACHELORS = 3

# Kept for identification and error analysis (E5); never used as model inputs.
ID_COLS = ["UNITID", "INSTNM", "CITY", "STABBR", "CURROPER"]

# Feature groups for the E1 ablation. Every model input must appear here.
FEATURE_GROUPS = {
    "admissions": ["ADM_RATE", "SAT_AVG"],
    "cost_spending": ["COSTT4_A", "TUITIONFEE_IN", "TUITFTE", "INEXPFTE", "AVGFACSAL", "PFTFAC"],
    "student_mix": ["PCTPELL", "PCTFLOAN", "UG25ABV", "PPTUG_EF", "PAR_ED_PCT_1STGEN", "UGDS"],
    "demographics": ["UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
                     "UGDS_2MOR", "UGDS_NRA", "UGDS_WOMEN"],
    "institution_type": ["CONTROL", "REGION", "LOCALE_TYPE", "HBCU"],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

# Encoded as one-hot; everything else is numeric.
CATEGORICAL = ["CONTROL", "REGION", "LOCALE_TYPE"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# Heavily right-skewed counts and dollar amounts, log-transformed for linear/SVM models.
SKEWED = ["UGDS", "TUITFTE", "INEXPFTE"]
# Removed in E2 (sensitive-feature comparison).
SENSITIVE = FEATURE_GROUPS["demographics"]

# Column-name patterns that describe outcomes measured after entry. None may be a
# model input; clean.py asserts this.
LEAKAGE_PREFIXES = ("C100", "C150", "C200", "D150", "RET_", "MD_EARN", "MN_EARN",
                    "PCT10_EARN", "PCT25_EARN", "PCT75_EARN", "PCT90_EARN", "GT_THRESHOLD",
                    "DEBT", "GRAD_DEBT", "WDRAW_DEBT", "RPY_", "CDR", "BBRR", "OMAWDP", "OMENR")

CONTROL_LABELS = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}
LOCALE_TYPE_LABELS = {1: "City", 2: "Suburb", 3: "Town", 4: "Rural"}
