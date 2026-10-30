# Cohort review

Source: College Scorecard "Most Recent Cohorts – Institution" file
(`Dataset/Most-Recent-Cohorts-Institution_10152026.zip`), checked against the official data
dictionary released **October 15, 2026** (`docs/reference/CollegeScorecardDataDictionary.xlsx`,
sheet `Most_Recent_Inst_Cohort_Map`). The cohort is rebuilt from the raw file by
`python -m src.clean`; the per-variable table is in [data_dictionary.md](data_dictionary.md).

> **Fix for the report:** the proposal says the file was "last updated on June 10, 2026".
> The file we downloaded is the **October 15, 2026** release (zip name and dictionary README).

## 1. Cohort construction

| Step | Institutions |
|---|---|
| Raw file | 6,278 (3,379 columns) |
| Predominantly bachelor's (`PREDDEG == 3`) | 1,947 |
| … reporting `RET_FT4` | 1,805 |
| … minus retention exactly 0% (12) or 100% (56) | **1,737** analysis cohort |
| Train / test (stratified by retention quartile, seed 42) | 1,389 / 348 |

The proposal's numbers reproduce exactly. Mean retention is 0.745.

The 68 excluded institutions are tiny (median undergraduate enrollment 153) and mostly private
(41 nonprofit, 25 for-profit, 2 public). They stay in `data/processed/cohort.csv` with
`split = "excluded_extreme"` for the E4 robustness check.

**At-risk label.** `at_risk = 1` when `RET_FT4 ≤ 0.6734`, the 25th percentile of the
**training** split. The cut-off comes from training rows only, so the test labels play no part
in defining the label. At-risk share: 18.7% for public, 25.4% for private nonprofit and 60.4%
for for-profit institutions.

## 2. Which years each variable describes

| Period | Variables |
|---|---|
| **Fall 2023 entering cohort** (returning Fall 2024) | `RET_FT4`: the target |
| Fall 2023 | `UG25ABV` |
| FY2024 / AY2023-24 | `TUITFTE`, `INEXPFTE` |
| Fall 2024 | `ADM_RATE`, `SAT_AVG`, `UGDS`, race/gender shares, `PPTUG_EF`, `PFTFAC`; `AVGFACSAL` (AY2024-25 salaries) |
| AY2024-25 | `COSTT4_A`, `PCTPELL`, `PCTFLOAN` |
| AY2025-26 | `TUITIONFEE_IN`, `CONTROL`, `REGION`, `LOCALE`, `HBCU` |
| AY2015-16 + 2016-17 pooled (NSLDS) | `PAR_ED_PCT_1STGEN` |

**What this means for the analysis**

1. **The features mostly describe the year *after* the cohort entered.** Admissions, enrollment
   mix, cost and Pell share describe the Fall 2024 class and the 2024-25 year, while the target
   describes students who entered in Fall 2023. This is not outcome leakage, because none of
   these variables is computed from the cohort's own retention. But the model learns
   associations between a school's *current profile* and its *recent* retention. It is not
   forecasting retention from information available before the cohort enrolled. Say this in the
   limitations section. Institutional characteristics change slowly, so the effect should be
   small; the optional temporal check (E5) using an earlier yearly file would test it.
2. **`ADM_RATE` and `SAT_AVG` describe the next admitted class (Fall 2024)**, not the Fall 2023
   cohort whose retention we predict.
3. **`PAR_ED_PCT_1STGEN` is stale and discontinued.** It comes from 2015-17 FAFSA filers (about
   eight years before the cohort), and 9.4% of values are missing or privacy-suppressed. Keep it
   for now, but E1/E4 should check whether dropping it changes the results. If not, drop it.
4. **Leakage exclusions.** Completion rates (`C150_4` covers the Fall 2019 cohort), earnings, debt,
   repayment, default and outcome-measure variables are never model inputs. `src/clean.py` keeps
   only an allow-list of features and asserts that none matches an outcome prefix
   (`config.LEAKAGE_PREFIXES`). `RET_PT4` (part-time retention) is excluded for the same reason.

## 3. Data quality findings

| Issue | Finding | Handling |
|---|---|---|
| Missing SAT | 42.0% missing (58% available; the proposal said 56%) | median imputation + missing indicator in the pipeline |
| Missing admission rate | 11.5% missing (88.5% available; the proposal said 87%) | same |
| Suppressed values | `PAR_ED_PCT_1STGEN` contains the string `PrivacySuppressed` | converted to NaN in `clean.py` |
| Units | `AVGFACSAL` is **monthly** salary (median ≈ $8.7k) | none needed for modelling; label figures correctly |
| Skew | `UGDS`, `TUITFTE`, `INEXPFTE` heavily right-skewed | `log1p` before scaling in linear/SVM pipelines |
| Empty columns | `PBI`, `HSI` are entirely empty for four-year schools in this release | not used; MSI-type analysis is limited to `HBCU` |
| Closed schools | 8 institutions have `CURROPER = 0` | kept (their retention is valid); flag in error analysis |
| Duplicates | `UNITID` unique; no duplicate rows | asserted in `clean.py` |
| Range checks | every share variable lies in [0, 1] | asserted in `clean.py` |
| Missing per row | median 0 features missing, max 11 | — |

## 4. Noise from small schools

Retention for tiny entering classes is noisy:

| Undergraduate enrollment | n | mean retention | std |
|---|---|---|---|
| ≤ 500 | 264 | 0.667 | 0.199 |
| 501–1,000 | 280 | 0.692 | 0.125 |
| 1,001–5,000 | 763 | 0.758 | 0.108 |
| > 5,000 | 430 | 0.803 | 0.117 |

Expect the largest errors among schools with 500 or fewer undergraduates. E5 should report
errors by size band, and robustness checks could weight by enrollment or drop schools below a
size threshold.

## 5. Subgroups for the fairness and error analyses (E2, E5)

| Group | n | mean retention |
|---|---|---|
| Public | 568 | 0.759 |
| Private nonprofit | 1,078 | 0.749 |
| Private for-profit | 91 | 0.607 |
| HBCU | 81 | 0.655 |
| Massachusetts | 65 | — |

For-profit (91) and HBCU (81) institutions are small groups. In the test split they number
roughly 18 and 16, so report subgroup metrics from cross-validated out-of-fold predictions on the
training set, not from the test set alone.
