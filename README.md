# Credit Default Risk Prediction (GCI World 2026, University of Tokyo)

Loan default prediction model built for the in-class competition of **GCI World 2026**, the Global Consumer Intelligence data science program of the **Matsuo-Iwasawa Lab, University of Tokyo**. The task is to predict whether a client will default on a loan, using a Home Credit Default Risk dataset.

**Result: top 7% of the public leaderboard (612th of 8,758 submissions), ROC-AUC 0.77201.**

## Results

| Version | Main change | Validation AUC | Public leaderboard |
| --- | --- | --- | --- |
| v1 | LightGBM + CatBoost with financial ratios | 0.7621 | n/a |
| v2 | Added XGBoost, 3 seeds | 0.7637 | 0.76703 (top ~16%) |
| v3 | Loan structure features and EXT_SOURCE imputation | 0.7668 | 0.76946 (top ~11%) |
| v5 | Tuned LightGBM (5 seeds) + XGBoost + CatBoost blend | 0.7685 | **0.77201 (top 7%)** |

Out-of-fold ROC-AUC of the final notebook (5-fold stratified CV):

| Model | OOF AUC |
| --- | --- |
| LightGBM (average of 5 seeds) | 0.76728 |
| XGBoost | 0.76753 |
| CatBoost | 0.76753 |
| **Blend (0.50 / 0.25 / 0.25)** | **0.76852** |

## Approach

**Data.** 171,202 training rows and 61,500 test rows. Only 8.07% of clients default, so the target is imbalanced. The metric is ROC-AUC and every model uses the same stratified 5-fold split (seed 42).

**Cleaning.**
- `DAYS_EMPLOYED = 365243` is a sentinel value (the same rows have `ORGANIZATION_TYPE = "XNA"`), so it was converted to null and flagged in a separate column.
- `XNA` in `CODE_GENDER` and `Unknown` in `NAME_FAMILY_STATUS` are disguised nulls.

**Feature engineering** (75 features, 8 categorical):
- **Financial ratios:** credit to income, annuity to income, credit to goods price, income per family member, age, and employment and registration time relative to age.
- **EXT_SOURCE combinations:** mean, min, max, std, product and a weighted sum of the three external scores, which are the strongest predictors in the data.
- **EXT_SOURCE imputation without the target:** `EXT_SOURCE_1` is about 70% null. For each score, a LightGBM regressor estimates it from the other raw columns, using out-of-fold predictions on known rows so the real value is never copied. This improved validation on all 3 seeds tested.
- **Loan structure:** `AMT_CREDIT / AMT_ANNUITY` approximates the number of payments, and its distribution has clear peaks, one per term and rate combination. The features include the nearest peak, the distance to it, the position between peaks, the standard term (6, 10, 12, 18, 24 months and so on) and the **implied interest rate**, solved from the annuity formula with vectorized bisection. This was the largest single gain: from 0.7631 to 0.7665 and from 0.7622 to 0.7660 on two different validation splits.

**Models.**
- LightGBM with hyperparameters from a short Optuna search (40 trials), averaged over 5 seeds.
- XGBoost (depth 4) and CatBoost (depth 5, `l2_leaf_reg` 20) with native categorical handling.
- Fixed, rounded blend weights (0.50 / 0.25 / 0.25). Nearby weight combinations give practically the same OOF AUC, so the weights were not tuned further to avoid overfitting.

**Validation discipline.** A change was kept only if it improved out-of-fold AUC on two different splits. The public leaderboard was not used to select features.

**Tried and discarded** (no validation gain): group statistics such as income relative to occupation, in-fold target encoding, mean target of the 500 nearest neighbours, date differences, counts of repeated applicant profiles, and residuals of the EXT_SOURCE estimate.

## How to run

1. Install the dependencies: `pip install -r requirements.txt`
2. Put the competition files in an `input/` folder next to the notebook: `train.csv`, `test.csv` and `sample_submission.csv`. They are not included in this repository because they belong to the GCI World course.
3. Run `credit_risk_solution.ipynb` from top to bottom. It takes about 1 hour on 8 cores and writes `output/submission.csv`.

The notebook also runs in Google Colab if you set `COLAB_DIR` to your Drive folder.

## Notes

- The ranking is from the **public** leaderboard. The final ranking on the full test set may differ slightly.
- Only `train.csv` and `test.csv` are used, with no external data and no manual labelling.
- The final version (v5) was developed with AI coding assistance (Claude Code).

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, matplotlib, seaborn

## Author

**Mateo Pozo García**, Systems and Computer Engineering student at Universidad Tecnológica del Perú (UTP), Lima.

[LinkedIn](https://www.linkedin.com/in/mateo-pozo-garcia/) | [GitHub](https://github.com/vuhn204)
