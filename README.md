# Credit Scoring Model

Predicts whether a borrower is likely to default within two years, 
using classification algorithms trained on financial history.
Built as part of the CodeAlpha Machine Learning Internship.

## Problem

Given financial data (income, debt ratio, credit utilization, payment 
history, etc.), predict whether a person will experience serious 
delinquency (90+ days late) within the next two years.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) 
(Kaggle) — 150,000 records, 10 features.

## Approach

1. **Data Cleaning**
   - Filled missing `MonthlyIncome` and `NumberOfDependents` with the 
     median, with flag columns marking what was originally missing.
   - Capped `DebtRatio` at 1 and `RevolvingUtilizationOfUnsecuredLines` 
     at 2 to control extreme outliers.
   - Identified and capped placeholder codes (96/98) in the late-payment 
     columns, flagging them separately since they correlated with a 
     much higher default rate.
   - Dropped 1 row with an invalid age of 0.

2. **Preprocessing**
   - Scaled all features with `StandardScaler`.
   - Split data 80/20 into train/test sets.

3. **Modeling**
   - Trained Logistic Regression and Random Forest models.
   - Addressed class imbalance (~6.7% default rate) using 
     `class_weight="balanced"`.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (plain) | 93.7% | 55.2% | 16.6% | 0.26 | 0.847 |
| **Logistic Regression (balanced)** | 80.3% | 21.1% | **73.6%** | 0.33 | **0.851** |
| Random Forest (balanced) | 93.6% | 52.3% | 15.6% | 0.24 | 0.829 |

**Final model: Logistic Regression with `class_weight="balanced"`.**

Raw accuracy is misleading here because only ~6.7% of borrowers actually 
default — a model that always predicts "no default" would already score 
93%+ accuracy while being useless. In credit scoring, missing a real 
defaulter is costlier than flagging a safe borrower, so recall matters 
more than raw accuracy. The balanced model catches ~74% of real 
defaulters, compared to ~17% for the unweighted model.

## Tech Stack

Python, pandas, scikit-learn, Google Colab

## Author

Martins (MartCruz17)
