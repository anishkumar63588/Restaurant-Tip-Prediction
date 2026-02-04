# Restaurant Tip Regression Project

## Restaurant Tip Prediction

## Overview
This notebook explores a restaurant tipping dataset (244 records) and applies regression techniques to model tipping behavior.

The workflow includes:
- Basic exploratory data analysis (EDA)
- Converting categorical variables to numeric features
- Correlation analysis and feature removal
- Training and evaluating regression models (with scaling pipelines)

## Dataset
The notebook uses the classic **tips** dataset with the following original columns:
- `total_bill` (float): total bill amount
- `tip` (float): tip amount
- `sex` (category): Male/Female
- `smoker` (category): Yes/No
- `day` (category): Thur/Fri/Sat/Sun
- `time` (category): Lunch/Dinner
- `size` (int): party size

Quick stats from `.describe()`:
- Mean `total_bill`: **19.79**
- Mean `tip`: **3.00**
- Mean `size`: **2.57**

No missing values were found (`244` non-null rows in all columns).

## EDA highlights (from outputs)
- **Tip distribution by sex** was visualized with histograms.
- **Tip vs total_bill** scatter plot suggests a positive relationship.

Correlation with `tip` (selected values from the notebook output):
- `total_bill`: **0.6757** (strongest)
- `size`: **0.4893**
- `is_weekend`: **0.1251**
- `is_dinner`: **0.1216**
- `is_male`: **0.0889**
- `smoker`: **0.0059** (smallest)

## Preprocessing (as implemented in the notebook)
Categorical columns were converted to numeric:
- `is_male`: Male → 1, Female → 0
- `smoker`: Yes → 1, No → 0
- `is_weekend`: Sun → 1, else → 0 (per notebook mapping)
- `is_dinner`: Dinner → 1, Lunch → 0

Then the original columns `sex`, `day`, and `time` were dropped.

Based on correlation, the notebook removes **`smoker`** due to the smallest correlation with `tip`.

## Models trained and evaluation results
Two regression approaches were trained using scikit-learn pipelines with scaling:

1) **Linear Regression + StandardScaler**
- Reported **MAE: 1.3753708722171603**

2) **Polynomial Regression (PolynomialFeatures) + StandardScaler + LinearRegression**
- Reported **MAE: 1.3766251499148412**

## Conclusion (based on notebook outputs)
- The dataset shows **a clear positive relationship** between `total_bill` and `tip`, and party `size` also correlates moderately with `tip`.
- After encoding categorical variables and removing `smoker`, the notebook compares linear and polynomial regression (both scaled).
- **Polynomial regression did not improve performance** over linear regression (MAE is nearly identical: ~1.375 vs ~1.377), suggesting the added complexity does not help on this dataset/settings.
- Overall, a **simple linear model performs as well as a more complex polynomial model** for the evaluated setup.

## Notes / potential improvements
- Consider using a clearer weekend mapping (typically Sat/Sun as weekend).
- Use cross-validation and additional metrics (R², RMSE) for more robust evaluation.
- Try tree-based models (Random Forest, Gradient Boosting) to capture non-linearities without manual polynomial features.

## How to run
1. Open the notebook: `Notebook - Regression Project.ipynb`
2. Run cells top-to-bottom.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib
- scikit-learn
