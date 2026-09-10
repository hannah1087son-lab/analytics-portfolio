# LendingClub Interest Rate Prediction

## Business Problem

Predict loan interest rates using only information available when a
borrower applies, supporting more consistent risk-based pricing while
avoiding post-origination data leakage.

## Data

-   100,000 training loans
-   10,000 test loans
-   38 modeling features
-   Interest-rate target

## Approach

Prepared missing values, created missingness indicators, engineered FICO
and utilization features, transformed skewed financial variables, and
used leakage-safe five-fold cross-validation. Compared decision tree,
random forest, neural network, LightGBM, XGBoost, and CatBoost
approaches.

## Results

CatBoost was the strongest standalone model with an out-of-fold RMSE of
3.845. SHAP and feature-importance analysis showed that FICO,
utilization, and debt burden were among the strongest drivers of
predicted rates.

## Business Takeaway

The model captured intuitive credit-risk relationships while maintaining
a realistic application-time feature set. The project reinforced the
importance of feature engineering, leakage prevention, model comparison,
and explainability.

## Tools

Python \| CatBoost \| XGBoost \| LightGBM \| Random Forest \| SHAP \|
Cross-Validation
