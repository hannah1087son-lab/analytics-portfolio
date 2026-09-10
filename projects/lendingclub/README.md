# LendingClub Interest Rate Prediction

![Project overview](images/project-overview.png)

## Result first

![Model comparison](images/model-comparison.png)

**CatBoost was the strongest standalone model at 3.845 OOF RMSE**,
followed closely by XGBoost at 3.867. Random Forest and the
neural-network benchmark performed worse.

## What I did

-   Built a leakage-safe modeling workflow using only information
    available at loan origination.
-   Cleaned numeric fields, handled missing values, added missingness
    flags, and engineered FICO, term, employment-length, and
    skewed-money features.
-   Compared boosted trees, bagged trees, a decision tree, and a
    neural-network baseline using consistent 5-fold validation.
-   Used SHAP and permutation importance to interpret the strongest
    drivers.

## Business insight

FICO, utilization, and debt-to-income were among the strongest signals.
The project showed how model accuracy, leakage prevention, and
explainability work together in a real pricing decision.

## Technical stack

`Python` · `CatBoost` · `XGBoost` · `LightGBM` · `Random Forest` ·
`SHAP` · `5-fold CV`

> **Code note:** I have not included reconstructed code. The original
> executable notebook was not available in the source files used to
> build this public portfolio. See [CODE_NOTE.md](CODE_NOTE.md).
