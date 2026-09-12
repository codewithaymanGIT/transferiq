# Model comparison

Generated 2026-09-12T12:28:58+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 11.27 | 16.99 | 7.70 | -0.044 | 102 | 26 |
| Linear Regression | 19.06 | 25.96 | 12.96 | -1.436 | 102 | 26 |
| Ridge | 17.49 | 23.89 | 13.53 | -1.062 | 102 | 26 |
| Random Forest | 15.75 | 20.76 | 12.93 | -0.558 | 102 | 26 |
| XGBoost | 17.84 | 23.13 | 14.50 | -0.934 | 102 | 26 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 14.78 | 20.98 | -0.084 | 0.086 |
| Linear Regression | 13.57 | 19.90 | -0.091 | 0.703 |
| Ridge | 13.23 | 19.73 | -0.034 | 0.548 |
| Random Forest | 12.89 | 19.15 | 0.063 | 0.267 |
| XGBoost | 14.23 | 20.25 | -0.090 | 0.447 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.430 |
| minutes | 0.185 |
| assists_per90 | 0.123 |
| goals_per90 | 0.099 |
| assists | 0.067 |
| goals | 0.067 |
| position=MF | 0.016 |
| position=DF | 0.008 |
| position=FW | 0.005 |
| position=GK | 0.002 |
