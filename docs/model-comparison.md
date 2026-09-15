# Model comparison

Generated 2026-09-14T11:15:26+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, tackles, interceptions, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 11.27 | 16.99 | 7.70 | -0.044 | 102 | 26 |
| Linear Regression | 19.00 | 25.71 | 14.46 | -1.389 | 102 | 26 |
| Ridge | 17.77 | 24.07 | 14.69 | -1.095 | 102 | 26 |
| Random Forest | 15.64 | 20.54 | 12.81 | -0.525 | 102 | 26 |
| XGBoost | 17.70 | 24.30 | 10.74 | -1.135 | 102 | 26 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 14.78 | 20.98 | -0.084 | 0.086 |
| Linear Regression | 13.66 | 20.00 | -0.093 | 0.676 |
| Ridge | 13.36 | 19.79 | -0.042 | 0.556 |
| Random Forest | 12.67 | 18.83 | 0.101 | 0.267 |
| XGBoost | 13.48 | 20.20 | -0.079 | 0.532 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.412 |
| minutes | 0.161 |
| assists_per90 | 0.096 |
| goals_per90 | 0.078 |
| assists | 0.064 |
| goals | 0.059 |
| interceptions | 0.056 |
| tackles | 0.051 |
| position=MF | 0.013 |
| position=DF | 0.005 |
| position=FW | 0.004 |
| position=GK | 0.001 |

## SHAP feature importance (mean |SHAP value|)

Real SHAP values from shap.TreeExplainer on the Random Forest model, fit on the full dataset. More rigorous than the Gini importances above since SHAP reflects each feature's actual average impact on individual predictions (in log-fee units), not just split frequency. Still descriptive of this one model, not a causal claim.

| Feature | Mean |SHAP value| |
|---|---|
| age_at_transfer | 0.4344 |
| minutes | 0.1787 |
| goals | 0.0691 |
| assists_per90 | 0.0626 |
| assists | 0.0585 |
| goals_per90 | 0.0574 |
| interceptions | 0.0377 |
| tackles | 0.0300 |
| position=MF | 0.0248 |
| position=FW | 0.0048 |
| position=DF | 0.0044 |
| position=GK | 0.0006 |
