# Model comparison

Generated 2026-09-15T20:11:48+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, tackles, interceptions, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 11.27 | 16.99 | 7.70 | -0.044 | 102 | 26 |
| Linear Regression | 19.25 | 25.78 | 14.42 | -1.403 | 102 | 26 |
| Ridge | 17.88 | 23.95 | 14.61 | -1.073 | 102 | 26 |
| Random Forest | 16.23 | 20.79 | 12.57 | -0.563 | 102 | 26 |
| XGBoost | 18.29 | 24.71 | 11.04 | -1.207 | 102 | 26 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 14.78 | 20.98 | -0.084 | 0.086 |
| Linear Regression | 13.85 | 20.34 | -0.122 | 0.671 |
| Ridge | 13.52 | 20.02 | -0.059 | 0.542 |
| Random Forest | 12.88 | 19.13 | 0.065 | 0.294 |
| XGBoost | 13.70 | 20.30 | -0.094 | 0.561 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.415 |
| minutes | 0.163 |
| assists_per90 | 0.101 |
| goals_per90 | 0.073 |
| goals | 0.063 |
| assists | 0.062 |
| interceptions | 0.055 |
| tackles | 0.052 |
| position=MF | 0.009 |
| position=DF | 0.004 |
| position=FW | 0.003 |
| position=GK | 0.000 |

## SHAP feature importance (mean |SHAP value|)

Real SHAP values from shap.TreeExplainer on the Random Forest model, fit on the full dataset. More rigorous than the Gini importances above since SHAP reflects each feature's actual average impact on individual predictions (in log-fee units), not just split frequency. Still descriptive of this one model, not a causal claim.

| Feature | Mean |SHAP value| |
|---|---|
| age_at_transfer | 0.4352 |
| minutes | 0.1819 |
| goals | 0.0684 |
| assists_per90 | 0.0653 |
| goals_per90 | 0.0550 |
| assists | 0.0531 |
| interceptions | 0.0376 |
| tackles | 0.0314 |
| position=MF | 0.0126 |
| position=DF | 0.0031 |
| position=FW | 0.0031 |
| position=GK | 0.0004 |
