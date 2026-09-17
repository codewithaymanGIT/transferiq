# Model comparison

Generated 2026-09-17T21:38:28+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, tackles, interceptions, is_top_six, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 17.19 | 26.19 | 11.05 | -0.221 | 148 | 37 |
| Linear Regression | 16.64 | 23.87 | 10.49 | -0.015 | 148 | 37 |
| Ridge | 14.88 | 22.86 | 8.98 | 0.070 | 148 | 37 |
| Random Forest | 16.71 | 23.88 | 8.62 | -0.015 | 148 | 37 |
| XGBoost | 17.49 | 24.10 | 12.95 | -0.034 | 148 | 37 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 13.38 | 19.18 | -0.094 | 0.071 |
| Linear Regression | 11.68 | 17.28 | 0.120 | 0.099 |
| Ridge | 11.30 | 17.09 | 0.135 | 0.074 |
| Random Forest | 11.84 | 16.98 | 0.140 | 0.112 |
| XGBoost | 12.63 | 17.49 | 0.068 | 0.216 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.317 |
| minutes | 0.165 |
| assists_per90 | 0.125 |
| goals_per90 | 0.101 |
| interceptions | 0.079 |
| tackles | 0.067 |
| goals | 0.062 |
| assists | 0.047 |
| is_top_six | 0.019 |
| position=MF | 0.007 |
| position=FW | 0.006 |
| position=DF | 0.005 |
| position=GK | 0.001 |

## SHAP feature importance (mean |SHAP value|)

Real SHAP values from shap.TreeExplainer on the Random Forest model, fit on the full dataset. More rigorous than the Gini importances above since SHAP reflects each feature's actual average impact on individual predictions (in log-fee units), not just split frequency. Still descriptive of this one model, not a causal claim.

| Feature | Mean |SHAP value| |
|---|---|
| age_at_transfer | 0.2959 |
| minutes | 0.1578 |
| assists_per90 | 0.0960 |
| interceptions | 0.0816 |
| goals_per90 | 0.0697 |
| goals | 0.0599 |
| tackles | 0.0561 |
| assists | 0.0356 |
| is_top_six | 0.0237 |
| position=FW | 0.0051 |
| position=MF | 0.0041 |
| position=DF | 0.0031 |
| position=GK | 0.0008 |
