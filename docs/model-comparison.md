# Model comparison

Generated 2026-09-27T17:50:50+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, tackles, interceptions, is_top_six, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 17.19 | 26.19 | 11.05 | -0.221 | 148 | 37 |
| Linear Regression | 16.73 | 24.17 | 10.10 | -0.040 | 148 | 37 |
| Ridge | 14.65 | 22.81 | 8.84 | 0.074 | 148 | 37 |
| Random Forest | 16.59 | 23.93 | 8.63 | -0.020 | 148 | 37 |
| XGBoost | 17.89 | 25.18 | 10.65 | -0.129 | 148 | 37 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 13.38 | 19.18 | -0.094 | 0.071 |
| Linear Regression | 11.76 | 17.50 | 0.097 | 0.094 |
| Ridge | 11.32 | 17.25 | 0.117 | 0.057 |
| Random Forest | 11.73 | 16.97 | 0.144 | 0.116 |
| XGBoost | 13.12 | 18.22 | -0.005 | 0.178 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.315 |
| minutes | 0.160 |
| assists_per90 | 0.125 |
| goals_per90 | 0.099 |
| tackles | 0.075 |
| interceptions | 0.074 |
| goals | 0.067 |
| assists | 0.050 |
| is_top_six | 0.017 |
| position=MF | 0.007 |
| position=DF | 0.005 |
| position=FW | 0.005 |
| position=GK | 0.001 |

## SHAP feature importance (mean |SHAP value|)

Real SHAP values from shap.TreeExplainer on the Random Forest model, fit on the full dataset. More rigorous than the Gini importances above since SHAP reflects each feature's actual average impact on individual predictions (in log-fee units), not just split frequency. Still descriptive of this one model, not a causal claim.

| Feature | Mean |SHAP value| |
|---|---|
| age_at_transfer | 0.2979 |
| minutes | 0.1567 |
| assists_per90 | 0.0976 |
| interceptions | 0.0762 |
| goals | 0.0731 |
| tackles | 0.0669 |
| goals_per90 | 0.0666 |
| assists | 0.0383 |
| is_top_six | 0.0219 |
| position=MF | 0.0041 |
| position=FW | 0.0035 |
| position=DF | 0.0028 |
| position=GK | 0.0010 |
