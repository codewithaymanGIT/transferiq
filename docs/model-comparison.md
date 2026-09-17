# Model comparison

Generated 2026-09-16T20:39:44+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

Features used: minutes, goals, assists, goals_per90, assists_per90, age_at_transfer, tackles, interceptions, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 17.19 | 26.19 | 11.05 | -0.221 | 148 | 37 |
| Linear Regression | 15.98 | 23.11 | 9.46 | 0.049 | 148 | 37 |
| Ridge | 14.53 | 22.36 | 8.33 | 0.110 | 148 | 37 |
| Random Forest | 16.92 | 24.17 | 8.45 | -0.040 | 148 | 37 |
| XGBoost | 16.76 | 23.14 | 11.88 | 0.047 | 148 | 37 |

## Cross-validated results (5-fold)

Averaged over multiple folds rather than one fixed 80/20 split -- at this sample size a single split's R2 is unstable enough (observed swings of more than a full point from adding a single feature) that it should not be trusted alone. This is a steadier, though still rough, read on whether a feature genuinely helps.

| Model | Mean MAE (EUR m) | Mean RMSE (EUR m) | Mean R2 | R2 std dev |
|---|---|---|---|---|
| Median baseline | 13.38 | 19.18 | -0.094 | 0.071 |
| Linear Regression | 11.84 | 17.48 | 0.088 | 0.043 |
| Ridge | 11.47 | 17.35 | 0.096 | 0.049 |
| Random Forest | 11.84 | 16.88 | 0.148 | 0.120 |
| XGBoost | 12.78 | 17.33 | 0.055 | 0.290 |

## Random Forest feature importances

Fit on the full dataset (not a held-out split) purely to see which features this particular model leans on most -- descriptive, not a validated or causal claim, and not the same as the metrics above. SHAP explanations (planned) will give a more rigorous per-prediction breakdown later.

| Feature | Importance |
|---|---|
| age_at_transfer | 0.319 |
| minutes | 0.169 |
| assists_per90 | 0.127 |
| goals_per90 | 0.107 |
| interceptions | 0.079 |
| tackles | 0.069 |
| goals | 0.062 |
| assists | 0.048 |
| position=MF | 0.008 |
| position=FW | 0.006 |
| position=DF | 0.005 |
| position=GK | 0.001 |

## SHAP feature importance (mean |SHAP value|)

Real SHAP values from shap.TreeExplainer on the Random Forest model, fit on the full dataset. More rigorous than the Gini importances above since SHAP reflects each feature's actual average impact on individual predictions (in log-fee units), not just split frequency. Still descriptive of this one model, not a causal claim.

| Feature | Mean |SHAP value| |
|---|---|
| age_at_transfer | 0.2954 |
| minutes | 0.1588 |
| assists_per90 | 0.0964 |
| interceptions | 0.0778 |
| goals_per90 | 0.0752 |
| tackles | 0.0600 |
| goals | 0.0600 |
| assists | 0.0369 |
| position=FW | 0.0045 |
| position=MF | 0.0045 |
| position=DF | 0.0028 |
| position=GK | 0.0010 |
