# Model comparison

Generated 2026-09-11T21:21:57+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

> **Negative result.** No model here beat a naive median-fee guess (every model's R2 is at or below the median baseline's R2). This is an honest finding, not a working valuation model -- treat it as evidence the current features/data aren't sufficient yet, not as real predictions.

Features used: minutes, goals, assists, goals_per90, assists_per90, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 11.27 | 16.99 | 7.70 | -0.044 | 102 | 26 |
| Linear Regression | 14.31 | 19.67 | 13.41 | -0.398 | 102 | 26 |
| Ridge | 14.37 | 19.82 | 13.37 | -0.420 | 102 | 26 |
| Random Forest | 14.02 | 19.22 | 12.79 | -0.335 | 102 | 26 |
| XGBoost | 15.58 | 21.32 | 10.97 | -0.643 | 102 | 26 |
