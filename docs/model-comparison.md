# Model comparison

Generated 2026-09-11T20:55:49+00:00 by `scripts/train/train_baseline_models.py` -- every number below comes from an actual sklearn/xgboost fit on real data, never hand-typed.

> **Proof-of-concept only.** This run used **70 rows** -- far below what's needed for a trustworthy comparison. Treat every metric here as illustrating that the pipeline works, not as a real signal of model quality. Re-run after loading more historical seasons (see README).

Features used: minutes, goals, assists, goals_per90, assists_per90, position (one-hot).

| Model | MAE (EUR m) | RMSE (EUR m) | Median AE (EUR m) | R² | n_train | n_test |
|---|---|---|---|---|---|---|
| Median baseline | 9.06 | 10.16 | 10.85 | -0.056 | 56 | 14 |
| Linear Regression | 6.89 | 9.02 | 6.59 | 0.168 | 56 | 14 |
| Ridge | 7.17 | 9.01 | 6.59 | 0.169 | 56 | 14 |
| Random Forest | 8.42 | 9.82 | 8.13 | 0.013 | 56 | 14 |
| XGBoost | 12.70 | 15.28 | 11.19 | -1.390 | 56 | 14 |
