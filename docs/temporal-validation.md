# Temporal (Walk-Forward) Validation

Out-of-time validation, distinct from the random 5-fold CV reported in
`model-comparison.md`. Trains only on seasons strictly before the test
season, simulating how the model is actually used: predicting fees for
transfers that have not happened yet, using only what was known before.

## Per-season breakdown

Early folds train on very little data (a handful of prior seasons) and
each test season alone is small (13-35 rows) -- individual per-season R2
swings a lot and should not be over-interpreted on its own. The pooled
metric below, combining every held-out prediction across all folds, is
the more reliable headline number.

| Test season | Train rows | Test rows | R2 | MAE (EUR m) |
|---|---|---|---|---|
| 2016-2017 | 33 | 24 | -0.113 | 8.90 |
| 2017-2018 | 57 | 25 | -0.222 | 15.50 |
| 2018-2019 | 82 | 13 | 0.095 | 13.32 |
| 2019-2020 | 95 | 20 | 0.114 | 10.81 |
| 2020-2021 | 115 | 16 | -0.040 | 11.46 |
| 2021-2022 | 131 | 19 | 0.127 | 15.03 |
| 2022-2023 | 150 | 35 | 0.013 | 12.87 |

## Pooled out-of-time result

- **R2: 0.025** across all 152 held-out predictions combined
- **MAE: 12.57m EUR**

Median-fee baseline under the same walk-forward folds (predicts the
training seasons' median fee for every test transfer):

| Model | Pooled R2 | Pooled MAE (EUR m) |
|---|---|---|
| Median baseline | -0.193 | 14.24 |
| Random Forest | 0.025 | 12.57 |

- **Interval coverage: 133/152 = 87.5%** of actual fees fell inside
  their 5th-95th percentile interval (a well-calibrated interval would hold about 90%).
  Quantiles are computed from each fold's training seasons only.
  Held-out rows are saved to `heldout_predictions.csv`.

For comparison, the random 5-fold CV reported in `model-comparison.md`
gives R2=0.140 on the same 185-row dataset. The gap between the two
(if any) is itself informative: a materially worse out-of-time result
would mean the model is leaning on patterns that don't hold up when
tested the way it will actually be used, which random shuffling can
hide.

No test-fold statistic (e.g. median for imputation) is ever computed
from the test fold itself -- only from the training fold available at
that point in time, to avoid leaking future information backward.
