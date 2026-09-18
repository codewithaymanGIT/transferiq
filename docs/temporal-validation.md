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
| 2016-2017 | 33 | 24 | -0.174 | 9.04 |
| 2017-2018 | 57 | 25 | -0.219 | 15.65 |
| 2018-2019 | 82 | 13 | 0.095 | 13.51 |
| 2019-2020 | 95 | 20 | 0.099 | 10.82 |
| 2020-2021 | 115 | 16 | -0.062 | 11.61 |
| 2021-2022 | 131 | 19 | 0.097 | 15.30 |
| 2022-2023 | 150 | 35 | 0.047 | 12.83 |

## Pooled out-of-time result

- **R2: 0.020** across all 152 held-out predictions combined
- **MAE: 12.67m EUR**

For comparison, the random 5-fold CV reported in `model-comparison.md`
gives R2=0.140 on the same 185-row dataset. The gap between the two
(if any) is itself informative: a materially worse out-of-time result
would mean the model is leaning on patterns that don't hold up when
tested the way it will actually be used, which random shuffling can
hide.

No test-fold statistic (e.g. median for imputation) is ever computed
from the test fold itself -- only from the training fold available at
that point in time, to avoid leaking future information backward.
