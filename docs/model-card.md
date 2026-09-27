# Model Validation Report: TransferIQ Transfer-Fee Model

Structured along the lines of a bank model-validation review: conceptual soundness, outcomes analysis, ongoing monitoring, and findings. Every figure below comes from a generated report in `docs/` (linked in each section), not from hand calculation.

## 1. Model summary

| Item | Detail |
|---|---|
| Purpose | Estimate the fee a Premier League player would command in a permanent transfer |
| Type | Random Forest regressor (scikit-learn), 200 trees, `random_state=42`, default depth |
| Target | log(1 + fee in EUR millions); predictions returned with `expm1` |
| Features | minutes, goals, assists, goals and assists per 90, age at transfer, tackles won, interceptions, `is_top_six`, position (one-hot) |
| Training data | 185 disclosed permanent transfers, each matched to the player's stats from the season before the transfer |
| Deployed version | `random_forest_v3_blended_<timestamp>`, a single model across all positions |
| Output | Point estimate, 5th to 95th percentile interval, confidence label (always "Low"), per-feature SHAP contributions |

## 2. Intended use and restrictions

- **Intended:** demonstrating an end-to-end, honestly validated valuation pipeline; directional, exploratory estimates.
- **Not for:** transfer negotiations, financial decisions, or any use where a wrong estimate has financial consequences.
- **Not validated for:** leagues other than the Premier League, or goalkeepers on their own (13 of the 185 training rows).
- **Not a market-value benchmark:** the `market_values` table exists in the schema but is empty, because no free, reliable source is integrated.

## 3. Data

- **Sources:** FBref for season and defensive stats and birth dates; the `ewenme/transfers` Transfermarkt export for fees (EUR millions). See `docs/data-sources/`.
- **Target population:** of 735 disclosed permanent transfers considered, 185 matched to prior-season stats. The other 550 were skipped because the prior season is not loaded, not imputed.
- **Exclusions:** loans, free transfers and undisclosed fees are excluded from the target, never recorded as zero. Players with no birth date are skipped, never given a guessed age.
- **Mid-season moves:** a player who changed club mid-season has one stats row per club. Counting stats are summed across stints, and club-level features come from the stint with the most minutes. This affects 12 of the 185 training rows.

## 4. Conceptual soundness

- **Leakage control:** a transfer in season S is joined only to stats from season S-1. This is enforced in `build_training_matrix.py` and covered by tests. In walk-forward folds, missing values in the test season are filled with the training seasons' median, never the test season's own.
- **Target transform:** fees are heavily right-skewed. Training on log fees stops a few very large transfers from dominating the fit.
- **Feature choice:** age is a well-documented driver of transfer fees. Defensive stats were added so defenders are not judged only on goals and assists. `is_top_six` is an objective fact about the club, not a reputation score.
- **Intervals:** bounds come from out-of-fold cross-validation residuals, not from the spread of the forest's trees, so they reflect how wrong the model actually was on held-out data.

## 5. Outcomes analysis

### 5.1 Benchmarking against simpler models (random 5-fold CV)

Source: [`model-comparison.md`](model-comparison.md).

| Model | Mean R² | R² std | Mean MAE (EUR m) |
|---|---|---|---|
| Median baseline | -0.094 | 0.071 | 13.38 |
| Linear Regression | 0.097 | 0.094 | 11.76 |
| Ridge | 0.117 | 0.057 | 11.32 |
| **Random Forest** | **0.144** | 0.116 | 11.73 |
| XGBoost | -0.005 | 0.178 | 13.12 |

Random Forest has the highest mean R², and Ridge has the lowest MAE with the tightest spread. The difference between them is inside the fold-to-fold spread, so choosing Random Forest over Ridge rests on its tree-based SHAP explanations, not on a clear accuracy advantage.

### 5.2 Out-of-time validation (walk-forward)

Source: [`temporal-validation.md`](temporal-validation.md). For each season, the model is trained on all earlier seasons only and tested on that season. The 152 held-out predictions are pooled.

| Model | Pooled R² | Pooled MAE (EUR m) |
|---|---|---|
| Median baseline, same folds | -0.193 | 14.24 |
| **Random Forest** | **0.025** | **12.57** |

Out-of-time R² (0.025) is far below random-CV R² (0.144). Shuffling seasons together lets the model learn fee inflation from years it would not yet have seen, so random CV overstates real forward-looking skill. The model still beats the naive baseline out of time on both R² and MAE, so the forward signal is small but real. Per-season R² ranges from -0.222 to 0.127 on 13 to 35 transfers each, so individual seasons should not be read on their own.

### 5.3 Uncertainty calibration

133 of 152 actual fees (87.5%) fell inside their 5th to 95th percentile interval, against a 90% target. At 90% true coverage, 136.8 of 152 would be expected with a standard deviation of 3.7, so 133 is about one standard deviation short. That is consistent with calibration, though slightly under. The intervals achieve this by being wide: the residual quantiles are -1.315 and +1.132 in log-fee space, so the upper bound is roughly 11 times the lower.

### 5.4 Where the model fails

Out of time, predictions span only EUR 3.2m to 38.94m, while actual fees run from EUR 0.11m to 117.5m.
- All 20 transfers under EUR 5m were overpredicted.
- None of the 23 transfers at EUR 40m or more was overpredicted. For those, the median prediction was EUR 18.79m against a median actual of EUR 55.0m.

The model pulls estimates toward the middle of the market, which is why R² stays near zero even though coverage is close to target. Expensive players are the most systematically undervalued.

## 6. Ongoing monitoring: feature stability

Source: [`stability-psi.md`](stability-psi.md). PSI compares each feature's live distribution (537 players with 2025-26 stats) against the training distribution, using training deciles, or exact values for discrete features.

| Feature | PSI | Reading |
|---|---|---|
| interceptions | 0.365 | significant shift |
| minutes | 0.290 | significant shift |
| assists | 0.273 | significant shift |
| tackles | 0.192 | moderate |
| age_at_transfer | 0.187 | moderate |
| assists_per90 | 0.157 | moderate |
| goals | 0.156 | moderate |
| goals_per90 | 0.118 | moderate |
| is_top_six | 0.032 | stable |

Thresholds are the usual rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant. Much of the shift is expected by design. Training rows are players later sold for a disclosed fee, who tend to be regular starters. The live population is every squad player with stats, including rotation players with few minutes. PSI therefore mixes this selection effect with any genuine market drift and cannot separate the two. That is itself a finding: the model is scoring many players unlike those it was trained on.

## 7. Findings

| # | Finding | Severity |
|---|---|---|
| 1 | Out-of-time R² is 0.025: weak forward-looking skill, though better than the naive baseline | High |
| 2 | Systematic compression: expensive transfers are underpredicted and cheap ones overpredicted | High |
| 3 | Live population differs from the training population (three features above PSI 0.25) | Medium |
| 4 | Small sample: 185 rows, 13 to 35 per test season, 13 goalkeepers | Medium |
| 5 | Interval coverage 87.5% against a 90% target, achieved with very wide bands and one global width | Low |
| 6 | Missing fee drivers: contract length, wages, injuries and selling-club context are not available | Medium |

## 8. Changes tested and reverted

- **Position-split models** (defensive DF/GK vs attacking MF/FW) scored CV R² of 0.112 and -0.002, both below the blended model's 0.140 at the time. Reverted.
- **`transfer_year` as a feature** made every model worse at this sample size. Dropped.
- **Mid-season stints:** these were previously scored separately in live predictions (some players valued twice) and chosen arbitrarily in training. Fixed by combining stints; the metrics above are after the fix.

## 9. Ethical and fairness considerations

- Only public, disclosed data is used. No personal, biometric or non-public data.
- Nationality is stored but never used as a feature.
- Every estimate is shown with its interval and a "Low" confidence label so a single number is never presented as more certain than it is.

## 10. Retraining and maintenance

- Club and squad data should be refreshed each transfer window by loading the newest FBref season. This is a data refresh, separate from retraining.
- Retrain (`generate_predictions.py`) when the training matrix changes materially, then re-run `temporal_validation.py` and `stability_psi.py`. There is no automated schedule.
- More matched historical seasons is the most likely way to improve out-of-time performance; retraining alone will not.
