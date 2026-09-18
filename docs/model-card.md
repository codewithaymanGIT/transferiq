# Model Card: TransferIQ Player Valuation Model

## Model details

- **Model type**: Random Forest Regressor (scikit-learn), 200 trees, default depth
- **Target**: log(transfer fee in EUR millions); predictions are transformed back with `expm1`
- **Features**: minutes, goals, assists, goals/assists per 90, age at transfer, tackles, interceptions, is_top_six (real, objective club fact), position (one-hot)
- **Training data size**: 185 real, disclosed permanent transfers with matched prior-season stats, spanning the 2014-15 through 2022-23 transfer windows
- **Current version**: `random_forest_v2_positiongrouped_*` (name is historical -- see Limitations; the deployed model is a single blended model, not position-split)

## Intended use

- Illustrating a real, working ML valuation pipeline: ingestion, cleaning, feature engineering, explainability, calibrated uncertainty
- Directional, exploratory estimates of Premier League player transfer value for portfolio/demonstration purposes
- A worked example of honest model reporting -- what a small, real dataset can and cannot support

## Out of scope

- **Not** suitable for actual transfer negotiations, financial decisions, or any use where a wrong estimate has real financial consequences
- **Not** validated for leagues other than the Premier League, or for goalkeeper valuations specifically (13 training rows -- see Limitations)
- **Not** a market-value benchmark -- no real external market-value data source is currently integrated (see README)

## Training data

- Source: FBref (performance stats, defensive stats) and a Transfermarkt transfer-history export (disclosed permanent fees only -- loans and undisclosed fees are excluded, not imputed as zero)
- 185 of 735 considered transfers were matched to real prior-season stats; the other 550 were skipped for lacking a matched prior season, not silently dropped
- Squad and club data reflects the 2025-2026 season as the most recently loaded (kept current, not a stale historical snapshot)

## Evaluation

Two different validation methods are reported, because they measure different things and give a very different picture:

| Method | R² | MAE (EUR m) | What it measures |
|---|---|---|---|
| Random 5-fold CV | 0.140 | 11.84 | How well the model fits transfers drawn from all years, shuffled together |
| **Walk-forward (out-of-time)** | **0.020** | **12.67** | How well the model predicts transfers it has never seen the *year* of, trained only on the past -- the way it would actually be used |

See `docs/temporal-validation.md` for the full per-season breakdown and methodology.

**This gap is the single most important fact about this model.** Random cross-validation gives a materially more flattering number than out-of-time validation, because shuffling lets the model implicitly learn from transfer-fee inflation patterns across years that it would not have access to when predicting a genuinely future transfer. The out-of-time R²=0.020 is close to zero -- the model captures very little that generalizes forward in time, even though the same architecture looks meaningfully better under random CV. Any claim about this model's real-world predictive power should cite the out-of-time number, not the random-CV number.

## Known limitations

1. **Weak out-of-time predictive power** (R²=0.020 -- see above). This is the headline limitation, not a minor caveat.
2. **Small sample size** (185 rows). Even the random-CV R²=0.140 carries wide uncertainty; per-season out-of-time folds range from 13 to 35 rows, too few to trust individually.
3. **Position-specific models were tried and reverted.** Splitting into defensive/attacking models measurably underperformed the blended model (R²=0.112 and -0.002 respectively vs. 0.140 blended) -- documented in `generate_predictions.py`'s module docstring as a real negative result, not hidden.
4. **`transfer_year` was tried as a direct feature and made the model worse** in earlier testing, despite carrying real signal (see the out-of-time gap above) -- 185 rows isn't enough for the model to safely separate a temporal trend from noise.
5. **No real market-value benchmark.** The `market_values` table exists in the schema but is empty; there is currently no reliable free data source for live market valuations (see README).
6. **Goalkeepers are underrepresented** (13 training rows) and effectively unvalidated on their own.
7. **Prediction intervals are a single global width**, calibrated from CV residuals, not a per-player or per-position interval -- the data doesn't support finer-grained intervals yet.

## Ethical and fairness considerations

- The model is trained only on real, disclosed, publicly reported data (FBref, Transfermarkt) -- no personal, biometric, or non-public data is used
- `is_top_six` and `age_at_transfer` are real, objective facts, not proxies for protected characteristics; nationality is stored but never used as a model feature
- Confidence is always reported as "Low" and intervals are shown alongside every point estimate specifically to avoid presenting a single number as more certain than it is

## Retraining cadence

- Squad/club data (`current_club_id`) should be refreshed each transfer window by pulling the newest FBref season (see README) -- this is a data freshness operation, separate from retraining
- Model retraining (`generate_predictions.py`) should be re-run whenever the training matrix changes materially (a new season's transfers matched, a new feature added) -- there is currently no automated retraining schedule; this is a manual step
- Given the out-of-time result above, retraining alone will not fix the core limitation -- meaningfully more historical data (more matched seasons) is the most likely lever to improve real forward-looking accuracy
