# Preregistration: FX Studies A and B

STATUS: APPROVED by the repo owner on 2026-11-14, before any real data was downloaded or evaluated.

CONFIG HASH: `92ba24b805ca5e57`

Written 2026-11-14 on branch `fx-study`. Every choice here is declared before
any forecast is run on real data. `fx_study.py` refuses to run unless this file
says `STATUS: APPROVED` and its `CONFIG` dict still hashes to the value above,
so the configuration cannot change after approval without it being visible.

## What has and has not been looked at

- **Looked at:** the first and last observation date of each FRED rate series
  (to confirm the series exist and are still updated). Nothing else about them.
- **Run:** the unit and integrity tests, and one dry run of the runner's
  plumbing. Both used synthetic random data only.
- **Not looked at:** any real FX price, any real residual, any real result. No
  yfinance FX data has been downloaded.

## Study A -- Uncovered interest parity vs. random walk

**Hypothesis (default: null).** A forecast equal to the lagged interest
differential does not beat a random walk at 1, 21 or 63 trading days on any of
the six pairs. This is the expected result: the failure of UIP at short
horizons (the forward premium puzzle; Meese and Rogoff 1983) is one of the
best-known findings in international finance.

**Pairs.** EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF. Daily closes from
yfinance (`EURUSD=X` etc.) through the existing `load_ohlcv`.

**Date range.** 2004-01-01 through 2026-03-31. The end date is set by the
rates, not chosen: the euro-area and UK series end at January 2026, and under
the lag rule a January value is the newest usable rate through March 2026.

**Rates.** OECD 3-month interbank rates from FRED, monthly, percent per year:

| Currency | FRED series | Currency | FRED series |
|---|---|---|---|
| USD | IR3TIB01USM156N | AUD | IR3TIB01AUM156N |
| EUR | IR3TIB01EZM156N | CAD | IR3TIB01CAM156N |
| JPY | IR3TIB01JPM156N | CHF | IR3TIB01CHM156N |
| GBP | IR3TIB01GBM156N | | |

**Rate lag.** The value for month M is usable from the first business day of
month M+2. Reason: a monthly average cannot exist before month M ends, and the
OECD publishes some weeks after that. Two months guarantees at least one full
month between the end of the month described and first use. A 3-month rate
moves slowly, so the cost of being conservative is small. If the newest usable
value is more than 3 months old the forecast is skipped and counted, never
filled.

**Forecaster.** For a pair quoted BASE/QUOTE, the predicted price k trading
days ahead is `last_close * (1 + i_quote*k/252) / (1 + i_base*k/252)`. The
high-rate currency is predicted to depreciate.

**Horizons and overlap.** 1, 21 and 63 trading days, each scored at the full
horizon only. Windows are **non-overlapping**: the step between forecasts
equals the horizon, so no overlap correction is needed. Every eval point is
used (no subsampling). Lookback is 252 bars. Expected sample per pair: about
5,500 forecasts at 1 day, 260 at 21 days, 88 at 63 days.

**Baselines.** The existing `RandomWalkForecaster` and `DriftForecaster`, run
through the same engine on the same eval points.

## Study B -- Triangular deviation mean reversion

**Hypothesis (default: null).** Deviations of a cross from the value implied by
its two USD legs do not predict the next day's cross return in a way that
survives a realistic retail cost.

**Triangles and signs.** Residual = log(cross) - [log(leg1) + sign2*log(leg2)].

| Cross | leg1 | leg2 | sign2 | Identity |
|---|---|---|---|---|
| EURJPY | EURUSD | USDJPY | +1 | EURJPY = EURUSD x USDJPY |
| EURGBP | EURUSD | GBPUSD | -1 | EURGBP = EURUSD / GBPUSD |
| GBPJPY | GBPUSD | USDJPY | +1 | GBPJPY = GBPUSD x USDJPY |

**Date range.** 2004-01-01 through 2026-10-31, restricted to dates on which
the cross and both legs all have a bar. Rows are dropped, never filled.

**Forecaster.** Predicted next-day log return of the cross = -residual, read
at the last observed bar. Horizon 1, step 1, lookback 252. No demeaning of the
residual and no fitted reversion speed: both would be tuning.

**Cost.** 1.5 pips on each of the three instruments, 4.5 pips in total for one
round trip, converted to a return at each day's prices (a pip is 0.01 for
JPY-quoted pairs and 0.0001 otherwise).

**Order of work.** Stage 1 prints and saves the residual distribution, the
mean cost and the fraction of days with |residual| > cost, **before** any
forecast is evaluated. Stage 2 runs the walk-forward evaluation.

**Gross and net.** For each forecast, gross = sign(predicted return) x actual
next-day cross return, and net = gross - cost. Both are reported over all days
and over "trade days" where |residual| > cost.

## Metrics and pass rules

All skill metrics come from the existing `evaluation.summarize`, unchanged.
Significance level is 0.05 throughout.

| Metric | Pass rule |
|---|---|
| Directional accuracy | Above the best naive constant baseline, one-sided binomial p < 0.05 |
| RMSE ratio vs random walk | Below 0.98 |
| IC (Spearman) | Positive, with p < 0.05 |
| Net return (Study B only) | Mean net return on trade days > 0, one-sided t-test p < 0.05, at least 30 trade days |

**A cell passes** only if it passes all three forecast metrics, the same rule
as the framework's `verdict()`.

**Study A passes** only if at least 2 of the 18 pair-horizon cells pass and
the median RMSE ratio across the 18 cells is below 1.0. Otherwise it is a null.

**Study B passes** only if at least 2 of the 3 triangles pass all three
forecast metrics **and** the net-return rule. A triangle that passes gross but
fails net is reported as a null, labelled "gross pass vanishes net of cost".

The word "edge" will not be applied to any result that fails these rules.

## Known consequences of these rules, declared now

These are properties of the design, stated up front so they cannot be
presented as findings later.

- **The RMSE bar is close to unreachable here.** A daily interest differential
  is about 1 basis point against roughly 60 of daily volatility, so the UIP
  1-day RMSE ratio will sit at about 1.000 whether or not UIP is true. The
  same holds for Study B unless residuals are far larger than the cost. The
  0.98 threshold is kept because the metrics are not to be modified.
- **UIP is nearly a constant-direction forecaster.** Its predicted sign only
  changes when the rate differential changes sign, so its directional accuracy
  will often equal the naive baseline exactly.
- **Random walk scores 0% on direction by construction.** It predicts zero
  change, which never matches a nonzero move. Its row is for the error
  comparison only.
- **A significantly negative IC for UIP is not a pass.** It would be the
  forward premium puzzle (carry) showing up. It will be reported as such. No
  carry forecaster will be tested in this study, because proposing one after
  seeing that result would be post-hoc.
- **Study B can show a spurious gross result.** If the cross close is sampled
  at a different instant from the legs, the cross carries noise the legs do
  not, and -residual predicts the next cross return mechanically. The
  synthetic dry run reproduced exactly this with pure timestamp noise. That is
  why the pass rule is net of cost.

## Data-snooping and validity risks

- UIP and triangular parity have been tested for decades; the nulls are the
  prior, and a pass would need replication before it meant anything.
- The 18 Study A cells are not independent: five pairs share the dollar and
  the horizons are nested. "Expected by chance" counts are a lower bound.
- The interest differential is highly persistent, which inflates the apparent
  significance of an IC at longer horizons (Stambaugh bias).
- FRED serves revised rates, not the values originally published.
- yfinance FX closes are indicative quotes with no fixed sampling time and no
  volume; early years contain gaps and bad ticks.
- This is pseudo out-of-sample: forecasts are reconstructed after the fact
  under no-lookahead constraints, not logged before the outcomes existed.
- The forecasters are deterministic, so a second run cannot differ unless the
  data vendor revises history. The raw downloads are saved with each run.

## Deviations and exclusions (decided now)

- A pair or triangle with a data error at run time is excluded and noted, not
  replaced.
- No outlier filtering in the primary results. If any daily |log return|
  exceeds 10%, the count is reported and the affected cells are also shown
  with those eval points removed. The unfiltered result is the one that counts.
- No parameter, date, pair, horizon, lag or threshold changes after approval.
  Any forced deviation will be disclosed in the report.

## Publication commitment

Full tables for both studies go into `results/fx_study/` and a summary into
`REPORT_FX.md`, whatever the outcome, including nulls and any result that
contradicts the hypotheses above.
