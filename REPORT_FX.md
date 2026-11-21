# FX Studies A and B: Report

Run 2026-11-14, stamp `20261114-123956`, config hash `92ba24b805ca5e57`,
under [PREREG_FX.md](PREREG_FX.md) (approved before any real data was
downloaded). No configuration was changed after approval. Full tables, raw
forecasts, the downloaded data and the console log are in `results/fx_study/`.

## Verdicts

- **Study A (interest parity): NULL.** 0 of 18 pair-horizon cells pass the preregistered rule. UIP is indistinguishable from a random walk.
- **Study B (triangular mean reversion): NULL.** 0 of 3 triangles pass the forecast metrics even before cost, and all three have worse forecast error than a random walk.

Neither result is an edge.

## What was tested

**Study A.** A forecast equal to the lagged 3-month interest differential
(the high-rate currency is predicted to depreciate), on six dollar pairs at 1,
21 and 63 trading days, 2004-01-01 to 2026-03-31, against the existing
random-walk and drift baselines. Rates are OECD monthly interbank series from
FRED; month M's value is used from the first business day of month M+2.
Multi-day horizons use non-overlapping windows scored at the full horizon.

**Study B.** A forecast that each cross (EURJPY, EURGBP, GBPJPY) closes its
whole gap to the value implied by its two dollar legs by the next day,
2004-01-01 to 2026-10-30, with a round-trip cost of 1.5 pips on each of three
instruments.

Every forecast went through the unchanged walk-forward engine and every skill
metric through the unchanged `summarize`. Both forecasters pass the spy,
poisoned-future and cheater checks in `tests/test_fx_forecasters.py`.

## Study A results

A cell passes only if it clears all three: direction p < 0.05 above the best
naive baseline, RMSE ratio vs random walk < 0.98, and positive IC with p < 0.05.

| Pair | Horizon | n | Dir. acc. | Best naive | Dir. p | RMSE vs RW | RMSE vs drift | IC | IC p | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 1 | 5517 | 49.0% | 50.6% | 0.993 | 0.9999 | 0.997 | +0.008 | 0.573 | No |
| EURUSD | 21 | 262 | 48.5% | 50.4% | 0.752 | 0.9977 | 0.933 | +0.049 | 0.431 | No |
| EURUSD | 63 | 87 | 54.0% | 50.6% | 0.296 | 0.9924 | 0.830 | +0.123 | 0.257 | No |
| USDJPY | 1 | 5517 | 48.2% | 51.9% | 1.000 | 1.0002 | 0.999 | -0.014 | 0.308 | No |
| USDJPY | 21 | 262 | 48.5% | 50.0% | 0.711 | 1.0046 | 0.986 | -0.013 | 0.836 | No |
| USDJPY | 63 | 87 | 46.0% | 55.2% | 0.966 | 1.0139 | 0.952 | +0.012 | 0.911 | No |
| GBPUSD | 1 | 5529 | 50.8% | 50.7% | 0.399 | 0.9998 | 0.997 | +0.015 | 0.259 | No |
| GBPUSD | 21 | 263 | 48.3% | 52.5% | 0.922 | 0.9952 | 0.937 | +0.078 | 0.209 | No |
| GBPUSD | 63 | 87 | 56.3% | 51.7% | 0.227 | 0.9862 | 0.826 | +0.156 | 0.150 | No |
| AUDUSD | 1 | 4916 | 48.6% | 50.9% | 0.999 | 1.0000 | 0.997 | -0.008 | 0.554 | No |
| AUDUSD | 21 | 234 | 47.9% | 50.4% | 0.802 | 1.0005 | 0.947 | -0.001 | 0.986 | No |
| AUDUSD | 63 | 78 | 46.2% | 53.8% | 0.930 | 1.0016 | 0.847 | -0.025 | 0.829 | No |
| USDCAD | 1 | 5532 | 49.7% | 50.1% | 0.700 | 0.9999 | 0.998 | +0.014 | 0.302 | No |
| USDCAD | 21 | 263 | 51.3% | 50.2% | 0.379 | 0.9979 | 0.954 | +0.071 | 0.251 | No |
| USDCAD | 63 | 87 | 51.7% | 51.7% | 0.543 | 0.9948 | 0.867 | +0.152 | 0.160 | No |
| USDCHF | 1 | 5530 | 48.5% | 51.1% | 1.000 | 0.9999 | 0.997 | +0.003 | 0.825 | No |
| USDCHF | 21 | 263 | 50.2% | 50.2% | 0.525 | 0.9975 | 0.944 | +0.041 | 0.510 | No |
| USDCHF | 63 | 87 | 55.2% | 55.2% | 0.544 | 0.9899 | 0.829 | +0.105 | 0.332 | No |

- **Per metric:** 0 of 18 cells pass direction, 0 pass error, 0 pass IC. Chance alone would produce about 0.9 per metric.
- **Median RMSE ratio vs random walk:** 0.9998. The best single cell is 0.9862 (GBPUSD, 63 days), short of the 0.98 bar.
- **No carry signal either:** no cell has a significantly negative IC, so this sample does not show the forward premium puzzle in these terms.
- **Baselines:** drift also passes 0 of 18 and is worse than a random walk on error in every cell (1.002 to 1.195). Random walk scores about 0% on direction by construction, as declared.
- **Skipped forecasts:** none. No rate was ever stale.

Observations that do not meet the pass rule and are not claimed as findings:
the UIP error ratio is below 1.0 in 12 of 18 cells and the IC is positive in
13 of 18, strongest at 63 days. None is individually significant, the cells
are not independent, and the pattern was noticed after the fact. UIP's lower
error than drift reflects how poor drift is, not skill in UIP.

## Study B results

**Stage 1, before any evaluation: residuals against cost.**

| Triangle | Days | Median abs. residual | Mean cost | Share of days with abs. residual > cost | Share of residuals below zero | Days over 50 bp | Largest |
|---|---|---|---|---|---|---|---|
| EURJPY | 5921 | 1.0 bp | 3.7 bp | 6.8% | 86% | 15 | 3056 bp |
| EURGBP | 5918 | 1.3 bp | 4.1 bp | 11.0% | 86% | 27 | 1440 bp |
| GBPJPY | 5918 | 1.6 bp | 3.3 bp | 15.3% | 92% | 16 | 1622 bp |

The typical residual is about a third of the cost. Two features say these are
not tradable gaps. Residuals are negative on 86 to 92% of days, a steady
offset consistent with the cross being sampled at a different moment from its
legs. And the few residuals that are large are enormous (up to 30%), which is
a bad tick, not an arbitrage. Lag-1 autocorrelation is about zero in all
three.

**Stage 2, forecast metrics (next-day cross return).**

| Triangle | n | Dir. acc. | Best naive | Dir. p | RMSE vs RW | IC | IC p | Gross pass |
|---|---|---|---|---|---|---|---|---|
| EURJPY | 5668 | 52.2% | 51.6% | 0.187 | 1.305 | +0.019 | 0.155 | No |
| EURGBP | 5665 | 49.2% | 50.3% | 0.957 | 1.096 | +0.003 | 0.850 | No |
| GBPJPY | 5665 | 52.0% | 51.9% | 0.432 | 1.082 | +0.015 | 0.274 | No |

**Gross and net of cost.** Mean return per forecast, in basis points.

| Triangle | Gross, all days | Net, all days | Trade days | Gross, trade days | Net, trade days | Net p |
|---|---|---|---|---|---|---|
| EURJPY | +1.60 | -2.13 | 352 | +6.68 | +3.02 | 0.240 |
| EURGBP | -0.12 | -4.22 | 529 | +5.35 | +1.35 | 0.333 |
| GBPJPY | +0.97 | -2.36 | 828 | -0.83 | -4.08 | 0.880 |

No triangle passes the forecast metrics gross, so the net rule never comes
into play. For completeness: net return on trade days is positive but not
significant for two triangles and negative for the third. Two gross figures
have p below 0.05 on their own (EURJPY all days, 0.044; EURGBP trade days,
0.043). They are 2 of 6 uncorrected one-sided tests, neither survives cost,
and they are not part of the pass rule.

## Limitations

- **Data vintage.** FRED serves revised rates, not the values first published. yfinance history can be revised too; the exact files used are saved under `results/fx_study/data/`.
- **Unsynchronised closes.** yfinance FX closes are indicative quotes with no fixed sampling time. Study B's residuals mostly measure that mismatch, so Study B says little about real triangular gaps, which live at intraday frequency.
- **Bad ticks.** Daily moves over 10% appear in the data:
  - EURUSD and USDJPY on 2008-12-08 and 09 (a spike that reverses the next day, almost certainly a bad tick), and USDJPY on 2008-10-09.
  - USDCHF on 2015-01-16 (real: the Swiss franc floor was removed).
  - EURGBP on 2022-10-10 and 11 (another reversing spike).
  - GBPJPY on 2016-06-27 (after the Brexit vote).
  - No filtering was applied, as preregistered. Removing the affected eval points changes no Study A cell's pass/fail, and the error ratios barely move. Study B's error ratios above 1 are driven largely by these ticks.
- **AUDUSD starts 2006-05-16.** yfinance has no earlier data, so that pair has about 11% fewer forecasts. It was kept, not replaced.
- **Pseudo out-of-sample.** Forecasts were reconstructed after the fact under no-lookahead constraints, not logged before the outcomes existed.
- **Small samples at 63 days.** 78 to 87 non-overlapping windows per pair gives little power; an IC of 0.15 is not detectable at that size.
- **Dependence.** Five pairs share the dollar and the horizons are nested, so the 18 cells are far fewer than 18 independent tests.
- **Persistent predictor.** The interest differential moves slowly, which biases IC significance upward at long horizons. That bias works in UIP's favour and it still did not pass.
- **Thresholds.** As declared in the preregistration, the 0.98 RMSE bar is close to unreachable for a predictor this small relative to daily volatility. A null here means "no detectable skill under this framework's rule", not "UIP is false".
- **Provenance.** The run was made from uncommitted working files, so the commit recorded in `run_meta.json` is the branch's starting point. The config hash in that file is what ties the run to the preregistration.
- **No trading claim.** The engine scores forecasts. The net-of-cost figures in Study B are a simple one-day sign rule, not a strategy backtest.
