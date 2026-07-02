# honest-forecast-eval

**Does an AI price-forecasting model actually predict anything? This framework finds out — and for the Kronos foundation model on daily equities, the answer is no.**

Most backtests are built to make a model look good. This one is built to make a model prove itself: walk-forward evaluation with strict no-lookahead guarantees, naive baselines every model must beat, significance tests on every metric, and a verdict that defaults to "no edge" until the evidence says otherwise.

## Headline finding

[Kronos](https://github.com/shiyu-coder/Kronos) is an open-source foundation model for financial candlesticks. Evaluated on ~4 years of daily bars across 12 diverse tickers (indices, mega-cap tech, financials, energy, staples, crypto), with ~150 walk-forward forecasts per ticker, **Kronos-small showed no forecasting edge**:

| Question | Result (run 1) | Result (run 2) |
|---|---|---|
| Beat the naive direction baseline (p<0.05)? | **0 / 12 tickers** | **0 / 12 tickers** |
| Beat a random walk on forecast error? | **0 / 12** | **0 / 12** |
| Statistically significant rank correlation (IC)? | **0 / 12** | **0 / 12** |
| Median RMSE vs random walk | **2.14x worse** | **2.14x worse** |

Two details worth more than the table:

- **The error result replicates almost exactly** (2.138x vs 2.144x across independent runs). Kronos's point forecasts are consistently ~2–4x noisier than simply predicting "no change."
- **The IC result demonstrates why single backtests lie.** Run 1 produced a suggestive median IC of +0.11, concentrated in correlated tech names. Run 2 — same code, same tickers — produced −0.04. The "signal" flipped sign because it was never signal. Noise wobbles between runs; real findings repeat. This framework is designed to catch exactly that.

None of this means Kronos is useless for every task, horizon, or asset class. It means: on daily equity bars, under an evaluation it cannot cheat, it demonstrates no predictive skill — which is also the correct prior for *any* model on near-random-walk data. Extraordinary claims should have to survive this harness.

## Why trust this harness?

Because it's tested against the two ways backtests lie:

1. **No path to the future.** `tests/test_no_lookahead.py` uses a spy forecaster to record every data window the evaluator hands out, and a poisoned-future tripwire, to prove no forecaster ever sees a bar at or after the ones it's scored on.
2. **The metrics provably detect skill.** A deliberate "cheater" forecaster with perfect future knowledge scores exactly 100% direction / 0.0 RMSE / IC 1.0. The grading rewards real skill — so a bad grade means bad skill, not a broken harness.

Every directional accuracy is reported against the *best* naive constant baseline (not a lazy 50%), with a binomial p-value. Every IC carries its own significance test. The multi-ticker aggregate states how many "wins" pure chance would produce, and flags that overlapping tickers are not independent tests.

## Quickstart

```bash
pip install -r requirements.txt

# Fast sanity check with a naive baseline (seconds, no GPU needed):
python run.py --ticker NVDA --model drift

# The real question (downloads Kronos weights from HuggingFace on first run):
python run.py --ticker NVDA --model kronos --model-pkg-dir .

# Full multi-ticker sweep with honest aggregate verdict:
python sweep.py --model kronos --model-pkg-dir .

# Integrity tests:
python tests/test_no_lookahead.py
```

## Evaluate your own model

The framework is model-agnostic. Implement one method:

```python
from forecasters.base import Forecaster

class MyModel(Forecaster):
    name = "my-model"
    def predict(self, history, horizon):
        # history: OHLCV DataFrame (DatetimeIndex, oldest->newest)
        # return: np.ndarray of the next `horizon` predicted CLOSE prices
        ...
```

Register it in `run.py`'s `build_forecaster()` and it drops into the identical walk-forward evaluation, baselines, and significance tests as everything else. If it has real edge, this harness will find it — and if it doesn't, this harness will say so.

## How the evaluation works

At each of ~150 points spread across history, the evaluator hands the model only the `lookback` bars *ending the day before*, asks for the next `horizon` closes, and only then compares against what actually happened. Metrics per horizon:

- **Directional accuracy vs best naive baseline** (always-up / always-down / coin-flip, whichever is strongest) with a one-sided binomial p-value.
- **RMSE ratio vs random walk** — error relative to predicting "no change." Below ~0.98 counts as a beat; 1.0 is parity.
- **Information Coefficient** (Spearman rank correlation of predicted vs actual returns) with its p-value.
- **A blunt verdict** that requires agreement across metrics and defaults to the null.

The sweep aggregates across tickers and asks the question single-ticker backtests dodge: did more tickers pass than chance alone predicts?

## Honest limitations

- **Forecast accuracy, not trading profitability.** No transaction costs, slippage, or position sizing are modeled. (Though a model that can't out-forecast "no change" has nothing for a trading strategy to monetize.)
- **The no-lookahead tests guarantee the evaluator's contract**, not the internals of third-party adapters — an adapter could in principle smuggle in outside data, as the cheater test itself demonstrates. Adapters are short and auditable by design.
- **Synthesized future timestamps** (business-day calendar) can differ from real trading calendars around holidays; the effect on Kronos's calendar features is negligible but nonzero.
- **One period, one frequency.** Results are for daily bars over ~2022–2026. Different horizons, frequencies, or regimes are open questions — which the framework exists to answer.

## Project structure

```
forecasters/   # the Forecaster interface + adapters (kronos, random-walk, drift)
evaluation/    # walk-forward engine + metrics + verdicts (model-agnostic)
data/          # OHLCV loading (yfinance)
tests/         # no-lookahead + metric-integrity tests
run.py         # single-ticker evaluation CLI
sweep.py       # multi-ticker sweep with honest aggregate
```

## Disclaimer

This is an educational evaluation tool. Nothing here is investment advice, and no result from this framework should be read as a recommendation to buy or sell anything.
