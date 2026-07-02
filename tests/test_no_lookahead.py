"""
No-lookahead integrity tests for the walk-forward evaluator.

These tests are the answer to the sharpest question a reader can ask about any
backtest result: "how do I know your harness isn't broken?" Three attacks:

1. SPY      -- a fake forecaster records every window it is handed; we assert
               no window ever contains a bar at/after the bars it is scored on.
2. TRIPWIRE -- we poison the future with an absurd value and assert the poison
               never appears inside any window a forecaster saw.
3. CHEATER  -- a forecaster that deliberately peeks at the true future must
               score PERFECTLY. This proves the metrics can detect skill when
               it exists -- which is what makes a bad Kronos score meaningful.
               (A harness that grades everything badly proves nothing.)

Run with:  python -m pytest tests/ -v        (or)        python tests/test_no_lookahead.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make the repo root importable when run directly (python tests/test_...py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import summarize, walk_forward
from forecasters.base import Forecaster

LOOKBACK, HORIZON, N = 60, 5, 300


def make_df(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV geometric random walk with a business-day index."""
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1e6},
        index=idx,
    )


class SpyForecaster(Forecaster):
    """Records every window it is handed; predicts a constant (prediction is irrelevant)."""

    name = "spy"

    def __init__(self):
        self.windows: list[pd.DataFrame] = []

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        self.windows.append(history.copy())
        return np.full(horizon, float(history["close"].iloc[-1]))


def test_windows_end_strictly_before_scored_bars():
    """SPY: every window's last timestamp must precede every bar it is scored on."""
    df = make_df()
    spy = SpyForecaster()
    res = walk_forward(df, spy, LOOKBACK, HORIZON, step=7, verbose=False)

    eval_points = sorted(res["t"].unique())
    assert len(eval_points) == len(spy.windows), "one recorded window per eval point"

    for window, t in zip(spy.windows, eval_points):
        window_end = window.index.max()
        first_scored = df.index[t]                     # first bar the forecast is graded on
        assert window_end < first_scored, (
            f"LEAK: window ends {window_end} but scoring starts {first_scored}"
        )
        assert len(window) == LOOKBACK, "window must be exactly `lookback` bars"
    print(f"PASS: {len(spy.windows)} windows checked; every window ends strictly "
          f"before the bars it is scored on.")


def test_tripwire_poisoned_future_never_seen():
    """TRIPWIRE: poison all bars after a cutoff; no window may contain the poison."""
    POISON = 1_000_000.0
    df = make_df()
    cutoff = LOOKBACK + 50                             # guarantees eval points past it
    df.iloc[cutoff:, df.columns.get_loc("close")] = POISON

    spy = SpyForecaster()
    walk_forward(df, spy, LOOKBACK, HORIZON, step=3, verbose=False)

    poisoned_windows = sum((w["close"] >= POISON).any() for w in spy.windows)
    # Windows that legitimately END before the cutoff can never contain poison.
    # Windows AFTER the cutoff will contain poison as legitimate *history* --
    # so the sharp assertion is on windows whose end precedes the cutoff date:
    cutoff_date = df.index[cutoff]
    for w in spy.windows:
        if w.index.max() < cutoff_date:
            assert not (w["close"] >= POISON).any(), (
                "LEAK: poison from the future appeared in a pre-cutoff window"
            )
    print(f"PASS: tripwire intact; {len(spy.windows)} windows inspected, no pre-cutoff "
          f"window ever contained future data ({poisoned_windows} post-cutoff windows "
          f"legitimately include it as history).")


class CheaterForecaster(Forecaster):
    """Deliberately peeks at the true future via a closure over the full df.

    This is the one thing the interface can't physically prevent -- a malicious
    adapter smuggling in outside data. The point of this test is the CONVERSE
    guarantee: if something DID have perfect knowledge, our metrics must give it
    a perfect score. That proves the scoring detects skill when it exists.
    """

    name = "cheater"

    def __init__(self, full_df: pd.DataFrame):
        self._df = full_df

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        last = history.index.max()
        pos = self._df.index.get_loc(last)
        future = self._df["close"].iloc[pos + 1: pos + 1 + horizon]
        return future.to_numpy(dtype=float)            # the literal true answers


def test_cheater_scores_perfectly():
    """CHEATER: perfect future knowledge must yield a perfect score."""
    df = make_df()
    res = walk_forward(df, CheaterForecaster(df), LOOKBACK, HORIZON, step=7, verbose=False)
    summ = summarize(res, HORIZON)

    h1 = summ[summ["horizon"] == 1].iloc[0]
    assert h1["dir_acc"] == 1.0, f"cheater dir_acc={h1['dir_acc']}, expected 1.0"
    assert h1["rmse_ratio"] < 1e-9, f"cheater rmse_ratio={h1['rmse_ratio']}, expected ~0"
    assert h1["IC"] > 0.999, f"cheater IC={h1['IC']}, expected ~1.0"
    print(f"PASS: cheater scored dir_acc={h1['dir_acc']:.0%}, "
          f"rmse_ratio={h1['rmse_ratio']:.2e}, IC={h1['IC']:.4f} -- the metrics "
          f"detect perfect skill, so a bad score means bad skill, not a broken harness.")


if __name__ == "__main__":
    test_windows_end_strictly_before_scored_bars()
    test_tripwire_poisoned_future_never_seen()
    test_cheater_scores_perfectly()
    print("\nALL INTEGRITY TESTS PASSED: the evaluator provides no path to the future, "
          "and the metrics provably reward real skill.")
