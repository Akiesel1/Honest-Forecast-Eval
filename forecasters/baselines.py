"""
Naive baseline forecasters.

These are NOT throwaway code -- they are the yardstick. In this framework a
model only "works" if it significantly beats these. They double as the
reference implementation of the Forecaster interface: if a ~4-line class can
satisfy the contract cleanly, the contract is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class RandomWalkForecaster(Forecaster):
    """Predicts "no change": every future close equals the last observed close.

    This is the honest null hypothesis for prices. Equities behave close to a
    random walk at short horizons, so beating this baseline on *error* is
    genuinely hard and genuinely meaningful.
    """

    name = "random-walk"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        last_close = float(history["close"].iloc[-1])
        return np.full(horizon, last_close)


class DriftForecaster(Forecaster):
    """Extrapolates the average per-bar drift over the lookback window.

    A hair less naive than random walk: it assumes the recent mean return
    continues. It still has zero real forecasting skill -- it's just the mean
    return compounded forward -- which is exactly why it's a useful second
    yardstick. If a "smart" model can't beat the average, it has no edge.
    """

    name = "drift"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        close = history["close"].to_numpy(dtype=float)
        last_close = close[-1]
        rets = np.diff(close) / close[:-1]                 # simple per-bar returns
        mean_ret = float(np.mean(rets)) if len(rets) else 0.0
        steps = np.arange(1, horizon + 1)
        return last_close * (1.0 + mean_ret) ** steps      # compound the mean forward
