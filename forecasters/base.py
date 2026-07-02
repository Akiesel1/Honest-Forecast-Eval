"""
The Forecaster interface: the single seam that makes this a *framework*
instead of a Kronos script.

Everything downstream -- the walk-forward loop, the metrics, the report --
talks to a Forecaster ONLY through this contract. It never knows, and never
needs to know, whether the forecaster is Kronos, a naive baseline, or some
model you write next year. That decoupling is the entire point: swap the
model, keep the (honest) evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Abstract base class every model adapter must implement.

    The contract is deliberately tiny -- one method, two arguments, one return
    type -- so that anything from a one-line baseline to a large neural net can
    satisfy it. Any model-specific configuration (temperature, sample count,
    device, model size, ...) belongs in the concrete class's __init__, NOT in
    predict(). That way the evaluator constructs a forecaster once and then
    calls .predict() identically, blind to what kind of model it holds.
    """

    #: Human-readable label used in reports and plot legends, e.g. "kronos-small".
    #: Override in every subclass.
    name: str = "unnamed-forecaster"

    @abstractmethod
    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        """Forecast the next `horizon` CLOSE prices.

        Parameters
        ----------
        history : pd.DataFrame
            The lookback window. Columns are OHLCV
            (open, high, low, close, volume), indexed by a DatetimeIndex,
            oldest row first and newest row last. The most recent observed
            close is ``history["close"].iloc[-1]``.

            Any timestamps a model needs are derived from this index -- they
            are NOT passed as separate arguments. Models that don't care about
            dates (e.g. baselines) simply ignore the index. This keeps each
            model's quirks encapsulated inside its own adapter.
        horizon : int
            Number of steps (bars) ahead to forecast.

        Returns
        -------
        np.ndarray
            A 1-D array of length ``horizon`` holding the predicted CLOSE price
            for each step h = 1..horizon, in order.

            Prices, not returns. The evaluator converts prices to returns
            itself, so the return math lives in exactly one place and every
            model speaks the same language.
        """
        raise NotImplementedError
