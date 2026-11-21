"""
Triangular-deviation mean-reversion forecaster.

A cross rate is pinned by its two USD legs. For example EURJPY should equal
EURUSD * USDJPY. The residual measures how far the quoted cross sits from that
synthetic value:

    residual = log(cross) - [ log(leg1) + sign2 * log(leg2) ]

sign2 is +1 when the legs MULTIPLY to the cross and -1 when they DIVIDE:

    EURJPY = EURUSD * USDJPY   ->  sign2 = +1
    EURGBP = EURUSD / GBPUSD   ->  sign2 = -1
    GBPJPY = GBPUSD * USDJPY   ->  sign2 = +1

The forecaster predicts the cross closes the whole gap by the next bar:
predicted next-day log return of the cross = -residual.

HONESTY NOTES
* Like the UIP adapter, this one holds data the evaluator does not pass (the
  two leg series). It only ever reads the legs AT the last observed bar's
  date, and tests/test_fx_forecasters.py proves that with poisoned future legs.
* Daily closes from a retail feed are not sampled at the same instant, so a
  residual may be timestamp noise rather than a tradable gap. Noise in the
  cross would also make "-residual" look predictive without being tradable.
  That is a property of the data, not something this class can fix; the study
  runner reports it and scores results net of cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster

#: The three preregistered triangles: yfinance tickers plus the leg-2 sign.
TRIANGLES = {
    "EURJPY": {"cross": "EURJPY=X", "leg1": "EURUSD=X", "leg2": "USDJPY=X", "sign2": +1},
    "EURGBP": {"cross": "EURGBP=X", "leg1": "EURUSD=X", "leg2": "GBPUSD=X", "sign2": -1},
    "GBPJPY": {"cross": "GBPJPY=X", "leg1": "GBPUSD=X", "leg2": "USDJPY=X", "sign2": +1},
}


def triangle_residual(cross, leg1, leg2, sign2: int):
    """log(cross) - [log(leg1) + sign2*log(leg2)]. Works on floats or Series."""
    if sign2 not in (+1, -1):
        raise ValueError("sign2 must be +1 (legs multiply) or -1 (legs divide).")
    return np.log(cross) - (np.log(leg1) + sign2 * np.log(leg2))


def pip_size(ticker: str) -> float:
    """One pip in price units: 0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    return 0.01 if ticker.replace("=X", "").upper().endswith("JPY") else 0.0001


def round_trip_cost(prices: dict, pips_per_leg: float = 1.5):
    """Cost of trading every leg of a triangle once round-trip, as a return.

    `prices` maps ticker -> price (float or Series). Crossing a spread of
    `pips_per_leg` on each instrument costs pips*pip_size/price of notional,
    summed over the instruments. Three instruments at 1.5 pips = 4.5 pips.
    """
    total = 0.0
    for ticker, price in prices.items():
        total = total + pips_per_leg * pip_size(ticker) / price
    return total


class TriangularForecaster(Forecaster):
    """Predicts the cross reverts fully to its synthetic value by the next bar.

    Parameters
    ----------
    leg1_close, leg2_close : pd.Series of closes for the two legs, indexed by
        date. Only the value at the last observed bar's date is ever read.
    sign2 : +1 if cross = leg1 * leg2, -1 if cross = leg1 / leg2.
    """

    name = "triangular"

    def __init__(self, leg1_close: pd.Series, leg2_close: pd.Series, sign2: int):
        if sign2 not in (+1, -1):
            raise ValueError("sign2 must be +1 or -1.")
        self._leg1 = leg1_close
        self._leg2 = leg2_close
        self.sign2 = int(sign2)
        #: One entry per forecast: the date read and the residual computed.
        self.access_log: list[dict] = []

    def _legs_at(self, asof: pd.Timestamp) -> tuple[float, float]:
        if asof not in self._leg1.index or asof not in self._leg2.index:
            raise ValueError(f"no leg prices on {asof.date()}")
        return float(self._leg1.loc[asof]), float(self._leg2.loc[asof])

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        asof = pd.Timestamp(history.index[-1])          # last OBSERVED bar
        cross = float(history["close"].iloc[-1])
        leg1, leg2 = self._legs_at(asof)
        resid = float(triangle_residual(cross, leg1, leg2, self.sign2))
        self.access_log.append({"asof": asof, "leg_date": asof, "residual": resid})
        # Full reversion by the next bar, then flat: every step sits at the
        # synthetic value implied by the legs as of `asof`.
        return np.full(horizon, cross * np.exp(-resid))
