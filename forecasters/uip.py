"""
Uncovered interest parity (UIP) forecaster.

UIP says the currency with the higher interest rate is expected to DEPRECIATE
by the interest differential. For a pair quoted BASE/QUOTE (price = units of
quote currency per one unit of base, e.g. EURUSD = dollars per euro):

    E[S(t+k)] / S(t) = (1 + i_quote * tau) / (1 + i_base * tau),   tau = k / 252

If the quote currency pays more, the pair is predicted to RISE (the quote
currency buys less). If the base currency pays more, it is predicted to fall.

HONESTY NOTE -- this adapter carries data the evaluator does not hand it.
The walk-forward engine only passes price history, so the interest-rate series
have to live inside the forecaster. That is exactly the "smuggled outside
data" path the README warns the engine cannot police. So the no-lookahead
guarantee for rates is enforced HERE, by a publication lag, and is tested in
tests/test_fx_forecasters.py with poisoned future rates. Every lookup is also
written to `access_log` so a test (or a sceptic) can audit what was used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


def rate_available_from(obs_date, lag_months: int = 2) -> pd.Timestamp:
    """First date on which a monthly rate observation may be used.

    FRED dates a monthly value at the first day of the month it describes
    (month M). The value cannot exist before month M has ended, and the OECD
    series used here are published some weeks after that. With lag_months=2 the
    value for month M becomes usable on the first business day of month M+2 --
    at least a full month after the month it describes closed.
    """
    month_start = pd.Timestamp(obs_date).to_period("M").to_timestamp()
    return pd.offsets.BDay().rollforward(month_start + pd.DateOffset(months=lag_months))


class UIPForecaster(Forecaster):
    """Predicts the next closes from the lagged interest differential alone.

    Parameters
    ----------
    base_rate, quote_rate : pd.Series
        Annualised interest rates in PERCENT (5.25 means 5.25%), indexed by
        observation month, for the pair's base and quote currencies.
    lag_months : publication lag; see `rate_available_from`.
    max_stale_months : if the newest usable observation describes a month more
        than this many months before the forecast date, predict() raises (the
        evaluator then skips that point) rather than forecast from a stale rate.
    periods_per_year : trading days per year used to scale the horizon.
    """

    name = "uip"

    def __init__(
        self,
        base_rate: pd.Series,
        quote_rate: pd.Series,
        lag_months: int = 2,
        max_stale_months: int = 3,
        periods_per_year: int = 252,
    ):
        self.lag_months = int(lag_months)
        self.max_stale_months = int(max_stale_months)
        self.periods_per_year = int(periods_per_year)
        self._base = self._prepare(base_rate)
        self._quote = self._prepare(quote_rate)
        #: One entry per rate lookup: which observation was used for which date.
        self.access_log: list[dict] = []

    def _prepare(self, rate: pd.Series):
        s = rate.dropna().sort_index()
        if s.empty:
            raise ValueError("Empty interest-rate series.")
        obs = pd.DatetimeIndex(s.index)
        avail = pd.DatetimeIndex([rate_available_from(d, self.lag_months) for d in obs])
        return obs, avail, s.to_numpy(dtype=float)

    def _lookup(self, prepared, asof: pd.Timestamp, which: str) -> float:
        obs, avail, values = prepared
        # Newest observation whose availability date is on or before `asof`.
        i = int(avail.searchsorted(asof, side="right")) - 1
        if i < 0:
            raise ValueError(f"no {which} rate published on or before {asof.date()}")
        stale = (asof.year - obs[i].year) * 12 + (asof.month - obs[i].month)
        if stale > self.max_stale_months:
            raise ValueError(
                f"{which} rate is {stale} months stale at {asof.date()} "
                f"(limit {self.max_stale_months})"
            )
        self.access_log.append({
            "asof": asof, "which": which, "obs_date": obs[i],
            "available_from": avail[i], "value": float(values[i]),
        })
        return float(values[i])

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        asof = pd.Timestamp(history.index[-1])          # last OBSERVED bar
        last_close = float(history["close"].iloc[-1])
        i_base = self._lookup(self._base, asof, "base") / 100.0
        i_quote = self._lookup(self._quote, asof, "quote") / 100.0

        tau = np.arange(1, horizon + 1) / self.periods_per_year
        log_ret = np.log1p(i_quote * tau) - np.log1p(i_base * tau)
        return last_close * np.exp(log_ret)
