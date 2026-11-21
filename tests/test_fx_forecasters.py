"""
Integrity tests for the two FX forecasters (UIP and triangular).

The existing tests in test_no_lookahead.py prove the EVALUATOR never hands a
forecaster the future. They cannot see inside an adapter, and both FX adapters
carry side data the evaluator does not pass (interest rates; triangle legs).
So each adapter is put through the same three attacks, aimed at its side data:

1. SPY      -- wrap the adapter in the existing SpyForecaster. Every window
               still ends strictly before the scored bars, AND the adapter's
               own access log shows it only read side data dated at or before
               the window's last bar.
2. TRIPWIRE -- poison the future (prices AND side data). Forecasts made before
               the cutoff must be bit-for-bit unchanged. A deliberately leaky
               variant must FAIL this check, which proves the tripwire has teeth.
3. CHEATER  -- in a synthetic world where the adapter's theory is exactly true,
               it must score perfectly. A bad score on real data then means the
               theory failed, not that the adapter or metrics are broken.

Plus unit tests for the sign conventions and the cost arithmetic.

Everything here is synthetic. No network, no real market data.

Run with:  python -m pytest tests/ -v     (or)     python tests/test_fx_forecasters.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)                    # so the existing test helpers import

from evaluation import summarize, walk_forward
from forecasters.triangular import (
    TRIANGLES,
    TriangularForecaster,
    pip_size,
    round_trip_cost,
    triangle_residual,
)
from forecasters.uip import UIPForecaster, rate_available_from
from test_no_lookahead import HORIZON, LOOKBACK, N, SpyForecaster, make_df

POISON = 1_000_000.0
# Bar 120 of the synthetic calendar is 2024-06-17: deliberately MID-month, so a
# forecaster that reads a month's rate before that month has ended gets caught.
CUTOFF = LOOKBACK + 60


class SpyOn(SpyForecaster):
    """The existing spy, but it returns the wrapped adapter's real forecast."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.name = f"spy+{inner.name}"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        super().predict(history, horizon)             # records the window
        return self.inner.predict(history, horizon)


def _assert_windows_precede_scoring(df, spy, res):
    eval_points = sorted(res["t"].unique())
    assert len(eval_points) == len(spy.windows), "one recorded window per eval point"
    for window, t in zip(spy.windows, eval_points):
        assert window.index.max() < df.index[t], "LEAK: window reaches the scored bars"


def _pre_cutoff_preds(df, forecaster, cutoff_date) -> pd.Series:
    """1-step predicted returns for every forecast made strictly before the cutoff."""
    res = walk_forward(df, forecaster, LOOKBACK, 1, step=1, max_evals=10_000, verbose=False)
    pred = res.set_index("t")["pred_ret"]
    keep = [t for t in pred.index if df.index[t - 1] < cutoff_date]
    assert len(keep) > 20, "need a meaningful number of pre-cutoff forecasts"
    return pred.loc[keep]


# --------------------------------------------------------------------------- UIP

def make_rates(seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Synthetic monthly rates (percent), covering the synthetic price calendar."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", "2025-06-01", freq="MS")
    base = pd.Series(2.0 + rng.normal(0, 1.0, len(idx)), index=idx)
    quote = pd.Series(3.0 + rng.normal(0, 1.0, len(idx)), index=idx)
    return base, quote


def test_rate_lag_rule():
    """Month M's value is usable from the first business day of month M+2."""
    assert rate_available_from("2024-01-01", 2) == pd.Timestamp("2024-03-01")   # a Friday
    assert rate_available_from("2024-04-01", 2) == pd.Timestamp("2024-06-03")   # 1 Jun is a Saturday
    assert rate_available_from("2024-04-17", 2) == pd.Timestamp("2024-06-03")   # any day in the month
    print("PASS: rate lag rule -- month M usable from first business day of M+2.")


def test_uip_sign_convention():
    """The high-rate currency is predicted to depreciate, scaled to the horizon."""
    idx = pd.date_range("2023-01-01", "2025-06-01", freq="MS")
    low, high = pd.Series(1.0, index=idx), pd.Series(5.0, index=idx)
    df = make_df()
    window = df.iloc[:LOOKBACK]
    last = float(window["close"].iloc[-1])

    # Quote currency pays more -> quote depreciates -> BASE/QUOTE price rises.
    up = UIPForecaster(base_rate=low, quote_rate=high).predict(window, 252)
    assert np.all(up > last) and np.all(np.diff(up) > 0)
    assert abs(up[-1] / last - 1.05 / 1.01) < 1e-12, "one year ahead = ratio of gross rates"
    assert abs(up[20] / last - (1 + 0.05 * 21 / 252) / (1 + 0.01 * 21 / 252)) < 1e-12

    # Base currency pays more -> base depreciates -> price falls.
    down = UIPForecaster(base_rate=high, quote_rate=low).predict(window, 252)
    assert np.all(down < last)

    # Equal rates -> no predicted change.
    flat = UIPForecaster(base_rate=low, quote_rate=low).predict(window, 5)
    assert np.allclose(flat, last)
    print("PASS: UIP sign convention -- high-rate currency predicted to depreciate.")


def test_uip_spy_only_reads_published_rates():
    """SPY: windows precede scoring, and every rate used was already public."""
    df = make_df()
    base, quote = make_rates()
    inner = UIPForecaster(base, quote)
    spy = SpyOn(inner)
    res = walk_forward(df, spy, LOOKBACK, HORIZON, step=7, verbose=False)

    _assert_windows_precede_scoring(df, spy, res)
    assert len(inner.access_log) == 2 * len(spy.windows), "one base + one quote lookup per forecast"
    for entry in inner.access_log:
        month_end = entry["obs_date"] + pd.offsets.MonthEnd(0)
        assert entry["available_from"] <= entry["asof"], "LEAK: rate used before its availability date"
        assert month_end < entry["asof"], "LEAK: rate used before the month it describes had ended"
    print(f"PASS: UIP spy -- {len(spy.windows)} windows, {len(inner.access_log)} rate lookups, "
          f"every rate was published before the forecast date.")


def _uip_poison_leaks(lag_months: int) -> bool:
    """True if poisoning the future changes any pre-cutoff UIP forecast."""
    df = make_df()
    base, quote = make_rates()
    cutoff_date = df.index[CUTOFF]

    dirty_df = df.copy()
    dirty_df.iloc[CUTOFF:, dirty_df.columns.get_loc("close")] = POISON
    # Poison every rate whose month had not fully ended before the cutoff:
    # nobody standing before the cutoff could have known those values.
    dirty_base, dirty_quote = base.copy(), quote.copy()
    unknowable = (base.index + pd.offsets.MonthEnd(0)) >= cutoff_date
    dirty_base[unknowable] = 500.0      # absurd but still a valid rate (percent)
    dirty_quote[unknowable] = 900.0

    clean = _pre_cutoff_preds(df, UIPForecaster(base, quote, lag_months=lag_months), cutoff_date)
    dirty = _pre_cutoff_preds(dirty_df, UIPForecaster(dirty_base, dirty_quote, lag_months=lag_months),
                              cutoff_date)
    return not np.array_equal(clean.to_numpy(), dirty.to_numpy())


def test_uip_tripwire_poisoned_future_never_used():
    """TRIPWIRE: poisoned future prices and rates leave pre-cutoff forecasts unchanged."""
    assert not _uip_poison_leaks(lag_months=2), "LEAK: UIP forecast changed when the future was poisoned"
    print("PASS: UIP tripwire -- pre-cutoff forecasts are identical with the future poisoned.")


def test_uip_tripwire_catches_a_leaky_variant():
    """The tripwire must FAIL a forecaster that reads a month's rate during that month."""
    assert _uip_poison_leaks(lag_months=0), "tripwire has no teeth: it missed a zero-lag forecaster"
    print("PASS: UIP tripwire has teeth -- a zero-lag variant is caught.")


def _reference_rate(series: pd.Series, d: pd.Timestamp) -> float:
    """Independent re-implementation of the lag rule (numpy business days)."""
    best = None
    for obs, val in series.items():
        shifted = (obs + pd.DateOffset(months=2)).date()
        avail = pd.Timestamp(np.busday_offset(np.datetime64(shifted), 0, roll="forward"))
        if avail <= d:
            best = val
    return best


def test_uip_scores_perfectly_when_uip_holds():
    """CHEATER: in a world where UIP is exactly true, the UIP forecaster is perfect."""
    base, quote = make_rates(seed=3)
    idx = pd.bdate_range("2024-01-01", periods=N)
    price = np.empty(N)
    price[0] = 100.0
    for k in range(N - 1):
        ib = _reference_rate(base, idx[k]) / 100.0
        iq = _reference_rate(quote, idx[k]) / 100.0
        price[k + 1] = price[k] * (1 + iq / 252) / (1 + ib / 252)
    df = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6},
                      index=idx)

    res = walk_forward(df, UIPForecaster(base, quote), LOOKBACK, 1, step=1,
                       max_evals=10_000, verbose=False)
    h1 = summarize(res, 1).iloc[0]
    assert h1["dir_acc"] == 1.0, f"dir_acc={h1['dir_acc']}"
    assert h1["rmse_ratio"] < 1e-9, f"rmse_ratio={h1['rmse_ratio']}"
    # Not > 0.999: rates are monthly, so forecasts within a month are TIED, and
    # float noise at the 1e-16 level breaks those ties differently in the
    # predicted and actual columns. Across months the ranking is exact.
    assert h1["IC"] > 0.99, f"IC={h1['IC']}"
    print(f"PASS: UIP in a UIP world scored dir_acc={h1['dir_acc']:.0%}, "
          f"rmse_ratio={h1['rmse_ratio']:.2e}, IC={h1['IC']:.4f}.")


# -------------------------------------------------------------------- triangular

def test_triangle_sign_conventions():
    """Each preregistered triangle has residual 0 at consistent prices, right sign otherwise."""
    px = {"EURUSD=X": 1.10, "USDJPY=X": 150.0, "GBPUSD=X": 1.25}
    fair = {"EURJPY": 1.10 * 150.0, "EURGBP": 1.10 / 1.25, "GBPJPY": 1.25 * 150.0}
    assert set(TRIANGLES) == set(fair)

    for name, tri in TRIANGLES.items():
        l1, l2 = px[tri["leg1"]], px[tri["leg2"]]
        assert abs(triangle_residual(fair[name], l1, l2, tri["sign2"])) < 1e-12, (
            f"{name}: residual should be zero at arbitrage-free prices")
        rich = triangle_residual(fair[name] * 1.01, l1, l2, tri["sign2"])
        assert abs(rich - np.log(1.01)) < 1e-12, f"{name}: cross 1% rich -> residual +log(1.01)"
        # The WRONG sign must not look arbitrage-free -- guards against a config typo.
        assert abs(triangle_residual(fair[name], l1, l2, -tri["sign2"])) > 0.1, name

    assert TRIANGLES["EURJPY"]["sign2"] == +1
    assert TRIANGLES["EURGBP"]["sign2"] == -1
    assert TRIANGLES["GBPJPY"]["sign2"] == +1
    print("PASS: triangle sign conventions -- EURJPY(+), EURGBP(-), GBPJPY(+).")


def test_triangular_predicts_minus_residual():
    """Predicted next-day log return of the cross is exactly -residual."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    leg1 = pd.Series([1.10, 1.10, 1.10], index=idx)
    leg2 = pd.Series([150.0, 150.0, 150.0], index=idx)
    for mult in (1.01, 0.99, 1.0):
        cross = 165.0 * mult
        hist = pd.DataFrame({"open": cross, "high": cross, "low": cross, "close": cross,
                             "volume": 0.0}, index=idx)
        pred = TriangularForecaster(leg1, leg2, +1).predict(hist, 3)
        assert np.allclose(np.log(pred / cross), -np.log(mult)), "predicted return must be -residual"
        assert np.allclose(pred, 165.0), "forecast is the synthetic value implied by the legs"
    print("PASS: triangular forecast = -residual (rich cross predicted to fall, cheap to rise).")


def test_pip_and_cost_arithmetic():
    """1.5 pips on each of three instruments, with JPY pairs using a 0.01 pip."""
    assert pip_size("EURUSD=X") == 0.0001 and pip_size("EURGBP=X") == 0.0001
    assert pip_size("USDJPY=X") == 0.01 and pip_size("GBPJPY=X") == 0.01
    cost = round_trip_cost({"EURJPY=X": 165.0, "EURUSD=X": 1.10, "USDJPY=X": 150.0}, 1.5)
    expected = 1.5 * (0.01 / 165.0 + 0.0001 / 1.10 + 0.01 / 150.0)
    assert abs(cost - expected) < 1e-15
    print(f"PASS: cost arithmetic -- EURJPY triangle round trip = {cost * 1e4:.2f} bp.")


def make_triangle(sign2: int = +1):
    """Synthetic legs plus a cross that reverts EXACTLY as the forecaster assumes."""
    leg1 = make_df(seed=1)["close"] / 100.0 * 1.10
    leg2 = make_df(seed=2)["close"] / 100.0 * 150.0
    synth = (leg1 * leg2 ** sign2).to_numpy()
    cross = np.empty(N)
    cross[0] = synth[0] * 1.01
    cross[1:] = synth[:-1]          # tomorrow's cross = today's synthetic value
    df = pd.DataFrame({"open": cross, "high": cross, "low": cross, "close": cross, "volume": 1e6},
                      index=leg1.index)
    return df, leg1, leg2


def test_triangular_spy_only_reads_same_day_legs():
    """SPY: windows precede scoring, and legs are read only at the window's last date."""
    df, leg1, leg2 = make_triangle()
    inner = TriangularForecaster(leg1, leg2, +1)
    spy = SpyOn(inner)
    res = walk_forward(df, spy, LOOKBACK, HORIZON, step=7, verbose=False)

    _assert_windows_precede_scoring(df, spy, res)
    assert len(inner.access_log) == len(spy.windows)
    for window, entry in zip(spy.windows, inner.access_log):
        assert entry["leg_date"] == window.index.max(), "LEAK: legs read at a date other than the last bar"
    print(f"PASS: triangular spy -- {len(spy.windows)} windows, legs always read at the "
          f"window's last bar.")


class LeakyTriangular(TriangularForecaster):
    """Deliberately broken: reads the legs one bar AHEAD of the last observed bar."""

    name = "leaky-triangular"

    def _legs_at(self, asof):
        pos = min(self._leg1.index.get_loc(asof) + 1, len(self._leg1) - 1)
        return float(self._leg1.iloc[pos]), float(self._leg2.iloc[pos])


def _triangular_poison_leaks(cls) -> bool:
    df, leg1, leg2 = make_triangle()
    cutoff_date = df.index[CUTOFF]
    dirty_df, dirty1, dirty2 = df.copy(), leg1.copy(), leg2.copy()
    dirty_df.iloc[CUTOFF:, dirty_df.columns.get_loc("close")] = POISON
    dirty1.iloc[CUTOFF:] = POISON
    dirty2.iloc[CUTOFF:] = POISON

    clean = _pre_cutoff_preds(df, cls(leg1, leg2, +1), cutoff_date)
    dirty = _pre_cutoff_preds(dirty_df, cls(dirty1, dirty2, +1), cutoff_date)
    return not np.array_equal(clean.to_numpy(), dirty.to_numpy())


def test_triangular_tripwire_poisoned_future_never_used():
    """TRIPWIRE: poisoned future cross and legs leave pre-cutoff forecasts unchanged."""
    assert not _triangular_poison_leaks(TriangularForecaster), (
        "LEAK: triangular forecast changed when the future was poisoned")
    print("PASS: triangular tripwire -- pre-cutoff forecasts are identical with the future poisoned.")


def test_triangular_tripwire_catches_a_leaky_variant():
    """The tripwire must FAIL a forecaster that reads tomorrow's legs."""
    assert _triangular_poison_leaks(LeakyTriangular), "tripwire has no teeth: it missed a one-bar leak"
    print("PASS: triangular tripwire has teeth -- a one-bar-ahead variant is caught.")


def test_triangular_scores_perfectly_when_reversion_is_exact():
    """CHEATER: where the cross truly reverts in one bar, the forecaster is perfect."""
    for sign2 in (+1, -1):
        df, leg1, leg2 = make_triangle(sign2)
        res = walk_forward(df, TriangularForecaster(leg1, leg2, sign2), LOOKBACK, 1, step=1,
                           max_evals=10_000, verbose=False)
        h1 = summarize(res, 1).iloc[0]
        assert h1["dir_acc"] == 1.0, f"sign2={sign2}: dir_acc={h1['dir_acc']}"
        assert h1["rmse_ratio"] < 1e-9, f"sign2={sign2}: rmse_ratio={h1['rmse_ratio']}"
        assert h1["IC"] > 0.999, f"sign2={sign2}: IC={h1['IC']}"
    print("PASS: triangular in an exact-reversion world scored 100% / ~0 RMSE / IC 1.0 "
          "for both leg signs.")


if __name__ == "__main__":
    test_rate_lag_rule()
    test_uip_sign_convention()
    test_uip_spy_only_reads_published_rates()
    test_uip_tripwire_poisoned_future_never_used()
    test_uip_tripwire_catches_a_leaky_variant()
    test_uip_scores_perfectly_when_uip_holds()
    test_triangle_sign_conventions()
    test_triangular_predicts_minus_residual()
    test_pip_and_cost_arithmetic()
    test_triangular_spy_only_reads_same_day_legs()
    test_triangular_tripwire_poisoned_future_never_used()
    test_triangular_tripwire_catches_a_leaky_variant()
    test_triangular_scores_perfectly_when_reversion_is_exact()
    print("\nALL FX INTEGRITY TESTS PASSED: neither FX forecaster reads its side data "
          "ahead of time, and each scores perfectly when its theory is exactly true.")
