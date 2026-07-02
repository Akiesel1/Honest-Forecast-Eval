"""
The walk-forward evaluation loop.

This is the model-agnostic heart of the framework. It rolls across history and,
at each point, asks a Forecaster for a prediction and compares it to what
actually happened. It calls ONLY `forecaster.predict(window, horizon)` -- it has
no idea, and no way to find out, whether that forecaster is Kronos, a baseline,
or something you write next year. The word "Kronos" does not appear in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasters.base import Forecaster


def walk_forward(
    df: pd.DataFrame,
    forecaster: Forecaster,
    lookback: int,
    horizon: int,
    step: int = 1,
    max_evals: int = 150,
    verbose: bool = True,
) -> pd.DataFrame:
    """Roll `forecaster` across `df` and record predicted vs actual returns.

    Parameters
    ----------
    df : OHLCV frame with a DatetimeIndex, oldest first.
    forecaster : any object satisfying the Forecaster interface.
    lookback : bars of history fed to the forecaster at each point.
    horizon : bars ahead to forecast and score.
    step : bars between successive evaluation points (bigger = faster).
    max_evals : cap on the number of evaluation points (runtime guard).

    Returns
    -------
    pd.DataFrame : one row per (evaluation point, horizon step) with the
        predicted and actual return, plus direction flags. This is the raw
        material the metrics functions summarize.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    # Each evaluation point t forecasts from the window ENDING at t-1, and is
    # scored against bars t .. t+horizon-1. The window is strictly BEFORE t, so
    # the forecaster can never peek at the bars it's being graded on. That
    # no-lookahead property is the whole reason a backtest means anything.
    eval_points = list(range(lookback, n - horizon, step))
    if len(eval_points) > max_evals:
        # Evenly subsample to honor the cap while keeping full-history coverage.
        sel = np.linspace(0, len(eval_points) - 1, max_evals).astype(int)
        eval_points = [eval_points[i] for i in sorted(set(sel))]
    if not eval_points:
        # A library raises; a script exits. This is a library now.
        raise ValueError(
            "Not enough data for even one walk-forward point. "
            "Lower lookback/horizon or supply more history."
        )

    records = []
    for k, t in enumerate(eval_points, 1):
        window = df.iloc[t - lookback:t]                  # bars [t-lookback .. t-1]
        anchor_close = float(window["close"].iloc[-1])    # last OBSERVED close

        try:
            pred_close = np.asarray(forecaster.predict(window, horizon), dtype=float)
        except Exception as e:
            if verbose:
                print(f"  [skip point {t}] {forecaster.name} failed: {e}")
            continue

        # Fail loudly if an adapter breaks its contract -- much better than
        # silently scoring garbage.
        if pred_close.shape[0] != horizon:
            raise ValueError(
                f"{forecaster.name} returned {pred_close.shape[0]} predictions, "
                f"expected {horizon}. Fix its predict() implementation."
            )

        # Convert prices -> returns HERE, against the anchor. This return math is
        # identical for every model, so it lives in exactly one place.
        for h in range(1, horizon + 1):
            actual_close = close[t + h - 1]               # bars t .. t+horizon-1
            actual_ret = (actual_close - anchor_close) / anchor_close
            pred_ret = (pred_close[h - 1] - anchor_close) / anchor_close
            records.append({
                "t": t, "horizon": h,
                "actual_ret": actual_ret, "pred_ret": pred_ret,
                "actual_up": int(actual_ret > 0),
                "pred_up": int(pred_ret > 0),
                "correct_dir": int(np.sign(pred_ret) == np.sign(actual_ret)),
            })

        if verbose and (k % 10 == 0 or k == len(eval_points)):
            print(f"  {k}/{len(eval_points)} forecasts done")

    res = pd.DataFrame(records)
    if res.empty:
        raise ValueError("Every forecast failed; nothing to evaluate.")
    return res
