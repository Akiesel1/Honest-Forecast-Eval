"""
KronosForecaster: the Kronos foundation model, wrapped to satisfy the
Forecaster interface.

This is where all the Kronos-specific machinery that used to be smeared across
`load_predictor()` and the middle of `walk_forward()` now lives -- and ONLY
here. The evaluator never sees any of it. From the outside this is just a
Forecaster whose .predict(history, horizon) returns predicted closes, exactly
like the baselines.

Note: torch and the bundled `model` package are imported lazily inside
__init__ so that merely importing this module (or using the baselines) never
requires torch to be installed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from .base import Forecaster

# (tokenizer_repo, model_repo, max_context) per Kronos size. Moved verbatim
# from the old MODEL_REPOS dict in run_backtest.py.
MODEL_REPOS = {
    "mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini", 2048),
    "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
    "base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base", 512),
}


def infer_future_index(history_index, horizon: int) -> pd.DatetimeIndex:
    """Synthesize the future timestamps Kronos wants to forecast into.

    The forecaster only ever sees the lookback window -- it must NOT see the
    real future bars, or it would be cheating (lookahead). But Kronos needs
    *some* future timestamps to build its calendar features (weekday, day,
    month). So we generate plausible ones from the cadence of the history:

    - Daily data  -> next `horizon` BUSINESS days (skips weekends), which
      matches an equity trading calendar closely.
    - Intraday / other -> extend by the typical spacing between bars.

    Honest caveat: because these are synthesized rather than the real trading
    dates, they can differ from reality around market holidays (a predicted
    "day"/"month" feature may be off by one session). This has a negligible
    effect on the forecast, but it's why the refactor should be validated
    against the original script on a real machine. If exactness is ever
    needed, swap this for a real market calendar (e.g. pandas_market_calendars)
    -- the only file that would change is this one.
    """
    idx = pd.DatetimeIndex(history_index)
    last = idx[-1]
    if len(idx) >= 2:
        median_delta = pd.Series(idx).diff().dropna().median()
    else:
        median_delta = pd.Timedelta(days=1)

    one_day = pd.Timedelta(days=1)
    if one_day * 0.9 <= median_delta <= one_day * 1.5:
        # Daily cadence: roll forward over business days.
        return pd.bdate_range(start=last + pd.offsets.BDay(1), periods=horizon)
    # Non-daily cadence: extend by the typical spacing.
    return pd.DatetimeIndex([last + median_delta * (i + 1) for i in range(horizon)])


class KronosForecaster(Forecaster):
    """Adapter around Kronos. All model config lives in the constructor."""

    def __init__(
        self,
        model_size: str = "small",
        sample_count: int = 5,
        T: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        device: str | None = None,
        seed: int = 42,
        model_pkg_dir: str | None = None,
    ):
        """Load Kronos once and hold it as state.

        This body is the old `load_predictor()`, moved verbatim except that its
        configuration is now constructor arguments instead of CLI flags.

        Parameters
        ----------
        model_size : "mini" | "small" | "base"
        sample_count : stochastic samples per forecast, averaged internally by
            Kronos (it is a sampling model, so more samples = steadier point
            forecast, but slower).
        T, top_p, top_k : Kronos sampling knobs. Held here, applied in predict.
        device : "cuda:0" | "mps" | "cpu" | None (None = Kronos auto-detects).
        seed : set once here for run reproducibility.
        model_pkg_dir : path to the folder that CONTAINS the bundled `model/`
            package (the one from scripts/model). If None, `model` must already
            be importable.
        """
        # Lazy, local imports so baselines never need torch.
        import torch

        if model_pkg_dir is not None:
            sys.path.insert(0, os.path.abspath(model_pkg_dir))
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as e:
            raise ImportError(
                "Could not import the bundled Kronos `model` package. Pass "
                "model_pkg_dir=<folder containing model/> or put it on sys.path."
            ) from e

        torch.manual_seed(seed)
        np.random.seed(seed)

        tok_repo, mdl_repo, max_ctx = MODEL_REPOS[model_size]
        self.max_context = max_ctx
        self.sample_count = sample_count
        self.T = T
        self.top_p = top_p
        self.top_k = top_k
        self.name = f"kronos-{model_size}"

        tokenizer = KronosTokenizer.from_pretrained(tok_repo)
        model = Kronos.from_pretrained(mdl_repo)
        self._predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_ctx
        )

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        """Forecast `horizon` closes. This is the old predict() call from
        inside walk_forward(), now self-contained.

        Two things that used to be the evaluator's job now happen here, because
        they are Kronos's business, not the loop's:
          1. Truncating the window to the model's max context.
          2. Building the x/y timestamp Series Kronos needs.
        """
        # 1) Never feed Kronos more context than it can take.
        hist = history.iloc[-self.max_context:]

        # 2) Timestamps: past come from the window's index; future are synthesized.
        x_ts = pd.Series(hist.index)
        y_ts = pd.Series(infer_future_index(hist.index, horizon))

        pred_df = self._predictor.predict(
            df=hist[["open", "high", "low", "close", "volume"]].copy(),
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=self.T,
            top_k=self.top_k,
            top_p=self.top_p,
            sample_count=self.sample_count,
            verbose=False,
        )
        # Contract: return exactly `horizon` predicted CLOSE prices as an array.
        return pred_df["close"].to_numpy(dtype=float)
