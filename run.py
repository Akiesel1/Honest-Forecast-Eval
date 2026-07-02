#!/usr/bin/env python3
"""
run.py -- the single entry point that ties the framework together.

    python run.py --ticker NVDA --model kronos      # the real question
    python run.py --ticker NVDA --model drift       # the yardstick, instantly
    python run.py --ticker SPY  --model random-walk

Same ticker, same walk-forward evaluation, same metrics -- change one word to
change the model. That interchangeability is the entire point of the refactor.
"""
from __future__ import annotations

import argparse
import os
import time

from data.loader import load_ohlcv
from evaluation import summarize, verdict, walk_forward
from forecasters import DriftForecaster, RandomWalkForecaster
from forecasters.base import Forecaster

# The registry of known models. Adding a model later = one entry here + one
# adapter file. Nothing else in the codebase changes.
MODEL_CHOICES = ["kronos", "random-walk", "drift"]


def build_forecaster(args: argparse.Namespace) -> Forecaster:
    """Map the --model flag to a concrete Forecaster instance (the factory).

    This is the ONLY place that knows how each model is constructed. Note the
    Kronos import is LAZY -- it happens inside this branch, so running a
    baseline (`--model drift`) never imports torch and works on a machine that
    doesn't even have it installed.
    """
    if args.model == "random-walk":
        return RandomWalkForecaster()
    if args.model == "drift":
        return DriftForecaster()
    if args.model == "kronos":
        from forecasters.kronos import KronosForecaster  # lazy: torch only if needed
        return KronosForecaster(
            model_size=args.model_size,
            sample_count=args.sample_count,
            model_pkg_dir=args.model_pkg_dir,
        )
    raise ValueError(f"Unknown model: {args.model}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward forecast-accuracy backtest")
    # What to test, and against which model.
    p.add_argument("--ticker", required=True, help="e.g. NVDA, SPY, BTC-USD")
    p.add_argument("--model", default="kronos", choices=MODEL_CHOICES)
    # Data window.
    p.add_argument("--years", type=float, default=4.0)
    p.add_argument("--interval", default="1d")
    # Evaluation knobs (model-agnostic).
    p.add_argument("--lookback", type=int, default=256, help="context bars per forecast")
    p.add_argument("--horizon", type=int, default=5, help="bars ahead to forecast")
    p.add_argument("--step", type=int, default=5, help="bars between eval points")
    p.add_argument("--max-evals", type=int, default=150, help="cap on eval points")
    # Kronos-only knobs (ignored by baselines).
    p.add_argument("--model-size", default="small", choices=["mini", "small", "base"])
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--model-pkg-dir", default=".",
                   help="Folder containing the bundled 'model' package (kronos only).")
    p.add_argument("--output-dir", default="out")
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_ohlcv(args.ticker, years=args.years, interval=args.interval)
    forecaster = build_forecaster(args)

    print(f"\nEvaluating '{forecaster.name}' on {args.ticker} "
          f"(lookback={args.lookback}, horizon={args.horizon})...")
    res = walk_forward(df, forecaster, args.lookback, args.horizon,
                       step=args.step, max_evals=args.max_evals)
    summ = summarize(res, args.horizon)

    # Persist raw + summary so runs are comparable and reproducible. The
    # timestamp makes every run self-preserving: a re-run can never silently
    # destroy the previous results -- which matters in a framework whose whole
    # thesis is "replicate and compare".
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.ticker}_{forecaster.name}_{stamp}"
    res.to_csv(os.path.join(args.output_dir, f"{tag}_raw.csv"), index=False)
    summ.to_csv(os.path.join(args.output_dir, f"{tag}_summary.csv"), index=False)

    import pandas as pd
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print(f"FORECAST-ACCURACY BACKTEST  |  {args.ticker}  |  model: {forecaster.name}")
    print("=" * 78)
    print(summ.to_string(index=False))
    print("-" * 78)
    print(verdict(summ))
    print("-" * 78)
    print(f"Saved: {os.path.join(args.output_dir, tag + '_summary.csv')}")
    print("\nReminder: one ticker over one window is an anecdote, not evidence. "
          "Re-run across several names and regimes -- and always compare the model "
          "against the baselines -- before trusting any result.")


if __name__ == "__main__":
    main()
