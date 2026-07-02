#!/usr/bin/env python3
"""
sweep.py -- run the same walk-forward evaluation across many tickers and report
an HONEST aggregate: does the model beat its baselines consistently, or only as
often as chance would predict?

    python sweep.py --model drift                      # fast sanity sweep
    python sweep.py --model kronos --model-pkg-dir .    # the real sweep

The forecaster is built ONCE and reused across every ticker, so Kronos weights
load a single time, not once per ticker.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data.loader import load_ohlcv
from evaluation import summarize, walk_forward
from run import build_forecaster  # reuse the factory

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA",
                   "JPM", "XOM", "KO", "PG", "WMT", "BTC-USD"]


def aggregate_report(table: pd.DataFrame, alpha: float) -> tuple[dict, str]:
    """Turn per-ticker headline rows into an honest aggregate + verdict.

    Factored out (and pure) so the multiple-comparisons logic -- the heart of
    the whole framework's honesty -- can be unit-tested without any network,
    torch, or model weights.
    """
    n = len(table)
    n_beat_dir = int(table["beat_dir"].sum())
    n_beat_err = int(table["beat_err"].sum())
    n_ic_sig = int(table["ic_sig"].sum()) if "ic_sig" in table.columns else 0
    expected_fp = alpha * n           # false positives expected by chance alone
    med_ic = float(table["IC"].median())
    med_ratio = float(table["rmse_ratio"].median())

    stats = {
        "n_tickers": n,
        "n_beat_dir": n_beat_dir,
        "expected_by_chance": expected_fp,
        "n_beat_err": n_beat_err,
        "n_ic_significant": n_ic_sig,
        "median_IC": med_ic,
        "median_rmse_ratio": med_ratio,
    }

    # The key comparison: did MORE tickers beat the baseline than chance predicts?
    beats_chance = n_beat_dir > np.ceil(expected_fp)
    if not beats_chance and med_ratio >= 1.0:
        verdict = (
            "VERDICT: No aggregate edge. The count of tickers that 'beat' the "
            f"direction baseline ({n_beat_dir}) is at or below what chance alone "
            f"predicts ({expected_fp:.1f}), median IC is ~0, and typical error is "
            "at or above random-walk. Across a diverse sample, no consistent "
            "forecasting skill on daily bars."
        )
    elif beats_chance and med_ratio < 1.0 and med_ic > 0.03:
        verdict = (
            "VERDICT: Possible weak signal worth deeper testing -- more tickers beat "
            f"baseline ({n_beat_dir}) than chance predicts ({expected_fp:.1f}), median "
            "error is below random-walk, and median IC is positive. NOT proof: re-test "
            "out-of-sample, across other periods, and net of realistic costs first."
        )
    else:
        verdict = (
            "VERDICT: Mixed / inconclusive. Some aggregate metrics lean one way but not "
            "consistently -- most likely noise. Expand the sample (more tickers, other "
            "periods) and re-test before reading anything into it."
        )
    return stats, verdict


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-ticker walk-forward sweep")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--model", default="kronos", choices=["kronos", "random-walk", "drift"])
    p.add_argument("--years", type=float, default=4.0)
    p.add_argument("--interval", default="1d")
    p.add_argument("--lookback", type=int, default=256)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--max-evals", type=int, default=150)
    p.add_argument("--model-size", default="small", choices=["mini", "small", "base"])
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--model-pkg-dir", default=".")
    p.add_argument("--output-dir", default="out")
    p.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    forecaster = build_forecaster(args)  # built ONCE, reused for every ticker
    print(f"Sweeping '{forecaster.name}' across {len(args.tickers)} tickers "
          f"(this can take a while for kronos)...\n")

    headline_rows, all_rows = [], []
    for i, ticker in enumerate(args.tickers, 1):
        print(f"[{i}/{len(args.tickers)}] {ticker} ...", flush=True)
        try:
            df = load_ohlcv(ticker, years=args.years, interval=args.interval)
            res = walk_forward(df, forecaster, args.lookback, args.horizon,
                               step=args.step, max_evals=args.max_evals, verbose=False)
            summ = summarize(res, args.horizon)
        except Exception as e:
            print(f"    skipped {ticker}: {e}")
            continue
        summ.insert(0, "ticker", ticker)
        all_rows.append(summ)
        h1 = summ[summ["horizon"] == 1].iloc[0]
        headline_rows.append({
            "ticker": ticker,
            "dir_acc": h1["dir_acc"],
            "baseline": h1["naive_baseline"],
            "edge": h1["edge_vs_baseline"],
            "p_value": h1["p_value"],
            "rmse_ratio": h1["rmse_ratio"],
            "IC": h1["IC"],
            "IC_p": h1["IC_p"],
            "beat_dir": bool((h1["p_value"] < args.alpha) and (h1["edge_vs_baseline"] > 0)),
            "beat_err": bool(h1["rmse_ratio"] < 0.98),
            "ic_sig": bool((h1["IC"] > 0) and (h1["IC_p"] < args.alpha)),
        })

    if not headline_rows:
        raise SystemExit("No tickers produced results.")

    table = pd.DataFrame(headline_rows)
    full = pd.concat(all_rows, ignore_index=True)
    # Timestamped so a re-run can never destroy a previous sweep's results --
    # the run-1 IC data for this project only survived by luck, in a chat log.
    stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    table_path = os.path.join(args.output_dir, f"sweep_{forecaster.name}_{stamp}_headline.csv")
    full_path = os.path.join(args.output_dir, f"sweep_{forecaster.name}_{stamp}_full.csv")
    table.to_csv(table_path, index=False)
    full.to_csv(full_path, index=False)

    stats, verdict = aggregate_report(table, args.alpha)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 92)
    print(f"MULTI-TICKER SWEEP  |  model: {forecaster.name}  |  horizon=1 headline")
    print("=" * 92)
    print(table.to_string(index=False))
    print("-" * 92)
    print(f"Tickers tested:                          {stats['n_tickers']}")
    print(f"Beat direction baseline (p<{args.alpha}):         {stats['n_beat_dir']}")
    print(f"  ...expected by CHANCE alone:           {stats['expected_by_chance']:.1f}")
    print(f"Beat random walk on error (ratio<0.98):  {stats['n_beat_err']}")
    print(f"Individually significant positive IC:    {stats['n_ic_significant']}"
          f"   (expected by chance: {stats['expected_by_chance']:.1f})")
    print(f"Median IC across tickers:                {stats['median_IC']:+.4f}")
    print(f"Median RMSE ratio across tickers:        {stats['median_rmse_ratio']:.3f}")
    print("-" * 92)
    print(verdict)
    print("CAVEAT: tickers overlap heavily (indices contain the mega-caps), so these "
          "are NOT independent tests; treat 'expected by chance' as a lower bound "
          "and clustered results (e.g. all tech leaning one way) as ~one draw, not many.")
    print("-" * 92)
    print(f"Saved: {table_path}\n       {full_path}")


if __name__ == "__main__":
    main()
