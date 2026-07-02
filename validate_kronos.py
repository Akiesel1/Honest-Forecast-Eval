#!/usr/bin/env python3
"""
Smoke-test the KronosForecaster adapter on ANY machine with torch + internet
(Windows, Linux, or Mac -- nothing here is OS-specific).

This does NOT check bit-for-bit equality with the old run_backtest.py: Kronos
samples stochastically, so no two runs are identical. It checks that the
adapter RUNS and returns sane, correctly-shaped output -- which is what
"validate the refactor" actually means here.

Usage (repo root, venv active):
    python validate_kronos.py --ticker NVDA --model-pkg-dir C:/path/to/repo
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--model", default="small", choices=["mini", "small", "base"])
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--sample-count", type=int, default=3,
                    help="Keep low (2-3) on CPU so it stays fast.")
    ap.add_argument("--model-pkg-dir", default=".",
                    help="Folder that CONTAINS the bundled 'model' package.")
    args = ap.parse_args()

    import pandas as pd
    import yfinance as yf
    from forecasters.kronos import KronosForecaster

    # 1) Pull a little daily history and shape it like the evaluator would.
    df = yf.download(args.ticker, period="2y", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise SystemExit(f"No data for {args.ticker} -- check the symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    window = df.iloc[-args.lookback:]
    last_close = float(window["close"].iloc[-1])
    print(f"Loaded {len(df)} bars of {args.ticker}; window = last {len(window)}.")
    print(f"Last observed close: {last_close:.2f}")

    # 2) Build the adapter (first run downloads weights) and forecast ONCE.
    print("Loading Kronos (first run downloads weights from HuggingFace)...")
    f = KronosForecaster(model_size=args.model,
                         sample_count=args.sample_count,
                         model_pkg_dir=args.model_pkg_dir)
    print(f"Device in use: {f._predictor.device}")
    preds = f.predict(window, horizon=args.horizon)

    # 3) Sanity checks (loose on purpose -- catches NaN / blow-ups, not noise).
    print(f"\nPredicted next {args.horizon} closes: "
          f"{[round(float(p), 2) for p in preds]}")
    ok_len = len(preds) == args.horizon
    ok_range = all(0.5 * last_close < float(p) < 2.0 * last_close for p in preds)
    print(f"  correct length ({args.horizon}): {ok_len}")
    print(f"  all within 0.5x-2x last close:  {ok_range}")
    print("\nPASS -- the adapter works." if (ok_len and ok_range)
          else "\nCHECK -- output looks off; paste this + any error back to me.")


if __name__ == "__main__":
    main()
