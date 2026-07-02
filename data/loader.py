"""
Data loading. One job: given a ticker, return a clean OHLCV DataFrame with a
DatetimeIndex. Knows nothing about models or evaluation -- it just fetches bars.
"""
from __future__ import annotations

import pandas as pd


def load_ohlcv(
    ticker: str,
    years: float = 4.0,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history from yfinance and return it clean and sorted.

    This is the old fetch_data(), moved out of the script. The only behavioral
    change: it raises ValueError on bad input instead of calling sys.exit(),
    because it's library code now.
    """
    import yfinance as yf

    if start is None:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
        start = (end_ts - pd.Timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")

    print(f"Fetching {ticker} {interval} from {start} to {end or 'today'} ...")
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}. Check symbol/dates/interval.")

    # yfinance sometimes returns MultiIndex columns even for one ticker; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: c.lower() for c in df.columns})
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing required columns {missing}. Got: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"Got {len(df)} bars: {df.index.min().date()} -> {df.index.max().date()}")
    return df
