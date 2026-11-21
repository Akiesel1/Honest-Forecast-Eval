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


# ---------------------------------------------------------------------------
# FX study additions. Same rule as load_ohlcv: fetch and clean, nothing else.
# No lagging, no model logic, no evaluation logic lives here.
# ---------------------------------------------------------------------------

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def load_fred_series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Fetch one FRED series (public CSV endpoint, no API key) as a clean Series.

    Index is the OBSERVATION date as FRED reports it (for monthly series, the
    first day of the month the value describes), oldest first.

    Two honesty notes for callers:

    * The observation date is NOT the date the value became public. Applying a
      publication lag is the forecaster's job, not the loader's.
    * This endpoint serves the latest REVISED values, not the original vintage.
      A live forecaster may have seen slightly different numbers.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    print(f"Fetching FRED series {series_id} ...")
    raw = pd.read_csv(url, na_values=["."])
    if series_id not in raw.columns or len(raw) == 0:
        raise ValueError(f"FRED returned no usable data for {series_id}. Got: {list(raw.columns)}")

    s = pd.Series(
        pd.to_numeric(raw[series_id], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(raw[raw.columns[0]]),
        name=series_id,
    ).dropna().sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    if s.empty:
        raise ValueError(f"FRED series {series_id} is empty for the requested dates.")
    print(f"Got {len(s)} observations: {s.index.min().date()} -> {s.index.max().date()}")
    return s


def align_on_common_dates(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict several OHLCV frames to the dates present in ALL of them.

    Used for FX triangles, where a residual only exists on days when the cross
    and both legs all have a bar. Rows are dropped, never filled.
    """
    if not frames:
        raise ValueError("No frames to align.")
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError("The supplied frames share no dates.")
    return {name: df.loc[common] for name, df in frames.items()}
