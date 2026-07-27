"""Binance public monthly kline archives (data.binance.vision).

No API key, no rate limit, no exchange account. Each month is cached to parquet
so a re-run is offline and instant.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KEEP = ["open", "high", "low", "close", "volume", "trades"]


def fetch_month(symbol: str, interval: str, month: str, cache_dir: str = "data",
                retries: int = 3) -> pd.DataFrame | None:
    """Return one month of klines, from cache when present. None if unavailable."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{symbol}-{interval}-{month}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  fail {month}: {exc}")
                return None
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 404:
            # Month predates the listing or is not yet published. Not an error.
            print(f"  miss {month} (404)")
            return None
        if resp.status_code != 200:
            if attempt == retries - 1:
                print(f"  fail {month} (HTTP {resp.status_code})")
                return None
            time.sleep(2 ** attempt)
            continue

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = zf.read(zf.namelist()[0]).decode()

        # Archives from ~2025 onward ship a header row; older ones do not.
        header = 0 if raw.lstrip().lower().startswith("open_time") else None
        df = pd.read_csv(io.StringIO(raw), header=header, names=KLINE_COLS)
        df.to_parquet(cached)
        print(f"  ok   {month}  ({len(df):,} bars)")
        return df
    return None


def load_binance(symbol: str, interval: str, start: str, end: str,
                 cache_dir: str = "data", verbose: bool = True) -> pd.DataFrame:
    """Load a contiguous range of months and validate the result."""
    if verbose:
        print(f"Loading {symbol} {interval} {start}..{end}")
    months = [d.strftime("%Y-%m") for d in pd.date_range(start, end, freq="MS")]
    parts = [p for p in (fetch_month(symbol, interval, m, cache_dir) for m in months)
             if p is not None and len(p)]
    if not parts:
        raise SystemExit(
            f"No data for {symbol} {interval} {start}..{end}. "
            f"Check the symbol spelling and that the range is not in the future."
        )

    df = pd.concat(parts, ignore_index=True)

    # Binance switched open_time from milliseconds to microseconds during 2025.
    # Detect per row rather than per file: a single concat can straddle both.
    ot = df["open_time"].astype("int64")
    unit_us = ot > 1e15
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if unit_us.any():
        ts[unit_us] = pd.to_datetime(ot[unit_us], unit="us")
    if (~unit_us).any():
        ts[~unit_us] = pd.to_datetime(ot[~unit_us], unit="ms")
    df["ts"] = ts

    df = df[["ts"] + KEEP].astype({c: "float64" for c in KEEP})
    df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")
    df.index.name = "ts"

    _validate(df, interval, verbose)
    return df


def _validate(df: pd.DataFrame, interval: str, verbose: bool) -> None:
    """Fail loudly on corrupt bars; warn on gaps."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("timestamps are not sorted after load")
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps survived deduplication")

    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} bars violate OHLC invariants, first at {df.index[bad][0]}"
        )

    step = pd.Timedelta(interval.replace("m", "min") if interval.endswith("m") else interval)
    gaps = df.index.to_series().diff().dropna()
    missing = int(((gaps / step) - 1).clip(lower=0).sum())
    if verbose:
        span = f"{df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}"
        note = f", {missing:,} missing bars in gaps" if missing else ", no gaps"
        print(f"  {len(df):,} bars  {span}{note}")
