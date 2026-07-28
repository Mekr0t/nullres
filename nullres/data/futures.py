"""Binance USD-M futures data: funding rates and open-interest metrics.

This is the first information in the repo that is NOT a transform of OHLCV.
Everything in `features/technical.py` is thirty-two views of four numbers;
`nullres features` showed almost none of them carry out of sample. Funding and
open interest measure something the price series cannot express: how much
leverage is deployed, and on which side.

    FUNDING RATE   Perpetual futures have no expiry, so an 8-hourly payment
                   tethers them to spot. Positive funding means longs pay
                   shorts — the crowd is long and paying to stay there. It is a
                   direct read on positioning, and it is a PRICE, so it is set
                   by people with money at risk.

    OPEN INTEREST  Total outstanding contracts. Rising OI into a rally means
                   new money; falling OI means an unwind. Same price move,
                   opposite implication.

    LONG/SHORT     Binance publishes account- and position-weighted long/short
    RATIOS         ratios, including a top-trader subset.

Availability (probed, not assumed):
    funding   monthly archives, BTCUSDT from 2020-01
    metrics   DAILY archives only, BTCUSDT from 2020-09; monthly 404s

Caveat worth stating plainly: these describe the PERPETUAL market, while the
bars elsewhere in this repo are spot. That is a legitimate pairing — futures
positioning predicting spot price is the whole idea — but they are different
venues, and the perp can dislocate from spot precisely when it matters most.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
METRICS_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

METRIC_COLS = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "top_trader_accounts_ls",
    "sum_toptrader_long_short_ratio": "top_trader_positions_ls",
    "count_long_short_ratio": "all_accounts_ls",
    "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
}


def _get(url: str, session: requests.Session | None = None, timeout: int = 45):
    getter = session.get if session else requests.get
    try:
        resp = getter(url, timeout=timeout)
    except requests.RequestException:
        return None
    return resp if resp.status_code == 200 else None


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    return pd.read_csv(io.StringIO(raw))


# --------------------------------------------------------------------------
# Funding rate
# --------------------------------------------------------------------------

def load_funding(symbol: str, start: str, end: str, cache_dir: str = "data",
                 verbose: bool = True) -> pd.DataFrame:
    """8-hourly funding rates, indexed by settlement time (UTC).

    The index is the moment the rate was SETTLED, which is the moment it became
    known. Callers must join with that in mind — see `features/derivatives.py`.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    months = [d.strftime("%Y-%m") for d in pd.date_range(start, end, freq="MS")]
    parts = []

    for month in months:
        cached = cache / f"{symbol}-funding-{month}.parquet"
        if cached.exists():
            parts.append(pd.read_parquet(cached))
            continue
        resp = _get(f"{FUNDING_URL}/{symbol}/{symbol}-fundingRate-{month}.zip")
        if resp is None:
            if verbose:
                print(f"  funding miss {month}")
            continue
        df = _read_zip_csv(resp.content)
        df.to_parquet(cached)
        parts.append(df)

    if not parts:
        raise SystemExit(f"no funding data for {symbol} {start}..{end}")

    df = pd.concat(parts, ignore_index=True)
    # Force nanosecond resolution. pandas 3.0 infers the unit from the source,
    # so string timestamps land as datetime64[ms] while kline data is [ns], and
    # merge_asof refuses to join across resolutions.
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms").astype("datetime64[ns]")
    out = (df[["ts", "last_funding_rate", "funding_interval_hours"]]
           .rename(columns={"last_funding_rate": "funding_rate",
                            "funding_interval_hours": "funding_hours"})
           .drop_duplicates("ts").sort_values("ts").set_index("ts"))
    out = out.astype({"funding_rate": "float64", "funding_hours": "float64"})
    if verbose:
        print(f"  funding: {len(out):,} settlements "
              f"{out.index[0]:%Y-%m-%d}..{out.index[-1]:%Y-%m-%d}")
    return out


# --------------------------------------------------------------------------
# Open interest and positioning metrics
# --------------------------------------------------------------------------

def _fetch_metrics_day(symbol: str, day: str, session: requests.Session):
    resp = _get(f"{METRICS_URL}/{symbol}/{symbol}-metrics-{day}.zip", session)
    if resp is None:
        return None
    return _read_zip_csv(resp.content)


def load_metrics(symbol: str, start: str, end: str, cache_dir: str = "data",
                 workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Open interest and long/short ratios at 5-minute granularity.

    Binance publishes these as one archive PER DAY (the monthly path 404s), so
    a six-year range is ~2,000 requests. They are fetched concurrently and
    cached one parquet per month — caching per day would leave 2,000 files in
    `data/` for no benefit.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(start, end, freq="MS")
    parts = []

    for month_start in months:
        tag = month_start.strftime("%Y-%m")
        cached = cache / f"{symbol}-metrics-{tag}.parquet"
        if cached.exists():
            parts.append(pd.read_parquet(cached))
            continue

        days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="D")
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                frames = list(pool.map(
                    lambda d: _fetch_metrics_day(symbol, d.strftime("%Y-%m-%d"), session),
                    days,
                ))
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            if verbose:
                print(f"  metrics miss {tag}")
            continue

        month_df = pd.concat(frames, ignore_index=True)
        month_df.to_parquet(cached)
        parts.append(month_df)
        if verbose:
            print(f"  metrics ok   {tag}  ({len(month_df):,} rows "
                  f"from {len(frames)} days)")

    if not parts:
        raise SystemExit(
            f"no metrics data for {symbol} {start}..{end}. "
            f"BTCUSDT metrics begin 2020-09; other symbols start later."
        )

    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["create_time"]).astype("datetime64[ns]")
    keep = ["ts"] + [c for c in METRIC_COLS if c in df.columns]
    out = (df[keep].rename(columns=METRIC_COLS)
           .drop_duplicates("ts").sort_values("ts").set_index("ts"))
    out = out.astype("float64")
    if verbose:
        print(f"  metrics: {len(out):,} rows "
              f"{out.index[0]:%Y-%m-%d}..{out.index[-1]:%Y-%m-%d}")
    return out
