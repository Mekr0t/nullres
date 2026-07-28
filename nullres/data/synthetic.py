"""Synthetic bars with known ground truth.

Two uses, both essential:

  `synthetic_bars`   — a geometric random walk. By construction there is NO edge.
                       Any strategy that profits on this after costs has a bug.
                       This is the single most useful test in the repo.

  `synthetic_bars(edge=...)` — a walk with a real, known autocorrelation. If your
                       pipeline CANNOT find this, it is too weak or mis-wired,
                       and a null result on real data tells you nothing.

Run both before trusting any result. A harness that fails either is not
measuring what you think it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FREQ = {"1m": "min", "5m": "5min", "15m": "15min", "30m": "30min",
         "1h": "h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
         "1d": "D", "1w": "W"}


def synthetic_funding(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Funding settlements every 8h, uncorrelated with future returns.

    Exists so the null control exercises the SAME code path as a real run.
    Without it the random-walk check runs on 32 features while the live config
    runs on 47, and a broken funding join would never reach the one test whose
    whole job is to catch fabricated edge.

    The values are noise by construction, so any strategy that profits from
    them on this data has found a bug in the join, not a signal.
    """
    rng = np.random.default_rng(seed + 101)
    idx = pd.date_range(bars.index[0], bars.index[-1] + pd.Timedelta("8h"), freq="8h")
    return pd.DataFrame(
        {
            "funding_rate": rng.normal(0.0001, 0.0003, len(idx)),
            "funding_hours": np.full(len(idx), 8.0),
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )


def synthetic_metrics(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Open interest and positioning ratios, also pure noise.

    Open interest is generated as a random walk with drift so it is
    non-stationary like the real thing — that way the stationarity discipline
    in `features/derivatives.py` is genuinely exercised.
    """
    rng = np.random.default_rng(seed + 202)
    idx = pd.date_range(bars.index[0], bars.index[-1] + pd.Timedelta("1h"), freq="1h")
    n = len(idx)
    oi = 1e5 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame(
        {
            "open_interest": oi,
            "oi_value": oi * 30_000,
            "top_trader_accounts_ls": np.abs(rng.normal(2.5, 0.4, n)),
            "top_trader_positions_ls": np.abs(rng.normal(1.2, 0.2, n)),
            "all_accounts_ls": np.abs(rng.normal(2.0, 0.3, n)),
            "taker_buy_sell_ratio": np.abs(rng.normal(1.0, 0.2, n)),
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )


def synthetic_bars(n: int = 40_000, seed: int = 0, interval: str = "1h",
                   sigma: float = 0.004, edge: float = 0.0,
                   start: str = "2020-01-01") -> pd.DataFrame:
    """Geometric random walk in OHLCV form.

    Args:
        edge: AR(1) coefficient on log returns. 0.0 means a pure martingale —
            unpredictable by construction. ~0.05 is a faint but genuinely
            learnable edge; real liquid markets sit near 0.0 to 0.02.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, sigma, n)

    if edge:
        logret = np.empty(n)
        logret[0] = shocks[0]
        for i in range(1, n):
            logret[i] = edge * logret[i - 1] + shocks[i]
    else:
        logret = shocks

    close = 30_000.0 * np.exp(np.cumsum(logret))

    # Build O/H/L around the close so OHLC invariants always hold.
    open_ = np.empty(n)
    open_[0] = close[0] * (1 + rng.normal(0, sigma / 8))
    open_[1:] = close[:-1] * (1 + rng.normal(0, sigma / 8, n - 1))
    hi_wick = np.abs(rng.normal(0, sigma / 2, n))
    lo_wick = np.abs(rng.normal(0, sigma / 2, n))
    high = np.maximum(open_, close) * (1 + hi_wick)
    low = np.minimum(open_, close) * (1 - lo_wick)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(500, 150, n)),
            "trades": np.abs(rng.normal(3_000, 800, n)),
        },
        index=pd.date_range(start, periods=n, freq=_interval_freq(interval), name="ts"),
    )


def _interval_freq(interval: str) -> str:
    """Bar spacing for an interval, refusing to guess.

    This used to fall back to hourly for anything unrecognised, silently. A 6h
    config would then generate HOURLY bars and annualise them at 1,460 bars per
    year — a null control mis-calibrated by a factor of six, in the one test the
    docs call the most useful in the repo. Failing loudly is the only safe
    behaviour for a calibration instrument.
    """
    if interval not in _FREQ:
        raise ValueError(
            f"no synthetic bar spacing defined for interval {interval!r}; "
            f"add it to _FREQ in nullres/data/synthetic.py "
            f"(known: {sorted(_FREQ)})"
        )
    return _FREQ[interval]
