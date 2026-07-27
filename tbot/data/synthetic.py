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
         "1h": "h", "2h": "2h", "4h": "4h", "12h": "12h", "1d": "D"}


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
        index=pd.date_range(start, periods=n, freq=_FREQ.get(interval, "h"), name="ts"),
    )
