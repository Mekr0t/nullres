"""Feature engineering.

Two hard rules, both enforced by `tbot.audit`:

  1. POINT-IN-TIME. Every value at bar t uses only bars <= t. In practice this
     means: rolling windows only, never `.shift(-k)`, never an expanding stat
     over the full sample, never `fillna(method="bfill")`, never a global
     mean/std for scaling. `audit.check_point_in_time` recomputes features on
     truncated data and asserts the last row is unchanged.

  2. STATIONARY. No raw price levels. BTC ran 4k -> 100k over this sample; a
     tree that learned "close > 60000" learned the calendar, not the market.
     Everything below is a ratio, a z-score, or a bounded oscillator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_DOC = {
    "ret_{n}":     "log return summed over the last n bars — momentum",
    "vol_{w}":     "rolling stdev of log returns — volatility regime",
    "volratio":    "short vol / long vol — is volatility expanding or calming",
    "z_{w}":       "close z-scored against its own rolling window — stretch",
    "rsi_14":      "Wilder RSI — bounded overbought/oversold oscillator",
    "atr_pct":     "ATR as a fraction of price — normalised true range",
    "hl_range":    "(high - low) / close — this bar's realised range",
    "upper_wick":  "fraction of the bar's range above the body — rejection above",
    "lower_wick":  "fraction of the bar's range below the body — rejection below",
    "body":        "signed body size / range — conviction within the bar",
    "donch_{w}":   "position within the rolling high-low channel, 0..1 — breakout",
    "macd_n":      "MACD histogram normalised by price — trend acceleration",
    "vol_z":       "volume z-scored over 72 bars — unusual participation",
    "trade_z":     "trade count z-scored over 72 bars — unusual activity",
    "avg_trade":   "volume per trade, z-scored — retail flow vs block flow",
    "amihud":      "|return| / volume, z-scored — illiquidity / price impact",
    "ret_skew_72": "skew of recent returns — asymmetry of the local distribution",
    "hour":        "UTC hour — session effects (Asia/Europe/US)",
    "dow":         "day of week — weekend liquidity effects",
}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_features(df: pd.DataFrame, funding=None, metrics=None) -> pd.DataFrame:
    """Return the feature matrix, indexed identically to `df`.

    Leading rows are NaN until the longest window fills; callers drop them.

    `funding` and `metrics` are optional Binance futures frames. When supplied,
    derivative features are appended — see `features/derivatives.py`, where the
    point-in-time join is the part that matters.
    """
    f = pd.DataFrame(index=df.index)
    close, high, low = df["close"], df["high"], df["low"]
    logret = np.log(close).diff()

    # --- momentum over several horizons -----------------------------------
    for lag in (1, 2, 3, 6, 12, 24, 72):
        f[f"ret_{lag}"] = logret.rolling(lag).sum()

    # --- volatility regime --------------------------------------------------
    for w in (12, 24, 72, 168):
        f[f"vol_{w}"] = logret.rolling(w).std()
    f["volratio"] = f["vol_12"] / f["vol_168"]

    # --- stretch / mean-reversion ------------------------------------------
    for w in (12, 24, 72, 168):
        f[f"z_{w}"] = _zscore(close, w)

    f["rsi_14"] = rsi(close)

    # --- bar shape ----------------------------------------------------------
    rng = (high - low).replace(0, np.nan)
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    f["atr_pct"] = atr(df) / close
    f["hl_range"] = rng / close
    f["upper_wick"] = (high - body_top) / rng
    f["lower_wick"] = (body_bot - low) / rng
    f["body"] = (close - df["open"]) / rng

    # --- channel position / breakout ---------------------------------------
    for w in (24, 72):
        hh = high.rolling(w).max()
        ll = low.rolling(w).min()
        f[f"donch_{w}"] = (close - ll) / (hh - ll).replace(0, np.nan)

    # --- trend acceleration -------------------------------------------------
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    f["macd_n"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    # --- flow ---------------------------------------------------------------
    f["vol_z"] = _zscore(df["volume"], 72)
    f["trade_z"] = _zscore(df["trades"], 72)
    f["avg_trade"] = _zscore(df["volume"] / df["trades"].replace(0, np.nan), 72)
    f["amihud"] = _zscore(logret.abs() / df["volume"].replace(0, np.nan), 72)

    # --- higher moments -----------------------------------------------------
    f["ret_skew_72"] = logret.rolling(72).skew()

    # --- calendar -----------------------------------------------------------
    f["hour"] = df.index.hour.astype("float64")
    f["dow"] = df.index.dayofweek.astype("float64")

    f = f.replace([np.inf, -np.inf], np.nan)

    if funding is not None or metrics is not None:
        from tbot.features.derivatives import build_derivative_features

        f = f.join(build_derivative_features(df, funding, metrics))
    return f
