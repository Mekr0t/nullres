"""Features from funding rates and open interest.

THE JOIN IS THE WHOLE PROBLEM. Everything else here is arithmetic.

A bar indexed at time T covers [T, T+interval) and CLOSES at T+interval. So a
funding settlement or OI reading may be used for bar T only if its timestamp is
strictly before T+interval. Joining on the bar's OPEN time throws away a bar of
information; joining on anything at or after the close is lookahead, and it is
the kind that produces a beautiful equity curve.

We use `merge_asof(direction="backward", allow_exact_matches=False)` against
the bar's close instant. Exact matches are excluded because a settlement
stamped exactly at the close is simultaneous with it, and "simultaneous" is not
"available".

The auxiliary frames are also CLIPPED to the bar range before joining. That is
not cosmetic: `audit.check_point_in_time` truncates the bars and recomputes, and
without clipping the funding frame would still hold future rows, so a bad join
direction would silently produce identical output and the check would pass. With
clipping, the audit covers this surface too — `tests/test_derivatives.py` proves
it by injecting a forward join and asserting it gets caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DERIVATIVE_DOC = {
    "funding":          "last settled 8h funding rate — positive means longs pay",
    "funding_ma_3":     "24h mean funding — persistent positioning, not one print",
    "funding_ma_21":    "7d mean funding — the standing cost of being long",
    "funding_z":        "funding vs its own 30-settlement history — crowding extreme",
    "funding_cum_7d":   "total funding paid over 7d — carry drag on a long",
    "oi_chg_6":         "log change in open interest over 6 bars — leverage building",
    "oi_chg_24":        "log change in open interest over 24 bars",
    "oi_z":             "open interest vs its own 168-bar history",
    "oi_price_div":     "sign(OI change) vs sign(price change) — new money or unwind",
    "ls_accounts":      "long/short ratio across all accounts — retail positioning",
    "ls_top_positions": "long/short ratio, top traders by position — smart money proxy",
    "ls_spread":        "top-trader ratio minus all-account ratio — who disagrees",
    "taker_ratio":      "taker buy/sell volume ratio — aggressive flow direction",
    "taker_z":          "taker ratio vs its 72-bar history",
}


def _bar_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing — robust to the odd missing bar."""
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError("cannot infer bar interval from a single bar")
    return pd.Timedelta(deltas.median())


def _asof(bars: pd.DataFrame, aux: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Point-in-time join of `aux` onto the bar index.

    Returns a frame indexed like `bars`, holding the most recent `aux` values
    strictly before each bar's close.
    """
    interval = _bar_interval(bars.index)
    close_time = (bars.index + interval).astype("datetime64[ns]")

    # merge_asof will not join datetime64[ms] against datetime64[ns], and
    # pandas 3.0 picks the resolution from whatever parsed the source. Normalise
    # defensively — aux can arrive from a loader, a test, or synthetic data.
    aux = aux.copy()
    aux.index = pd.DatetimeIndex(aux.index).astype("datetime64[ns]")

    # Clip to what exists as of the final bar's close, so truncating the bars
    # also truncates the auxiliary data. See the module docstring.
    aux = aux[aux.index < close_time[-1]]
    if aux.empty:
        return pd.DataFrame(np.nan, index=bars.index, columns=cols)

    left = pd.DataFrame({"close_time": close_time}, index=bars.index)
    joined = pd.merge_asof(
        left.reset_index(names="bar_ts").sort_values("close_time"),
        aux[cols].reset_index(names="aux_ts").sort_values("aux_ts"),
        left_on="close_time",
        right_on="aux_ts",
        direction="backward",
        allow_exact_matches=False,
    )
    return joined.set_index("bar_ts")[cols].reindex(bars.index)


def build_derivative_features(bars: pd.DataFrame,
                              funding: pd.DataFrame | None = None,
                              metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stationary features from funding and open-interest data.

    Levels are avoided throughout. Open interest grew ~10x over this sample; a
    model that learned "OI > 400k" learned the calendar, exactly as it would
    have from raw price.
    """
    f = pd.DataFrame(index=bars.index)
    logclose = np.log(bars["close"])

    if funding is not None and len(funding):
        joined = _asof(bars, funding, ["funding_rate"])
        rate = joined["funding_rate"]
        f["funding"] = rate
        # Settlements are 8-hourly; on a 4h bar 3 settlements is ~24h. Windows
        # are expressed in BARS, so they scale with the configured timeframe.
        f["funding_ma_3"] = rate.rolling(3).mean()
        f["funding_ma_21"] = rate.rolling(21).mean()
        f["funding_z"] = (
            (rate - rate.rolling(90).mean()) / rate.rolling(90).std()
        )
        f["funding_cum_7d"] = rate.rolling(42).sum()

    if metrics is not None and len(metrics):
        cols = [c for c in ["open_interest", "all_accounts_ls",
                            "top_trader_positions_ls", "taker_buy_sell_ratio"]
                if c in metrics.columns]
        joined = _asof(bars, metrics, cols)

        if "open_interest" in joined:
            oi = joined["open_interest"].replace(0, np.nan)
            log_oi = np.log(oi)
            f["oi_chg_6"] = log_oi.diff(6)
            f["oi_chg_24"] = log_oi.diff(24)
            f["oi_z"] = (log_oi - log_oi.rolling(168).mean()) / log_oi.rolling(168).std()
            # Same price move, opposite meaning depending on whether positions
            # are being opened or closed.
            f["oi_price_div"] = np.sign(log_oi.diff(24)) * np.sign(logclose.diff(24))

        if "all_accounts_ls" in joined:
            f["ls_accounts"] = joined["all_accounts_ls"]
        if "top_trader_positions_ls" in joined:
            f["ls_top_positions"] = joined["top_trader_positions_ls"]
        if {"all_accounts_ls", "top_trader_positions_ls"} <= set(joined.columns):
            f["ls_spread"] = (joined["top_trader_positions_ls"]
                              - joined["all_accounts_ls"])
        if "taker_buy_sell_ratio" in joined:
            taker = joined["taker_buy_sell_ratio"]
            f["taker_ratio"] = taker
            f["taker_z"] = (
                (taker - taker.rolling(72).mean()) / taker.rolling(72).std()
            )

    return f.replace([np.inf, -np.inf], np.nan)
