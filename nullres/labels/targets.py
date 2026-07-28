"""Label construction.

Every label returns a frame with a uniform contract:

    y        int    0/1 target, NaN where the bar is unlabelled (dropped later)
    t_end    int    positional index of the bar at which the label RESOLVES
    ret      float  the log return the label is derived from, for diagnostics
    sigma    float  volatility estimate at decision time, known at bar t

`t_end` is the load-bearing column. A label spanning bars t..t+20 must not sit
in a training set whose test window begins at t+5 — the training label already
contains the answer to the test period. `nullres.validation` purges on this column.
A fixed purge constant is only correct when every label has the same horizon,
which stops being true the moment you use barriers.

On label choice: `next_bar_sign` is the honest version of the baseline's label,
and it is almost pure noise. A 1h BTC bar's empirical mean absolute move is
~0.40% against a ~0.24% round trip, so you are asking a model to call a coin
flip well enough to clear 60% of the move. (`nullres budget` quotes 45% instead,
because it uses the Gaussian `E|move|` of 0.54%; fat tails make the real move
smaller and the real bar higher — see docs/03.) `triple_barrier` instead asks a
question worth answering —
"does price travel 1.5 sigma up before it travels 1.5 sigma down" — which has
a real, if small, autocorrelation structure and a payoff that exceeds costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigma(close: pd.Series, window: int) -> pd.Series:
    """Per-bar volatility estimate, knowable at the close of each bar."""
    return np.log(close).diff().rolling(window).std()


def next_bar_sign(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """1 if the next bar's close exceeds this one's. The baseline's label.

    Kept for comparison, not recommended. Resolves one bar ahead.
    """
    logret = np.log(df["close"]).diff()
    fwd = logret.shift(-1)
    n = len(df)
    return pd.DataFrame(
        {
            "y": (fwd > 0).astype("float64").where(fwd.notna()),
            "t_end": np.minimum(np.arange(n) + 1, n - 1),
            "ret": fwd,
            "sigma": _sigma(df["close"], cfg.vol_window),
        },
        index=df.index,
    )


def fwd_return(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sign of the vol-scaled return over `horizon` bars.

    Bars whose move is smaller than `deadband` sigma are left unlabelled. That
    matters: without it, roughly half the training set is noise the model tries
    to fit, and the fit it finds is spurious.
    """
    n = len(df)
    logclose = np.log(df["close"])
    fwd = logclose.shift(-cfg.horizon) - logclose
    sigma = _sigma(df["close"], cfg.vol_window)
    scale = sigma * np.sqrt(cfg.horizon)

    scaled = fwd / scale.replace(0, np.nan)
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y[scaled > cfg.deadband] = 1.0
    y[scaled < -cfg.deadband] = 0.0
    y[fwd.isna()] = np.nan

    return pd.DataFrame(
        {
            "y": y,
            "t_end": np.minimum(np.arange(n) + cfg.horizon, n - 1),
            "ret": fwd,
            "sigma": sigma,
        },
        index=df.index,
    )


def triple_barrier(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """López de Prado triple barrier, vectorised over the horizon.

    From the close of bar t, place a profit barrier at +upper*sigma and a stop
    at -lower*sigma, plus a vertical barrier `horizon` bars out. Label 1 if the
    upper barrier is touched first, 0 if the lower is, and by the sign of the
    realised return if the vertical barrier is reached first.

    The barriers are volatility-scaled, so the label means the same thing in a
    calm 2023 and a violent March 2020 — a fixed 1% target is a different
    question in each regime, and mixing the two is why fixed-percent labels
    train models that only work in the regime that dominated the sample.

    When both barriers fall inside one bar, OHLC cannot tell us which came
    first. We assume the STOP hit first. That is pessimistic by design: the
    alternative silently inflates every result you will ever produce here.
    """
    n = len(df)
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    sigma = _sigma(df["close"], cfg.vol_window)
    sig = sigma.to_numpy(dtype="float64")

    upper = close * np.exp(cfg.upper * sig)
    lower = close * np.exp(-cfg.lower * sig)

    idx = np.arange(n)
    side = np.zeros(n)                     # +1 profit, -1 stop, 0 unresolved
    t_end = np.minimum(idx + cfg.horizon, n - 1)
    open_ = np.ones(n, dtype=bool) & np.isfinite(sig)

    for h in range(1, cfg.horizon + 1):
        j = np.minimum(idx + h, n - 1)
        in_range = (idx + h) < n
        live = open_ & in_range
        if not live.any():
            break

        hit_dn = live & (low[j] <= lower)
        hit_up = live & (high[j] >= upper) & ~hit_dn   # stop wins ties

        newly = hit_up | hit_dn
        side[newly] = np.where(hit_up[newly], 1.0, -1.0)
        t_end[newly] = j[newly]
        open_ &= ~newly

    # Vertical barrier: unresolved paths fall back to the sign of the return.
    logclose = np.log(close)
    ret = logclose[t_end] - logclose
    y = np.where(side > 0, 1.0, np.where(side < 0, 0.0, (ret > 0).astype(float)))

    # A label that runs off the end of the sample never resolved — drop it.
    unresolved_tail = (idx + cfg.horizon) >= n
    y = np.where(unresolved_tail | ~np.isfinite(sig), np.nan, y)

    return pd.DataFrame(
        {"y": y, "t_end": t_end, "ret": ret, "sigma": sigma.to_numpy()},
        index=df.index,
    )


BUILDERS = {
    "triple_barrier": triple_barrier,
    "fwd_return": fwd_return,
    "next_bar_sign": next_bar_sign,
}


def build_label(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if cfg.kind not in BUILDERS:
        raise ValueError(f"unknown label {cfg.kind!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[cfg.kind](df, cfg)
