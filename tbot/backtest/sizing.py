"""Signal -> position. The step that decides whether an edge survives.

The baseline mapped `proba > 0.52` straight to a position and changed its mind
15,527 times in 47,502 bars. At 12bps a side that is ~18.6 in log cost — the
entire account, several times over, regardless of how good the model was.

Three mechanisms fix that, and all three are point-in-time safe:

  HYSTERESIS  Enter long above `long_entry`, but do not exit until the signal
              falls below the lower `long_exit`. A single band makes the
              position chatter every time the signal grazes the threshold.

  MIN HOLD    A hard floor on bars between state changes. This alone caps
              turnover at n/min_hold and is the bluntest, most reliable lever
              you have.

  VOL TARGET  Scale exposure by 1/volatility so risk, not notional, is constant.
              Improves risk-adjusted return and cuts trading in violent regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_min_hold(desired: pd.Series, min_hold: int) -> pd.Series:
    """Enforce a minimum bars-between-changes floor on an explicit position series.

    Used when a strategy already knows the side it wants (meta-labelling, rule
    ensembles) and only needs turnover control, not threshold logic.
    """
    # A NaN means "no prediction here" — carry the last known intent forward
    # rather than forcing a flat, which would show up as spurious turnover.
    d = desired.ffill().fillna(0.0).to_numpy(dtype="float64")
    out = np.zeros(len(d))
    current, held = 0.0, min_hold
    for i in range(len(d)):
        if d[i] != current and held >= min_hold:
            current, held = d[i], 0
        else:
            held += 1
        out[i] = current
    return pd.Series(out, index=desired.index)


def apply_vol_target(pos: pd.Series, sigma: pd.Series, cfg,
                     bars_per_year: int) -> pd.Series:
    """Scale exposure so annualised risk, not notional, is held constant."""
    if cfg.vol_target <= 0:
        return pos.clip(-cfg.max_leverage, cfg.max_leverage)
    ann_vol = sigma.to_numpy(dtype="float64") * np.sqrt(bars_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(ann_vol > 0, cfg.vol_target / ann_vol, 0.0)
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0)
    return (pos * np.clip(scale, 0.0, cfg.max_leverage)).clip(
        -cfg.max_leverage, cfg.max_leverage
    )


def signal_to_position(proba: pd.Series, cfg, sigma: pd.Series | None = None,
                       bars_per_year: int = 8_760) -> pd.Series:
    """Convert P(up) into a target position in [-max_leverage, +max_leverage].

    `proba` is indexed by bar; NaN means "no opinion", which holds the current
    position rather than forcing an exit.
    """
    p = proba.to_numpy(dtype="float64")
    n = len(p)
    state = np.zeros(n, dtype="float64")

    current = 0.0
    bars_held = 0

    for i in range(n):
        pi = p[i]
        desired = current

        if np.isfinite(pi):
            if current <= 0.0 and pi >= cfg.long_entry:
                desired = 1.0
            elif current >= 0.0 and cfg.allow_short and pi <= cfg.short_entry:
                desired = -1.0
            elif current > 0.0 and pi < cfg.long_exit:
                desired = 0.0
            elif current < 0.0 and pi > cfg.short_exit:
                desired = 0.0

        if desired != current and bars_held >= cfg.min_hold:
            current = desired
            bars_held = 0
        else:
            bars_held += 1

        state[i] = current

    pos = pd.Series(state, index=proba.index)

    if cfg.vol_target > 0:
        if sigma is None:
            raise ValueError("vol_target requires a sigma series")
        return apply_vol_target(pos, sigma, cfg, bars_per_year)
    return pos.clip(-cfg.max_leverage, cfg.max_leverage)
