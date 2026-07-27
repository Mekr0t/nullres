"""Rule-based strategies — the benchmarks any model has to clear.

These are deliberately simple and deliberately not tuned. Their job is to set
the bar. A tuned rule is not a benchmark, it is another overfit strategy with
fewer parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tbot.backtest.sizing import apply_rebalance_band
from tbot.strategies.base import Context, mask_to_oos


class BuyHold:
    name = "buy_hold"

    def positions(self, ctx: Context) -> pd.Series:
        return mask_to_oos(pd.Series(1.0, index=ctx.bars.index), ctx)


class SMACross:
    """Long when the fast average is above the slow one, flat otherwise.

    The oldest systematic strategy there is. It trades rarely, so costs barely
    register, which is exactly why it is hard to beat.
    """

    name = "sma_cross"

    def __init__(self, fast: int = 50, slow: int = 200, allow_short: bool = False):
        self.fast, self.slow, self.allow_short = fast, slow, allow_short

    def positions(self, ctx: Context) -> pd.Series:
        close = ctx.bars["close"]
        f = close.rolling(self.fast).mean()
        s = close.rolling(self.slow).mean()
        raw = np.where(f > s, 1.0, -1.0 if self.allow_short else 0.0)
        pos = pd.Series(raw, index=close.index).where(s.notna(), 0.0)
        return mask_to_oos(pos, ctx)


class DonchianBreakout:
    """Long on a new N-bar high, flat on a new M-bar low. Classic trend following."""

    name = "donchian"

    def __init__(self, entry: int = 96, exit: int = 48):
        self.entry, self.exit = entry, exit

    def positions(self, ctx: Context) -> pd.Series:
        high, low, close = ctx.bars["high"], ctx.bars["low"], ctx.bars["close"]
        # shift(1) so the channel is built from bars strictly BEFORE this one;
        # comparing today's close to a high that includes today is circular.
        upper = high.rolling(self.entry).max().shift(1)
        lower = low.rolling(self.exit).min().shift(1)

        state = np.zeros(len(close))
        current = 0.0
        c, u, l = close.to_numpy(), upper.to_numpy(), lower.to_numpy()
        for i in range(len(c)):
            if np.isfinite(u[i]) and c[i] > u[i]:
                current = 1.0
            elif np.isfinite(l[i]) and c[i] < l[i]:
                current = 0.0
            state[i] = current
        return mask_to_oos(pd.Series(state, index=close.index), ctx)


class VolTargetHold:
    """Always long, but sized so that RISK is constant rather than notional.

    This strategy makes no directional claim at all. It exists because of a
    measured asymmetry in the data:

        lag-1 autocorrelation of returns      -0.029    (noise)
        lag-1 autocorrelation of |returns|    +0.227    (strong)
        lag-1 autocorrelation of 30-bar vol   +0.992    (near-deterministic)

    Direction is unpredictable; volatility is extremely persistent. So rather
    than guessing which way the market goes, hold it continuously and vary the
    size by 1/sigma — cutting exposure when the market is violent and restoring
    it when it calms.

    Note what this can and cannot do. It does not improve expected return; a
    lower-volatility path with the same drift compounds better, but the edge
    comes from risk management, not prediction. Judge it on Sharpe and
    drawdown, and expect total return at or slightly below buy & hold.

    `max_leverage=1.0` by default, so in calm regimes it is simply long and
    never borrows. That makes it deliverable in a spot account with no margin.
    """

    name = "vol_target"

    def __init__(self, target: float = 0.50, vol_window: int = 30,
                 band: float = 0.10, max_leverage: float = 1.0):
        self.target = target
        self.vol_window = vol_window
        self.band = band
        self.max_leverage = max_leverage

    def positions(self, ctx: Context) -> pd.Series:
        logret = np.log(ctx.bars["close"]).diff()
        # Rolling stdev at bar t uses bars <= t only.
        sigma = logret.rolling(self.vol_window).std()
        annualised = sigma * np.sqrt(ctx.cfg.data.bars_per_year)

        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (self.target / annualised).replace([np.inf, -np.inf], np.nan)
        raw = raw.clip(0.0, self.max_leverage).fillna(0.0)

        return mask_to_oos(apply_rebalance_band(raw, self.band), ctx)


class MeanReversionZ:
    """Fade stretched moves: long when the z-score is deeply negative, and vice versa.

    Works in ranging regimes, gets destroyed in trending ones. Included partly
    as a benchmark and partly because its failure mode is instructive.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 72, entry: float = 2.0, exit: float = 0.5):
        self.window, self.entry, self.exit = window, entry, exit

    def positions(self, ctx: Context) -> pd.Series:
        close = ctx.bars["close"]
        z = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        state = np.zeros(len(z))
        current = 0.0
        zv = z.to_numpy()
        for i in range(len(zv)):
            if not np.isfinite(zv[i]):
                state[i] = current
                continue
            if current == 0.0:
                if zv[i] <= -self.entry:
                    current = 1.0
                elif zv[i] >= self.entry:
                    current = -1.0
            elif current > 0 and zv[i] >= -self.exit:
                current = 0.0
            elif current < 0 and zv[i] <= self.exit:
                current = 0.0
            state[i] = current
        return mask_to_oos(pd.Series(state, index=z.index), ctx)
