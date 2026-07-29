"""Vectorised backtest with explicit execution timing.

The timing convention, stated once and obeyed everywhere:

    position[t]  is decided using information up to the CLOSE of bar t
    the resulting trade is FILLED at the OPEN of bar t+1
    so position[t] earns the return from OPEN[t+1] to OPEN[t+2]
    and pays cost on |position[t] - position[t-1]| at the fill

Assuming a fill at the close of the bar you just predicted is the second most
common way to invent a profitable strategy that does not exist. There is no
mechanism by which you observe a bar's close and also trade at it.

What this engine does NOT model, and you should not forget: partial fills,
order book depth (it assumes your size is small enough not to move the market),
funding rates on perpetuals, exchange downtime, and the fact that slippage is
worst exactly when your signal is strongest.

One approximation worth naming: cost is subtracted in log space
(`net = gross - turnover * rate`) although `rate` is a simple fraction. The
exact charge is `log(1 - rate)`, which is very slightly larger. The gap is
`rate^2/2` per unit turnover — about 3e-6 at 24bps, so under half a percent of
equity even at a thousand round trips, and invisible at the turnover levels
anything here survives. It flatters high-turnover strategies fractionally, which
are the ones already dying by an order of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series          # cumulative equity, starts at 1.0
    returns: pd.Series         # per-bar NET log returns
    gross: pd.Series           # per-bar log returns before cost
    position: pd.Series        # target position per bar
    turnover: pd.Series        # |change in position| per bar
    cost: pd.Series            # per-bar cost in log terms

    @property
    def n_trades(self) -> int:
        return int((self.turnover > 1e-12).sum())

    @property
    def total_cost(self) -> float:
        return float(self.cost.sum())


def backtest(bars: pd.DataFrame, position: pd.Series, cost_cfg) -> BacktestResult:
    """Run the position series against the bars and charge realistic costs."""
    if not position.index.equals(bars.index):
        raise ValueError("position and bars must share an index")

    pos = position.astype("float64").fillna(0.0)

    log_open = np.log(bars["open"])
    # OPEN[t+1] -> OPEN[t+2]. The last two bars have no forward fill and earn 0.
    fwd = (log_open.shift(-2) - log_open.shift(-1)).fillna(0.0)

    turnover = pos.diff()
    turnover.iloc[0] = pos.iloc[0]        # opening the first position is a trade
    turnover = turnover.abs()

    rate = (cost_cfg.fee_bps + cost_cfg.slippage_bps) / 10_000.0
    cost = turnover * rate

    gross = pos * fwd
    net = gross - cost
    equity = np.exp(net.cumsum())

    return BacktestResult(
        equity=equity, returns=net, gross=gross,
        position=pos, turnover=turnover, cost=cost,
    )


def restrict(result: BacktestResult, mask) -> BacktestResult:
    """Narrow a result to a subset of bars, rebasing equity to the first of them.

    Strategies are evaluated over the whole frame with positions ZEROED outside
    the out-of-sample window, not absent from it. A result therefore carries a
    block of structural zero returns, and summarising across them is not a
    harmless dilution: zeros scale the mean by their share `f` and the standard
    deviation by `sqrt(f)`, so the reported Sharpe comes out as

        sharpe_full = sqrt(f) * sharpe_oos

    which is 0.88x on the 4h config — every strategy understated by the same
    12%. The t-statistic is immune (the factor cancels top and bottom), which is
    why this survived a test suite that checks t-stats.

    **The closing trade is charged, not dropped.** A position still open on the
    last in-window bar has to be liquidated, and that trade lands on the first
    bar AFTER the window — so simply masking discarded it, leaving every book
    unbilled for getting out and `n_trades` short by one. Buy & hold reported a
    single trade for a round trip.

    Nothing needs to be invented to fix it: the exit already exists in the
    unmasked result, priced at the configured rate. Its cost and turnover are
    moved onto the final in-window bar, while its RETURN stays outside — you pay
    to leave, you do not earn the bar you left in. When the window runs to the
    end of the frame there is no following bar and nothing to re-attribute.
    """
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    r = result.returns[mask]
    turnover, cost = result.turnover[mask], result.cost[mask]

    if selected.size:
        exit_bar = selected[-1] + 1
        if exit_bar < len(result.cost):
            exit_cost = float(result.cost.iloc[exit_bar])
            if exit_cost > 0:
                turnover = turnover.copy()
                cost = cost.copy()
                turnover.iloc[-1] += float(result.turnover.iloc[exit_bar])
                cost.iloc[-1] += exit_cost
                r = r.copy()
                r.iloc[-1] -= exit_cost

    return BacktestResult(
        equity=np.exp(r.cumsum()),
        returns=r,
        gross=result.gross[mask],
        position=result.position[mask],
        turnover=turnover,
        cost=cost,
    )


def buy_and_hold(bars: pd.DataFrame, cost_cfg) -> BacktestResult:
    """Reference strategy: fully long from the first fill, never trades again."""
    pos = pd.Series(1.0, index=bars.index)
    return backtest(bars, pos, cost_cfg)
