"""The backtest engine must be boringly, verifiably correct."""

import numpy as np
import pandas as pd
import pytest

from nullres.backtest.engine import backtest
from nullres.config import CostConfig


def bars_from_opens(opens):
    n = len(opens)
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o * 1.01 for o in opens],
            "low": [o * 0.99 for o in opens],
            "close": opens,
            "volume": [1.0] * n,
            "trades": [1.0] * n,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


NO_COST = CostConfig(fee_bps=0.0, slippage_bps=0.0)


def test_flat_position_earns_nothing():
    bars = bars_from_opens([100, 110, 120, 130, 140])
    pos = pd.Series(0.0, index=bars.index)
    result = backtest(bars, pos, NO_COST)
    assert result.equity.iloc[-1] == pytest.approx(1.0)
    assert result.total_cost == 0.0


def test_return_uses_next_open_to_the_one_after():
    """position[t] must earn log(open[t+2]) - log(open[t+1]), nothing else."""
    opens = [100.0, 100.0, 110.0, 110.0, 110.0]
    bars = bars_from_opens(opens)

    pos = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=bars.index)
    result = backtest(bars, pos, NO_COST)
    # Held at bar 0 -> earns open[1] -> open[2] == log(110/100).
    assert result.gross.iloc[0] == pytest.approx(np.log(110 / 100))
    assert result.gross.iloc[1:].abs().sum() == pytest.approx(0.0)


def test_cannot_trade_the_bar_it_predicts():
    """A position taken at the bar of a jump must NOT capture that jump.

    This is the check that catches the classic 'fill at the close you just
    predicted' bug. The jump happens between open[1] and open[2]... so only a
    position decided at bar 0 gets it. A position decided at bar 1 is too late.
    """
    bars = bars_from_opens([100.0, 100.0, 200.0, 200.0, 200.0])
    late = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=bars.index)
    result = backtest(bars, late, NO_COST)
    assert result.equity.iloc[-1] == pytest.approx(1.0), "captured a move it could not have traded"


def test_costs_charged_on_every_change():
    bars = bars_from_opens([100.0] * 6)
    pos = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, 0.0], index=bars.index)
    cost_cfg = CostConfig(fee_bps=10.0, slippage_bps=2.0)
    result = backtest(bars, pos, cost_cfg)

    # turnover: open 1.0, hold 0, flip 2.0, hold 0, close 1.0, hold 0 -> 4.0
    assert result.turnover.sum() == pytest.approx(4.0)
    assert result.total_cost == pytest.approx(4.0 * 12 / 10_000)
    assert result.n_trades == 3


def test_flat_price_with_costs_only_loses():
    bars = bars_from_opens([100.0] * 50)
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], 50), index=bars.index)
    result = backtest(bars, pos, CostConfig(fee_bps=10.0, slippage_bps=2.0))
    assert result.equity.iloc[-1] < 1.0, "trading a flat market must lose to fees"


def test_engine_is_causal():
    """Rewriting the future must not change the past's equity curve."""
    rng = np.random.default_rng(1)
    opens = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    bars_a = bars_from_opens(opens)

    tampered = list(opens)
    tampered[150:] = [x * 5 for x in tampered[150:]]
    bars_b = bars_from_opens(tampered)

    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], 200), index=bars_a.index)
    eq_a = backtest(bars_a, pos, NO_COST).equity
    eq_b = backtest(bars_b, pos, NO_COST).equity
    # Bar 147 is the last one whose return window (open[148]->open[149]) lies
    # entirely before the tampering; bar 148 already reaches open[150].
    assert np.allclose(eq_a.iloc[:148], eq_b.iloc[:148])
    assert not np.allclose(eq_a.iloc[148:], eq_b.iloc[148:])


def test_index_mismatch_is_rejected():
    bars = bars_from_opens([100.0] * 5)
    pos = pd.Series(1.0, index=pd.date_range("2021-01-01", periods=5, freq="h"))
    with pytest.raises(ValueError, match="share an index"):
        backtest(bars, pos, NO_COST)
