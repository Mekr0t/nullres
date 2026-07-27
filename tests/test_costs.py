"""The cost arithmetic that should precede any modelling work."""

import numpy as np
import pytest

from tbot.costs import (
    breakeven_hold,
    expected_abs_move,
    required_accuracy,
    round_trip_cost,
)

SIGMA_1H_BTC = 0.0067      # measured on BTCUSDT 1h, 2020-2025


def test_round_trip_charges_both_sides():
    assert round_trip_cost(10.0, 2.0) == pytest.approx(0.0024)


def test_expected_move_scales_with_sqrt_time():
    a = expected_abs_move(0.01, 1)
    b = expected_abs_move(0.01, 100)
    assert b / a == pytest.approx(10.0)


def test_short_holds_demand_implausible_accuracy():
    """The finding that explains the baseline's -100%."""
    need = required_accuracy(SIGMA_1H_BTC, hold_bars=1, fee_bps=10, slippage_bps=2)
    assert need > 0.70, "flipping every bar should require an absurd hit rate"


def test_longer_holds_demand_less_accuracy():
    holds = [1, 6, 24, 168, 720]
    needs = [required_accuracy(SIGMA_1H_BTC, h, 10, 2) for h in holds]
    assert needs == sorted(needs, reverse=True)
    assert needs[-1] < 0.52


def test_breakeven_hold_inverts_required_accuracy():
    """The two functions must agree — they are the same equation rearranged."""
    for acc in (0.51, 0.53, 0.57):
        h = breakeven_hold(SIGMA_1H_BTC, acc, 10, 2)
        assert required_accuracy(SIGMA_1H_BTC, h, 10, 2) == pytest.approx(acc, rel=1e-9)


def test_no_edge_can_never_break_even():
    assert breakeven_hold(SIGMA_1H_BTC, 0.50, 10, 2) == float("inf")
    assert breakeven_hold(SIGMA_1H_BTC, 0.40, 10, 2) == float("inf")


def test_zero_cost_breaks_even_instantly():
    assert breakeven_hold(SIGMA_1H_BTC, 0.51, 0.0, 0.0) == pytest.approx(0.0)


def test_a_51_percent_model_needs_weeks_at_1h():
    """The headline number for hourly crypto, pinned so it cannot silently drift."""
    h = breakeven_hold(SIGMA_1H_BTC, 0.51, 10, 2)
    assert 300 < h < 800, f"expected several hundred bars, got {h:.0f}"
