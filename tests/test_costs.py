"""The cost arithmetic that should precede any modelling work."""

import numpy as np
import pytest

from nullres.costs import (
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


def test_gaussian_move_is_wrong_in_both_directions_so_measure_it():
    """The error changes sign with the horizon, so no constant can fix it.

    Fat tails make sigma large relative to a typical SHORT move, so the formula
    overstates it — and the modelled accuracy requirement comes out too
    forgiving, which is the dangerous direction for a tool that exists to kill
    ideas. Aggregation pulls longer horizons toward normal and drift pushes them
    past it, so the same formula understates long moves.
    """
    from nullres.costs import empirical_abs_move, expected_abs_move

    rng = np.random.default_rng(0)
    # Fat-tailed, driftless: Student-t has the same variance story, heavier tails.
    fat = rng.standard_t(df=3, size=200_000) * 0.004

    sigma = float(fat.std())
    short_ratio = empirical_abs_move(fat, 1) / expected_abs_move(sigma, 1)
    long_ratio = empirical_abs_move(fat, 200) / expected_abs_move(sigma, 200)

    assert short_ratio < 0.95, "fat tails must make the typical short move smaller"
    assert long_ratio > short_ratio, "aggregation pulls the ratio back toward 1"


def test_empirical_breakeven_agrees_with_the_closed_form_on_gaussian_data():
    """The bisection must reproduce the algebra when the assumption holds."""
    from nullres.costs import breakeven_hold_empirical

    rng = np.random.default_rng(1)
    normal = rng.normal(0.0, 0.006, 400_000)

    closed = breakeven_hold(float(normal.std()), 0.52, 10, 2)
    measured = breakeven_hold_empirical(normal, 0.52, 10, 2)
    assert measured == pytest.approx(closed, rel=0.15), (
        f"on Gaussian data the two must agree: {closed:.0f} vs {measured:.0f}"
    )


def test_empirical_move_grows_with_the_horizon():
    """Monotonicity is what makes the bisection in breakeven_hold valid."""
    from nullres.costs import empirical_abs_move

    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, 50_000)
    moves = [empirical_abs_move(r, h) for h in (1, 5, 20, 100, 500)]
    assert moves == sorted(moves)


def test_budget_table_reports_duration_at_the_configured_bar_size():
    """The printed `~duration` column has to know how long a bar is.

    It did not — it divided bars by 24, i.e. assumed hourly. `nullres budget`
    on the 1d config therefore claimed 51% accuracy broke even in 0.9 days,
    against a true ~21 days. The number that command exists to deliver, wrong
    on every config except the one it was written against.
    """
    from nullres.costs import budget_table

    printed = {}
    for interval, hours, bars_per_year in (("1h", 1, 8_760), ("4h", 4, 2_190),
                                           ("1d", 24, 365)):
        sigma = SIGMA_1H_BTC * np.sqrt(hours)
        table = budget_table(sigma, 10, 2, hours_per_bar=8_760 / bars_per_year)
        row = next(ln for ln in table.splitlines() if ln.strip().startswith("51%"))
        printed[interval] = row.split()[-2:]

    days = [float(v[0]) for v in printed.values()]
    assert all(v[1] == "days" for v in printed.values()), printed
    assert max(days) / min(days) < 1.05, (
        f"duration must be timeframe-invariant, got {printed}"
    )
    assert 18 < days[0] < 24, printed


def test_format_duration_switches_units():
    from nullres.costs import format_duration

    assert format_duration(6) == "6.0 hours"
    assert format_duration(24 * 21) == "21.0 days"
    assert "months" in format_duration(24 * 200)
    assert format_duration(float("inf")) == "never"


def test_breakeven_duration_is_invariant_to_timeframe():
    """Moving to a slower timeframe does NOT reduce costs. Easy to get wrong.

    Volatility scales as sqrt(bar length), so break-even bars scale as
    1/bar_length, so break-even *duration* is constant. Measured on BTCUSDT:

        1h   sigma 0.6715%   ->  502 bars  = 20.9 days
        4h   sigma 1.3430%   ->  133 bars  = 22.2 days
        1d   sigma 3.2912%   ->   21 bars  = 21.0 days

    What matters is how long you hold in wall-clock time, not how many bars
    that happens to be. Choose a timeframe for signal quality and sample size;
    choose a holding period to beat costs. They are separate decisions.
    """
    hours_per_bar = {"1h": 1, "4h": 4, "1d": 24}
    durations = {}
    for name, hours in hours_per_bar.items():
        sigma = SIGMA_1H_BTC * np.sqrt(hours)      # sqrt-of-time scaling
        bars = breakeven_hold(sigma, 0.51, 10, 2)
        durations[name] = bars * hours             # in hours

    values = list(durations.values())
    assert max(values) / min(values) < 1.01, (
        f"break-even duration should be timeframe-invariant, got {durations}"
    )
