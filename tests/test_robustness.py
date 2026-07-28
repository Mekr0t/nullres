"""The falsification battery must be able to fail things — including good-looking ones."""

import numpy as np
import pandas as pd
import pytest

from nullres.robustness import _valid, verdict


def grid_df(sharpes):
    return pd.DataFrame({"entry": range(len(sharpes)), "sharpe": sharpes,
                         "total_return": sharpes, "n_trades": [10] * len(sharpes)})


def stability_df(excess):
    return pd.DataFrame({
        "period": [str(2020 + i) for i in range(len(excess))],
        "sharpe": [1.0] * len(excess),
        "sharpe_hold": [1.0 - e for e in excess],
        "excess_sharpe": excess,
        "total_return": [0.1] * len(excess),
        "total_return_hold": [0.1] * len(excess),
        "n_trades": [10] * len(excess),
        "bars": [1000] * len(excess),
    })


def transfer_df(sharpes, vs_hold=None):
    return pd.DataFrame({"symbol": [f"S{i}" for i in range(len(sharpes))],
                         "sharpe": sharpes, "total_return": sharpes,
                         "n_trades": [10] * len(sharpes),
                         "vs_hold": sharpes if vs_hold is None else vs_hold,
                         "note": [""] * len(sharpes)})


def test_transfer_judges_excess_not_absolute_return():
    """Positive on every symbol, worse than holding every symbol.

    The measured case: BNBUSDT scored Sharpe 0.03 — technically positive, and
    0.71 below simply holding it. In a sample where every asset rose, "made
    money" is not evidence the rule did anything.
    """
    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.17, 0.03, 0.20], vs_hold=[-0.02, -0.71, -0.15]),
    )
    assert not ok
    assert any("TRANSFER FAIL" in n for n in notes)


def smooth_grid(values_2d):
    """Grid frame from a 2D list, for the sign-flip tests."""
    rows = []
    for i, row in enumerate(values_2d):
        for j, v in enumerate(row):
            rows.append({"a": i, "b": j, "sharpe": v,
                         "total_return": v, "n_trades": 10})
    return pd.DataFrame(rows)


def test_sign_flip_rate_is_zero_on_a_coherent_region():
    from nullres.robustness import sign_flip_rate

    grid = smooth_grid([[0.3, 0.4, 0.5], [0.4, 0.6, 0.5], [0.2, 0.3, 0.4]])
    assert sign_flip_rate(grid, ["a", "b"]) == 0.0


def test_sign_flip_rate_is_high_on_a_checkerboard():
    """The case that fooled the old 'count positive cells' criterion."""
    from nullres.robustness import sign_flip_rate

    grid = smooth_grid([[0.8, -0.2, 0.9], [-0.7, 0.6, -0.3], [0.5, -0.4, 0.7]])
    assert sign_flip_rate(grid, ["a", "b"]) == 1.0


def test_checkerboard_is_killed_even_when_mostly_positive():
    """60% of cells positive, median positive — but the sign is noise."""
    from nullres.robustness import sign_flip_rate

    grid = smooth_grid([[0.8, -0.2, 0.9], [0.7, 0.6, -0.3], [0.5, -0.4, 0.7]])
    flips = sign_flip_rate(grid, ["a", "b"])
    assert (grid["sharpe"] > 0).mean() > 0.6 and grid["sharpe"].median() > 0

    ok, notes = verdict(
        grid,
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
        flip_rate=flips,
    )
    assert not ok
    assert any("noise field" in n for n in notes)


def test_noise_baseline_scales_with_the_positive_fraction():
    """A fixed flip threshold is wrong; the baseline is 2p(1-p).

    A grid that is 95% positive cannot flip more than ~10% of the time no
    matter how the signs are arranged, so a flat 30% cutoff would pass it
    unconditionally. The random-placement baseline adapts.
    """
    from nullres.robustness import _is_noise_field

    # 50/50 split scattered at random -> expected 0.50
    assert _is_noise_field(0.5, 0.50)
    assert not _is_noise_field(0.5, 0.15)

    # 95% positive -> expected 0.095; 8% flips is NOT smooth relative to that
    assert _is_noise_field(0.95, 0.09)
    assert not _is_noise_field(0.95, 0.02)

    # Effectively single-signed grids have nothing to test.
    assert not _is_noise_field(1.0, 0.0)
    assert not _is_noise_field(0.99, 0.0)


def test_the_real_ml_grids_are_indistinguishable_from_noise():
    """Measured on BTCUSDT 4h with derivatives — the observed values.

    ml_direction: 75% positive, 39% flips vs 37.5% expected
    ml_meta:      60% positive, 48% flips vs 48.0% expected
    """
    from nullres.robustness import _is_noise_field

    assert _is_noise_field(0.75, 0.39)
    assert _is_noise_field(0.60, 0.48)


def test_partial_periods_are_excluded_from_stability():
    """An OOS window starting in December must not create a one-month 'year'.

    Annualising one month produced a buy & hold Sharpe of -4.68 off two trades,
    which then counted as a full observation in the stability verdict.
    """
    import numpy as np
    from nullres.backtest.engine import backtest
    from nullres.backtest.metrics import by_period
    from nullres.config import CostConfig

    idx = pd.date_range("2021-12-01", "2023-12-31", freq="4h")
    opens = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx))))
    bars = pd.DataFrame({"open": opens, "high": opens * 1.01, "low": opens * 0.99,
                         "close": opens, "volume": 1.0, "trades": 1.0}, index=idx)
    result = backtest(bars, pd.Series(1.0, index=idx),
                      CostConfig(fee_bps=0.0, slippage_bps=0.0))

    periods = by_period(result, bars_per_year=2_190)["period"].tolist()
    assert "2021" not in periods, "one month of December must not count as a year"
    assert {"2022", "2023"} <= set(periods)


def test_a_broadly_robust_strategy_survives():
    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert ok, notes


def test_isolated_parameter_spike_is_killed():
    """One brilliant cell surrounded by losses is a fitting artefact."""
    ok, notes = verdict(
        grid_df([-0.2, -0.1, 1.9, -0.3, -0.15]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert not ok
    assert any("NEIGHBOURHOOD FAIL" in n for n in notes)


def test_one_good_year_is_killed():
    """The donchian 4h case: a strong average carried by a single period."""
    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.21, 0.56, -0.50, -0.21, -0.27]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert not ok
    assert any("STABILITY FAIL" in n for n in notes)


def test_strategy_that_only_works_on_one_symbol_is_killed():
    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([-0.2, -0.1, 0.4]),
    )
    assert not ok
    assert any("TRANSFER FAIL" in n for n in notes)


def test_stability_judges_excess_not_absolute_return():
    """A long-only filter is positive in a bull market by construction.

    Being profitable is not the bar; beating the thing you'd have done anyway is.
    """
    # Positive every year, but worse than holding every year.
    losing_to_hold = stability_df([-0.1, -0.2, -0.15, -0.3, -0.05])
    assert (losing_to_hold["sharpe"] > 0).all()      # absolute test would pass

    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        losing_to_hold,
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert not ok, "a strategy that never beats buy & hold must not survive"


def test_empty_inputs_fail_closed():
    """Missing evidence is not passing evidence."""
    ok, notes = verdict(
        grid_df([0.4, 0.5, 0.45]),
        stability_df([]),
        transfer_df([np.nan, np.nan]),
    )
    assert not ok
    assert any("STABILITY FAIL" in n for n in notes)
    assert any("TRANSFER FAIL" in n for n in notes)


@pytest.mark.parametrize("name,combo,expected", [
    ("sma_cross", {"fast": 50, "slow": 200}, True),
    ("sma_cross", {"fast": 200, "slow": 50}, False),
    ("sma_cross", {"fast": 100, "slow": 100}, False),
    ("donchian", {"entry": 96, "exit": 48}, True),
    ("donchian", {"entry": 48, "exit": 96}, False),
])
def test_invalid_parameter_combinations_are_rejected(name, combo, expected):
    assert _valid(name, combo) is expected


def test_strategy_params_flow_from_config():
    """A rule's parameters must be reproducible from the config alone."""
    from nullres.config import load_config
    from nullres.strategies import build

    cfg = load_config("configs/btc_4h.toml")
    assert cfg.params["donchian"] == {"entry": 96, "exit": 48}
    strategy = build("donchian", cfg.params["donchian"])
    assert strategy.entry == 96 and strategy.exit == 48


def test_bad_strategy_params_are_rejected_clearly():
    from nullres.strategies import build

    with pytest.raises(ValueError, match="bad params"):
        build("donchian", {"nonexistent_param": 1})
