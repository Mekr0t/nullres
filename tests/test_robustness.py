"""The falsification battery must be able to fail things — including good-looking ones."""

import numpy as np
import pandas as pd
import pytest

from tbot.robustness import _valid, verdict


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


def transfer_df(sharpes):
    return pd.DataFrame({"symbol": [f"S{i}" for i in range(len(sharpes))],
                         "sharpe": sharpes, "total_return": sharpes,
                         "n_trades": [10] * len(sharpes), "vs_hold": sharpes,
                         "note": [""] * len(sharpes)})


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
    from tbot.config import load_config
    from tbot.strategies import build

    cfg = load_config("configs/btc_4h.toml")
    assert cfg.params["donchian"] == {"entry": 96, "exit": 48}
    strategy = build("donchian", cfg.params["donchian"])
    assert strategy.entry == 96 and strategy.exit == 48


def test_bad_strategy_params_are_rejected_clearly():
    from tbot.strategies import build

    with pytest.raises(ValueError, match="bad params"):
        build("donchian", {"nonexistent_param": 1})
