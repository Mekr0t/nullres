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
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.17, 0.03, 0.20], vs_hold=[-0.02, -0.71, -0.15]),
    )
    assert outcome != "SURVIVED"
    assert any("TRANSFER" in n for n in notes)


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

    outcome, notes = verdict(
        grid,
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
        flip_rate=flips,
    )
    assert outcome == "KILLED"
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


def test_a_year_the_strategy_sat_flat_is_scored_not_dropped():
    """`by_period` drops a zero-variance period; the stability test must not.

    A year with no position has no variance, so it never reached the merge and
    vanished from the denominator. Live on the 1d config, where sma_cross and
    donchian are flat through 2022 — the year a flat book most obviously beats a
    collapsing benchmark. Which way the omission cuts depends on what the
    benchmark did that year, so the verdict moved for reasons unrelated to the
    strategy.
    """
    import numpy as np

    from nullres.config import load_config
    from nullres.robustness import period_stability
    from nullres.strategies.base import Context

    idx = pd.date_range("2021-01-01", "2023-12-31", freq="4h")
    rng = np.random.default_rng(0)
    opens = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    bars = pd.DataFrame({"open": opens, "high": opens * 1.01, "low": opens * 0.99,
                         "close": opens, "volume": 1.0, "trades": 1.0}, index=idx)

    label = pd.DataFrame({"sigma": 0.01, "ret": 0.0, "y": 1.0, "t_end": 0}, index=idx)
    cfg = load_config("configs/btc_4h.toml")
    ctx = Context(bars=bars, features=pd.DataFrame(index=idx), label=label,
                  cfg=cfg, oos_mask=pd.Series(True, index=idx), verbose=False)

    # Flat across the whole of 2022, having already gone flat mid-2021. The
    # exit must fall OUTSIDE 2022: a position change inside the year leaves a
    # cost tick, and that lone non-zero gives the year variance enough to
    # survive `by_period`. The real 1d case is this one — flat throughout, no
    # transition, so every return in the year is exactly zero.
    flat_year = pd.Series(1.0, index=idx)
    flat_year[flat_year.index >= pd.Timestamp("2021-07-01")] = 0.0
    flat_year[flat_year.index.year >= 2023] = 1.0

    class FlatIn2022:
        name = "flat_in_2022"

        def positions(self, _ctx):
            return flat_year

    import nullres.robustness as R

    real_build = R.build_strategy
    R.build_strategy = lambda name, params=None: (
        FlatIn2022() if name == "flat_in_2022" else real_build(name, params)
    )
    try:
        stability = period_stability(cfg, "flat_in_2022", ctx=ctx)
    finally:
        R.build_strategy = real_build

    assert "2022" in set(stability["period"]), "the flat year was dropped"
    row = stability.set_index("period").loc["2022"]
    assert row["sharpe"] == 0.0, "a flat year returned nothing, so Sharpe is 0"
    assert row["n_trades"] == 0
    # Excess is simply minus the benchmark's — the sign follows what hold did.
    assert row["excess_sharpe"] == pytest.approx(-row["sharpe_hold"])


def test_a_barely_active_period_is_not_scored_as_a_loss():
    """One cost tick is not a year's performance.

    A period where the book was flat except for a single position change has
    exactly one non-zero return — the fee. That tick is then the period's whole
    variance, and annualising it produces a confident Sharpe of -1.00 for a
    strategy that did nothing. Sibling of the partial-period bug `min_coverage`
    already handles.
    """
    import numpy as np

    from nullres.backtest.engine import backtest
    from nullres.backtest.metrics import by_period
    from nullres.config import CostConfig

    idx = pd.date_range("2022-01-01", "2023-12-31", freq="4h")
    opens = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(idx))))
    bars = pd.DataFrame({"open": opens, "high": opens * 1.01, "low": opens * 0.99,
                         "close": opens, "volume": 1.0, "trades": 1.0}, index=idx)

    # Long through 2022, then flat for all of 2023 apart from the exit itself.
    pos = pd.Series(0.0, index=idx)
    pos[pos.index.year == 2022] = 1.0
    result = backtest(bars, pos, CostConfig(fee_bps=10.0, slippage_bps=2.0))

    scored = by_period(result, bars_per_year=2_190)
    assert "2022" in set(scored["period"])
    assert "2023" not in set(scored["period"]), (
        "a year whose only non-zero return is one fee must not be scored"
    )


def test_a_broadly_robust_strategy_survives():
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert outcome == "SURVIVED", notes


def test_isolated_parameter_spike_is_killed():
    """One brilliant cell surrounded by losses is a fitting artefact."""
    outcome, notes = verdict(
        grid_df([-0.2, -0.1, 1.9, -0.3, -0.15]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert outcome == "KILLED"
    assert any("NEIGHBOURHOOD FAIL" in n for n in notes)


def test_one_good_year_is_inconclusive_not_killed():
    """The donchian 4h case, and the reason the verdict gained a third state.

    Excess Sharpes +0.21 +0.56 -0.50 -0.21 -0.27 beat hold in 2 of 5 years, so
    the old count gate returned KILLED. But their mean is -0.04 with p≈0.84:
    the evidence cannot separate this from a strategy exactly as good as
    holding, and at n=5 the gate fires on such a strategy half the time.
    Calling that KILLED published a coin flip as a finding.

    Killing donchian is still the right *decision* — the graveyard makes it on
    the magnitudes and on 2022 carrying the whole result. The machine simply no
    longer claims to have proven it.
    """
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.21, 0.56, -0.50, -0.21, -0.27]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert outcome == "INCONCLUSIVE"
    assert any("STABILITY INCONCLUSIVE" in n for n in notes)
    assert any("50% of the time" in n for n in notes), "must report its own power"


def test_a_consistently_worse_strategy_is_still_killed():
    """The magnitude test has to stay able to kill. sma_cross 4h is the case.

    Excess Sharpes -0.37 -0.05 -0.34 -0.29 -0.25: never beats hold, and the mean
    is decisively below zero (p≈0.01). That is evidence, not an unlucky count.
    """
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([-0.37, -0.05, -0.34, -0.29, -0.25]),
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert outcome == "KILLED"
    assert any("STABILITY FAIL" in n and "decisively worse" in n for n in notes)


def test_strategy_that_only_works_on_one_symbol_is_not_waved_through():
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([-0.2, -0.1, 0.4]),
    )
    assert outcome != "SURVIVED"
    assert any("TRANSFER" in n for n in notes)


def test_stability_judges_excess_not_absolute_return():
    """A long-only filter is positive in a bull market by construction.

    Being profitable is not the bar; beating the thing you'd have done anyway is.
    """
    # Positive every year, but worse than holding every year.
    losing_to_hold = stability_df([-0.1, -0.2, -0.15, -0.3, -0.05])
    assert (losing_to_hold["sharpe"] > 0).all()      # absolute test would pass

    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45, 0.6, 0.38]),
        losing_to_hold,
        transfer_df([0.3, 0.4, 0.35]),
    )
    assert outcome == "KILLED", "a strategy that never beats buy & hold must not survive"


def test_empty_inputs_fail_closed():
    """Missing evidence is not passing evidence — and not INCONCLUSIVE either.

    A battery that produced no numbers has not run. That is an error, and it
    must stay distinguishable from one that ran and could not decide.
    """
    outcome, notes = verdict(
        grid_df([0.4, 0.5, 0.45]),
        stability_df([]),
        transfer_df([np.nan, np.nan]),
    )
    assert outcome == "KILLED"
    assert any("STABILITY FAIL" in n for n in notes)
    assert any("TRANSFER FAIL" in n for n in notes)


def test_noise_field_test_uses_a_real_null_not_a_constant():
    """`0.8 x expected` was a significance level with no argument behind it.

    The flip count has a computable null — approximately Binomial(pairs,
    2p(1-p)) — so "is this grid smoother than chance" is a question with an
    actual test rather than a tunable constant.
    """
    from nullres.robustness import _is_noise_field

    # 50% positive -> expected flip rate 0.50. A grid that flips at exactly the
    # random rate carries no information, however many pairs it has.
    assert _is_noise_field(0.5, 0.50, n_pairs=31)
    # A perfectly smooth grid over the same pairs is decisively not random.
    assert not _is_noise_field(0.5, 0.0, n_pairs=31)

    # The measured derivatives case: 39% observed against 38% expected.
    assert _is_noise_field(0.75, 0.39, n_pairs=31)

    # Too few pairs to establish anything: the test must not claim "noise",
    # because it could not have proven smoothness either. The weak-count branch
    # of `verdict` is what catches these.
    assert not _is_noise_field(0.5, 0.33, n_pairs=3)

    # A 97%-positive grid can only flip ~6% of the time however it is arranged,
    # so zero flips would not be significant and the test has no power. It must
    # not condemn a strong result for having too few signs to shuffle.
    assert not _is_noise_field(0.97, 0.06, n_pairs=31)


def test_a_lucky_maximum_in_a_smooth_grid_is_not_a_pass():
    """A cell towering over its own neighbours is the shape of a fitted parameter.

    This needs no independence assumption: it asks where the headline sits
    inside its own surface, not whether the surface samples anything.
    """
    from nullres.robustness import sign_flip_pairs, sign_flip_rate

    # Smooth and all-positive, but one cell is wildly above the rest.
    grid = smooth_grid([[0.10, 0.12, 0.11], [0.09, 2.40, 0.13], [0.11, 0.10, 0.12]])
    flips = sign_flip_rate(grid, ["a", "b"])
    pairs = sign_flip_pairs(grid, ["a", "b"])

    outcome, notes = verdict(
        grid,
        stability_df([0.2, 0.3, 0.1, 0.25, 0.15]),
        transfer_df([0.3, 0.4, 0.35]),
        flip_rate=flips, flip_pairs=pairs,
    )
    assert outcome == "INCONCLUSIVE"
    assert any("NEIGHBOURHOOD INCONCLUSIVE" in n and "interquartile" in n
               for n in notes)


def test_sign_flip_pairs_counts_the_denominator():
    from nullres.robustness import sign_flip_pairs

    grid = smooth_grid([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # 3x3: 2 vertical pairs per column x 3, plus 2 horizontal per row x 3.
    assert sign_flip_pairs(grid, ["a", "b"]) == 12


def test_count_gate_power_is_reported_honestly():
    """The numbers that motivated the rework, pinned so they cannot be forgotten."""
    from nullres.robustness import count_gate_power

    assert count_gate_power(5) == pytest.approx(0.50, abs=0.01)
    assert count_gate_power(4) == pytest.approx(0.3125, abs=0.01)
    assert count_gate_power(3) == pytest.approx(0.50, abs=0.01)
    # Even 20 observations is not a strong gate at a 60% threshold: 12 of 20 is
    # barely above half, and a coin flip reaches it a quarter of the time. More
    # data helps far less than the round number suggests.
    assert count_gate_power(20) == pytest.approx(0.25, abs=0.01)
    assert count_gate_power(100) < 0.05


def test_magnitude_separates_what_the_count_conflates():
    """donchian and mean_reversion both beat hold in 40% of years on the 4h data.

    Their mean excess Sharpes are -0.04 and -1.13. Counting signs scores them
    identically; the magnitude does not.
    """
    from nullres.robustness import excess_magnitude

    donchian = [0.21, 0.56, -0.50, -0.21, -0.27]
    mean_rev = [-1.13, -2.2, 0.4, -1.9, 0.4]
    assert (np.array(donchian) > 0).mean() == (np.array(mean_rev) > 0).mean()

    d_mean, _ = excess_magnitude(donchian)
    m_mean, _ = excess_magnitude(mean_rev)
    assert d_mean > m_mean + 0.5, "the magnitudes are nothing alike"


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


def test_set_can_override_a_rule_parameter():
    """`params` is a dict, not a dataclass, and --set used to crash on it.

    `--set params.donchian.entry=48` died with a bare
    `AttributeError: 'dict' object has no attribute 'donchian'` — no clean
    message, and rule parameters are exactly what you want to vary from the
    command line.
    """
    from nullres.cli import _apply_override
    from nullres.config import load_config

    cfg = load_config("configs/btc_4h.toml")
    _apply_override(cfg, "params.donchian.entry=48")
    assert cfg.params["donchian"] == {"entry": 48, "exit": 48}
    assert isinstance(cfg.params["donchian"]["entry"], int), "type came from TOML"

    # A strategy with no params block yet.
    _apply_override(cfg, "params.mean_reversion.entry=2.5")
    assert cfg.params["mean_reversion"] == {"entry": 2.5}

    for bad in ("params.donchian=48", "params.a.b.c=1"):
        with pytest.raises(SystemExit, match="params"):
            _apply_override(cfg, bad)
