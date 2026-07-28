"""Cross-sectional panel — new leak surfaces, new failure modes.

The single-asset engine's invariants do not carry over. A panel can leak in
ways a time series cannot: across symbols at the same timestamp, or by quietly
dropping the assets that died.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbot.config import SplitConfig
from tbot.crosssec import (
    UNIVERSE_2021_12,
    _cross_sectional_rank,
    _relative_label,
    backtest_panel,
    panel_positions,
    time_folds,
)


def make_log_open(n=400, symbols=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", name="ts")
    return pd.DataFrame(
        {s: np.log(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))) for s in symbols},
        index=idx,
    )


def test_universe_contains_a_delisted_symbol():
    """Survivorship guard. If this fails the whole exercise is fiction."""
    assert "LUNAUSDT" in UNIVERSE_2021_12, (
        "the fixed universe must include an asset that died, or the "
        "cross-sectional result is a test of survivors"
    )


def test_ranks_are_computed_within_a_timestamp_not_across_time():
    """A rank must depend only on the other symbols at that instant."""
    idx = pd.date_range("2022-01-01", periods=3, freq="4h", name="ts")
    frame = pd.concat(
        {"A": pd.DataFrame({"f": [1.0, 100.0, 1.0]}, index=idx),
         "B": pd.DataFrame({"f": [2.0, 200.0, 0.5]}, index=idx)},
        names=["symbol", "ts"],
    ).swaplevel().sort_index()
    # Only 2 symbols, so relax the minimum-width filter for this unit test.
    ranked = frame.groupby(level="ts").rank(pct=True)

    # B > A at t0 and t1 despite the 100x level change; A > B at t2.
    assert ranked.loc[(idx[0], "B"), "f"] > ranked.loc[(idx[0], "A"), "f"]
    assert ranked.loc[(idx[1], "B"), "f"] > ranked.loc[(idx[1], "A"), "f"]
    assert ranked.loc[(idx[2], "A"), "f"] > ranked.loc[(idx[2], "B"), "f"]


def test_a_market_wide_move_does_not_change_any_rank():
    """The point of ranking: the common factor cancels.

    When everything sells off together each symbol's return z-score collapses,
    but the ordering is untouched — which is why a dollar-neutral book built on
    ranks is not implicitly betting on the market.
    """
    base = make_log_open()
    frame = pd.concat({s: pd.DataFrame({"r": base[s].diff()}) for s in base.columns},
                      names=["symbol", "ts"]).swaplevel().sort_index()
    shocked = base.add(-0.5, axis=0)      # every symbol down 50 log %
    frame2 = pd.concat({s: pd.DataFrame({"r": shocked[s].diff()}) for s in shocked.columns},
                       names=["symbol", "ts"]).swaplevel().sort_index()

    a = frame.groupby(level="ts").rank(pct=True).dropna()
    b = frame2.groupby(level="ts").rank(pct=True).dropna()
    pd.testing.assert_frame_equal(a, b)


def test_label_is_balanced_in_every_regime():
    """Half the universe beats its own median, in a crash as in a boom."""
    rising = make_log_open(seed=1).add(np.linspace(0, 2, 400), axis=0)
    crashing = make_log_open(seed=1).add(np.linspace(0, -2, 400), axis=0)
    for frame in (rising, crashing):
        y = _relative_label(frame, horizon=12)
        assert 0.45 < y.mean() < 0.55, f"base rate {y.mean():.3f} is not balanced"


def test_label_never_uses_the_bar_it_is_decided_on():
    """Return runs open[t+1] -> open[t+1+h]; tampering earlier must not matter."""
    base = make_log_open(seed=2)
    y_base = _relative_label(base, horizon=12)

    tampered = base.copy()
    tampered.iloc[:100] += 5.0          # rewrite the distant past
    y_after = _relative_label(tampered, horizon=12)

    late = y_base.index.get_level_values("ts") > base.index[120]
    pd.testing.assert_series_equal(y_base[late], y_after[y_after.index.isin(y_base[late].index)])


def test_time_folds_keep_all_symbols_of_a_timestamp_together():
    """A positional split would put BTC's Tuesday in train and ETH's in test."""
    times = pd.date_range("2022-01-01", periods=3_000, freq="4h")
    cfg = SplitConfig(n_folds=4, min_train=1_000, embargo=0)
    for train_times, test_times in time_folds(times, cfg, horizon=42):
        assert train_times.max() < test_times.min()
        assert not set(train_times) & set(test_times)


def test_time_folds_purge_the_label_horizon():
    times = pd.date_range("2022-01-01", periods=3_000, freq="4h")
    cfg = SplitConfig(n_folds=4, min_train=1_000, embargo=10)
    horizon = 42
    step = times[1] - times[0]
    for train_times, test_times in time_folds(times, cfg, horizon):
        gap = (test_times.min() - train_times.max()) / step
        assert gap >= horizon + 10


def test_book_is_dollar_neutral_and_delisted_symbols_are_flat():
    from tbot.crosssec import Panel

    symbols = ["A", "B", "C", "D", "E", "F"]
    times = pd.date_range("2022-01-01", periods=100, freq="4h")
    ret = pd.DataFrame(0.01, index=times, columns=symbols)
    ret.loc[times[50]:, "F"] = np.nan          # F delists halfway

    rng = np.random.default_rng(0)
    proba = pd.Series(
        rng.uniform(0, 1, len(times) * len(symbols)),
        index=pd.MultiIndex.from_product([times, symbols], names=["ts", "symbol"]),
    )
    panel = Panel(features=pd.DataFrame(), y=pd.Series(dtype=float), ret_next=ret,
                  funding=pd.DataFrame(0.0, index=times, columns=symbols),
                  times=times, horizon=12, symbols=symbols)

    pos = panel_positions(proba, panel, top_k=2, rebalance=10)
    assert np.allclose(pos.sum(axis=1), 0.0, atol=1e-12), "book is not dollar neutral"
    assert (pos.loc[times[50]:, "F"] == 0.0).all(), "held a delisted symbol"
    # Neutrality must survive the delisting, not just hold before it.
    assert np.allclose(pos.loc[times[50]:].sum(axis=1), 0.0, atol=1e-12)


def test_losing_a_leg_to_delisting_goes_flat_rather_than_directional():
    """The bug this caught: zeroing a dead short leaves the book net long.

    A 'market-neutral' strategy that quietly becomes directional when an asset
    dies is worse than a directional one, because nothing in the reported
    metrics says so.
    """
    from tbot.crosssec import _neutralise

    book = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    assert _neutralise(book).sum() == pytest.approx(0.0)

    # Both shorts delist.
    orphaned = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0})
    assert (_neutralise(orphaned) == 0.0).all(), "ran a directional book"

    # One short survives: rescale it to carry the full short side.
    partial = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": 0.0})
    out = _neutralise(partial)
    assert out.sum() == pytest.approx(0.0)
    assert out["C"] == pytest.approx(-1.0)


def test_long_only_book_is_not_forced_flat():
    """Neutralisation must not break the allow_short=False case."""
    from tbot.crosssec import _neutralise

    book = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    out = _neutralise(book, allow_short=False)
    assert out.sum() == pytest.approx(1.0)


def test_benchmarks_are_restricted_to_the_out_of_sample_window():
    """An unmasked benchmark carries periods the model never traded.

    This flattered the first cross-sectional result badly: equal_weight ran from
    2021-12 while the model only traded from 2022-10, so the benchmark absorbed
    the entire bear market and reported -83%. The strategy looked good by
    comparison for reasons that had nothing to do with the strategy.
    """
    from tbot.config import CostConfig
    from tbot.crosssec import Panel, benchmarks

    symbols = ["BTCUSDT", "A", "B"]
    times = pd.date_range("2022-01-01", periods=200, freq="4h")
    ret = pd.DataFrame(-0.05, index=times, columns=symbols)   # brutal early period
    ret.loc[times[100]:] = 0.01
    panel = Panel(features=pd.DataFrame(), y=pd.Series(dtype=float), ret_next=ret,
                  funding=pd.DataFrame(0.0, index=times, columns=symbols),
                  times=times, horizon=12, symbols=symbols)

    late = times[100:]
    masked = benchmarks(panel, CostConfig(0.0, 0.0), oos_times=late)
    unmasked = benchmarks(panel, CostConfig(0.0, 0.0))

    assert masked["equal_weight"].equity.iloc[-1] > 1.0, "should only see the good half"
    assert unmasked["equal_weight"].equity.iloc[-1] < 1.0
    # Nothing may be held before the window opens.
    assert masked["equal_weight"].position.loc[times[:100]].abs().sum() == 0.0


def test_static_reference_benchmark_is_dollar_neutral():
    """long BTC / short alts must be neutral, or it is not comparable."""
    from tbot.config import CostConfig
    from tbot.crosssec import Panel, benchmarks

    symbols = ["BTCUSDT", "A", "B", "C"]
    times = pd.date_range("2022-01-01", periods=50, freq="4h")
    panel = Panel(features=pd.DataFrame(), y=pd.Series(dtype=float),
                  ret_next=pd.DataFrame(0.01, index=times, columns=symbols),
                  funding=pd.DataFrame(0.0, index=times, columns=symbols),
                  times=times, horizon=12, symbols=symbols)

    out = benchmarks(panel, CostConfig(0.0, 0.0))
    assert "static_vs_alts" in out
    # Gross exposure 2, net 0: one unit long BTC, one unit spread across alts.
    assert out["static_vs_alts"].position.iloc[10] == pytest.approx(2.0)


def test_funding_is_charged_on_held_positions():
    from tbot.config import CostConfig
    from tbot.crosssec import Panel

    symbols = ["A", "B"]
    times = pd.date_range("2022-01-01", periods=50, freq="4h")
    panel = Panel(
        features=pd.DataFrame(), y=pd.Series(dtype=float),
        ret_next=pd.DataFrame(0.0, index=times, columns=symbols),
        funding=pd.DataFrame({"A": 0.001, "B": 0.0}, index=times),
        times=times, horizon=12, symbols=symbols,
    )
    pos = pd.DataFrame({"A": 1.0, "B": 0.0}, index=times)
    free = CostConfig(fee_bps=0.0, slippage_bps=0.0)

    with_funding = backtest_panel(pos, panel, free, charge_funding=True)
    without = backtest_panel(pos, panel, free, charge_funding=False)
    assert with_funding.equity.iloc[-1] < without.equity.iloc[-1], (
        "a long paying positive funding must lose money on a flat market"
    )
