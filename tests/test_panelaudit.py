"""The controls that decide whether cross-sectional skill is real.

These run on a tiny hand-built panel: the arithmetic has to be checkable by
eye, because a verification tool that is itself unverified is worth nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nullres.panelaudit import (
    delisted_share,
    per_symbol_accuracy,
    pnl_contribution,
    tail_census,
)


class FakePanel:
    """Minimal stand-in for `crosssec.Panel` — only the fields these read."""

    def __init__(self, ret_next, delisted=None, y=None):
        self.ret_next = ret_next
        self.delisted = delisted or {}
        self.y = y
        self.times = ret_next.index
        self.symbols = list(ret_next.columns)


def panel_of(returns: dict[str, list[float]], delisted=None) -> FakePanel:
    idx = pd.date_range("2022-01-01", periods=len(next(iter(returns.values()))),
                        freq="4h")
    return FakePanel(pd.DataFrame(returns, index=idx), delisted)


def test_pnl_contribution_attributes_to_the_symbol_that_earned_it():
    panel = panel_of({"AAA": [0.10, 0.10], "BBB": [-0.05, -0.05]})
    positions = pd.DataFrame({"AAA": [1.0, 1.0], "BBB": [1.0, 1.0]},
                             index=panel.times)

    contribution = pnl_contribution(positions, panel)
    assert contribution["AAA"] == pytest.approx(0.20)
    assert contribution["BBB"] == pytest.approx(-0.10)


def test_delisted_share_does_not_net_winners_against_losers():
    """The documented choice: this measures activity, not profitability.

    A book that made a fortune on one dying coin and lost it on another nets to
    zero and would look untouched by delisting — when its whole outcome hinged
    on them. For a survivorship control that is the wrong answer.
    """
    panel = panel_of(
        {"ALIVE": [0.10, 0.0], "DEAD1": [0.30, 0.0], "DEAD2": [-0.30, 0.0]},
        delisted={"DEAD1": pd.Timestamp("2022-06-01"),
                  "DEAD2": pd.Timestamp("2022-07-01")},
    )
    positions = pd.DataFrame({"ALIVE": [1.0, 0.0], "DEAD1": [1.0, 0.0],
                              "DEAD2": [1.0, 0.0]}, index=panel.times)

    # Netted, the two dead names cancel exactly; in absolute terms they are 6/7
    # of everything the book did.
    assert pnl_contribution(positions, panel)[["DEAD1", "DEAD2"]].sum() == \
        pytest.approx(0.0)
    assert delisted_share(positions, panel) == pytest.approx(0.6 / 0.7)


def test_delisted_share_is_zero_without_corpses():
    panel = panel_of({"AAA": [0.1, 0.1], "BBB": [0.1, 0.1]})
    positions = pd.DataFrame(1.0, index=panel.times, columns=["AAA", "BBB"])
    assert delisted_share(positions, panel) == 0.0


def test_per_symbol_accuracy_finds_the_symbol_the_model_actually_knows():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-01", periods=4, freq="4h"), ["GOOD", "BAD"]],
        names=["ts", "symbol"],
    )
    y = pd.Series([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], index=idx)
    # Right about GOOD every time, wrong about BAD every time.
    proba = pd.Series([0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9], index=idx)

    panel = FakePanel(pd.DataFrame(), y=y)
    accuracy = per_symbol_accuracy(proba, panel)
    assert accuracy["GOOD"] == pytest.approx(1.0)
    assert accuracy["BAD"] < 0.6
    assert accuracy.index[0] == "GOOD", "sorted best first"


def test_tail_census_reports_expectation_not_just_observation():
    """Zero hits means nothing until you know how many chance predicted."""
    n = 1_000
    returns = {"AAA": [0.0] * n, "BBB": [0.0] * n}
    returns["AAA"][10] = float(np.log(2.0))          # a +100% bar
    panel = panel_of(returns)

    # Short BBB throughout; never short AAA, so the one extreme move is missed.
    positions = pd.DataFrame({"AAA": [0.0] * n, "BBB": [-1.0] * n},
                             index=panel.times)

    census = tail_census(positions, panel, threshold=0.65)
    assert census["extreme_moves"] == 1
    assert census["short_name_bars"] == n
    assert census["actual_hits"] == 0
    # 1 extreme in 2,000 symbol-bars, 1,000 short-name-bars -> 0.5 expected.
    assert census["expected_hits"] == pytest.approx(0.5)


def test_tail_census_counts_a_hit_when_the_book_was_short_it():
    n = 100
    returns = {"AAA": [0.0] * n, "BBB": [0.0] * n}
    returns["BBB"][5] = float(np.log(2.0))
    panel = panel_of(returns)
    positions = pd.DataFrame({"AAA": [0.0] * n, "BBB": [-1.0] * n},
                             index=panel.times)

    assert tail_census(positions, panel, threshold=0.65)["actual_hits"] == 1


def test_shuffled_labels_do_not_depend_on_group_order():
    """A control whose answer moves with iteration order is not a control.

    Sharing one generator across timestamps made each bar's permutation depend
    on how many groups preceded it, so anything that reordered the panel
    silently changed the reported AUC. Seeding per timestamp fixes the
    permutation to the bar it belongs to.
    """
    from nullres.panelaudit import shuffled_label_auc

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-01", periods=6, freq="4h"),
         ["AAA", "BBB", "CCC", "DDD"]],
        names=["ts", "symbol"],
    )
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, len(idx)).astype(float), index=idx)

    captured = {}

    def capture(panel, cfg, verbose=True):
        captured[len(captured)] = panel.y.copy()
        return None, [{"auc": 0.5}]

    import nullres.crosssec as C

    real = C.fit_predict_panel
    C.fit_predict_panel = capture
    try:
        base = FakePanel(pd.DataFrame(index=idx.get_level_values("ts").unique()),
                         y=y)
        base.features = pd.DataFrame(index=idx)
        base.funding = pd.DataFrame()
        base.horizon = 1
        shuffled_label_auc(base, cfg=None)

        # Same panel, timestamps presented in reverse: permutations must match.
        reversed_panel = FakePanel(base.ret_next, y=y.iloc[::-1])
        reversed_panel.features = base.features
        reversed_panel.funding = base.funding
        reversed_panel.horizon = 1
        shuffled_label_auc(reversed_panel, cfg=None)
    finally:
        C.fit_predict_panel = real

    first, second = captured[0], captured[1].reindex(captured[0].index)
    assert first.equals(second), "permutation changed with group order"
