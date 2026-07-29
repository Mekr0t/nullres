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
    assert accuracy.loc["GOOD", "accuracy"] == pytest.approx(1.0)
    assert accuracy.loc["BAD", "accuracy"] < 0.6
    assert accuracy.loc["GOOD", "n"] == 4, "the count it rests on is reported"
    # GOOD never loses, so it has one class and no AUC to compute. That is a
    # real state, not an error, and it must be NaN rather than a fabricated 0.5.
    assert np.isnan(accuracy.loc["GOOD", "auc"])


def test_per_symbol_accuracy_reports_counts_so_thin_symbols_can_be_excluded():
    """A spread across barely-scored symbols measures the screen, not the model.

    The wide panel produced a spread of 0.685 against the narrow panel's 0.093 —
    which reads as damning until you notice the extremes were symbols scored on
    a few dozen bars, where a coin flip reaches 0.86 without trying.
    """
    ts = pd.date_range("2022-01-01", periods=500, freq="4h")
    rows, proba_rows = [], []
    for t in ts:
        rows.append((t, "THICK"))
        proba_rows.append(0.9)
    # THIN appears on only 6 bars, and happens to be right on all of them.
    for t in ts[:6]:
        rows.append((t, "THIN"))
        proba_rows.append(0.9)

    idx = pd.MultiIndex.from_tuples(rows, names=["ts", "symbol"])
    proba = pd.Series(proba_rows, index=idx)
    y = pd.Series(
        [1.0 if s == "THIN" else float(i % 2) for i, (_, s) in enumerate(rows)],
        index=idx,
    )

    accuracy = per_symbol_accuracy(proba, FakePanel(pd.DataFrame(), y=y))
    assert accuracy.loc["THIN", "n"] == 6
    assert accuracy.loc["THIN", "accuracy"] == pytest.approx(1.0)
    assert accuracy.loc["THICK", "n"] == 500

    # Filtering on n is what makes the spread meaningful.
    thick = accuracy[accuracy["n"] >= 200]
    assert list(thick.index) == ["THICK"], "the thin symbol must be excludable"


def test_per_symbol_lift_strips_out_a_symbols_own_base_rate():
    """High accuracy on a persistently-lagging coin is not skill.

    The label is "beats the cross-sectional median", so a symbol that usually
    lags has a lopsided base rate of its own. A model that learned only "this
    one usually lags" scores that base rate without knowing anything, and the
    raw accuracy spread reads it as concentrated skill.
    """
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="4h")
    idx = pd.MultiIndex.from_product([ts, ["DRIFTER", "REAL"]],
                                     names=["ts", "symbol"])

    y, proba = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        # DRIFTER loses to the median 90% of the time; the model just always
        # says "loses" and inherits that base rate for free.
        loses = i % 10 != 0
        y.append(0.0 if loses else 1.0)
        proba.append(0.1)
        # REAL is a coin flip the model calls right 70% of the time.
        truth = float(rng.integers(0, 2))
        y.append(truth)
        proba.append(0.9 if (truth == 1.0) == (rng.random() < 0.7) else 0.1)

    panel = FakePanel(pd.DataFrame(), y=pd.Series(y, index=idx))
    out = per_symbol_accuracy(pd.Series(proba, index=idx), panel)

    assert out.loc["DRIFTER", "accuracy"] == pytest.approx(0.9, abs=0.01)
    assert out.loc["DRIFTER", "base_rate"] == pytest.approx(0.9, abs=0.01)
    assert abs(out.loc["DRIFTER", "lift"]) < 0.02, "drift is not skill"
    assert out.loc["REAL", "lift"] > 0.1, "genuine skill must still show"


def test_per_symbol_auc_is_the_base_rate_free_read():
    """AUC is what survives both confounds.

    Raw accuracy is inflated by a symbol's own base rate. Lift corrects that but
    penalises a cross-sectional model for a constraint it cannot escape — it
    must rank, so it cannot predict the majority class for every symbol. AUC is
    immune to both: 0.5 means the model cannot separate this symbol's good bars
    from its bad ones, whatever its tendency and whatever the ranking forced.
    """
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="4h")
    idx = pd.MultiIndex.from_product([ts, ["DRIFTER", "REAL"]],
                                     names=["ts", "symbol"])

    y, proba = [], []
    rng = np.random.default_rng(1)
    for i in range(n):
        # DRIFTER beats the median 85% of the time. The model always scores it
        # high — right most of the time, but with no discrimination at all.
        wins = i % 20 != 0 and i % 19 != 0
        y.append(1.0 if wins else 0.0)
        proba.append(0.8)          # constant: every pair ties, so AUC is 0.5
        # REAL is balanced, and the model's score genuinely tracks the outcome.
        truth = float(rng.integers(0, 2))
        y.append(truth)
        proba.append(0.75 if truth else 0.25)

    out = per_symbol_accuracy(pd.Series(proba, index=idx),
                              FakePanel(pd.DataFrame(), y=pd.Series(y, index=idx)))

    assert out.loc["DRIFTER", "accuracy"] > 0.8, "base rate carries the accuracy"
    assert out.loc["DRIFTER", "auc"] == pytest.approx(0.5), (
        "a constant score has no discrimination whatever its hit rate"
    )
    assert out.loc["REAL", "auc"] > 0.95, "genuine discrimination shows"
    assert out.index[0] == "REAL", "sorted by AUC, not accuracy"


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


def test_tail_curve_prices_a_hit_against_the_book_s_own_weight():
    """The ruin arithmetic the graveyard did by hand, made mechanical.

    "One UNFI-type event (+274% in 4h) against a -0.5 weight is -137% of
    capital" is a calculation, not a judgement, and it belongs in code where it
    updates itself when the book changes.
    """
    from nullres.panelaudit import tail_curve

    n = 1_000
    returns = {"AAA": [0.0] * n, "BBB": [0.0] * n}
    returns["AAA"][10] = float(np.log(2.0))       # +100% in one bar
    panel = panel_of(returns)
    # Dollar-neutral k=1 book: long AAA, short BBB at half weight each.
    positions = pd.DataFrame({"AAA": [0.5] * n, "BBB": [-0.5] * n},
                             index=panel.times)

    curve = tail_curve(positions, panel, thresholds=(0.5, 1.0, 2.0))
    assert curve.attrs["worst_short_weight"] == pytest.approx(0.5)

    # A +200% move against a -0.5 weight costs 100% of capital: ruin.
    row = curve.set_index("move").loc[2.0]
    assert row["cost_of_one"] == pytest.approx(1.0)
    assert bool(row["ruinous"]) is True
    # A +50% move costs 25% — survivable.
    assert bool(curve.set_index("move").loc[0.5]["ruinous"]) is False


def test_an_unobserved_move_size_is_bounded_not_zero():
    """Concluding a tail probability is zero from non-observation is the error.

    The panel contains no +200% four-hour move, so the empirical rate is 0 and
    the expected-hits column read 0.00 for the level that would wipe the book
    out — which looks like safety and is really an empty sample. The rule of
    three bounds an unobserved rate at 3/N instead of asserting it is zero.
    """
    from nullres.panelaudit import tail_curve

    n = 1_000
    panel = panel_of({"AAA": [0.0] * n, "BBB": [0.0] * n})
    positions = pd.DataFrame({"AAA": [0.5] * n, "BBB": [-0.5] * n},
                             index=panel.times)

    curve = tail_curve(positions, panel, thresholds=(2.0,)).iloc[0]
    assert curve["occurrences"] == 0
    assert bool(curve["estimated"]) is False
    # 3 / 2000 observed symbol-bars, times 1000 short-name-bars.
    assert curve["expected_hits"] == pytest.approx(3 / 2_000 * 1_000)
    assert curve["expected_hits"] > 0, "an unseen event is not an impossible one"


def test_tail_curve_sweeps_rather_than_trusting_one_threshold():
    """A rate at one move size is one point on a steeply falling curve."""
    from nullres.panelaudit import tail_curve

    n = 2_000
    rng = np.random.default_rng(4)
    moves = rng.normal(0, 0.05, n)
    moves[5], moves[50], moves[500] = np.log(1.3), np.log(1.8), np.log(3.0)
    panel = panel_of({"AAA": list(moves), "BBB": [0.0] * n})
    positions = pd.DataFrame({"AAA": [0.5] * n, "BBB": [-0.5] * n},
                             index=panel.times)

    curve = tail_curve(positions, panel, thresholds=(0.25, 0.5, 1.0, 2.0))
    # Rarer moves must be rarer, and cost more when they land.
    assert list(curve["occurrences"]) == sorted(curve["occurrences"], reverse=True)
    assert list(curve["cost_of_one"]) == sorted(curve["cost_of_one"])


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
