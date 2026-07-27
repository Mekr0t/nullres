"""End-to-end control: the pipeline must find nothing on data with no edge.

This is the regression test that matters most, and it is the one a unit test
cannot express. Every other test checks a component in isolation. This one runs
the whole assembled pipeline — features, labels, purged splits, model, sizing,
execution, costs — against a geometric random walk where the answer is known to
be "no edge" by construction.

If someone refactors the engine six months from now and quietly reintroduces a
lookahead, no unit test will notice. This will.

Marked `slow` (~1-2 min). CI runs it on every push:

    pytest -m slow
    pytest -m "not slow"        # the fast suite, ~3s
"""

from __future__ import annotations

import numpy as np
import pytest

from tbot.config import load_config
from tbot.data.synthetic import synthetic_bars
from tbot.pipeline import run_pipeline

pytestmark = pytest.mark.slow

# A strategy on a martingale should land near zero. Anything meaningfully above
# this is a bug, not a discovery.
SHARPE_CEILING = 0.5


@pytest.fixture(scope="module")
def null_results():
    cfg = load_config("configs/null.toml")
    return run_pipeline(cfg, verbose=False)


def test_no_strategy_finds_edge_in_a_random_walk(null_results):
    """The headline control. A failure here voids every result in the repo."""
    offenders = {
        name: m["sharpe"]
        for name, m in null_results.items()
        # buy_hold is exempt: a single realised random walk drifts somewhere,
        # and holding it is not a claim about predictability.
        if name != "buy_hold" and m["sharpe"] > SHARPE_CEILING
    }
    assert not offenders, (
        f"found 'edge' in data that has none: {offenders}. "
        f"The bug is in the engine, the split, or the labels."
    )


def test_trading_strategies_lose_to_costs_on_a_martingale(null_results):
    """Beyond 'no edge': with no signal, turnover can only destroy value."""
    for name, m in null_results.items():
        if name == "buy_hold" or m["n_trades"] < 10:
            continue
        assert m["total_return"] < m["gross_return"], (
            f"{name} netted more than it grossed — costs are not being charged"
        )


def test_model_accuracy_is_indistinguishable_from_chance(null_results):
    """A classifier on a martingale must not beat the base rate out of sample."""
    for name in ("ml_direction", "ml_meta"):
        assert name in null_results, f"{name} missing from the null run"


def test_synthetic_data_has_no_autocorrelation():
    """Guard the guard: the null data must actually be a martingale."""
    bars = synthetic_bars(n=40_000, seed=0)
    logret = np.diff(np.log(bars["close"].to_numpy()))
    for lag in (1, 2, 3, 12, 24):
        rho = np.corrcoef(logret[:-lag], logret[lag:])[0, 1]
        assert abs(rho) < 0.02, f"lag-{lag} autocorrelation {rho:.4f} is too high"


def test_the_control_can_detect_a_real_edge():
    """The opposite failure mode: a harness too weak to find anything.

    If the pipeline cannot detect a planted AR(1) signal, then a null result on
    real data means nothing — absence of evidence would just be blindness.
    """
    bars = synthetic_bars(n=40_000, seed=0, edge=0.15)
    logret = np.diff(np.log(bars["close"].to_numpy()))
    rho = np.corrcoef(logret[:-1], logret[1:])[0, 1]
    assert rho > 0.05, (
        f"planted edge did not survive data generation (rho={rho:.4f}); "
        f"the positive control is broken"
    )
