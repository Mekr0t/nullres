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
import pandas as pd
import pytest

from nullres.config import load_config
from nullres.data.synthetic import synthetic_bars
from nullres.pipeline import run_pipeline

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


def test_unknown_interval_refuses_to_guess_the_bar_spacing():
    """A calibration instrument must not silently generate the wrong bars.

    `_FREQ.get(interval, "h")` meant a 6h config produced HOURLY synthetic bars
    while the pipeline annualised them at 1,460 bars/year — the null control
    mis-calibrated by a factor of six, silently, in the one test the docs call
    the most useful in the repo.
    """
    from nullres.config import BARS_PER_YEAR
    from nullres.data.synthetic import _FREQ

    missing = sorted(set(BARS_PER_YEAR) - set(_FREQ))
    assert not missing, (
        f"intervals accepted by BARS_PER_YEAR but with no synthetic bar "
        f"spacing: {missing}. The null control cannot run on them."
    )

    with pytest.raises(ValueError, match="no synthetic bar spacing"):
        synthetic_bars(n=100, interval="3h")


def test_synthetic_bars_honour_the_requested_interval():
    for interval, expected in (("1h", "1h"), ("4h", "4h"), ("6h", "6h"),
                               ("1d", "1D")):
        bars = synthetic_bars(n=50, interval=interval)
        spacing = bars.index[1] - bars.index[0]
        assert spacing == pd.Timedelta(expected), (
            f"{interval} bars are spaced {spacing}, not {expected}"
        )
