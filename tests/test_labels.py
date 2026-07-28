"""Label construction — including that labels resolve when they claim to."""

import numpy as np
import pandas as pd
import pytest

from nullres.config import LabelConfig
from nullres.data.synthetic import synthetic_bars
from nullres.labels import build_label


def ramp_bars(n=500, step=0.002):
    close = 100 * np.exp(np.arange(n) * step)
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.0001, "low": close * 0.9999,
            "close": close, "volume": [1.0] * n, "trades": [1.0] * n,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


def test_relentless_uptrend_labels_are_all_long():
    cfg = LabelConfig(kind="triple_barrier", horizon=24, vol_window=20)
    out = build_label(ramp_bars(), cfg)
    resolved = out["y"].dropna()
    assert len(resolved) > 100
    assert (resolved == 1.0).all(), "a monotonic ramp must never hit a stop"


def test_downtrend_labels_are_all_short():
    cfg = LabelConfig(kind="triple_barrier", horizon=24, vol_window=20)
    bars = ramp_bars(step=-0.002)
    resolved = build_label(bars, cfg)["y"].dropna()
    assert (resolved == 0.0).all()


@pytest.mark.parametrize("kind", ["triple_barrier", "fwd_return", "next_bar_sign"])
def test_t_end_is_bounded_by_the_horizon(kind):
    cfg = LabelConfig(kind=kind, horizon=24, vol_window=72)
    bars = synthetic_bars(n=3_000)
    out = build_label(bars, cfg)
    positions = np.arange(len(out))
    t_end = out["t_end"].to_numpy()
    assert (t_end >= positions).all()
    horizon = 1 if kind == "next_bar_sign" else cfg.horizon
    assert (t_end <= positions + horizon).all()


def test_triple_barrier_only_reads_bars_within_its_window():
    """Mutating bars beyond t+horizon must not change the label at t."""
    cfg = LabelConfig(kind="triple_barrier", horizon=24, vol_window=72)
    bars = synthetic_bars(n=2_000, seed=3)
    base = build_label(bars, cfg)

    tampered = bars.copy()
    cut = 1_000
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(col)] *= 3.0
    after = build_label(tampered, cfg)

    # Labels for bars up to cut-horizon-1 resolve strictly before the tampering.
    safe = slice(0, cut - cfg.horizon - 1)
    pd.testing.assert_series_equal(base["y"].iloc[safe], after["y"].iloc[safe])


def test_deadband_leaves_small_moves_unlabelled():
    bars = synthetic_bars(n=5_000, seed=1)
    wide = build_label(bars, LabelConfig(kind="fwd_return", horizon=12, deadband=1.0))
    none = build_label(bars, LabelConfig(kind="fwd_return", horizon=12, deadband=0.0))
    assert wide["y"].notna().sum() < none["y"].notna().sum()


def test_tail_labels_are_dropped_not_guessed():
    """The last `horizon` bars never resolved and must not carry a label."""
    cfg = LabelConfig(kind="triple_barrier", horizon=24, vol_window=72)
    out = build_label(synthetic_bars(n=1_000), cfg)
    assert out["y"].iloc[-cfg.horizon:].isna().all()


def test_synthetic_labels_are_near_balanced():
    """On a random walk with symmetric barriers, classes should be ~50/50."""
    cfg = LabelConfig(kind="triple_barrier", horizon=24, vol_window=72)
    y = build_label(synthetic_bars(n=20_000, seed=7), cfg)["y"].dropna()
    # Ties resolve to the stop, so a mild tilt below 0.5 is expected and correct.
    assert 0.40 < y.mean() < 0.55
