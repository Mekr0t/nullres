"""The audit has to catch leaks that a human review would miss.

A leak detector that only passes on clean data proves nothing. Each test here
injects a specific, realistic bug and asserts the audit fails.
"""

import numpy as np
import pandas as pd

from tbot.audit import check_label_leakage, check_point_in_time
from tbot.config import LabelConfig
from tbot.data.synthetic import synthetic_bars
from tbot.features import build_features
from tbot.labels import build_label


def test_clean_features_pass_point_in_time():
    bars = synthetic_bars(n=4_000, seed=2)
    check = check_point_in_time(bars, probes=3)
    assert check.passed, check.detail


def test_negative_shift_is_caught():
    """The textbook lookahead: a feature that peeks one bar ahead."""
    def leaky(df):
        f = build_features(df)
        f["tomorrow"] = np.log(df["close"]).diff().shift(-1)
        return f

    check = check_point_in_time(synthetic_bars(n=4_000, seed=2), builder=leaky, probes=3)
    assert not check.passed
    assert "tomorrow" in check.detail


def test_full_sample_scaling_is_caught():
    """Subtler and far more common: normalising with the whole sample's mean."""
    def leaky(df):
        f = build_features(df)
        close = df["close"]
        f["global_z"] = (close - close.mean()) / close.std()
        return f

    check = check_point_in_time(synthetic_bars(n=4_000, seed=2), builder=leaky, probes=3)
    assert not check.passed
    assert "global_z" in check.detail


def test_backfilled_resample_is_caught():
    """A daily series joined onto hourly bars with bfill instead of ffill.

    Every hour of Monday gets Tuesday's close. Easy to write, invisible in a
    plot, and it produces a beautiful equity curve.
    """
    def leaky(df):
        f = build_features(df)
        daily = df["close"].resample("D").last()
        f["vs_daily"] = df["close"] / daily.reindex(df.index, method="bfill")
        return f

    check = check_point_in_time(synthetic_bars(n=4_000, seed=2), builder=leaky, probes=5)
    assert not check.passed
    assert "vs_daily" in check.detail


def test_forward_filled_resample_passes():
    """Control: the same join done correctly must not trip the check."""
    def clean(df):
        f = build_features(df)
        daily = df["close"].resample("D").last()
        f["vs_daily"] = df["close"] / daily.shift(1).reindex(df.index, method="ffill")
        return f

    check = check_point_in_time(synthetic_bars(n=4_000, seed=2), builder=clean, probes=5)
    assert check.passed, check.detail


def test_baseline_leaky_label_is_caught_by_auc():
    """The exact bug in the original baseline.py.

    `label_leaky` marks whether THIS bar rose, and `ret_1` is that same
    quantity. The single-feature AUC check pins it at ~1.0.
    """
    bars = synthetic_bars(n=6_000, seed=4)
    features = build_features(bars)
    y_leaky = (np.log(bars["close"]).diff() > 0).astype(float)

    keep = features.notna().all(axis=1)
    check = check_label_leakage(features[keep], y_leaky[keep])

    assert not check.passed
    assert "ret_1" in check.detail


def test_honest_label_passes_auc_check():
    bars = synthetic_bars(n=6_000, seed=4)
    features = build_features(bars)
    label = build_label(bars, LabelConfig(kind="triple_barrier", horizon=24))
    keep = features.notna().all(axis=1) & label["y"].notna()
    check = check_label_leakage(features[keep], label["y"][keep])
    assert check.passed, check.detail
