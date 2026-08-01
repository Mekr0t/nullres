"""Position sizing — the turnover controls have to actually control turnover."""

import numpy as np
import pandas as pd

from nullres.backtest.sizing import apply_min_hold, signal_to_position
from nullres.config import SizingConfig


def series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="h"))


def test_min_hold_caps_the_number_of_changes():
    rng = np.random.default_rng(0)
    noisy = series(rng.uniform(0.3, 0.7, 2_000))

    loose = signal_to_position(noisy, SizingConfig(min_hold=1))
    tight = signal_to_position(noisy, SizingConfig(min_hold=48))

    changes_loose = int((loose.diff().abs() > 0).sum())
    changes_tight = int((tight.diff().abs() > 0).sum())

    assert changes_tight < changes_loose / 4
    assert changes_tight <= len(noisy) / 48 + 1


def test_hysteresis_prevents_chatter_at_the_threshold():
    """A signal oscillating around the entry band must not flip every bar."""
    cfg = SizingConfig(long_entry=0.56, long_exit=0.50, allow_short=False, min_hold=0)
    oscillating = series([0.57, 0.55, 0.57, 0.55] * 50)
    pos = signal_to_position(oscillating, cfg)
    # 0.55 is below entry but above exit, so the long is entered once at bar 0
    # and then held for the whole series — one trade, not two hundred.
    assert (pos == 1.0).all()


def test_single_band_would_chatter():
    """Control for the test above: collapse the bands and chatter returns."""
    cfg = SizingConfig(long_entry=0.56, long_exit=0.56, allow_short=False, min_hold=0)
    pos = signal_to_position(series([0.57, 0.55] * 50), cfg)
    assert int((pos.diff().abs() > 0).sum()) > 50


def test_nan_signal_holds_rather_than_exits():
    cfg = SizingConfig(long_entry=0.56, allow_short=False, min_hold=0)
    pos = signal_to_position(series([0.9, np.nan, np.nan, 0.9]), cfg)
    assert (pos == 1.0).all()


def test_shorts_can_be_disabled():
    cfg = SizingConfig(allow_short=False, min_hold=0)
    pos = signal_to_position(series([0.1] * 20), cfg)
    assert (pos >= 0).all()

    cfg_short = SizingConfig(allow_short=True, min_hold=0)
    assert (signal_to_position(series([0.1] * 20), cfg_short) < 0).any()


def test_positions_respect_max_leverage():
    cfg = SizingConfig(max_leverage=0.5, min_hold=0)
    pos = signal_to_position(series([0.99] * 50), cfg)
    assert pos.abs().max() <= 0.5


def test_vol_target_shrinks_exposure_when_volatile():
    cfg = SizingConfig(vol_target=0.3, min_hold=0, max_leverage=1.0)
    proba = series([0.9] * 100)
    calm = series([0.001] * 100)      # low per-bar sigma
    wild = series([0.05] * 100)       # high per-bar sigma
    calm_pos = signal_to_position(proba, cfg, sigma=calm, bars_per_year=8_760)
    wild_pos = signal_to_position(proba, cfg, sigma=wild, bars_per_year=8_760)
    assert wild_pos.abs().mean() < calm_pos.abs().mean()


def test_apply_min_hold_respects_the_floor():
    desired = series([1.0, -1.0] * 100)
    held = apply_min_hold(desired, min_hold=10)
    change_points = np.flatnonzero(held.diff().abs().fillna(0).to_numpy() > 0)
    gaps = np.diff(change_points)
    assert (gaps >= 10).all()
