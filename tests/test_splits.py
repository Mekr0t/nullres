"""Splitting is where leakage hides. These invariants are non-negotiable."""

import numpy as np
import pytest

from tbot.config import SplitConfig
from tbot.validation import purged_walk_forward, remap_t_end, uniqueness_weights


def horizon_t_end(n, h):
    return np.minimum(np.arange(n) + h, n - 1)


def test_train_always_precedes_test():
    t_end = horizon_t_end(20_000, 24)
    cfg = SplitConfig(n_folds=5, min_train=5_000)
    folds = list(purged_walk_forward(t_end, cfg))
    assert folds
    for train, test in folds:
        assert train.max() < test.min(), "a training row sits inside the test window"


def test_purge_removes_overlapping_labels():
    """No training label may resolve at or after the test window opens."""
    n, h = 20_000, 50
    t_end = horizon_t_end(n, h)
    cfg = SplitConfig(n_folds=5, min_train=5_000, embargo=0)
    for train, test in purged_walk_forward(t_end, cfg):
        assert t_end[train].max() < test.min()


def test_embargo_widens_the_gap():
    n, h = 20_000, 24
    t_end = horizon_t_end(n, h)
    strict = SplitConfig(n_folds=5, min_train=5_000, embargo=500)
    for train, test in purged_walk_forward(t_end, strict):
        assert t_end[train].max() < test.min() - 500


def test_longer_horizons_purge_more_rows():
    n = 20_000
    cfg = SplitConfig(n_folds=5, min_train=5_000)
    short = sum(len(tr) for tr, _ in purged_walk_forward(horizon_t_end(n, 1), cfg))
    long = sum(len(tr) for tr, _ in purged_walk_forward(horizon_t_end(n, 200), cfg))
    assert long < short, "a 200-bar label must cost more training rows than a 1-bar one"


def test_rolling_window_bounds_training_size():
    t_end = horizon_t_end(30_000, 24)
    cfg = SplitConfig(scheme="rolling", n_folds=5, min_train=5_000, train_window=3_000)
    for train, _ in purged_walk_forward(t_end, cfg):
        assert len(train) <= 3_000


def test_test_folds_are_disjoint_and_ordered():
    t_end = horizon_t_end(20_000, 24)
    cfg = SplitConfig(n_folds=6, min_train=5_000)
    seen = []
    for _, test in purged_walk_forward(t_end, cfg):
        assert not set(test) & set(seen)
        seen.extend(test.tolist())
    assert seen == sorted(seen)


def test_remap_never_points_backwards():
    n = 1_000
    t_end = horizon_t_end(n, 10)
    keep = np.ones(n, dtype=bool)
    keep[::3] = False           # drop a third of the rows
    mapped = remap_t_end(t_end, keep)
    positions = np.arange(len(mapped))
    assert (mapped >= positions).all(), "a label cannot resolve before its own bar"
    assert mapped.max() <= len(mapped) - 1


def test_too_little_data_raises_clearly():
    with pytest.raises(ValueError, match="min_train"):
        list(purged_walk_forward(horizon_t_end(100, 5), SplitConfig(min_train=5_000)))


def test_uniqueness_weights_reflect_overlap():
    n = 1_000
    isolated = uniqueness_weights(np.arange(n), n)          # 1-bar labels
    overlapping = uniqueness_weights(horizon_t_end(n, 50), n)
    assert isolated.mean() == pytest.approx(1.0, abs=1e-9)
    assert overlapping.mean() < 0.2, "50-bar overlapping labels should be heavily discounted"
    assert (overlapping > 0).all() and (overlapping <= 1.0 + 1e-9).all()
