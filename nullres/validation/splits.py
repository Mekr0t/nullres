"""Purged, embargoed walk-forward splitting.

Ordinary K-fold on time series is nonsense: it trains on Friday to predict
Wednesday. Walk-forward fixes the direction but not the overlap — a label at
bar t that resolves at t+24 still contains information about bars t+1..t+24,
so if the test window starts at t+5, that training row has already seen the
answer. Purging removes those rows.

The embargo goes further and drops training rows that merely END shortly
before the test window opens. Serial correlation in features means a row from
five bars before the boundary is nearly the same row as one inside it. Set
`embargo` to roughly the feature memory (the longest rolling window) when you
want to be strict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

from nullres.errors import ConfigError, InsufficientDataError

if TYPE_CHECKING:                       # annotation only; no runtime import
    from nullres.config import SplitConfig


def remap_t_end(t_end: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Translate label end positions from the raw frame to the filtered frame.

    Rows are dropped for NaN features or unlabelled bars, which renumbers every
    position. A label that ended on a dropped bar is mapped forward to the next
    surviving bar, which can only lengthen the purge — the safe direction.
    """
    surviving = np.flatnonzero(keep)
    if surviving.size == 0:
        return np.array([], dtype=np.int64)
    mapped = np.searchsorted(surviving, t_end[keep], side="left")
    return np.clip(mapped, np.arange(surviving.size), surviving.size - 1)


def purged_walk_forward(t_end: np.ndarray, cfg: SplitConfig
                        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs of positional indices.

    Args:
        t_end: for each row, the position at which its label resolves.
        cfg: a SplitConfig.
    """
    n = len(t_end)
    if n <= cfg.min_train:
        raise InsufficientDataError(
            f"only {n:,} usable rows but min_train is {cfg.min_train:,}; "
            f"lower split.min_train or widen the date range"
        )

    fold = (n - cfg.min_train) // cfg.n_folds
    if fold < 100:
        raise InsufficientDataError(
            f"folds of {fold} rows are too small to measure anything; "
            f"reduce split.n_folds or add data"
        )

    for k in range(cfg.n_folds):
        test_start = cfg.min_train + k * fold
        test_end = min(test_start + fold, n)
        if test_end - test_start < 100:
            break

        if cfg.scheme == "rolling":
            train_lo = max(0, test_start - cfg.train_window)
        elif cfg.scheme == "expanding":
            train_lo = 0
        else:
            raise ConfigError(f"unknown split scheme {cfg.scheme!r}")

        cand = np.arange(train_lo, test_start)
        # Purge: the label must have fully resolved before the embargo window.
        train = cand[t_end[cand] < test_start - cfg.embargo]
        if train.size < 100:
            continue
        yield train, np.arange(test_start, test_end)


def describe_folds(t_end: np.ndarray, cfg, index=None) -> list[dict]:
    """Fold summary for logging — sizes, date ranges, and rows purged."""
    rows = []
    for i, (tr, te) in enumerate(purged_walk_forward(t_end, cfg), start=1):
        naive = te[0] - (max(0, te[0] - cfg.train_window) if cfg.scheme == "rolling" else 0)
        row = {
            "fold": i,
            "train": len(tr),
            "test": len(te),
            "purged": naive - len(tr),
        }
        if index is not None:
            row["test_from"] = str(index[te[0]])[:10]
            row["test_to"] = str(index[te[-1]])[:10]
        rows.append(row)
    return rows
