"""Sample weights for overlapping labels.

With a 24-bar horizon, 24 consecutive training rows describe almost the same
stretch of price. Treating them as 24 independent observations tells the model
it has far more evidence than it does, and it will happily overfit to match.

`uniqueness_weights` down-weights each row by how many other labels overlap it,
following López de Prado's average-uniqueness construction. A row whose window
is shared with 23 others carries roughly 1/24 the weight of an isolated one.
"""

from __future__ import annotations

import numpy as np


def uniqueness_weights(t_end: np.ndarray, n: int | None = None) -> np.ndarray:
    """Average uniqueness of each label, in (0, 1].

    Args:
        t_end: position at which each row's label resolves (inclusive).
        n: total bars; defaults to len(t_end).
    """
    t_end = np.asarray(t_end, dtype=np.int64)
    m = len(t_end)
    n = m if n is None else n
    starts = np.arange(m, dtype=np.int64)
    ends = np.clip(t_end, starts, n - 1)

    # Concurrency: how many label windows cover each bar, via a difference array.
    diff = np.zeros(n + 1, dtype=np.float64)
    np.add.at(diff, starts, 1.0)
    np.add.at(diff, ends + 1, -1.0)
    concurrency = np.cumsum(diff)[:n]
    concurrency[concurrency < 1.0] = 1.0

    # Mean of 1/concurrency over each window, via a prefix sum.
    prefix = np.concatenate([[0.0], np.cumsum(1.0 / concurrency)])
    spans = (ends - starts + 1).astype(np.float64)
    return (prefix[ends + 1] - prefix[starts]) / spans
