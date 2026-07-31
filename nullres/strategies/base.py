from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class Context:
    """Everything a strategy is allowed to see.

    Note what is absent: there is no handle on the future, and `oos_mask` marks
    the bars a strategy is permitted to be judged on. Rule strategies could in
    principle trade the whole sample, but they are masked to the same window as
    the ML strategies so the comparison is fair — a rule evaluated over six
    years against a model evaluated over five is not a comparison.
    """

    bars: pd.DataFrame
    features: pd.DataFrame
    label: pd.DataFrame
    cfg: object
    oos_mask: pd.Series
    diagnostics: dict = field(default_factory=dict)
    verbose: bool = True

    @property
    def sigma(self) -> pd.Series:
        return self.label["sigma"]


class Strategy(Protocol):
    name: str

    def positions(self, ctx: Context) -> pd.Series:
        """Target position per bar, decided at that bar's close."""
        ...


def strategy_fingerprint(obj) -> str:
    """A stable identity for a strategy instance, including its parameters.

    `repr` will not do: these are plain classes, so the default repr embeds
    `id(obj)` and changes every run. This reads the instance dictionary
    instead, recursing into nested strategies — `MLMeta` holds a primary rule
    whose own parameters decide what the model is trained on.
    """
    parts = []
    for name, value in sorted(vars(obj).items()):
        inner = (strategy_fingerprint(value)
                 if hasattr(value, "positions") else repr(value))
        parts.append(f"{name}={inner}")
    return f"{type(obj).__name__}({','.join(parts)})"


def cached_proba(ctx: Context, key: str, compute, extra: str = ""):
    """Memoise walk-forward predictions across runs that share a context.

    `nullres sweep` varies only sizing, which cannot change the model's output,
    so refitting 25 times would be pure waste. The cache key includes the
    label, split and model config, so any change that WOULD alter the
    predictions misses the cache instead of silently returning stale ones.

    `extra` is for anything else that feeds the feature matrix. It exists
    because the fingerprint was incomplete: `MLMeta` appends `primary_side` to
    the features, which depends on the parameters of its primary rule, and none
    of those appeared in the key. Two `MLMeta` strategies with different
    primaries, evaluated against one prepared context, would have taken each
    other's predictions — silently, since a cache hit looks exactly like a fast
    computation. Nothing in the shipped configs varies the primary, so this was
    a trap rather than a live bug, which is the kind that survives longest.
    """
    fingerprint = (repr(ctx.cfg.label), repr(ctx.cfg.split), repr(ctx.cfg.model),
                   extra)
    slot = ctx.diagnostics.setdefault(key, {})
    if slot.get("fingerprint") == fingerprint and "proba" in slot:
        return slot["proba"], slot.get("folds", [])

    proba, folds = compute()
    slot.update(fingerprint=fingerprint, proba=proba, folds=folds)
    return proba, folds


def mask_to_oos(pos: pd.Series, ctx: Context) -> pd.Series:
    """Zero out any position outside the out-of-sample window."""
    return pos.where(ctx.oos_mask, 0.0).fillna(0.0)


def crossover_state(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 while fast is above slow, -1 while below. Point-in-time safe."""
    return pd.Series(np.where(fast > slow, 1.0, -1.0), index=fast.index)
