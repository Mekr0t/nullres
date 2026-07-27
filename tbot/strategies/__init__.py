"""Strategy registry.

Every strategy consumes a `Context` and returns a position series over the same
index. Rule-based strategies exist to be beaten: if a gradient-boosted model on
23 features cannot outperform a 50/200 moving-average cross after costs, the
model has not learned anything worth the complexity, and you should say so.
"""

from tbot.strategies.base import Context, Strategy
from tbot.strategies.ml import MLDirection, MLMeta
from tbot.strategies.rules import (
    BuyHold,
    DonchianBreakout,
    MeanReversionZ,
    SMACross,
)

REGISTRY: dict[str, type[Strategy]] = {
    "buy_hold": BuyHold,
    "sma_cross": SMACross,
    "donchian": DonchianBreakout,
    "mean_reversion": MeanReversionZ,
    "ml_direction": MLDirection,
    "ml_meta": MLMeta,
}


def build(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(REGISTRY)}")
    try:
        return REGISTRY[name](**(params or {}))
    except TypeError as exc:
        raise ValueError(f"bad params for strategy {name!r}: {exc}") from exc


__all__ = ["REGISTRY", "build", "Strategy", "Context"]
