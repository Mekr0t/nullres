"""Typed configuration loaded from TOML.

TOML is read with the stdlib `tomllib`, so config costs no dependency. (It
landed in 3.11; the project's tested floor is 3.13 — see `requires-python`.)
Every experiment is fully described by one file in `configs/` — that file is the
unit of reproducibility. If a number influenced a result, it belongs here and
not in a function default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from nullres.errors import ConfigError

# Bars per year, used to annualise Sharpe and CAGR.
BARS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460,
    "12h": 730, "1d": 365, "1w": 52,
}


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start: str = "2020-01"
    end: str = "2025-12"
    cache_dir: str = "data"
    source: str = "binance"          # binance | synthetic
    # Join Binance USD-M futures funding rates / open-interest metrics onto the
    # bars. The only information in this repo that is not a transform of OHLCV.
    # Ignored for synthetic data, which has no derivatives market.
    funding: bool = False
    metrics: bool = False

    @property
    def bars_per_year(self) -> int:
        if self.interval not in BARS_PER_YEAR:
            raise ConfigError(
                f"unknown interval {self.interval!r}; "
                f"add it to BARS_PER_YEAR in nullres/config.py"
            )
        return BARS_PER_YEAR[self.interval]


@dataclass
class LabelConfig:
    kind: str = "triple_barrier"     # triple_barrier | fwd_return | next_bar_sign
    horizon: int = 24                # max bars to hold / lookahead
    vol_window: int = 72             # bars used to estimate volatility
    upper: float = 1.5               # profit barrier, in units of sigma
    lower: float = 1.5               # stop barrier, in units of sigma
    # fwd_return only: |return| must exceed this many sigma to count as a move.
    # Bars below it are dropped, which stops the model learning to predict noise.
    deadband: float = 0.5


@dataclass
class SplitConfig:
    n_folds: int = 6
    scheme: str = "expanding"        # expanding | rolling
    min_train: int = 5_000
    train_window: int = 20_000       # rolling scheme only
    embargo: int = 0                 # extra bars dropped after the purge


@dataclass
class ModelConfig:
    kind: str = "hgb"                # hgb | logistic
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    l2: float = 1.0
    min_samples_leaf: int = 200
    seed: int = 0


@dataclass
class CostConfig:
    fee_bps: float = 10.0            # Binance spot taker, one side
    slippage_bps: float = 2.0        # one side


@dataclass
class SizingConfig:
    """Signal -> position. This is where a real edge is kept or destroyed."""
    long_entry: float = 0.56
    long_exit: float = 0.50          # hysteresis: exit band != entry band
    short_entry: float = 0.44
    short_exit: float = 0.50
    allow_short: bool = True
    min_hold: int = 6                # bars a position must be held before flipping
    # SINGLE-ASSET ONLY. Clips the position to +/-this. `nullres xsec` never
    # reads it: a dollar-neutral book is 100% long and 100% short, so its gross
    # is 2.0 by construction and capping it at 1.0 would be a different
    # strategy, not a safer one. Read `gross_exposure` in the results table to
    # see what a book is actually carrying.
    max_leverage: float = 1.0
    vol_target: float = 0.0          # >0 scales exposure to this annualised vol


@dataclass
class RunConfig:
    name: str = "unnamed"
    data: DataConfig = field(default_factory=DataConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    strategies: list[str] = field(default_factory=lambda: ["buy_hold"])
    # Per-strategy keyword arguments, e.g. [params.donchian] entry = 96.
    # Rule parameters belong in the config for the same reason everything else
    # does: a number that influenced a result has to be recorded somewhere the
    # result can be reproduced from.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    out_dir: str = "reports"
    # Parameter combinations explored BEFORE the run ledger existed. The
    # deflated Sharpe needs the true number of variants tried, and the ledger
    # only knows about runs it recorded. Undercounting flatters every result,
    # so an honest estimate of prior work belongs here.
    prior_trials: int = 0


def _build(cls: type, raw: dict[str, Any], path: str = "") -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys.

    Silently ignoring a typo'd config key is how you spend a week backtesting a
    parameter you thought you had changed.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        loc = path or cls.__name__
        raise ConfigError(
            f"unknown config key(s) in [{loc}]: {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    # `from __future__ import annotations` makes field.type a string, so the
    # real classes have to be resolved before nested sections can be detected.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        ftype = hints.get(key)
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[key] = _build(ftype, value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = _build(RunConfig, raw)
    if cfg.name == "unnamed":
        cfg.name = path.stem
    return cfg
