"""Market data loading.

Everything here returns the same contract: a DataFrame indexed by UTC timestamp
with float columns [open, high, low, close, volume, trades], strictly increasing
index, no duplicates. `load_bars` is the only entry point callers should need.
"""

from tbot.data.binance import fetch_month, load_binance
from tbot.data.synthetic import synthetic_bars

OHLCV = ["open", "high", "low", "close", "volume", "trades"]


def load_bars(cfg):
    """Dispatch on cfg.source and return bars matching the OHLCV contract."""
    if cfg.source == "synthetic":
        return synthetic_bars(interval=cfg.interval)
    if cfg.source == "binance":
        return load_binance(cfg.symbol, cfg.interval, cfg.start, cfg.end, cfg.cache_dir)
    raise ValueError(f"unknown data source {cfg.source!r}")


__all__ = ["load_bars", "load_binance", "fetch_month", "synthetic_bars", "OHLCV"]
