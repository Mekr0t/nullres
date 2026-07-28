"""Market data loading.

Everything here returns the same contract: a DataFrame indexed by UTC timestamp
with float columns [open, high, low, close, volume, trades], strictly increasing
index, no duplicates. `load_bars` is the only entry point callers should need.
"""

from nullres.data.binance import fetch_month, load_binance
from nullres.data.futures import load_funding, load_metrics
from nullres.data.synthetic import synthetic_bars, synthetic_funding, synthetic_metrics

OHLCV = ["open", "high", "low", "close", "volume", "trades"]


def load_bars(cfg):
    """Dispatch on cfg.source and return bars matching the OHLCV contract."""
    if cfg.source == "synthetic":
        return synthetic_bars(interval=cfg.interval)
    if cfg.source == "binance":
        return load_binance(cfg.symbol, cfg.interval, cfg.start, cfg.end, cfg.cache_dir)
    raise ValueError(f"unknown data source {cfg.source!r}")


def load_auxiliary(cfg, verbose: bool = True, bars=None):
    """Return (funding, metrics), either of which may be None.

    For synthetic data the auxiliary frames are generated as pure noise rather
    than skipped. That keeps the null control running the SAME feature pipeline
    as a live config — otherwise the random-walk check would exercise 32
    features while the real run uses 47, and a broken funding join would never
    reach the one test designed to catch fabricated edge.
    """
    if cfg.source == "synthetic":
        if bars is None or not (cfg.funding or cfg.metrics):
            return None, None
        return (
            synthetic_funding(bars) if cfg.funding else None,
            synthetic_metrics(bars) if cfg.metrics else None,
        )
    if cfg.source != "binance":
        return None, None
    funding = (load_funding(cfg.symbol, cfg.start, cfg.end, cfg.cache_dir, verbose)
               if cfg.funding else None)
    metrics = (load_metrics(cfg.symbol, cfg.start, cfg.end, cfg.cache_dir,
                            verbose=verbose) if cfg.metrics else None)
    return funding, metrics


__all__ = [
    "load_bars", "load_auxiliary", "load_binance", "fetch_month",
    "load_funding", "load_metrics", "synthetic_bars", "synthetic_funding",
    "synthetic_metrics", "OHLCV",
]
