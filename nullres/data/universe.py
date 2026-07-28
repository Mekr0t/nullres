"""Point-in-time universe construction.

Writing down a list of symbols from memory is hindsight dressed as data — you
will recall the ones that survived. The universe here is built mechanically:
enumerate everything the archive holds, then ask each symbol whether it had
data in a given month. A coin that listed in 2023 fails that test; a coin that
died in 2022 passes it, and belongs in the sample.

Liquidity screening is a separate problem and a subtler one. Ranking by
full-sample average volume is lookahead — it knows which coins would go on to
matter. `liquidity_screen` ranks on a TRAILING window only, so the universe at
each bar is the one you could actually have chosen at that bar.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIXES = {
    "um": "data/futures/um/monthly/klines/",
    "spot": "data/spot/monthly/klines/",
}
KLINES = {
    "um": "https://data.binance.vision/data/futures/um/monthly/klines",
    "spot": "https://data.binance.vision/data/spot/monthly/klines",
}


def list_symbols(market: str = "um", pattern: str = r"[A-Z0-9]+USDT") -> list[str]:
    """Every symbol the archive holds for `market`."""
    prefix = PREFIXES[market]
    out, marker = [], None
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        resp = requests.get(LISTING, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"s3": root.tag.split("}")[0].strip("{")}
        got = [p.find("s3:Prefix", ns).text for p in root.findall("s3:CommonPrefixes", ns)]
        out.extend(s[len(prefix):].strip("/") for s in got)
        truncated = root.find("s3:IsTruncated", ns)
        if truncated is None or truncated.text != "true" or not got:
            break
        marker = got[-1]
    return sorted(s for s in out if re.fullmatch(pattern, s))


def universe_as_of(month: str, interval: str = "4h", market: str = "um",
                   workers: int = 24, cache_dir: str = "data",
                   verbose: bool = True) -> list[str]:
    """Symbols that were trading in `month` — nothing about what came after.

    Cached, because 787 HEAD requests is rude to repeat and the answer for a
    past month never changes.
    """
    cache = Path(cache_dir) / f"universe-{market}-{interval}-{month}.txt"
    if cache.exists():
        return [s for s in cache.read_text().split() if s]

    candidates = list_symbols(market)
    if verbose:
        print(f"  {len(candidates):,} symbols in archive; testing {month}")

    base = KLINES[market]

    def existed(symbol: str) -> str | None:
        url = f"{base}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
        try:
            return symbol if requests.head(url, timeout=30).status_code == 200 else None
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        live = [s for s in pool.map(existed, candidates) if s]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(live))
    if verbose:
        print(f"  {len(live):,} were trading in {month}")
    return live


def delisted_from_cache(symbols: list[str], interval: str, end: str,
                        cache_dir: str = "data", market: str = "um",
                        grace_months: int = 2) -> dict[str, str]:
    """Symbols whose cached archive stops well before the sample ends.

    Works entirely off local parquet files, so the survivorship check runs
    offline and costs nothing. `grace_months` absorbs the normal lag between
    the end of a range and the archive catching up — without it, every symbol
    looks delisted in the current month.
    """
    tag = "" if market == "spot" else f"{market}-"
    cutoff = (pd.Period(end, freq="M") - grace_months).strftime("%Y-%m")

    out: dict[str, str] = {}
    for symbol in symbols:
        months = sorted(
            p.stem.rsplit("-", 2)[-2] + "-" + p.stem.rsplit("-", 2)[-1]
            for p in Path(cache_dir).glob(f"{tag}{symbol}-{interval}-*.parquet")
        )
        if months and months[-1] < cutoff:
            out[symbol] = months[-1]
    return out


def liquidity_screen(volumes: pd.DataFrame, top_n: int = 40,
                     window: int = 180, min_history: int = 180) -> pd.DataFrame:
    """Boolean mask: is this symbol in the top-N by TRAILING dollar volume?

    The trailing window is what makes this point-in-time. Screening on
    full-sample volume would quietly select the coins that went on to become
    important — a survivorship bias that hides inside what looks like ordinary
    data hygiene.

    Args:
        volumes: ts x symbol quote volume per bar.
        window: bars of history the ranking is computed over.
        min_history: a symbol needs at least this much history to be eligible,
            so newly listed coins are not ranked on three days of launch hype.
    """
    trailing = volumes.rolling(window, min_periods=min_history).mean()
    ranks = trailing.rank(axis=1, ascending=False, na_option="bottom")
    eligible = trailing.notna()
    return (ranks <= top_n) & eligible
