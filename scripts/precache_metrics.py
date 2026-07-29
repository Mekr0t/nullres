"""Pre-cache open-interest metrics for a wide cross-sectional universe.

`nullres xsec --universe ... --top-n 40` refuses to start if metrics are missing
for more than a handful of symbols, because Binance publishes them ONE ARCHIVE
PER DAY — a wide panel needs roughly `symbols x months x 30` requests, which is
hours of quiet network traffic with no output. `_guard_metrics_fetch` raises
rather than starting that silently. This is the deliberate version it tells you
to run.

    python scripts/precache_metrics.py --config configs/xsec_4h.toml \
        --universe 2021-12 --top-n 40

It is **resumable**: `load_metrics` caches one parquet per month and skips any
month already on disk, so re-running after an interruption costs only the months
that had not finished. Interrupt it freely.

Two things keep it from being a naive loop over the universe:

  ONLY WHAT IS TRADED   Metrics are needed for symbols that pass the liquidity
                        screen, not every symbol in the archive. On the 2021-12
                        universe that is ~123 of 136, and the rest would be
                        thousands of requests for data no book can act on.

  ONLY WHEN IT TRADED   A coin listed in 2023 has no metrics for 2021, and
                        asking anyway costs 30 requests per empty month — every
                        time you resume, because an empty month caches nothing.
                        Each symbol is fetched over its own bar range.

Bars must already be cached (`nullres xsec` fetches them, and they are cheap
monthly archives). This only fills in the expensive part.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nullres.config import load_config                      # noqa: E402
from nullres.data.binance import load_binance                # noqa: E402
from nullres.data.futures import load_metrics                # noqa: E402
from nullres.data.universe import liquidity_screen, universe_as_of  # noqa: E402

# Index products, not assets — the same exclusion `nullres xsec` applies.
EXCLUDE = {"BTCDOMUSDT", "DEFIUSDT"}


def tradable_symbols(cfg, month: str, top_n: int, screen_window: int = 180):
    """The symbols a top-N book could ever hold, with their bar ranges."""
    data = cfg.data
    symbols = [s for s in universe_as_of(month, data.interval) if s not in EXCLUDE]
    print(f"{len(symbols)} symbols in the {month} universe (excluding "
          f"{len(EXCLUDE)} index products)")

    bars: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = load_binance(symbol, data.interval, data.start, data.end,
                             data.cache_dir, verbose=False, market="um",
                             required=False)
        if frame is not None and len(frame) >= 500:
            bars[symbol] = frame
    print(f"{len(bars)} have usable cached bars")

    if not top_n:
        return {s: (f.index[0], f.index[-1]) for s, f in bars.items()}

    times = pd.DatetimeIndex(sorted(set().union(*(b.index for b in bars.values()))))
    dollar_volume = pd.DataFrame(
        {s: b["volume"] * b["close"] for s, b in bars.items()}
    ).reindex(times)
    screen = liquidity_screen(dollar_volume, top_n=top_n, window=screen_window)

    keep = {s: (bars[s].index[0], bars[s].index[-1])
            for s in bars if screen[s].any()}
    print(f"{len(keep)} ever qualify for the top-{top_n} liquidity screen — "
          f"only these need metrics")
    return keep


def cached_plan(cfg, month: str, top_n: int, refresh: bool = False):
    """Symbol ranges, computed once and remembered.

    Working out which symbols qualify means loading bars for the whole universe,
    and for delisted symbols that means requesting kline months which do not
    exist. A 404 is not cached, so every resume repeated thousands of failing
    requests before the first byte of metrics was fetched — on a job that exists
    to be resumed. The answer cannot change for a past month, so it is written
    down.
    """
    import json

    path = Path(cfg.data.cache_dir) / f"precache-plan-{month}-top{top_n}.json"
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        print(f"reusing the symbol plan in {path.name} "
              f"({len(raw)} symbols; --refresh-plan to rebuild)")
        return {s: (pd.Timestamp(a), pd.Timestamp(b)) for s, (a, b) in raw.items()}

    targets = tradable_symbols(cfg, month, top_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {s: [str(a), str(b)] for s, (a, b) in targets.items()}, indent=1))
    print(f"symbol plan written to {path.name}")
    return targets


def missing_months(symbol: str, start, end, cache_dir: str) -> int:
    cache = Path(cache_dir)
    months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                           end, freq="MS")
    return sum(
        not (cache / f"{symbol}-metrics-{m:%Y-%m}.parquet").exists() for m in months
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", default="configs/xsec_4h.toml")
    parser.add_argument("--universe", default="2021-12", metavar="YYYY-MM")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent daily requests per month")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing and exit")
    parser.add_argument("--refresh-plan", action="store_true",
                        help="recompute which symbols qualify instead of "
                             "reusing the cached plan")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="fetch only the first N symbols — use it to prove "
                             "the thing works before starting a multi-hour run")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    targets = cached_plan(cfg, args.universe, args.top_n,
                          refresh=args.refresh_plan)

    # Clip each symbol to the config window AND to when it actually traded.
    plan = []
    for symbol, (first, last) in sorted(targets.items()):
        start = max(pd.Timestamp(first), pd.Timestamp(cfg.data.start))
        end = min(pd.Timestamp(last), pd.Timestamp(cfg.data.end)
                  + pd.offsets.MonthEnd(0))
        if start >= end:
            continue
        gap = missing_months(symbol, start, end, cfg.data.cache_dir)
        if gap:
            plan.append((symbol, start, end, gap))

    if args.limit:
        plan = plan[:args.limit]

    total_months = sum(g for _, _, _, g in plan)
    print(f"\n{len(plan)} symbols need metrics, {total_months} symbol-months "
          f"missing (~{total_months * 30:,} requests)")
    if args.dry_run or not plan:
        if not plan:
            print("Nothing to fetch — the cache is complete for this universe.")
        return 0

    started = time.time()
    done_months = 0
    for i, (symbol, start, end, gap) in enumerate(plan, start=1):
        elapsed = time.time() - started
        rate = done_months / elapsed if done_months and elapsed else 0
        eta = f", eta {(total_months - done_months) / rate / 60:.0f} min" if rate else ""
        print(f"\n[{i}/{len(plan)}] {symbol}  {start:%Y-%m}..{end:%Y-%m}  "
              f"{gap} month(s) missing{eta}", flush=True)
        try:
            load_metrics(symbol, f"{start:%Y-%m}", f"{end:%Y-%m}",
                         cfg.data.cache_dir, workers=args.workers, verbose=False)
        except SystemExit as exc:
            # No metrics exist for this symbol at all. Normal for coins that
            # predate the metrics archive; not a reason to abandon the run.
            print(f"  skipped: {exc}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  FAILED ({type(exc).__name__}: {exc}) — continuing; "
                  f"re-run to retry this symbol")
        done_months += gap

    print(f"\nDone in {(time.time() - started) / 60:.0f} min. Re-run to verify "
          f"the cache is complete, then:")
    print(f"  python -m nullres xsec --config {args.config} "
          f"--universe {args.universe} --top-n {args.top_n} --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
