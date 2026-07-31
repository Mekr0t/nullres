"""Writing to the parquet cache without leaving corpses behind.

`crosssec._guard_metrics_fetch` refuses to start a download it estimates at
four hours, on the grounds that a silent multi-hour fetch is not something a
tool should do to you. The corollary went unhandled: a download that long WILL
be interrupted — Ctrl-C, a laptop lid, an OOM kill — and `DataFrame.to_parquet`
writes in place. An interrupt part-way through leaves a truncated file at
exactly the path every later run treats as authoritative.

That failure is worse than a missing file in three ways. It is permanent, since
nothing ever rewrites a path that already exists. It is silent until the next
read, which may be days later. And it surfaces as a parquet decode error deep
inside pyarrow, naming neither the cache nor the download that produced it — so
the obvious reading is "the library is broken", not "delete this one file".

Writing to a temporary file in the same directory and renaming it into place
fixes it. `os.replace` is atomic on POSIX and on Windows, so a reader sees
either the previous file or the complete new one, never a partial write.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def write_parquet_atomic(df: pd.DataFrame, path: str | Path) -> Path:
    """Write `df` to `path` so that readers never observe a partial file.

    The temporary file is created beside the target rather than in the system
    temp directory, because `os.replace` is only atomic within one filesystem
    and a cache directory may well be on a different mount.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt is the single most
        # likely way to land here, and it is not an Exception subclass. Leaving
        # the temp file behind on Ctrl-C would recreate the litter this exists
        # to prevent, one `.tmp` per abandoned download.
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_parquet_or_discard(path: str | Path) -> pd.DataFrame | None:
    """Read a cached parquet, deleting and reporting it if it is unreadable.

    Files written before `write_parquet_atomic` may already be truncated, and a
    corrupt cache entry should cost one re-download rather than an afternoon of
    reading tracebacks. Returning None means "treat this as a cache miss".
    """
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:                              # noqa: BLE001
        log.warning("  discarding unreadable cache file %s (%s: %s); "
                    "it will be re-fetched", path.name, type(exc).__name__, exc)
        path.unlink(missing_ok=True)
        return None


__all__ = ["write_parquet_atomic", "read_parquet_or_discard"]
