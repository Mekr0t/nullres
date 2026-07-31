"""The exception hierarchy.

A library must never kill the host process. Six places used to `raise
SystemExit` — the data loaders, the panel builder, the alignment step — which
is fine for a script and unusable from anything else: an import of `nullres`
could terminate a notebook, a web request, or another library's test run,
with no chance to catch it.

The smell was already visible from inside. `robustness.cross_symbol` has to
survive a symbol whose data is missing, so it carried
`except (SystemExit, ValueError)` — catching the interpreter's own shutdown
signal to keep a loop going. That is what a missing exception type looks like.

Three failure modes are worth distinguishing, because a caller responds
differently to each:

    ConfigError            You asked for something that does not exist or does
                           not parse. Nothing will fix it but editing the
                           config or the call.

    DataUnavailableError   The data is not here. It might be a symbol that
                           never listed, a month the archive does not publish,
                           or a cache that has not been filled yet. Fetching
                           more may fix it.

    InsufficientDataError  The data is here and there is not enough of it to
                           measure anything. Widen the date range, shorten the
                           windows, or lower `min_train`.

`ConfigError` and `InsufficientDataError` also subclass `ValueError`, because
that is what they were before this module existed and callers that already
catch `ValueError` should keep working. New code should catch `NullresError`,
or one of the three.

Each type carries the exit code the CLI returns for it, so the mapping lives
with the error rather than in a table in `cli.py` that drifts from it.
"""

from __future__ import annotations


class NullresError(Exception):
    """Base for every error this package raises deliberately.

    Catching this catches everything nullres considers a *handled* failure. It
    does not catch bugs — an index mismatch between a position series and its
    bars stays a bare `ValueError`, because there is no configuration that
    causes it and no caller that should recover from it.
    """

    exit_code = 1


class ConfigError(NullresError, ValueError):
    """A config value, override, or name that cannot be honoured.

    Unknown TOML keys, unknown strategies, unknown intervals, malformed
    `--set` overrides. The distinguishing property is that no amount of data
    or compute changes the answer.
    """

    exit_code = 2


class DataUnavailableError(NullresError):
    """Required market data could not be obtained.

    Deliberately NOT a `ValueError`: nothing was wrong with what you asked
    for, the bytes are simply not there. That distinction is what lets
    `cross_symbol` record "no data for this symbol" as a per-symbol note and
    carry on, rather than treating it as a bad request.
    """

    exit_code = 3


class InsufficientDataError(NullresError, ValueError):
    """There is data, but not enough of it to measure anything.

    Raised by the splitters when the folds would be too small to mean
    anything, and by `pipeline.prepare` when alignment leaves too few bars.
    Reporting a Sharpe off eighty bars would be worse than refusing.
    """

    exit_code = 4


__all__ = [
    "NullresError",
    "ConfigError",
    "DataUnavailableError",
    "InsufficientDataError",
]
