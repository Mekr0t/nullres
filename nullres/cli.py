"""Command line interface.

    nullres fetch    --config configs/btc_1h.toml     download and cache bars
    nullres budget   --config configs/btc_1h.toml     accuracy needed to beat costs
    nullres audit    --config configs/btc_1h.toml     leakage + null-data checks
    nullres run      --config configs/btc_1h.toml     backtest every strategy
    nullres sweep    --config configs/btc_1h.toml     threshold sensitivity surface
    nullres features --config configs/btc_1h.toml     out-of-sample importances
    nullres ablate   --config configs/btc_4h_deriv.toml --ablate derivatives
                                                   matched-sample A/B on AUC
    nullres xsec     --config configs/xsec_4h.toml    cross-sectional long/short
    nullres log                                       the run ledger
    nullres robust   --config configs/btc_4h.toml -s donchian
                                                   three falsification tests:
                                                   parameter neighbourhood,
                                                   sub-period stability vs hold,
                                                   cross-symbol transfer

Any option can be overridden without editing the file:

    nullres run -c configs/btc_1h.toml --set sizing.min_hold=168

Every command here is three lines: render the header, call `nullres.api`,
render the result. Nothing in this file computes anything. If you find yourself
adding a calculation to a `cmd_*` function, it belongs in `nullres.api`; if you
find yourself adding a format string, it belongs in `nullres.report`. That
boundary is what makes each command callable from Python — the CLI is one of
two front ends, not the only way in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from nullres import api, report
from nullres.config import load_config
from nullres.errors import ConfigError, NullresError


def _configure_logging(level: int = logging.INFO) -> None:
    """Send library log records to stdout, formatted as bare messages.

    The library modules log; only this file prints. That split is what lets
    `nullres` be imported by something with its own logging setup — a notebook,
    a scheduler, another library — without hijacking its stdout.

    A handler is attached to the `nullres` logger rather than the root, and
    `propagate` is turned off, so configuring us never disturbs anyone else's
    handlers. `%(message)s` and stdout are deliberate: this replaced `print`,
    and terminal output has to come out byte-identical to what it was.

    The NullHandler that `nullres/__init__.py` attaches at import does NOT
    count as already-configured. Testing a bare `if not logger.handlers` here
    silently swallowed every progress line — the NullHandler satisfied it, the
    StreamHandler was never added, and the command still exited 0 with the
    results table intact. Only a diff against the previous output caught it.
    """
    logger = logging.getLogger("nullres")
    logger.setLevel(level)
    logger.propagate = False
    configured = any(not isinstance(h, logging.NullHandler)
                     for h in logger.handlers)
    if not configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


# Default counterparties for `robust`'s cross-symbol transfer test.
TRANSFER_SYMBOLS = ("ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")


def _split_symbols(raw: str | None) -> list[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def cmd_log(cfg, args) -> int:
    """The evidence ledger. The graveyard explains; this one remembers."""
    print(report.format_ledger(api.ledger(verdict=args.verdict, limit=args.limit)))
    return 0


def cmd_fetch(cfg, args) -> int:
    print(report.format_fetch(api.fetch(cfg)))
    return 0


def cmd_budget(cfg, args) -> int:
    """What accuracy would this instrument and cost structure actually require?

    Run this FIRST, before building features. It is arithmetic, it takes two
    seconds, and it will tell you whether the thing you are about to attempt is
    possible at all.
    """
    print(report.budget_header(cfg))
    print(report.budget_body(api.budget(cfg)))
    return 0


def cmd_run(cfg, args) -> int:
    print(report.run_header(cfg, api.killed_warning(cfg)))
    result = api.run(cfg, n_trials=args.trials, ablate=args.ablate)
    print(report.run_body(result))

    if args.save:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{cfg.name}.json"
        path.write_text(json.dumps(result.metrics, indent=2, default=float))
        print(f"\nwrote {path}")

    print("\n".join(report.ledger_line(result.record)))
    return 0


def cmd_audit(cfg, args) -> int:
    print(report.audit_header(cfg))
    result = api.audit(cfg)
    print(report.audit_body(result))
    return result.exit_code


def cmd_sweep(cfg, args) -> int:
    """Threshold sensitivity — read the SHAPE, not the peak."""
    print(report.sweep_header(cfg))
    print(report.sweep_body(api.sweep(cfg, args.strategy)))
    return 0


def cmd_robust(cfg, args) -> int:
    """Try three times to kill a strategy that looked good once."""
    symbols = _split_symbols(args.symbols) or list(TRANSFER_SYMBOLS)
    print(report.robust_header(cfg, args.strategy, api.killed_warning(cfg)))
    result = api.robust(cfg, args.strategy, symbols,
                        transfer_start=args.transfer_start)
    print(report.robust_body(result))
    return 0


def cmd_ablate(cfg, args) -> int:
    """Does a feature group improve DISCRIMINATION, on matched samples?"""
    group = args.ablate or "derivatives"
    print(report.ablate_header(cfg, group))
    print(report.ablate_body(api.ablate(cfg, group)))
    return 0


def cmd_features(cfg, args) -> int:
    print(report.features_header(cfg))
    print(report.features_body(api.feature_importance(cfg)))
    return 0


def cmd_xsec(cfg, args) -> int:
    """Cross-sectional long/short on a panel of symbols."""
    # Resolved up front so the header can state the universe size before
    # `load_panel` runs — that call is the slowest thing in the repo, and a
    # banner that arrives after it is a banner nobody reads. Enumeration is
    # cached, so paying for it here costs nothing on a re-run.
    symbols, hardcoded = api.resolve_universe(
        cfg, _split_symbols(args.symbols) or None, args.universe)
    print(report.xsec_header(cfg, len(symbols), args.universe))

    result = api.xsec(cfg, symbols=symbols, universe_month=args.universe,
                      hardcoded=hardcoded, top_n=args.top_n, top_k=args.top_k,
                      rebalance=args.rebalance, verify=args.verify,
                      n_trials=args.trials)
    print(report.xsec_body(result))
    return 0


COMMANDS = {
    "fetch": cmd_fetch,
    "log": cmd_log,
    "budget": cmd_budget,
    "run": cmd_run,
    "audit": cmd_audit,
    "sweep": cmd_sweep,
    "robust": cmd_robust,
    "ablate": cmd_ablate,
    "xsec": cmd_xsec,
    "features": cmd_features,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nullres", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", "-c", default="configs/btc_1h.toml")
    parser.add_argument("--strategy", "-s", default="ml_meta",
                        help="sweep/robust: which strategy to operate on")
    parser.add_argument("--save", action="store_true", help="run: write metrics JSON")
    parser.add_argument("--trials", type=int, default=None,
                        help="override the multiple-testing trial count used by "
                             "deflated_sharpe (default: read from the run ledger)")
    parser.add_argument("--verdict", default=None,
                        choices=["KILLED", "SURVIVED", "INCONCLUSIVE"],
                        help="log: show only runs with this verdict")
    parser.add_argument("--limit", type=int, default=25,
                        help="log: how many recent runs to show")
    parser.add_argument("--ablate", default=None, metavar="GROUP",
                        help="run: drop a feature group (e.g. 'derivatives') "
                             "after row alignment, for a matched-sample A/B")
    # Default is None, not the list, so "was this passed?" is answerable.
    # `cmd_xsec` needs to know, and used to find out by scanning sys.argv for a
    # string starting with "--symbols" — which read the HOST process's command
    # line whenever argv was empty, since `[] or sys.argv[1:]` skips the empty
    # list. argparse already has a sentinel for this; use it.
    parser.add_argument("--symbols", default=None,
                        help=f"robust: symbols for the cross-symbol transfer test "
                             f"(default: {','.join(TRANSFER_SYMBOLS)}); "
                             f"xsec: use these instead of the fixed universe")
    parser.add_argument("--top-k", type=int, default=None,
                        help="xsec: symbols long and short per side")
    parser.add_argument("--universe", default=None, metavar="YYYY-MM",
                        help="xsec: enumerate the universe from the archive as "
                             "of this month instead of the hardcoded 11")
    parser.add_argument("--top-n", type=int, default=None,
                        help="xsec: keep the top-N by trailing dollar volume")
    parser.add_argument("--rebalance", type=int, default=42,
                        help="xsec: bars between book rebalances (turnover control)")
    parser.add_argument("--verify", action="store_true",
                        help="xsec: run the controls that decide whether panel "
                             "skill is real — shuffled labels, survivors-only, "
                             "per-symbol spread, delisted P&L share, tail census")
    parser.add_argument("--transfer-start", default=None, metavar="YYYY-MM",
                        help="robust: force a common start date across symbols "
                             "(auxiliary archives begin at different dates)")
    parser.add_argument("--set", action="append", default=[], metavar="a.b=v",
                        help="override a config value, e.g. --set sizing.min_hold=12")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress progress logging (results still print)")
    parser.add_argument("--debug", action="store_true",
                        help="log everything, and re-raise errors with a traceback")
    args = parser.parse_args(argv)

    _configure_logging(logging.WARNING if args.quiet else
                       logging.DEBUG if args.debug else logging.INFO)

    pd.set_option("display.width", 200)
    try:
        cfg = load_config(args.config)
        for override in args.set:
            _apply_override(cfg, override)
        return COMMANDS[args.command](cfg, args)
    except NullresError as exc:
        # The library raises; the CLI is the only layer that decides what an
        # error costs the process. Each error type carries its own exit code,
        # so the mapping cannot drift from the hierarchy.
        if args.debug:
            raise
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return exc.exit_code


def _apply_override(cfg, spec: str) -> None:
    """Apply `section.key=value`, parsing the value as TOML would."""
    if "=" not in spec:
        raise ConfigError(f"bad --set {spec!r}, expected section.key=value")
    path, raw = spec.split("=", 1)
    parts = path.split(".")

    # `params` is a plain dict of per-strategy kwargs, not a dataclass, so
    # `--set params.donchian.entry=48` used to die on `getattr(dict, ...)` with
    # a bare AttributeError. Rule parameters are exactly what you want to
    # override from the command line, so handle the dict branch explicitly.
    if parts[0] == "params":
        if len(parts) != 3:
            raise ConfigError(
                f"bad --set {spec!r}: expected params.<strategy>.<key>=value"
            )
        _, strategy, key = parts
        existing = cfg.params.get(strategy, {}).get(key)
        cfg.params.setdefault(strategy, {})[key] = _coerce(raw, existing)
        return

    target = cfg
    for part in parts[:-1]:
        target = getattr(target, part)
    key = parts[-1]
    if not hasattr(target, key):
        raise ConfigError(f"unknown config path {path!r}")

    setattr(target, key, _coerce(raw, getattr(target, key)))


def _coerce(raw: str, current):
    """Parse `raw` to match the type of the value it replaces."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [v.strip() for v in raw.split(",")]
    if current is None:
        # An override for a key the config never declared: guess from the text,
        # since there is no existing value to match.
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                continue
    return raw


if __name__ == "__main__":
    sys.exit(main())
