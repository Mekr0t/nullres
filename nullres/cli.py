"""A research harness that tries to disprove trading strategies.

The command list lives in `COMMAND_HELP` and the options in `build_parser`, so
`nullres --help` is generated rather than transcribed. This docstring used to
carry a hand-written copy of both, which had already drifted — it advertised
`ablate --ablate derivatives`, a flag that no longer exists.

Every command here is three lines: render the header, call `nullres.api`,
render the result. Nothing in this file computes anything. If you find yourself
adding a calculation to a `cmd_*` function, it belongs in `nullres.api`; if you
find yourself adding a format string, it belongs in `nullres.report`. That
boundary is what makes each command callable from Python — the CLI is one of
two front ends, not the only way in.

Options are attached per subcommand for the same reason. The flat parser this
replaced accepted every flag for every command, so `nullres log --top-k 99
--verify` ran happily and ignored both. A silently dropped flag is worse than
a rejected one: the output looks like an answer to the question you thought
you asked.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from nullres import __version__, api, report
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
    group = args.group
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


# Every command is (name, handler, one-line summary, option groups). The
# summary is what `nullres --help` lists, so it has to say what the command
# decides, not what it computes.
#
# Options are attached per command rather than globally. That is the whole
# point of the subparser rewrite: the flat parser accepted every flag for every
# command, so `nullres log --top-k 99 --verify` ran happily and ignored both.
# A flag that is silently dropped is worse than one that errors, because the
# output looks like an answer to the question you thought you asked.
COMMAND_HELP = {
    "fetch": "download and cache bars, plus any configured futures data",
    "budget": "what accuracy this instrument and cost structure require",
    "audit": "five mechanical leak checks — run before believing anything",
    "run": "backtest every configured strategy against buy & hold",
    "sweep": "entry threshold vs holding period; read the shape, not the peak",
    "features": "permutation importance, out of sample",
    "ablate": "matched-sample A/B on AUC for one feature group",
    "xsec": "cross-sectional long/short on a panel of symbols",
    "robust": "three attempts to falsify a strategy that looked good once",
    "log": "the run ledger — what has already been tried, and killed",
}

EPILOG = """\
examples:
  nullres budget -c configs/btc_1h.toml            can this idea pay for itself?
  nullres audit  -c configs/btc_1h.toml            is the harness lying to me?
  nullres run    -c configs/btc_1h.toml --save     backtest, with baselines
  nullres robust -c configs/btc_4h.toml -s donchian
  nullres xsec   -c configs/xsec_4h.toml --universe 2021-12 --top-n 40 --verify
  nullres log --verdict KILLED                     what have I already killed?

Any config value can be overridden without editing the file:
  nullres run -c configs/btc_1h.toml --set sizing.min_hold=168

`nullres <command> --help` shows the options that command actually reads.
"""


def _add_config_options(parser, default: str = "configs/btc_1h.toml") -> None:
    parser.add_argument("--config", "-c", default=default, metavar="PATH",
                        help=f"experiment config [{default}]")
    parser.add_argument("--set", action="append", default=[], metavar="a.b=v",
                        help="override a config value, repeatable")


def _add_trials_option(parser) -> None:
    parser.add_argument("--trials", type=int, default=None, metavar="N",
                        help="override the multiple-testing count used by "
                             "deflated_sharpe (default: read from the ledger)")


def build_parser() -> argparse.ArgumentParser:
    """The command line, as a parser per command.

    Shared options are attached through small helpers rather than a parent
    parser so each subcommand's `--help` lists them in a sensible order.
    """
    parser = argparse.ArgumentParser(
        prog="nullres", description=__doc__.split("\n\n")[0], epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version=f"nullres {__version__}")
    # Global, because they shape output rather than the experiment.
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress progress logging (results still print)")
    parser.add_argument("--debug", action="store_true",
                        help="log everything, and re-raise errors with a traceback")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>",
                                       required=True)

    def add(name: str) -> argparse.ArgumentParser:
        # Not ArgumentDefaultsHelpFormatter: it appends "(default: None)" to
        # every optional flag, and doubles up on the ones whose help already
        # explains the default in words — "(default: read from the ledger)
        # (default: None)". Defaults worth knowing are written into the help
        # text in [brackets]; the rest are None and saying so is noise.
        return subparsers.add_parser(
            name, help=COMMAND_HELP[name], description=COMMAND_HELP[name],
            formatter_class=argparse.RawDescriptionHelpFormatter)

    for name in ("fetch", "budget", "audit", "features"):
        _add_config_options(add(name))

    run = add("run")
    _add_config_options(run)
    _add_trials_option(run)
    run.add_argument("--save", action="store_true",
                     help="write the metrics to <out_dir>/<name>.json")
    run.add_argument("--ablate", default=None, metavar="GROUP",
                     help="drop a feature group after row alignment, so rows, "
                          "splits and benchmark stay identical")

    sweep = add("sweep")
    _add_config_options(sweep)
    sweep.add_argument("--strategy", "-s", default="ml_meta",
                       help="which strategy to sweep [ml_meta]")

    ablate = add("ablate")
    _add_config_options(ablate)
    ablate.add_argument("--group", "-g", default="derivatives", metavar="GROUP",
                        help="feature group to ablate [derivatives]")

    robust = add("robust")
    _add_config_options(robust)
    robust.add_argument("--strategy", "-s", default="ml_meta",
                        help="which strategy to try to falsify [ml_meta]")
    robust.add_argument("--symbols", default=None, metavar="A,B,C",
                        help=f"symbols for the cross-symbol transfer test "
                             f"(default: {','.join(TRANSFER_SYMBOLS)})")
    robust.add_argument("--transfer-start", default=None, metavar="YYYY-MM",
                        help="force a common start date across symbols, so the "
                             "test compares assets and not eras")

    # A single-asset 1h config would load and then produce a slow, meaningless
    # panel, so this one command carries its own default.
    xsec = add("xsec")
    _add_config_options(xsec, "configs/xsec_4h.toml")
    _add_trials_option(xsec)
    xsec.add_argument("--symbols", default=None, metavar="A,B,C",
                      help="use these symbols instead of the fixed 2021-12 universe")
    xsec.add_argument("--universe", default=None, metavar="YYYY-MM",
                      help="enumerate the universe from the archive as of this "
                           "month — the survivorship-honest option")
    xsec.add_argument("--top-n", type=int, default=None, metavar="N",
                      help="keep the top-N by TRAILING dollar volume")
    xsec.add_argument("--top-k", type=int, default=None, metavar="K",
                      help="symbols long and short per side (default: sweep k)")
    xsec.add_argument("--rebalance", type=int, default=42, metavar="BARS",
                      help="bars between book rebalances [42]")
    xsec.add_argument("--verify", action="store_true",
                      help="run the controls that decide whether panel skill is "
                           "real — shuffled labels, survivors-only, per-symbol "
                           "spread, delisted P&L share, tail census")

    log = add("log")
    log.add_argument("--verdict", default=None,
                     choices=["KILLED", "SURVIVED", "INCONCLUSIVE"],
                     help="show only runs with this verdict")
    log.add_argument("--limit", type=int, default=25, metavar="N",
                     help="how many recent runs to show [25]")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    _configure_logging(logging.WARNING if args.quiet else
                       logging.DEBUG if args.debug else logging.INFO)

    pd.set_option("display.width", 200)
    try:
        # `log` reads the ledger and nothing else, so it has no --config to
        # load. Loading one anyway made `nullres log` fail on a checkout
        # without configs/btc_1h.toml — the one command that should work
        # anywhere, refusing to run over a dependency it never touches.
        cfg = None
        if hasattr(args, "config"):
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
