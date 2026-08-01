"""The library must not kill its host, and must not print behind its back.

Two properties that no unit test covered, and that both silently regress the
moment someone adds a `raise SystemExit` or a `print` to a library module for
convenience during debugging. They are cheap to assert mechanically and
expensive to notice by hand.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from nullres.errors import (
    ConfigError,
    DataUnavailableError,
    InsufficientDataError,
    NullresError,
)

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "nullres"

# The CLI is the process boundary: it is allowed to print and to exit.
PRESENTATION = {"nullres/cli.py"}


def _library_modules() -> list[Path]:
    return sorted(
        p for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.relative_to(REPO).as_posix() not in PRESENTATION
        and p.name != "__main__.py"
    )


def test_no_library_module_raises_systemexit():
    """A library that raises SystemExit can terminate a notebook or a web request.

    Six modules used to. `robustness.cross_symbol` had to carry
    `except (SystemExit, ValueError)` to survive a symbol with no archive —
    catching the interpreter's shutdown signal to keep a loop running, which is
    what a missing exception type looks like from the inside.
    """
    offenders = []
    for path in _library_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            name = exc.func if isinstance(exc, ast.Call) else exc
            if isinstance(name, ast.Name) and name.id == "SystemExit":
                offenders.append(
                    f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                )
    assert not offenders, (
        "library modules raising SystemExit:\n  " + "\n  ".join(offenders) +
        "\n\nRaise a NullresError subclass instead; nullres/cli.py maps it to "
        "an exit code."
    )


def test_no_library_module_prints():
    """Progress belongs on a logger the caller can silence, redirect, or capture.

    `print` in a library writes to whatever stdout happens to be, which for an
    importer is their output. 157 calls used to.
    """
    offenders = []
    for path in _library_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                offenders.append(
                    f"{path.relative_to(REPO).as_posix()}:{node.lineno}"
                )
    assert not offenders, (
        "library modules calling print():\n  " + "\n  ".join(offenders) +
        "\n\nUse `log = logging.getLogger(__name__)` and log.info(...). Only "
        "nullres/cli.py prints."
    )


def test_importing_nullres_attaches_no_handler():
    """A library that configures logging at import time steals the decision.

    Checked in a subprocess, because this is a claim about a fresh interpreter.
    Asserting on the live logger would instead measure whatever earlier tests
    left attached — `cli.main` legitimately adds a StreamHandler, and caplog
    adds its own — so the in-process version passed only when it happened to
    run first.
    """
    import subprocess
    import sys

    probe = (
        "import logging, sys, nullres;"
        "real = [h for h in logging.getLogger('nullres').handlers"
        "        if not isinstance(h, logging.NullHandler)];"
        "print(real)"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", (
        f"importing nullres attached {out.stdout.strip()} to the 'nullres' "
        f"logger. Only cli._configure_logging may do that."
    )


def test_importing_nullres_prints_nothing():
    """Import must be silent as well as handler-free."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", "import nullres"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    assert out.stdout == "" and out.stderr == "", (
        f"importing nullres wrote stdout={out.stdout!r} stderr={out.stderr!r}"
    )


@pytest.mark.parametrize("error", [ConfigError, DataUnavailableError,
                                   InsufficientDataError])
def test_every_error_is_a_nullres_error(error):
    assert issubclass(error, NullresError)


def test_exit_codes_are_distinct_and_nonzero():
    """A caller scripting nullres has to be able to tell the failures apart."""
    codes = {e: e.exit_code for e in
             (NullresError, ConfigError, DataUnavailableError,
              InsufficientDataError)}
    assert all(c > 0 for c in codes.values()), codes
    assert len(set(codes.values())) == len(codes), f"duplicate exit codes: {codes}"


def test_config_and_insufficient_data_stay_valueerrors():
    """Backwards compatibility: these were ValueErrors before the hierarchy."""
    assert issubclass(ConfigError, ValueError)
    assert issubclass(InsufficientDataError, ValueError)
    # DataUnavailableError deliberately is NOT: nothing was wrong with the
    # request, the bytes are simply absent.
    assert not issubclass(DataUnavailableError, ValueError)


def test_cli_actually_emits_progress_to_stdout(capsys):
    """The handler must survive the NullHandler attached at import.

    `_configure_logging` first read `if not logger.handlers`, which the
    import-time NullHandler satisfies — so no StreamHandler was ever attached
    and every progress line vanished. The command still exited 0 with a full
    results table, so nothing failed; the output was simply gone. Moving
    `print` to `logging` is only safe if something asserts the messages still
    come out.
    """
    from nullres.cli import _configure_logging

    logger = logging.getLogger("nullres")
    saved = list(logger.handlers)
    logger.handlers = [logging.NullHandler()]
    try:
        _configure_logging()
        logging.getLogger("nullres.test").info("progress line")
        assert "progress line" in capsys.readouterr().out
    finally:
        logger.handlers = saved


def test_cli_maps_errors_to_their_exit_code():
    """`nullres run -c does-not-exist.toml` must not traceback."""
    from nullres.cli import main

    code = main(["run", "--config", "configs/btc_1h.toml",
                 "--set", "sizing.no_such_key=1"])
    assert code == ConfigError.exit_code


@pytest.mark.parametrize("path,expect", [
    ("configs/does-not-exist.toml", "no config file"),
    ("README.md", "not valid TOML"),
])
def test_a_bad_config_path_is_a_config_error_not_a_traceback(path, expect):
    """`nullres run -c typo.toml` used to end in a raw FileNotFoundError from
    inside pathlib, which tells the reader where Python gave up rather than
    what they got wrong."""
    from nullres.config import load_config

    with pytest.raises(ConfigError, match=expect):
        load_config(path)


def test_cli_argv_does_not_leak_the_host_command_line():
    """`main([])` must parse [] — not fall through to sys.argv.

    `[] or sys.argv[1:]` skips the empty list, so an empty argv read the
    HOST process's command line. Under pytest that meant nullres saw pytest's
    flags. argparse raises SystemExit(2) on a missing positional, which is the
    correct answer for an empty command line.
    """
    from nullres.cli import main

    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


# ---------------------------------------------------------------------------
# The command line rejects what it cannot honour
# ---------------------------------------------------------------------------

def _parser():
    from nullres.cli import build_parser

    return build_parser()


@pytest.mark.parametrize("argv", [
    ["log", "--top-k", "99"],           # xsec option on the ledger command
    ["log", "--verify"],
    ["run", "--verify"],                # xsec option on run
    ["run", "--top-n", "40"],
    ["budget", "--strategy", "donchian"],   # robust/sweep option on budget
    ["xsec", "--save"],                 # run option on xsec
    ["audit", "--trials", "5"],         # audit does not deflate anything
    ["fetch", "--rebalance", "10"],
])
def test_a_flag_a_command_cannot_honour_is_rejected(argv):
    """The flat parser accepted every flag for every command and ignored the
    ones that did not apply, so `nullres log --top-k 99 --verify` ran happily
    and answered a different question than the one asked."""
    from nullres.cli import main

    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2


@pytest.mark.parametrize("argv", [
    ["run", "-c", "configs/null.toml", "--save", "--trials", "7"],
    ["run", "-c", "configs/null.toml", "--ablate", "derivatives"],
    ["xsec", "--universe", "2021-12", "--top-n", "40", "--verify"],
    ["xsec", "--symbols", "BTCUSDT,ETHUSDT", "--top-k", "2"],
    ["robust", "-s", "donchian", "--transfer-start", "2021-12"],
    ["ablate", "--group", "derivatives"],
    ["sweep", "-s", "ml_meta"],
    ["log", "--verdict", "KILLED", "--limit", "5"],
    ["budget", "--set", "cost.fee_bps=20"],
])
def test_every_documented_invocation_parses(argv):
    """Parsing only — these must not have become unreachable in the rewrite."""
    args = _parser().parse_args(argv)
    assert args.command == argv[0]


def test_every_command_is_reachable_from_the_parser():
    from nullres.cli import COMMANDS

    for name in COMMANDS:
        assert _parser().parse_args([name]).command == name


def test_help_lists_every_command():
    from nullres.cli import COMMANDS

    text = _parser().format_help()
    missing = [n for n in COMMANDS if n not in text]
    assert not missing, f"commands absent from `nullres --help`: {missing}"


def test_each_command_summarises_itself():
    """A command with no summary is invisible in `--help`."""
    from nullres.cli import COMMAND_HELP, COMMANDS

    assert set(COMMAND_HELP) == set(COMMANDS)
    assert all(COMMAND_HELP[n].strip() for n in COMMANDS)


def test_no_command_is_missing_its_config_option():
    """Everything except `log` operates on an experiment, so needs --config."""
    from nullres.cli import COMMANDS

    for name in COMMANDS:
        args = _parser().parse_args([name])
        assert hasattr(args, "config") is (name != "log"), name
