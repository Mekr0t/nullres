"""The API returns data; the CLI is a thin layer over it.

Two properties worth pinning. First, every command is reachable from Python and
hands back numbers rather than a transcript — that is the whole point of the
split, and it regresses the moment someone puts a calculation back into a
`cmd_*` function. Second, the formatters are pure: same result object in, same
string out, no file reads and no clock.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from nullres import api, report, results
from nullres.config import load_config

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/null.toml")


# ---------------------------------------------------------------------------
# The split is real, not cosmetic
# ---------------------------------------------------------------------------

def test_cli_commands_contain_no_computation():
    """A `cmd_*` function may wire and print. It may not calculate.

    Before the split, `cmd_xsec` was 190 lines that loaded a panel, fitted a
    walk-forward, built books, swept costs and printed — so none of it could be
    called, tested, or documented without a terminal. The guard is crude on
    purpose: command bodies should be short enough that any real logic in them
    is obvious.
    """
    from nullres import cli

    too_long = {}
    for name, fn in vars(cli).items():
        if not name.startswith("cmd_") or not callable(fn):
            continue
        body = inspect.getsource(fn)          # module level, so already at col 0
        statements = [n for n in ast.walk(ast.parse(body))
                      if isinstance(n, (ast.For, ast.While))]
        lines = len([ln for ln in body.splitlines() if ln.strip()
                     and not ln.strip().startswith("#")])
        if lines > 25 or statements:
            too_long[name] = (lines, len(statements))
    assert not too_long, (
        f"CLI commands doing real work: {too_long}. Move the computation into "
        f"nullres/api.py and the rendering into nullres/report.py."
    )


def test_every_command_has_an_api_function():
    from nullres import cli

    missing = [name for name in cli.COMMANDS
               if not hasattr(api, {"log": "ledger",
                                    "features": "feature_importance"}.get(name, name))]
    assert not missing, f"commands with no api entry point: {missing}"


def test_report_module_computes_nothing():
    """Formatters must not load data, fit models, or touch the ledger."""
    source = (REPO / "nullres" / "report.py").read_text(encoding="utf-8")
    banned = ["load_bars", "load_panel", "prepare(", "run_pipeline",
              "load_runs", "record_run", "open(", "datetime.now"]
    found = [b for b in banned if b in source]
    assert not found, (
        f"nullres/report.py references {found}. Formatters take a result "
        f"object and return a string; anything else belongs in api.py."
    )


# ---------------------------------------------------------------------------
# Results carry numbers
# ---------------------------------------------------------------------------

def test_budget_returns_numbers_not_text(cfg):
    result = api.budget(cfg)
    assert isinstance(result, results.BudgetResult)
    assert result.sigma > 0
    assert len(result.logret) > 1_000
    assert result.hours_per_bar == pytest.approx(1.0)   # 1h config


@pytest.fixture(scope="module")
def null_run(cfg):
    """One pipeline run, shared by every slow test below.

    These used to be three tests calling `api.run` four times between them —
    164 seconds to assert things that differ only in what they read off the
    same result. `test_null_control` already shares its run through a
    module-scoped fixture; this follows it.

    `n_trials=1` means no deflation, so `deflated_sharpe == sharpe` here and
    the correction can be applied afterwards without paying for a second run.
    """
    return api.run(cfg, n_trials=1, record=False, verbose=False)


@pytest.mark.slow
def test_run_returns_a_result_per_strategy(cfg, null_run):
    """Shape only.

    Whether a random walk yields an edge is `test_null_control`'s question and
    it asks it more thoroughly — over the same synthetic config, against the
    same Sharpe ceiling. Repeating it here would cost a second full pipeline to
    assert something already asserted.
    """
    assert isinstance(null_run, results.RunResult)
    assert "buy_hold" in null_run.metrics
    assert set(cfg.strategies) <= set(null_run.metrics)
    for metrics in null_run.metrics.values():
        assert {"sharpe", "deflated_sharpe", "t_stat", "n_trades"} <= set(metrics)
    assert null_run.benchmark is null_run.metrics["buy_hold"]


@pytest.mark.slow
def test_survivors_reads_the_deflated_sharpe_not_the_raw_one(null_run):
    """The correction is what empties the list, not the raw Sharpe.

    On this random walk two strategies post a small positive Sharpe, which is
    what noise looks like. Deflating against a real trial count removes them.
    Re-deflating the recorded Sharpes tests exactly that, without a second run:
    `deflated_sharpe` is a pure function of (sharpe, n_obs, bars_per_year,
    n_trials), and only the last of those changes.
    """
    from nullres.backtest.metrics import deflated_sharpe

    assert null_run.survivors, (
        "undeflated, some strategy should look positive on a random walk — "
        "otherwise this test cannot show that deflation is what removes them"
    )
    bars_per_year = null_run.cfg.data.bars_per_year
    still_up = [
        name for name, m in null_run.metrics.items()
        if name != "buy_hold"
        and deflated_sharpe(m["sharpe"], m["bars"], bars_per_year, 220) > 0
    ]
    assert not still_up, f"survived deflation at 220 trials: {still_up}"


@pytest.mark.slow
def test_record_false_leaves_the_result_unlogged(null_run):
    """The ledger feeds `deflated_sharpe`, so a caller re-deriving a number for
    a plot must be able to avoid inflating its own correction."""
    assert null_run.record is None


def test_record_false_never_touches_the_ledger_directory(cfg, tmp_path, monkeypatch):
    """The mechanism behind the assertion above, tested without a pipeline."""
    monkeypatch.chdir(tmp_path)          # cfg is loaded before this
    (tmp_path / "runs").mkdir()
    assert api._record(cfg, "run", enabled=False) is None
    assert not list((tmp_path / "runs").glob("*.json"))
    # ...and that the directory is only quiet because the flag was honoured.
    assert api._record(cfg, "run", enabled=True) is not None
    assert list((tmp_path / "runs").glob("*.json"))


def test_ledger_view_is_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view = api.ledger()
    assert isinstance(view, results.LedgerView)
    assert view.n_total == 0 and view.runs == []


def test_resolve_universe_flags_a_hardcoded_list(cfg):
    """A hardcoded universe is how a sample ends up filtered by survival."""
    symbols, hardcoded = api.resolve_universe(cfg)
    assert hardcoded is True and "LUNAUSDT" in symbols

    explicit, hardcoded = api.resolve_universe(cfg, ["BTCUSDT", "ETHUSDT"])
    assert explicit == ["BTCUSDT", "ETHUSDT"] and hardcoded is False


# ---------------------------------------------------------------------------
# Formatters are pure functions of their input
# ---------------------------------------------------------------------------

def _fake_metrics(**over):
    base = {"total_return": 0.1, "cagr": 0.05, "sharpe": 0.5,
            "deflated_sharpe": 0.2, "max_dd": -0.2, "calmar": 0.25,
            "t_stat": 1.5, "n_trades": 12, "cost_drag": 0.01, "exposure": 0.9,
            "gross_exposure": 1.0, "peak_exposure": 1.0}
    return {**base, **over}


def test_run_body_is_deterministic(cfg):
    result = results.RunResult(
        cfg=cfg, metrics={"buy_hold": _fake_metrics(),
                          "donchian": _fake_metrics(deflated_sharpe=-0.3)},
        n_trials=220)
    assert report.run_body(result) == report.run_body(result)


def test_run_body_flags_strategies_that_do_not_beat_luck(cfg):
    result = results.RunResult(
        cfg=cfg, metrics={"donchian": _fake_metrics(deflated_sharpe=-0.3)},
        n_trials=220)
    text = report.run_body(result)
    assert "indistinguishable from luck" in text
    assert "220 variants" in text


def test_format_run_includes_the_header(cfg):
    result = results.RunResult(cfg=cfg, metrics={"buy_hold": _fake_metrics()},
                               n_trials=1)
    assert cfg.name in report.format_run(result)


def test_sweep_grid_pivots_to_entry_by_hold(cfg):
    cells = pd.DataFrame([{"entry": e, "hold": h, "sharpe": e * h}
                          for e in (0.52, 0.54) for h in (12, 48)])
    result = results.SweepResult(cfg=cfg, strategy="donchian",
                                 entries=[0.52, 0.54], holds=[12, 48],
                                 cells=cells)
    assert result.n_cells == 4
    assert result.grid.loc[0.54, 48] == pytest.approx(0.54 * 48)
    assert "tried 4 variants" in report.sweep_body(result)


def test_audit_result_separates_skipped_from_failed(cfg):
    from nullres.audit import Check

    result = results.AuditResult(cfg=cfg, checks=[
        Check("a", True, "ok"),
        Check("b", False, "broken"),
        Check("c", True, "not applicable", applicable=False),
    ])
    assert [c.name for c in result.failed] == ["b"]
    assert [c.name for c in result.skipped] == ["c"]
    assert result.passed is False and result.exit_code == 1

    text = report.audit_body(result)
    assert "1 CHECK(S) FAILED" in text
    assert "have NOT been ruled out" in text


def test_audit_result_that_passes_says_what_that_does_not_mean(cfg):
    from nullres.audit import Check

    result = results.AuditResult(cfg=cfg, checks=[Check("a", True, "ok")])
    assert result.passed and result.exit_code == 0
    assert "not lying to you" in report.audit_body(result)
