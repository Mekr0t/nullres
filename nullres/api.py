"""The programmatic entry points. One function per command, returning data.

    from nullres import load_config, run
    result = run(load_config("configs/btc_4h.toml"))
    result.metrics["donchian"]["sharpe"]

Every function here computes and returns a `nullres.results` object. None of
them print, and none of them format — `nullres.report` does that, and the CLI
is a thin layer over the two. That separation is what makes the commands
callable from a notebook, testable without capturing stdout, and documentable
without pasting a terminal transcript.

**These functions append to the run ledger by default.** That is not an
accident of implementation: the ledger is what `deflated_sharpe` reads to find
out how many variants were tried, and a run that goes unrecorded undercounts
the exposure and flatters every result that follows it. Pass `record=False`
when you are genuinely not testing a hypothesis — re-deriving a number for a
plot, say — and be honest about which case you are in.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from nullres import audit as audit_mod
from nullres.config import CostConfig, RunConfig
from nullres.data import load_auxiliary, load_bars
from nullres.pipeline import (
    ablate as ablate_ctx, prepare, run_pipeline, trials_caveat, trials_so_far,
)
from nullres.results import (
    AblationResult, AuditResult, BudgetResult, FeatureImportanceResult,
    FetchResult, LedgerView, PanelVerification, RobustnessResult, RunResult,
    SweepResult, XsecResult,
)

log = logging.getLogger(__name__)

# `sweep` explores these axes. The hold range spans the break-even table from
# `budget`, because at these cost levels that is the axis that decides the
# outcome.
SWEEP_ENTRIES = [0.52, 0.54, 0.56, 0.58, 0.60]
SWEEP_HOLDS = [12, 48, 168, 336, 720]

# Slippage levels for the xsec cost-sensitivity table. The config's own rate is
# prepended. 8bps is about right for BTC perps and fiction for thin alts —
# which is exactly what a wide long/short book shorts.
COST_LEVELS = (25.0, 50.0, 100.0)


def _record(cfg, command: str, metrics=None, verdict=None, notes: str = "",
            variants: int = 1, enabled: bool = True):
    """Append to the run ledger. Never fatal.

    A bookkeeping bug destroying a backtest that already ran is the wrong
    trade. The catch is deliberately broad: the first version caught only
    OSError, so a signature mismatch took down every `run` after the results
    had already been computed and printed.
    """
    if not enabled:
        return None
    from nullres.runlog import record_run

    try:
        return record_run(cfg, command, metrics=metrics, verdict=verdict,
                          notes=notes, variants=variants)
    except Exception as exc:                              # noqa: BLE001
        log.warning("  (run log write failed: %s: %s)", type(exc).__name__, exc)
        return None


def killed_warning(cfg) -> str:
    """Has something within a few parameters of this config already been killed?

    Empty string when there is nothing to say. This is the reason the machine
    ledger exists: nobody re-reads a 500-line graveyard before every
    experiment, and eighteen months from now the dead end gets re-run.
    """
    from nullres.runlog import find_similar, format_warning, load_runs

    try:
        return format_warning(find_similar(cfg, load_runs()))
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch(cfg: RunConfig) -> FetchResult:
    """Download and cache bars, plus any configured auxiliary data."""
    bars = load_bars(cfg.data)
    funding = metrics = None
    if cfg.data.funding or cfg.data.metrics:
        funding, metrics = load_auxiliary(cfg.data)
    return FetchResult(cfg=cfg, bars=bars, funding=funding, metrics=metrics)


def ledger(verdict: str | None = None, limit: int = 25) -> LedgerView:
    """Read the run ledger. `verdict` filters; `limit` is a display hint."""
    from nullres.runlog import count_trials, load_runs, unrecorded_variants

    runs = load_runs()
    shown = [r for r in runs if r.verdict == verdict.upper()] if verdict else runs
    return LedgerView(
        runs=shown,
        n_total=len(runs),
        n_configs=len({r.config_hash for r in runs}),
        n_trials=count_trials(runs),
        killed=sum(1 for r in runs if r.verdict == "KILLED"),
        survived=sum(1 for r in runs if r.verdict == "SURVIVED"),
        inconclusive=sum(1 for r in runs if r.verdict == "INCONCLUSIVE"),
        unrecorded_variants=unrecorded_variants(runs),
        verdict_filter=verdict,
        limit=limit,
    )


def budget(cfg: RunConfig) -> BudgetResult:
    """What accuracy would this instrument and cost structure actually require?

    Run this FIRST. It is arithmetic, it takes two seconds, and it will tell
    you whether the thing you are about to attempt is possible at all.
    """
    bars = load_bars(cfg.data)
    logret = np.log(bars["close"]).diff()
    return BudgetResult(
        cfg=cfg,
        sigma=float(logret.std()),
        logret=logret.dropna().to_numpy(),
        # Bars carry no duration on their own; the break-even table is only
        # readable once they are converted to wall-clock at this bar size.
        hours_per_bar=8_760 / cfg.data.bars_per_year,
        min_hold=cfg.sizing.min_hold,
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run(cfg: RunConfig, n_trials: int | None = None, ablate: str | None = None,
        record: bool = True, verbose: bool = True) -> RunResult:
    """Backtest every configured strategy, plus buy & hold, out of sample.

    Args:
        n_trials: override the multiple-testing count. Defaults to reading the
            ledger, which is the honest answer — see `pipeline.trials_so_far`.
        ablate: drop a feature group after row alignment, so the rows, splits
            and benchmark stay byte-identical and the only variable is the
            feature set. Turning the data off in the config is NOT a controlled
            comparison; see `pipeline.ablate`.
    """
    warning = killed_warning(cfg)
    if n_trials is None:
        n_trials = trials_so_far(cfg, extra=len(cfg.strategies) + 1, command="run")

    before = after = 0
    if ablate:
        ctx = prepare(cfg, verbose=verbose)
        before = ctx.features.shape[1]
        ctx = ablate_ctx(ctx, ablate)
        after = ctx.features.shape[1]
        # Logged rather than returned: this is context for the work that
        # follows, and it belongs beside the pipeline's own progress lines
        # rather than at the end with the results.
        log.info("\nABLATION: dropped %d %s features (%d -> %d). Rows, splits "
                 "and benchmark are unchanged.", before - after, ablate,
                 before, after)
        metrics = run_pipeline(cfg, ctx=ctx, n_trials=n_trials, verbose=verbose)
    else:
        metrics = run_pipeline(cfg, n_trials=n_trials, verbose=verbose)

    record_obj = _record(cfg, "run", variants=len(metrics), enabled=record, metrics={
        "n_trials_used": n_trials,
        **{name: {k: m[k] for k in ("total_return", "sharpe", "max_dd", "t_stat",
                                    "n_trades", "deflated_sharpe")}
           for name, m in metrics.items()},
    })
    return RunResult(cfg=cfg, metrics=metrics, n_trials=n_trials,
                     trials_caveat=trials_caveat(), killed_warning=warning,
                     ablated=ablate, features_before=before,
                     features_after=after, record=record_obj)


def audit(cfg: RunConfig, verbose: bool = True) -> AuditResult:
    """The five mechanical leak checks.

    Run before believing any result. It takes a minute and it has a much better
    record than intuition.
    """
    from nullres.features import build_features as _build

    ctx = prepare(cfg, verbose=verbose)
    checks = []

    # Rebuild through the same auxiliary data the pipeline used, so the
    # funding/OI join is covered by the truncation test rather than exempt.
    funding, metrics = load_auxiliary(cfg.data, verbose=False, bars=ctx.bars)
    log.info("\n1/5 point-in-time feature check (recomputing on truncated history)")
    checks.append(audit_mod.check_point_in_time(
        ctx.bars, builder=lambda d: _build(d, funding=funding, metrics=metrics)
    ))

    log.info("2/5 single-feature AUC against the label")
    checks.append(audit_mod.check_label_leakage(ctx.features, ctx.label["y"]))

    log.info("3/5 shuffled-label control (retraining on permuted targets)")
    checks.append(audit_mod.check_shuffled_label(
        ctx.features, ctx.label["y"],
        ctx.label["t_end"].to_numpy(dtype=np.int64), cfg.split, cfg.model,
    ))

    log.info("4/5 null data (running the full pipeline on a random walk)")
    checks.append(audit_mod.check_null_data(run_pipeline, cfg))

    log.info("5/5 survivorship (does this universe contain assets that died?)")
    # Real bars carry a real end date, so the single-asset form of the question
    # is answerable. Synthetic bars run for however many were generated, which
    # says nothing about any instrument, so it stays n/a there.
    dates = ({"last_bar": ctx.bars.index[-1], "sample_end": cfg.data.end}
             if cfg.data.source == "binance" else {})
    checks.append(audit_mod.check_survivorship([cfg.data.symbol], delisted={},
                                               **dates))
    return AuditResult(cfg=cfg, checks=checks)


def sweep(cfg: RunConfig, strategy: str, entries=None, holds=None,
          record: bool = True, verbose: bool = True) -> SweepResult:
    """Threshold sensitivity — read the SHAPE, not the peak.

    A real edge degrades smoothly as the entry threshold moves. A lone spike
    surrounded by losses is a fitting artefact, and picking it is how you turn
    a backtest into fiction.
    """
    entries = list(entries or SWEEP_ENTRIES)
    holds = list(holds or SWEEP_HOLDS)
    ctx = prepare(cfg, verbose=verbose)

    rows = []
    for entry in entries:
        for hold in holds:
            trial = copy.deepcopy(cfg)
            trial.strategies = [strategy]
            trial.sizing.long_entry = entry
            trial.sizing.short_entry = 1 - entry
            trial.sizing.min_hold = hold
            res = run_pipeline(trial, verbose=False, ctx=ctx)
            rows.append({"entry": entry, "hold": hold,
                         "sharpe": res[strategy]["sharpe"]})

    cells = pd.DataFrame(rows)
    record_obj = _record(cfg, "sweep", notes=f"strategy={strategy}",
                         variants=len(cells), enabled=record)
    return SweepResult(cfg=cfg, strategy=strategy, entries=entries, holds=holds,
                       cells=cells, record=record_obj)


def robust(cfg: RunConfig, strategy: str, symbols: list[str],
           transfer_start: str | None = None, record: bool = True,
           verbose: bool = False) -> RobustnessResult:
    """Three independent attempts to kill a strategy that looked good once.

    Neighbourhood, sub-period stability, and cross-symbol transfer. Passing all
    three does not make a strategy real — the only test that does is forward
    paper trading — but failing any one of them is cheap information.
    """
    from nullres.robustness import (
        cross_symbol, grid_for, hold_sharpe, parameter_neighbourhood,
        period_stability, sign_flip_pairs, sign_flip_rate, verdict as decide,
    )

    warning = killed_warning(cfg)
    ctx = prepare(cfg, verbose=verbose)
    params = cfg.params.get(strategy, {})

    grid = parameter_neighbourhood(cfg, strategy, ctx=ctx)
    grid_def, kind = grid_for(strategy)
    keys = list(grid_def)
    flips = sign_flip_rate(grid, keys)
    pairs = sign_flip_pairs(grid, keys)

    stability = period_stability(cfg, strategy, ctx=ctx)
    transfer = cross_symbol(cfg, strategy, symbols, start=transfer_start)

    # The bar the grid must clear is buy & hold's Sharpe over the same window
    # AND the same statistic — not the mean of its per-year Sharpes, which is a
    # different statistic and a materially higher bar.
    bench = hold_sharpe(cfg, ctx)
    outcome, notes = decide(grid, stability, transfer, benchmark_sharpe=bench,
                            flip_rate=flips, flip_pairs=pairs)

    # The strategy is pinned into the logged config on purpose: `robust` takes
    # it from the caller rather than the file, so without this every config
    # would inherit the verdict of whichever strategy was tested last, and
    # killing donchian would warn you off ml_meta on the same data.
    logged = copy.deepcopy(cfg)
    logged.strategies = [strategy]
    record_obj = _record(
        logged, "robust", verdict=outcome, variants=len(grid) + len(transfer),
        notes=f"strategy={strategy}; " + " | ".join(notes), enabled=record,
        metrics={
            "strategy": strategy,
            "neighbourhood_positive": float((grid["sharpe"] > 0).mean()),
            "neighbourhood_median": float(grid["sharpe"].median()),
            "sign_flip_rate": float(flips) if np.isfinite(flips) else None,
            "years_beating_hold": (float((stability["excess_sharpe"] > 0).mean())
                                   if not stability.empty else None),
            "symbols_beating_hold": (float((transfer["vs_hold"] > 0).mean())
                                     if "vs_hold" in transfer else None),
        })
    return RobustnessResult(
        cfg=cfg, strategy=strategy, params=params, grid=grid, grid_keys=keys,
        grid_kind=kind, flip_rate=flips, flip_pairs=pairs, stability=stability,
        transfer=transfer, benchmark_sharpe=bench, verdict=outcome, notes=notes,
        symbols=symbols, transfer_start=transfer_start, killed_warning=warning,
        record=record_obj)


def ablate(cfg: RunConfig, group: str = "derivatives", record: bool = True,
           verbose: bool = False) -> AblationResult:
    """Matched-sample A/B on AUC for one feature group.

    Sharpe cannot answer this question. With ~80 trades an equity curve swings
    from -0.68 to +0.43 on feature sets whose AUC differs by one point.
    """
    from scipy import stats as sps

    from nullres.models.classifier import fit_predict_walk_forward

    full = prepare(cfg, verbose=verbose)
    reduced = ablate_ctx(prepare(cfg, verbose=verbose), group)
    t_end = full.label["t_end"].to_numpy(dtype=np.int64)
    y = full.label["y"]

    scores = []
    for X in (full.features, reduced.features):
        _, reports = fit_predict_walk_forward(X, y, t_end, cfg.split, cfg.model,
                                              verbose=False)
        scores.append(np.array([r["auc"] for r in reports]))
    with_group, without = scores

    t_stat, p_value = sps.ttest_rel(with_group, without)
    record_obj = _record(cfg, "ablate", notes=f"group={group}", variants=2,
                         enabled=record, metrics={
                             "group": group,
                             "auc_with": float(with_group.mean()),
                             "auc_without": float(without.mean()),
                             "auc_delta": float((with_group - without).mean()),
                             "t_stat": float(t_stat),
                             "p_value": float(p_value),
                         })
    return AblationResult(
        cfg=cfg, group=group, n_rows=len(full.features),
        features_with=full.features.shape[1],
        features_without=reduced.features.shape[1],
        auc_with=with_group, auc_without=without,
        t_stat=float(t_stat), p_value=float(p_value), record=record_obj)


def feature_importance(cfg: RunConfig, verbose: bool = True
                       ) -> FeatureImportanceResult:
    """Permutation importance on the last fold's test window.

    In-sample importances tell you what the model memorised. This tells you
    what carried out of sample, which is a much shorter list.
    """
    from nullres.features import DERIVATIVE_DOC
    from nullres.models.classifier import feature_importance as permute

    ctx = prepare(cfg, verbose=verbose)
    importances = permute(ctx.features, ctx.label["y"],
                          ctx.label["t_end"].to_numpy(dtype=np.int64),
                          cfg.split, cfg.model)
    return FeatureImportanceResult(cfg=cfg, importances=importances,
                                   derivative_names=set(DERIVATIVE_DOC))


# ---------------------------------------------------------------------------
# Cross-sectional
# ---------------------------------------------------------------------------

def resolve_universe(cfg: RunConfig, symbols: list[str] | None = None,
                     universe_month: str | None = None) -> tuple[list[str], bool]:
    """Decide which symbols a panel covers. Returns (symbols, hardcoded).

    `hardcoded` is True when the universe came from a literal list rather than
    being enumerated from the archive as of a date. It is not a detail:
    `audit.check_survivorship` reports it, because a hardcoded list is exactly
    how a universe ends up filtered by survival.

    Split out from `xsec` so a caller can know the universe size before paying
    for `load_panel`, which is the slowest thing in this repository.
    """
    from nullres.crosssec import UNIVERSE_2021_12

    if symbols:
        return list(symbols), False
    if universe_month:
        from nullres.data.universe import universe_as_of

        # Index products, not single assets: BTCDOM is BTC dominance, DEFI a
        # basket. Ranking them against single coins is not a cross-section.
        exclude = {"BTCDOMUSDT", "DEFIUSDT"}
        return [s for s in universe_as_of(universe_month, cfg.data.interval)
                if s not in exclude], False
    return list(UNIVERSE_2021_12), True


def xsec(cfg: RunConfig, symbols: list[str] | None = None,
         universe_month: str | None = None, top_n: int | None = None,
         top_k: int | None = None, rebalance: int = 42, verify: bool = False,
         n_trials: int | None = None, hardcoded: bool | None = None,
         record: bool = True, verbose: bool = True) -> XsecResult:
    """Cross-sectional long/short on a panel of symbols.

    Args:
        symbols: explicit universe. Overrides `universe_month`.
        universe_month: enumerate the universe mechanically from the archive as
            of this month (YYYY-MM), including symbols that later died. This is
            the survivorship-honest option; a hardcoded list is not.
        top_k: symbols long and short per side. Defaults to a sweep, because
            the point of a wide universe is that the same signal can be
            expressed through diversification instead of concentration.
        verify: run the controls in `panelaudit`. Expensive — two of them refit
            the entire walk-forward.
    """
    from nullres.crosssec import (
        backtest_panel, benchmarks, fit_predict_panel, load_panel,
        panel_positions,
    )

    resolved, was_hardcoded = resolve_universe(cfg, symbols, universe_month)
    if hardcoded is None:
        hardcoded = was_hardcoded
    symbols = resolved

    panel = load_panel(cfg, symbols, top_n=top_n, verbose=verbose)
    survivorship = audit_mod.check_survivorship(
        panel.symbols, panel.delisted, point_in_time=symbols, hardcoded=hardcoded)

    if verbose:
        log.info("\nwalk-forward fit (folds split on TIME; all symbols move "
                 "together)")
    proba, reports = fit_predict_panel(panel, cfg, verbose=verbose)

    # Every book is judged on the SAME window the model was scored on.
    oos_times = pd.DatetimeIndex(
        proba.dropna().index.get_level_values("ts").unique()).sort_values()
    # Books hold nothing before the first test fold opens, so every metric has
    # to be measured on the out-of-sample bars only — averaging across the flat
    # pre-OOS block multiplies Sharpe by sqrt(oos fraction).
    oos_mask = pd.Series(panel.times.isin(oos_times), index=panel.times)

    if top_k:
        ks = (top_k,)
    else:
        width = top_n or len(panel.symbols)
        ks = (2, 3, 4) if width < 12 else (2, 5, 10, 15)

    books = benchmarks(panel, cfg.cost, oos_times, rebalance=rebalance)
    if n_trials is None:
        n_trials = trials_so_far(cfg, extra=len(books) + len(ks), command="xsec")

    from nullres.backtest.metrics import by_period, summarize

    results = {name: summarize(result, cfg.data.bars_per_year,
                               n_trials=n_trials, mask=oos_mask)
               for name, result in books.items()}

    # Built once and reused by the cost sweep below. These used to be computed
    # here and then AGAIN for the cost table — on a wide panel that is minutes
    # of duplicated work for identical output.
    positions = {k: panel_positions(proba, panel, top_k=k, rebalance=rebalance)
                 for k in ks}

    stability = stability_k = None
    for k in ks:
        result = backtest_panel(positions[k], panel, cfg.cost)
        results[f"longshort_k{k}"] = summarize(
            result, cfg.data.bars_per_year, n_trials=n_trials, mask=oos_mask)
        if stability is None:
            stability = by_period(result, cfg.data.bars_per_year, mask=oos_mask)
            stability_k = k

    cost_sensitivity = _cost_sensitivity(cfg, panel, positions, ks, oos_mask,
                                         oos_times, rebalance)

    verification = None
    if verify:
        verification = verify_panel(
            panel, cfg, proba, positions[ks[0]],
            mean_auc=float(np.nanmean([r["auc"] for r in reports])),
            nominal_weight=1.0 / ks[0])

    record_obj = _record(
        cfg, "xsec", notes=f"{len(panel.symbols)} symbols, top_n={top_n}",
        variants=len(results), enabled=record, metrics={
            "n_symbols": len(panel.symbols),
            "n_delisted": len(panel.delisted),
            "n_features": int(panel.features.shape[1]),
            "mean_auc": float(np.nanmean([r["auc"] for r in reports])),
            "n_trials_used": n_trials,
            **{name: {k: m[k] for k in ("total_return", "sharpe", "max_dd",
                                        "t_stat", "n_trades", "deflated_sharpe")}
               for name, m in results.items()},
        })
    return XsecResult(
        cfg=cfg, panel=panel, proba=proba, fold_reports=reports,
        requested_symbols=symbols,
        survivorship=survivorship, oos_times=oos_times, oos_mask=oos_mask,
        books=results, ks=ks, positions=positions, stability=stability,
        stability_k=stability_k, cost_sensitivity=cost_sensitivity,
        n_trials=n_trials, trials_caveat=trials_caveat(),
        universe_month=universe_month, verification=verification,
        record=record_obj)


def _cost_sensitivity(cfg, panel, positions, ks, oos_mask, oos_times,
                      rebalance) -> pd.DataFrame:
    """Sharpe by slippage level. For a wide crypto panel this IS the result.

    The book makes its money shorting thin alts, and the difference between
    8bps and 60bps all-in is the difference between 239x and nothing.
    """
    from nullres.backtest.metrics import summarize
    from nullres.crosssec import backtest_panel, benchmarks

    rows = []
    for slip in (cfg.cost.slippage_bps, *COST_LEVELS):
        trial_cost = CostConfig(fee_bps=cfg.cost.fee_bps, slippage_bps=float(slip))
        row: dict[str, Any] = {"slippage_bps": float(slip)}
        for k in ks:
            m = summarize(backtest_panel(positions[k], panel, trial_cost),
                          cfg.data.bars_per_year, mask=oos_mask)
            row[f"k={k}"] = m["sharpe"]
        static = benchmarks(panel, trial_cost, oos_times,
                            rebalance=rebalance).get("static_vs_alts")
        if static is not None:
            row["static"] = summarize(static, cfg.data.bars_per_year,
                                      mask=oos_mask)["sharpe"]
        rows.append(row)
    return pd.DataFrame(rows)


def verify_panel(panel, cfg, proba, positions, mean_auc: float,
                 min_obs: int = 200,
                 nominal_weight: float | None = None) -> PanelVerification:
    """Run every cross-sectional control and return the numbers.

    These were run once by hand and quoted in RESEARCH.md, which meant the
    numbers underneath the project's strongest result were the only ones no
    command could regenerate. That is exactly backwards.
    """
    from nullres.panelaudit import (
        concentration, delisted_share, per_symbol_accuracy, pnl_contribution,
        shuffled_label_auc, survivors_only_auc, tail_census, tail_curve,
    )

    return PanelVerification(
        mean_auc=mean_auc,
        shuffled_auc=shuffled_label_auc(panel, cfg),
        survivors_auc=survivors_only_auc(panel, cfg),
        per_symbol=per_symbol_accuracy(proba, panel),
        delisted_share=delisted_share(positions, panel),
        contribution=pnl_contribution(positions, panel),
        tail_curve=tail_curve(positions, panel),
        tail_census=tail_census(positions, panel),
        concentration=(concentration(positions, nominal_weight)
                       if nominal_weight else None),
        min_obs=min_obs,
        nominal_weight=nominal_weight,
    )


__all__ = [
    "fetch", "ledger", "budget", "run", "audit", "sweep", "robust", "ablate",
    "feature_importance", "xsec", "resolve_universe", "verify_panel",
    "killed_warning",
]
