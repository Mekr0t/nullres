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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nullres import audit as audit_mod
from nullres.backtest.metrics import format_table
from nullres.config import CostConfig, load_config
from nullres.data import load_bars
from nullres.pipeline import prepare, run_pipeline, trials_caveat, trials_so_far


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def _warn_if_already_killed(cfg) -> None:
    """Check the run log before spending compute on a known dead end."""
    from nullres.runlog import find_similar, format_warning, load_runs

    warning = format_warning(find_similar(cfg, load_runs()))
    if warning:
        print(f"\n{warning}\n")


def _record(cfg, command, metrics=None, verdict=None, notes="",
            variants: int = 1) -> None:
    """Append to the run ledger. Never fatal — a logging failure must not
    invalidate a result that has already been computed.

    The catch is deliberately broad. A bookkeeping bug destroying a backtest
    that already ran is the wrong trade, and the first version of this caught
    only OSError, so a signature mismatch took down every `run` after printing
    the full results.
    """
    from nullres.runlog import record_run

    try:
        record = record_run(cfg, command, metrics=metrics, verdict=verdict,
                            notes=notes, variants=variants)
    except Exception as exc:                              # noqa: BLE001
        print(f"  (run log write failed: {type(exc).__name__}: {exc})")
        return
    dirty = " +uncommitted" if record.git_dirty else ""
    print(f"\nlogged as runs/ [{record.short_id}]  "
          f"config {record.config_hash}  git {record.git_sha}{dirty}")


def cmd_log(cfg, args) -> int:
    """The evidence ledger. The graveyard explains; this one remembers."""
    from nullres.runlog import load_runs

    runs = load_runs()
    if not runs:
        print("No runs recorded yet. Every run/robust/xsec/ablate appends one.")
        return 0

    if args.verdict:
        runs = [r for r in runs if r.verdict == args.verdict.upper()]

    _banner(f"RUN LOG — {len(runs)} record(s)")
    print(f"\n  {'date':<12}{'id':<10}{'config':<18}{'command':<9}"
          f"{'verdict':<14}git")
    print("  " + "-" * 74)
    for record in runs[-args.limit:]:
        verdict = record.verdict or "-"
        dirty = "+" if record.git_dirty else " "
        print(f"  {record.timestamp[:10]:<12}{record.short_id:<10}"
              f"{record.config_name[:17]:<18}{record.command:<9}"
              f"{verdict:<14}{record.git_sha}{dirty}")

    from nullres.runlog import count_trials, unrecorded_variants

    killed = sum(1 for r in runs if r.verdict == "KILLED")
    survived = sum(1 for r in runs if r.verdict == "SURVIVED")
    unsettled = sum(1 for r in runs if r.verdict == "INCONCLUSIVE")
    configs = len({r.config_hash for r in runs})
    print(f"\n  {len(runs)} runs over {configs} distinct configs — "
          f"{killed} KILLED, {survived} SURVIVED, {unsettled} INCONCLUSIVE")
    print(f"  {count_trials(runs)} distinct trials (re-running the same config "
          f"and command is one look, not two)")

    unknown = unrecorded_variants(runs)
    if unknown:
        print(f"  {unknown} record(s) predate variant recording and count as 1 "
              f"each — the real total is higher")

    print("\n  That trial count is your multiple-testing exposure. Every variant")
    print("  tried is a chance to find something by luck, which is what")
    print("  `deflated_sharpe` discounts. A long log is not a productivity")
    print("  metric — it is a reason to trust the best result less.")
    return 0


def cmd_fetch(cfg, args) -> int:
    from nullres.data import load_auxiliary

    bars = load_bars(cfg.data)
    print(f"\n{len(bars):,} bars cached in {cfg.data.cache_dir}/")
    print(bars.head(3).to_string())

    if cfg.data.funding or cfg.data.metrics:
        print("\nfutures auxiliary data")
        funding, metrics = load_auxiliary(cfg.data)
        for name, frame in (("funding", funding), ("metrics", metrics)):
            if frame is not None:
                print(f"\n{name}: {len(frame):,} rows, "
                      f"columns {list(frame.columns)}")
                print(frame.head(3).to_string())
    return 0


def cmd_run(cfg, args) -> int:
    _banner(f"{cfg.name}: {cfg.data.symbol} {cfg.data.interval} "
            f"{cfg.data.start}..{cfg.data.end}")
    print(f"label={cfg.label.kind}(h={cfg.label.horizon}) "
          f"model={cfg.model.kind} "
          f"cost={cfg.cost.fee_bps + cfg.cost.slippage_bps:.0f}bps/side "
          f"min_hold={cfg.sizing.min_hold}")
    _warn_if_already_killed(cfg)

    n_trials = args.trials or trials_so_far(
        cfg, extra=len(cfg.strategies) + 1, command="run")

    if args.ablate:
        from nullres.pipeline import ablate

        ctx = prepare(cfg)
        before = ctx.features.shape[1]
        ctx = ablate(ctx, args.ablate)
        print(f"\nABLATION: dropped {before - ctx.features.shape[1]} "
              f"{args.ablate} features ({before} -> {ctx.features.shape[1]}). "
              f"Rows, splits and benchmark are unchanged.")
        results = run_pipeline(cfg, ctx=ctx, n_trials=n_trials)
    else:
        results = run_pipeline(cfg, n_trials=n_trials)

    _banner("RESULTS (out-of-sample only)")
    print(format_table(results))

    bh = results.get("buy_hold", {})
    print(f"\nDeflated Sharpe (adjusted for {n_trials:,} variants tried across "
          f"the whole run ledger,")
    print(f"not just this run — see `nullres log`):")
    caveat = trials_caveat()
    if caveat:
        print(caveat)
    for name, m in results.items():
        verdict = "" if m["deflated_sharpe"] > 0 else "   <- indistinguishable from luck"
        print(f"  {name:<18}{m['deflated_sharpe']:>7.2f}{verdict}")

    print("\nRead this before celebrating:")
    print(f"  - buy & hold returned {bh.get('total_return', 0):.1%} "
          f"at sharpe {bh.get('sharpe', 0):.2f} for zero effort.")
    print("  - a strategy is only interesting if it beats that RISK-ADJUSTED,")
    print("    survives `nullres audit`, and has a t-stat above ~3.")
    print("  - these are in-sample-of-the-research-process results. The only")
    print("    honest test left is forward paper trading on bars you have never seen.")

    if args.save:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{cfg.name}.json"
        path.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {path}")

    _record(cfg, "run", variants=len(results), metrics={
        "n_trials_used": n_trials,
        **{name: {k: m[k] for k in ("total_return", "sharpe", "max_dd", "t_stat",
                                    "n_trades", "deflated_sharpe")}
           for name, m in results.items()},
    })
    return 0


def cmd_audit(cfg, args) -> int:
    _banner(f"AUDIT: {cfg.name}")
    ctx = prepare(cfg)
    checks = []

    print("\n1/5 point-in-time feature check (recomputing on truncated history)")
    # Rebuild through the same auxiliary data the pipeline used, so the
    # funding/OI join is covered by the truncation test rather than exempt.
    from nullres.data import load_auxiliary
    from nullres.features import build_features as _bf

    funding, metrics = load_auxiliary(cfg.data, verbose=False, bars=ctx.bars)
    checks.append(audit_mod.check_point_in_time(
        ctx.bars, builder=lambda d: _bf(d, funding=funding, metrics=metrics)
    ))

    print("2/5 single-feature AUC against the label")
    checks.append(audit_mod.check_label_leakage(ctx.features, ctx.label["y"]))

    print("3/5 shuffled-label control (retraining on permuted targets)")
    checks.append(audit_mod.check_shuffled_label(
        ctx.features, ctx.label["y"],
        ctx.label["t_end"].to_numpy(dtype=np.int64),
        cfg.split, cfg.model,
    ))

    print("4/5 null data (running the full pipeline on a random walk)")
    checks.append(audit_mod.check_null_data(run_pipeline, cfg))

    print("5/5 survivorship (does this universe contain assets that died?)")
    # Real bars carry a real end date, so the single-asset form of the question
    # is answerable. Synthetic bars run for however many were generated, which
    # says nothing about any instrument, so it stays n/a there.
    dates = ({"last_bar": ctx.bars.index[-1], "sample_end": cfg.data.end}
             if cfg.data.source == "binance" else {})
    checks.append(audit_mod.check_survivorship([cfg.data.symbol], delisted={},
                                               **dates))

    _banner("AUDIT RESULTS")
    for check in checks:
        print(check)

    # A check that could not apply is neither a pass nor a failure.
    failed = [c for c in checks if c.applicable and not c.passed]
    skipped = [c for c in checks if not c.applicable]
    if skipped:
        print(f"\n{len(skipped)} of {len(checks)} check(s) did not apply to this "
              f"config, so {len(checks) - len(skipped)} actually ran and the rest "
              f"have NOT been ruled out.")
    if failed:
        print(f"\n{len(failed)} CHECK(S) FAILED — results from this config are not "
              f"trustworthy until these are resolved.")
        return 1
    print("\nAll checks passed. This does not mean the strategy makes money; it")
    print("means the measurement apparatus is not lying to you.")
    return 0


def cmd_sweep(cfg, args) -> int:
    """Threshold sensitivity — read the SHAPE, not the peak.

    A real edge degrades smoothly as you move the entry threshold. A lone spike
    surrounded by losses is a fitting artefact, and picking it is how you turn a
    backtest into fiction.
    """
    import copy

    _banner(f"SWEEP: {cfg.name} — entry threshold vs min_hold")
    ctx = prepare(cfg)

    # The hold range spans the break-even table from `nullres budget`, because
    # that is the axis that actually decides the outcome at these cost levels.
    entries = [0.52, 0.54, 0.56, 0.58, 0.60]
    holds = [12, 48, 168, 336, 720]
    strategy = args.strategy

    print(f"\nstrategy={strategy}   cell = out-of-sample Sharpe\n")
    header = "entry \\ hold" + "".join(f"{h:>9}" for h in holds)
    print(header)
    print("-" * len(header))

    for entry in entries:
        cells = []
        for hold in holds:
            trial = copy.deepcopy(cfg)
            trial.strategies = [strategy]
            trial.sizing.long_entry = entry
            trial.sizing.short_entry = 1 - entry
            trial.sizing.min_hold = hold
            res = run_pipeline(trial, verbose=False, ctx=ctx)
            cells.append(f"{res[strategy]['sharpe']:>9.2f}")
        print(f"{entry:<12.2f}" + "".join(cells))

    cells = len(entries) * len(holds)
    print("\nIf the best cell is isolated, you found noise. If a contiguous")
    print("region is positive, you may have found something — but you have now")
    print(f"tried {cells} variants, so deflate accordingly.")
    _record(cfg, "sweep", notes=f"strategy={strategy}", variants=cells)
    return 0


def cmd_budget(cfg, args) -> int:
    """What accuracy would this instrument and cost structure actually require?

    Run this FIRST, before building features. It is arithmetic, it takes two
    seconds, and it will tell you whether the thing you are about to attempt is
    possible at all.
    """
    from nullres.costs import budget_table, format_duration

    _banner(f"COST BUDGET: {cfg.data.symbol} {cfg.data.interval}")
    bars = load_bars(cfg.data)
    logret = np.log(bars["close"]).diff()
    sigma = float(logret.std())

    # The modelled columns assume Gaussian returns; the measured ones do not.
    # The gap is not a constant that could be divided out — it changes sign with
    # the horizon (see `costs.expected_abs_move`), which is why the table shows
    # both rather than applying a calibration factor.
    measured_returns = logret.dropna().to_numpy()

    # Bars carry no duration on their own; the break-even table is only readable
    # once they are converted to wall-clock time at this config's bar size.
    hours_per_bar = 8_760 / cfg.data.bars_per_year

    print()
    print(budget_table(sigma, cfg.cost.fee_bps, cfg.cost.slippage_bps,
                       hours_per_bar=hours_per_bar, logret=measured_returns))

    print(f"\nYour config holds a position for at least {cfg.sizing.min_hold} bars "
          f"({format_duration(cfg.sizing.min_hold * hours_per_bar)}).")
    print("If that is far below the break-even row for the accuracy you actually")
    print("achieve, no amount of feature engineering will save the strategy — the")
    print("costs are structurally larger than the signal.")
    return 0


def cmd_robust(cfg, args) -> int:
    """Try three times to kill a strategy that looked good once."""
    from nullres.robustness import (
        cross_symbol, grid_for, hold_sharpe, parameter_neighbourhood,
        period_stability, pivot_grid, sign_flip_pairs, sign_flip_rate, verdict,
    )

    strategy = args.strategy
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    _banner(f"ROBUSTNESS: {strategy} on {cfg.data.symbol} {cfg.data.interval}")
    _warn_if_already_killed(cfg)
    ctx = prepare(cfg, verbose=False)
    params = cfg.params.get(strategy, {})
    print(f"\nreported parameters: {params or 'defaults'}")

    print("\n--- 1/3 parameter neighbourhood " + "-" * 44)
    print("Does the result survive a change to its own parameters?\n")
    grid = parameter_neighbourhood(cfg, strategy, ctx=ctx)
    grid_def, kind = grid_for(strategy)
    if kind == "sizing":
        print("(varying [sizing]; the model fit is identical across the grid)\n")
    print(pivot_grid(grid, list(grid_def)))
    flips = sign_flip_rate(grid, list(grid_def))
    pairs = sign_flip_pairs(grid, list(grid_def))
    print(f"\n  {(grid['sharpe'] > 0).mean():.0%} positive, "
          f"median {grid['sharpe'].median():.2f}, "
          f"sign flips across {flips:.0%} of {pairs} adjacent cell pairs")

    print("\n--- 2/3 sub-period stability " + "-" * 47)
    print("Is it profitable every year, or did one year carry it?\n")
    stability = period_stability(cfg, strategy, ctx=ctx)
    print(f"  {'year':<7}{'total':>9}{'sharpe':>8}{'|':>3}{'hold':>9}{'hold sh':>9}"
          f"{'|':>3}{'excess':>8}{'trades':>8}")
    for _, row in stability.iterrows():
        print(f"  {row['period']:<7}{row['total_return']:>9.1%}{row['sharpe']:>8.2f}"
              f"{'|':>3}{row['total_return_hold']:>9.1%}{row['sharpe_hold']:>9.2f}"
              f"{'|':>3}{row['excess_sharpe']:>8.2f}{row['n_trades']:>8,}")

    print("\n--- 3/3 cross-symbol transfer " + "-" * 46)
    print(f"Does it work on anything other than {cfg.data.symbol}?\n")
    if args.transfer_start:
        print(f"  window forced to {args.transfer_start} onward for every symbol, "
              f"including {cfg.data.symbol}, so this compares assets and not eras\n")
    transfer = cross_symbol(cfg, strategy, symbols, start=args.transfer_start)
    print(f"  {'symbol':<12}{'total':>10}{'sharpe':>9}{'vs hold':>9}{'trades':>8}  note")
    for _, row in transfer.iterrows():
        if pd.isna(row["sharpe"]):
            print(f"  {row['symbol']:<12}{'—':>10}{'—':>9}{'—':>9}{'—':>8}  {row['note']}")
        else:
            print(f"  {row['symbol']:<12}{row['total_return']:>10.1%}"
                  f"{row['sharpe']:>9.2f}{row['vs_hold']:>9.2f}{row['n_trades']:>8,}")

    # The bar the grid must clear is buy & hold's Sharpe over the same window
    # and computed the same way — NOT the mean of its per-year Sharpes, which
    # is a different statistic and a materially higher bar.
    bench = hold_sharpe(cfg, ctx)
    outcome, notes = verdict(grid, stability, transfer, benchmark_sharpe=bench,
                             flip_rate=flips, flip_pairs=pairs)
    _banner("VERDICT: " + outcome)
    for note in notes:
        print(f"  {note}")
    if outcome == "SURVIVED":
        print("\nThis strategy survived three attempts to falsify it. That earns it")
        print("a forward paper-trading run — not capital, and not confidence.")
    elif outcome == "KILLED":
        print("\nThis strategy is not worth further work in its current form.")
        print("That is a cheap answer to have obtained today rather than in six months.")
    else:
        print("\nThe battery ran and did not settle it. That is a statement about")
        print("the evidence, not about the strategy: at four or five observations")
        print("these gates cannot separate 'worse than holding' from 'too little")
        print("data to tell'. Judge it on the magnitudes above and on reasoning")
        print("the machine does not have — and do not read it as encouragement.")

    # The verdict is what makes the ledger useful: it is the field `find_similar`
    # matches on when warning about a re-run.
    #
    # The strategy is pinned into the logged config on purpose. `robust` takes it
    # from --strategy rather than the file, so without this every config would
    # inherit the verdict of whichever strategy happened to be tested last, and
    # killing donchian would warn you off ml_meta on the same data.
    import copy as _copy

    logged = _copy.deepcopy(cfg)
    logged.strategies = [strategy]
    # Only KILLED propagates to the near-miss warning (`find_similar` filters on
    # it). An INCONCLUSIVE run must not warn a future config off a dead end that
    # was never established.
    _record(logged, "robust", verdict=outcome,
            variants=len(grid) + len(transfer),
            notes=f"strategy={strategy}; " + " | ".join(notes),
            metrics={
                "strategy": strategy,
                "neighbourhood_positive": float((grid["sharpe"] > 0).mean()),
                "neighbourhood_median": float(grid["sharpe"].median()),
                "sign_flip_rate": float(flips) if flips == flips else None,
                "years_beating_hold": (float((stability["excess_sharpe"] > 0).mean())
                                       if not stability.empty else None),
                "symbols_beating_hold": (float((transfer["vs_hold"] > 0).mean())
                                         if "vs_hold" in transfer else None),
            })
    return 0


def cmd_ablate(cfg, args) -> int:
    """Does a feature group improve DISCRIMINATION, on matched samples?

    Sharpe cannot answer this. With ~80 trades an equity curve is decided by
    which handful of positions happened to land, and it will swing from -0.68
    to +0.43 on feature sets whose AUC differs by one percentage point. AUC
    uses every labelled bar, so it is the metric that can distinguish "the
    model knows more" from "the model got luckier".
    """
    from scipy import stats as sps

    from nullres.models.classifier import fit_predict_walk_forward
    from nullres.pipeline import ablate

    group = args.ablate or "derivatives"
    _banner(f"ABLATION: {group} on {cfg.name}")

    full = prepare(cfg, verbose=False)
    reduced = ablate(prepare(cfg, verbose=False), group)
    t_end = full.label["t_end"].to_numpy(dtype=np.int64)
    y = full.label["y"]

    print(f"\n{len(full.features):,} rows, identical folds and benchmark. "
          f"{full.features.shape[1]} features vs {reduced.features.shape[1]}.\n")

    scores = {}
    for name, X in (("with " + group, full.features),
                    ("without " + group, reduced.features)):
        _, reports = fit_predict_walk_forward(
            X, y, t_end, cfg.split, cfg.model, verbose=False
        )
        aucs = np.array([r["auc"] for r in reports])
        scores[name] = aucs
        print(f"  {name:<24}" + "  ".join(f"{a:.4f}" for a in aucs))
        print(f"  {'':<24}mean {aucs.mean():.4f}   std {aucs.std(ddof=1):.4f}   "
              f"folds above 0.5: {(aucs > 0.5).sum()}/{len(aucs)}\n")

    a, b = list(scores.values())
    diff = a - b
    print(f"  {'per-fold difference':<24}" + "  ".join(f"{d:+.4f}" for d in diff))
    print(f"  {'':<24}mean {diff.mean():+.4f}")

    t_stat, p_value = sps.ttest_rel(a, b)
    print(f"\n  paired t-test over {len(a)} folds: t = {t_stat:.2f}, p = {p_value:.3f}")
    if p_value > 0.05:
        print("  NOT significant. With this few folds that is the expected outcome")
        print("  even for a real effect — it is a reason to gather more evidence,")
        print("  not a reason to believe the difference is zero.")

    _record(cfg, "ablate", notes=f"group={group}", variants=2, metrics={
        "group": group,
        "auc_with": float(a.mean()),
        "auc_without": float(b.mean()),
        "auc_delta": float(diff.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    })
    return 0


def cmd_xsec(cfg, args) -> int:
    """Cross-sectional long/short on a panel of symbols."""
    from nullres.backtest.metrics import by_period, format_table, summarize
    from nullres.crosssec import (
        UNIVERSE_2021_12, backtest_panel, benchmarks,
        fit_predict_panel, load_panel, panel_positions,
    )

    if args.symbols_given:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        from nullres.data.universe import universe_as_of

        # Index products, not single assets: BTCDOM is BTC dominance, DEFI a basket.
        exclude = {"BTCDOMUSDT", "DEFIUSDT"}
        symbols = [s for s in universe_as_of(args.universe, cfg.data.interval)
                   if s not in exclude]
    else:
        symbols = UNIVERSE_2021_12

    _banner(f"CROSS-SECTIONAL: {len(symbols)} symbols, {cfg.data.interval}")
    if args.universe:
        print(f"universe enumerated mechanically from the archive as of "
              f"{args.universe} —")
        print("every symbol that traded that month, including those that later died")
    else:
        print("universe fixed as of 2021-12, NOT chosen from today's survivors")
    print()

    panel = load_panel(cfg, symbols, top_n=args.top_n)
    print(f"\n{len(panel.times):,} timestamps | {panel.features.shape[1]} ranked "
          f"features | {len(panel.features):,} panel rows")
    print(f"base rate {panel.y.mean():.4f} "
          f"(balanced by construction — half the universe beats the median)")
    for sym, last in panel.delisted.items():
        print(f"  DELISTED: {sym} last traded {last:%Y-%m-%d} — held to the end")

    survivorship = audit_mod.check_survivorship(
        panel.symbols, panel.delisted,
        point_in_time=symbols,
        hardcoded=not args.universe and not args.symbols_given,
    )
    print(f"\n{survivorship}")
    if not survivorship.passed:
        print("\n  Every number below is suspect until this is fixed. A universe")
        print("  filtered by survival cannot lose money on the things that died.")

    print("\nwalk-forward fit (folds split on TIME; all symbols move together)")
    proba, reports = fit_predict_panel(panel, cfg)
    aucs = np.array([r["auc"] for r in reports])
    print(f"\n  mean AUC {np.nanmean(aucs):.4f}   "
          f"folds above 0.5: {(aucs > 0.5).sum()}/{len(aucs)}")

    # Every book is judged on the SAME window the model was scored on.
    oos_times = pd.DatetimeIndex(
        proba.dropna().index.get_level_values("ts").unique()
    ).sort_values()
    print(f"  out-of-sample window: {oos_times[0]:%Y-%m-%d} .. "
          f"{oos_times[-1]:%Y-%m-%d} ({len(oos_times):,} bars)")

    # Books hold nothing before the first test fold opens, so every metric has
    # to be measured on the out-of-sample bars only. Averaging across the flat
    # pre-OOS block multiplies Sharpe by sqrt(oos fraction) — see
    # `engine.restrict`. `benchmarks` already zeroes positions there; this makes
    # the measurement agree with the intent.
    oos_mask = pd.Series(panel.times.isin(oos_times), index=panel.times)

    # This command produced the strongest result in the project, and it was the
    # one place the multiple-testing correction never reached: `summarize` was
    # called without `n_trials`, so `deflated_sharpe` returned the raw Sharpe and
    # every deflation figure in the docs had to be worked out by hand. The
    # correction is worth least on the results you were never going to question.
    if args.top_k:
        ks = (args.top_k,)
    else:
        # The point of a wide universe is that the same signal can be expressed
        # through diversification instead of concentration. Sweeping k from
        # narrow to wide is how you see whether that actually helps.
        width = args.top_n or len(panel.symbols)
        ks = (2, 3, 4) if width < 12 else (2, 5, 10, 15)

    # Books are built before the trial count so `extra` is the number this run
    # actually evaluates, matching what gets recorded in the ledger.
    books = benchmarks(panel, cfg.cost, oos_times, rebalance=args.rebalance)
    n_trials = args.trials or trials_so_far(
        cfg, extra=len(books) + len(ks), command="xsec")

    results = {}
    for name, result in books.items():
        results[name] = summarize(result, cfg.data.bars_per_year,
                                  n_trials=n_trials, mask=oos_mask)

    stability = None
    for k in ks:
        positions = panel_positions(proba, panel, top_k=k, rebalance=args.rebalance)
        result = backtest_panel(positions, panel, cfg.cost)
        results[f"longshort_k{k}"] = summarize(result, cfg.data.bars_per_year,
                                               n_trials=n_trials, mask=oos_mask)
        if stability is None:
            stability = by_period(result, cfg.data.bars_per_year, mask=oos_mask)
            stability_k = k

    _banner("RESULTS")
    print(format_table(results))

    # Gross notional is the number that decides whether margin is involved, and
    # a dollar-neutral book hides it: net is 0 and `expo` reads 100% whether you
    # are carrying 1x or 5x.
    book = max(results.items(), key=lambda kv: kv[1].get("gross_exposure", 0.0))
    if book[1].get("gross_exposure", 0.0) > 1.01:
        print(f"\nGross exposure: {book[0]} carries "
              f"{book[1]['gross_exposure']:.2f}x notional "
              f"(peak {book[1]['peak_exposure']:.2f}x) against zero net. "
              f"`sizing.max_leverage`\nis a single-asset clip and does not apply "
              f"here — a long/short book is 100% each\nway by construction. It "
              f"still needs margin, and the engine models none.")

    print(f"\nDeflated Sharpe (adjusted for {n_trials:,} variants across the run "
          f"ledger):")
    caveat = trials_caveat()
    if caveat:
        print(caveat)
    for name, m in results.items():
        tail = "" if m["deflated_sharpe"] > 0 else "   <- indistinguishable from luck"
        print(f"  {name:<18}{m['deflated_sharpe']:>7.2f}{tail}")

    if stability is not None and not stability.empty:
        print(f"\nper-year (k={stability_k}):")
        print(f"  {'year':<8}{'total':>10}{'sharpe':>9}{'trades':>8}")
        for _, row in stability.iterrows():
            print(f"  {row['period']:<8}{row['total_return']:>10.1%}"
                  f"{row['sharpe']:>9.2f}{row['n_trades']:>8,}")

    # Cost sensitivity is not optional context here — it is the result. A wide
    # crypto panel makes its money shorting thin alts, and the difference
    # between 8bps and 60bps all-in is the difference between 239x and nothing.
    print("\n--- cost sensitivity " + "-" * 55)
    print("The config charges "
          f"{cfg.cost.fee_bps + cfg.cost.slippage_bps:.0f}bps/side. That is about right "
          "for BTC perps and\nfiction for thin alts — which is exactly what this "
          "book shorts.\n")
    header = f"  {'slip bps':>9}" + "".join(f"{f'k={k}':>12}" for k in ks) + f"{'static':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    positions_by_k = {
        k: panel_positions(proba, panel, top_k=k, rebalance=args.rebalance)
        for k in ks
    }
    for slip in (cfg.cost.slippage_bps, 25.0, 50.0, 100.0):
        trial_cost = CostConfig(fee_bps=cfg.cost.fee_bps, slippage_bps=float(slip))
        cells = []
        for k in ks:
            m = summarize(backtest_panel(positions_by_k[k], panel, trial_cost),
                          cfg.data.bars_per_year, mask=oos_mask)
            cells.append(f"{m['sharpe']:>12.2f}")
        static = benchmarks(panel, trial_cost, oos_times,
                            rebalance=args.rebalance).get("static_vs_alts")
        tail = (f"{summarize(static, cfg.data.bars_per_year, mask=oos_mask)['sharpe']:>12.2f}"
                if static else "")
        print(f"  {slip:>9.0f}" + "".join(cells) + tail)
    print("\n  (Sharpe. If the model's column decays fast while static holds up,")
    print("   the durable part of the result needs no model.)")

    if args.verify:
        from nullres.panelaudit import format_report

        print(format_report(panel, cfg, proba, positions_by_k[ks[0]],
                            float(np.nanmean(aucs))))

    print("\nRead this against static_vs_alts, not equal_weight. A model that")
    print("only learned 'the lowest-volatility member outperforms' has learned")
    print("long-BTC/short-alts under another name — and that book needs no model")
    print("and three trades. Beating equal_weight is not evidence of anything.")

    _record(cfg, "xsec", notes=f"{len(panel.symbols)} symbols, top_n={args.top_n}",
            variants=len(results),
            metrics={
                "n_symbols": len(panel.symbols),
                "n_delisted": len(panel.delisted),
                "n_features": int(panel.features.shape[1]),
                "mean_auc": float(np.nanmean(aucs)),
                "n_trials_used": n_trials,
                **{name: {k: m[k] for k in ("total_return", "sharpe", "max_dd",
                                            "t_stat", "n_trades",
                                            "deflated_sharpe")}
                   for name, m in results.items()},
            })
    return 0


def cmd_features(cfg, args) -> int:
    from nullres.models.classifier import feature_importance

    _banner(f"FEATURE IMPORTANCE: {cfg.name}")
    ctx = prepare(cfg)
    print("\npermutation importance (AUC drop) on the final fold's test window\n")
    imp = feature_importance(
        ctx.features, ctx.label["y"],
        ctx.label["t_end"].to_numpy(dtype=np.int64),
        cfg.split, cfg.model,
    )

    from nullres.features import DERIVATIVE_DOC

    derived = set(DERIVATIVE_DOC)
    for name, value in imp.items():
        bar = "#" * max(0, int(value * 400))
        # Mark features that came from funding / open interest, so it is
        # immediately visible whether the new data earned its place.
        tag = "*" if name in derived else " "
        print(f" {tag}{name:<22}{value:>8.4f}  {bar}")

    if derived & set(imp.index):
        top = [n for n in imp.index[:10] if n in derived]
        print(f"\n  * = derivatives (funding / open interest). "
              f"{len(top)} of the top 10.")
    print("\nValues near zero mean the feature carried nothing out of sample.")
    print("Most of them will be near zero. That is the normal result.")
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
    parser.add_argument("--symbols", default="ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT",
                        help="robust: symbols for the cross-symbol transfer test")
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
    args = parser.parse_args(argv)

    # xsec defaults to the fixed 2021-12 universe unless symbols are given.
    args.symbols_given = any(a.startswith("--symbols") for a in (argv or sys.argv[1:]))

    cfg = load_config(args.config)
    for override in args.set:
        _apply_override(cfg, override)

    pd.set_option("display.width", 200)
    return COMMANDS[args.command](cfg, args)


def _apply_override(cfg, spec: str) -> None:
    """Apply `section.key=value`, parsing the value as TOML would."""
    if "=" not in spec:
        raise SystemExit(f"bad --set {spec!r}, expected section.key=value")
    path, raw = spec.split("=", 1)
    parts = path.split(".")

    # `params` is a plain dict of per-strategy kwargs, not a dataclass, so
    # `--set params.donchian.entry=48` used to die on `getattr(dict, ...)` with
    # a bare AttributeError. Rule parameters are exactly what you want to
    # override from the command line, so handle the dict branch explicitly.
    if parts[0] == "params":
        if len(parts) != 3:
            raise SystemExit(
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
        raise SystemExit(f"unknown config path {path!r}")

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
