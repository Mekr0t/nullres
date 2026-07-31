"""Rendering. Every function takes a result object and returns a string.

Nothing here computes anything, reads a file, or prints. That is the point: the
text a command emits is now a pure function of what the command measured, so it
can be tested by comparing strings instead of by capturing stdout, and changing
how a number is displayed cannot change the number.

Commands that take minutes have their output split in two. `*_header` renders
what is known before the work starts — which config, which symbols, whether
this looks like something already killed — so it can be shown while the user
waits. `format_*` renders the whole report, header included, for a caller that
just wants the text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nullres.backtest.metrics import format_table
from nullres.results import (
    AblationResult, AuditResult, BudgetResult, FeatureImportanceResult,
    FetchResult, LedgerView, PanelVerification, RobustnessResult, RunResult,
    SweepResult, XsecResult,
)

WIDTH = 78


def _banner(text: str) -> list[str]:
    return ["", "=" * WIDTH, text, "=" * WIDTH]


def _warning_block(warning: str) -> list[str]:
    return ["", warning, ""] if warning else []


def ledger_line(record) -> list[str]:
    """The "logged as runs/ [id]" trailer, or nothing if the write failed.

    Public because `nullres run` prints it after the --save line rather
    than as part of the body, so the CLI has to place it itself.
    """
    if record is None:
        return []
    dirty = " +uncommitted" if record.git_dirty else ""
    return ["", f"logged as runs/ [{record.short_id}]  "
                f"config {record.config_hash}  git {record.git_sha}{dirty}"]


def _deflated_block(metrics: dict[str, dict], n_trials: int, caveat: str,
                    lead: str) -> list[str]:
    """The multiple-testing correction, and what it does not cover.

    At or below zero means: you found nothing, you just looked a lot of times.
    """
    lines = ["", lead]
    if caveat:
        lines.append(caveat)
    for name, m in metrics.items():
        tail = "" if m["deflated_sharpe"] > 0 else "   <- indistinguishable from luck"
        lines.append(f"  {name:<18}{m['deflated_sharpe']:>7.2f}{tail}")
    return lines


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_header(cfg, killed_warning: str = "") -> str:
    """Config line and near-miss warning, printable before the backtest starts."""
    lines = _banner(f"{cfg.name}: {cfg.data.symbol} {cfg.data.interval} "
                    f"{cfg.data.start}..{cfg.data.end}")
    lines.append(f"label={cfg.label.kind}(h={cfg.label.horizon}) "
                 f"model={cfg.model.kind} "
                 f"cost={cfg.cost.fee_bps + cfg.cost.slippage_bps:.0f}bps/side "
                 f"min_hold={cfg.sizing.min_hold}")
    lines += _warning_block(killed_warning)
    return "\n".join(lines)


def run_body(result: RunResult) -> str:
    """Results table, deflated Sharpe, and what the numbers do not prove."""
    lines = _banner("RESULTS (out-of-sample only)")
    lines.append(format_table(result.metrics))
    lines += _deflated_block(
        result.metrics, result.n_trials, result.trials_caveat,
        f"Deflated Sharpe (adjusted for {result.n_trials:,} variants tried "
        f"across the whole run ledger,\nnot just this run — see `nullres log`):")

    bh = result.benchmark
    lines += [
        "",
        "Read this before celebrating:",
        f"  - buy & hold returned {bh.get('total_return', 0):.1%} "
        f"at sharpe {bh.get('sharpe', 0):.2f} for zero effort.",
        "  - a strategy is only interesting if it beats that RISK-ADJUSTED,",
        "    survives `nullres audit`, and has a t-stat above ~3.",
        "  - these are in-sample-of-the-research-process results. The only",
        "    honest test left is forward paper trading on bars you have never seen.",
    ]
    return "\n".join(lines)


def format_run(result: RunResult) -> str:
    """The complete `nullres run` report."""
    return (run_header(result.cfg, result.killed_warning) + "\n"
            + run_body(result) + "\n".join(ledger_line(result.record)))


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit_header(cfg) -> str:
    """Which config is being audited."""
    return "\n".join(_banner(f"AUDIT: {cfg.name}"))


def audit_body(result: AuditResult) -> str:
    """Each check's verdict, plus what a pass does and does not mean."""
    lines = _banner("AUDIT RESULTS")
    lines += [str(check) for check in result.checks]

    if result.skipped:
        lines += ["",
                  f"{len(result.skipped)} of {len(result.checks)} check(s) did "
                  f"not apply to this config, so "
                  f"{len(result.checks) - len(result.skipped)} actually ran and "
                  f"the rest have NOT been ruled out."]
    if result.failed:
        lines += ["",
                  f"{len(result.failed)} CHECK(S) FAILED — results from this "
                  f"config are not trustworthy until these are resolved."]
    else:
        lines += ["",
                  "All checks passed. This does not mean the strategy makes "
                  "money; it",
                  "means the measurement apparatus is not lying to you."]
    return "\n".join(lines)


def format_audit(result: AuditResult) -> str:
    """The complete `nullres audit` report."""
    return audit_header(result.cfg) + "\n" + audit_body(result)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

def budget_header(cfg) -> str:
    """Which instrument and timeframe the arithmetic is for."""
    return "\n".join(_banner(f"COST BUDGET: {cfg.data.symbol} {cfg.data.interval}"))


def budget_body(result: BudgetResult) -> str:
    """The two break-even tables, measured against modelled."""
    from nullres.costs import budget_table, format_duration

    cfg = result.cfg
    lines = ["", budget_table(result.sigma, cfg.cost.fee_bps,
                              cfg.cost.slippage_bps,
                              hours_per_bar=result.hours_per_bar,
                              logret=result.logret)]
    lines += [
        "",
        f"Your config holds a position for at least {result.min_hold} bars "
        f"({format_duration(result.min_hold * result.hours_per_bar)}).",
        "If that is far below the break-even row for the accuracy you actually",
        "achieve, no amount of feature engineering will save the strategy — the",
        "costs are structurally larger than the signal.",
    ]
    return "\n".join(lines)


def format_budget(result: BudgetResult) -> str:
    """The complete `nullres budget` report."""
    return budget_header(result.cfg) + "\n" + budget_body(result)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep_header(cfg) -> str:
    """Which config is being swept."""
    return "\n".join(_banner(f"SWEEP: {cfg.name} — entry threshold vs min_hold"))


def sweep_body(result: SweepResult) -> str:
    """The entry x hold Sharpe surface. Read the shape, not the peak."""
    grid = result.grid
    header = "entry \\ hold" + "".join(f"{h:>9}" for h in result.holds)
    lines = ["", f"strategy={result.strategy}   cell = out-of-sample Sharpe", "",
             header, "-" * len(header)]
    for entry in result.entries:
        cells = "".join(f"{grid.loc[entry, h]:>9.2f}" for h in result.holds)
        lines.append(f"{entry:<12.2f}" + cells)
    lines += [
        "",
        "If the best cell is isolated, you found noise. If a contiguous",
        "region is positive, you may have found something — but you have now",
        f"tried {result.n_cells} variants, so deflate accordingly.",
    ]
    lines += ledger_line(result.record)
    return "\n".join(lines)


def format_sweep(result: SweepResult) -> str:
    """The complete `nullres sweep` report."""
    return sweep_header(result.cfg) + "\n" + sweep_body(result)


# ---------------------------------------------------------------------------
# robust
# ---------------------------------------------------------------------------

def robust_header(cfg, strategy: str, killed_warning: str = "") -> str:
    """Strategy, symbol, and any near-miss warning."""
    lines = _banner(f"ROBUSTNESS: {strategy} on {cfg.data.symbol} "
                    f"{cfg.data.interval}")
    lines += _warning_block(killed_warning)
    return "\n".join(lines)


def robust_body(result: RobustnessResult) -> str:
    """All three falsification tests, the verdict, and its reasoning."""
    from nullres.robustness import pivot_grid

    lines = ["", f"reported parameters: {result.params or 'defaults'}"]

    lines += ["", "--- 1/3 parameter neighbourhood " + "-" * 44,
              "Does the result survive a change to its own parameters?", ""]
    if result.grid_kind == "sizing":
        lines += ["(varying [sizing]; the model fit is identical across the grid)",
                  ""]
    lines.append(pivot_grid(result.grid, result.grid_keys))
    lines.append(f"\n  {(result.grid['sharpe'] > 0).mean():.0%} positive, "
                 f"median {result.grid['sharpe'].median():.2f}, "
                 f"sign flips across {result.flip_rate:.0%} of "
                 f"{result.flip_pairs} adjacent cell pairs")

    lines += ["", "--- 2/3 sub-period stability " + "-" * 47,
              "Is it profitable every year, or did one year carry it?", ""]
    lines.append(f"  {'year':<7}{'total':>9}{'sharpe':>8}{'|':>3}{'hold':>9}"
                 f"{'hold sh':>9}{'|':>3}{'excess':>8}{'trades':>8}")
    for _, row in result.stability.iterrows():
        lines.append(
            f"  {row['period']:<7}{row['total_return']:>9.1%}{row['sharpe']:>8.2f}"
            f"{'|':>3}{row['total_return_hold']:>9.1%}{row['sharpe_hold']:>9.2f}"
            f"{'|':>3}{row['excess_sharpe']:>8.2f}{row['n_trades']:>8,}")

    lines += ["", "--- 3/3 cross-symbol transfer " + "-" * 46,
              f"Does it work on anything other than {result.cfg.data.symbol}?", ""]
    if result.transfer_start:
        lines += [f"  window forced to {result.transfer_start} onward for every "
                  f"symbol, including {result.cfg.data.symbol}, so this compares "
                  f"assets and not eras", ""]
    lines.append(f"  {'symbol':<12}{'total':>10}{'sharpe':>9}{'vs hold':>9}"
                 f"{'trades':>8}  note")
    for _, row in result.transfer.iterrows():
        if pd.isna(row["sharpe"]):
            lines.append(f"  {row['symbol']:<12}{'—':>10}{'—':>9}{'—':>9}"
                         f"{'—':>8}  {row['note']}")
        else:
            lines.append(f"  {row['symbol']:<12}{row['total_return']:>10.1%}"
                         f"{row['sharpe']:>9.2f}{row['vs_hold']:>9.2f}"
                         f"{row['n_trades']:>8,}")

    lines += _banner("VERDICT: " + result.verdict)
    lines += [f"  {note}" for note in result.notes]

    if result.verdict == "SURVIVED":
        lines += ["",
                  "This strategy survived three attempts to falsify it. That earns it",
                  "a forward paper-trading run — not capital, and not confidence."]
    elif result.verdict == "KILLED":
        lines += ["",
                  "This strategy is not worth further work in its current form.",
                  "That is a cheap answer to have obtained today rather than in "
                  "six months."]
    else:
        lines += ["",
                  "The battery ran and did not settle it. That is a statement about",
                  "the evidence, not about the strategy: at four or five observations",
                  "these gates cannot separate 'worse than holding' from 'too little",
                  "data to tell'. Judge it on the magnitudes above and on reasoning",
                  "the machine does not have — and do not read it as encouragement."]
    lines += ledger_line(result.record)
    return "\n".join(lines)


def format_robust(result: RobustnessResult) -> str:
    """The complete `nullres robust` report."""
    return (robust_header(result.cfg, result.strategy, result.killed_warning)
            + "\n" + robust_body(result))


# ---------------------------------------------------------------------------
# ablate
# ---------------------------------------------------------------------------

def ablate_header(cfg, group: str) -> str:
    """Which feature group is being ablated, on which config."""
    return "\n".join(_banner(f"ABLATION: {group} on {cfg.name}"))


def ablate_body(result: AblationResult) -> str:
    """Per-fold AUC with and without the group, and the paired t-test."""
    lines = ["",
             f"{result.n_rows:,} rows, identical folds and benchmark. "
             f"{result.features_with} features vs {result.features_without}.",
             ""]
    for name, aucs in ((f"with {result.group}", result.auc_with),
                       (f"without {result.group}", result.auc_without)):
        lines.append(f"  {name:<24}" + "  ".join(f"{a:.4f}" for a in aucs))
        lines.append(f"  {'':<24}mean {aucs.mean():.4f}   "
                     f"std {aucs.std(ddof=1):.4f}   "
                     f"folds above 0.5: {(aucs > 0.5).sum()}/{len(aucs)}")
        lines.append("")

    diff = result.delta
    lines.append(f"  {'per-fold difference':<24}"
                 + "  ".join(f"{d:+.4f}" for d in diff))
    lines.append(f"  {'':<24}mean {diff.mean():+.4f}")
    lines.append(f"\n  paired t-test over {len(diff)} folds: "
                 f"t = {result.t_stat:.2f}, p = {result.p_value:.3f}")
    if not result.significant:
        lines += [
            "  NOT significant. With this few folds that is the expected outcome",
            "  even for a real effect — it is a reason to gather more evidence,",
            "  not a reason to believe the difference is zero.",
        ]
    lines += ledger_line(result.record)
    return "\n".join(lines)


def format_ablate(result: AblationResult) -> str:
    """The complete `nullres ablate` report."""
    return ablate_header(result.cfg, result.group) + "\n" + ablate_body(result)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def features_header(cfg) -> str:
    """Which config the importances come from."""
    return "\n".join(_banner(f"FEATURE IMPORTANCE: {cfg.name}"))


def features_body(result: FeatureImportanceResult) -> str:
    """Permutation importance per feature, with derivatives marked."""
    lines = ["",
             "permutation importance (AUC drop) on the final fold's test window",
             ""]
    for name, value in result.importances.items():
        bar = "#" * max(0, int(value * 400))
        # Mark features that came from funding / open interest, so it is
        # immediately visible whether the new data earned its place.
        tag = "*" if name in result.derivative_names else " "
        lines.append(f" {tag}{name:<22}{value:>8.4f}  {bar}")

    if result.derivative_names & set(result.importances.index):
        lines.append(f"\n  * = derivatives (funding / open interest). "
                     f"{len(result.derivatives_in_top_10)} of the top 10.")
    lines += ["",
              "Values near zero mean the feature carried nothing out of sample.",
              "Most of them will be near zero. That is the normal result."]
    return "\n".join(lines)


def format_features(result: FeatureImportanceResult) -> str:
    """The complete `nullres features` report."""
    return features_header(result.cfg) + "\n" + features_body(result)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def format_fetch(result: FetchResult) -> str:
    """What is now in the cache."""
    lines = ["", f"{len(result.bars):,} bars cached in {result.cfg.data.cache_dir}/",
             result.bars.head(3).to_string()]
    if result.cfg.data.funding or result.cfg.data.metrics:
        lines += ["", "futures auxiliary data"]
        for name, frame in (("funding", result.funding), ("metrics", result.metrics)):
            if frame is not None:
                lines += ["", f"{name}: {len(frame):,} rows, "
                              f"columns {list(frame.columns)}",
                          frame.head(3).to_string()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------

def format_ledger(view: LedgerView) -> str:
    """The run ledger, and why a long one is a reason for less confidence."""
    if not view.runs and not view.n_total:
        return "No runs recorded yet. Every run/robust/xsec/ablate appends one."

    lines = _banner(f"RUN LOG — {len(view.runs)} record(s)")
    lines.append(f"\n  {'date':<12}{'id':<10}{'config':<18}{'command':<9}"
                 f"{'verdict':<14}git")
    lines.append("  " + "-" * 74)
    for record in view.runs[-view.limit:]:
        verdict = record.verdict or "-"
        dirty = "+" if record.git_dirty else " "
        lines.append(f"  {record.timestamp[:10]:<12}{record.short_id:<10}"
                     f"{record.config_name[:17]:<18}{record.command:<9}"
                     f"{verdict:<14}{record.git_sha}{dirty}")

    lines.append(f"\n  {view.n_total} runs over {view.n_configs} distinct "
                 f"configs — {view.killed} KILLED, {view.survived} SURVIVED, "
                 f"{view.inconclusive} INCONCLUSIVE")
    lines.append(f"  {view.n_trials} distinct trials (re-running the same config "
                 f"and command is one look, not two)")
    if view.unrecorded_variants:
        lines.append(f"  {view.unrecorded_variants} record(s) predate variant "
                     f"recording and count as 1 each — the real total is higher")

    lines += ["",
              "  That trial count is your multiple-testing exposure. Every variant",
              "  tried is a chance to find something by luck, which is what",
              "  `deflated_sharpe` discounts. A long log is not a productivity",
              "  metric — it is a reason to trust the best result less."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# xsec
# ---------------------------------------------------------------------------

def xsec_header(cfg, n_symbols: int, universe_month: str | None = None) -> str:
    """Universe size and how it was chosen — enumerated or hardcoded."""
    lines = _banner(f"CROSS-SECTIONAL: {n_symbols} symbols, {cfg.data.interval}")
    if universe_month:
        lines += [f"universe enumerated mechanically from the archive as of "
                  f"{universe_month} —",
                  "every symbol that traded that month, including those that "
                  "later died"]
    else:
        lines.append("universe fixed as of 2021-12, NOT chosen from today's "
                     "survivors")
    lines.append("")
    return "\n".join(lines)


def xsec_body(result: XsecResult) -> str:
    """Panel summary, survivorship, books, cost sensitivity, verification."""
    cfg, panel = result.cfg, result.panel
    aucs = np.array([r["auc"] for r in result.fold_reports])

    lines = ["",
             f"{len(panel.times):,} timestamps | {panel.features.shape[1]} ranked "
             f"features | {len(panel.features):,} panel rows",
             f"base rate {panel.y.mean():.4f} "
             f"(balanced by construction — half the universe beats the median)"]
    for sym, last in panel.delisted.items():
        lines.append(f"  DELISTED: {sym} last traded {last:%Y-%m-%d} — "
                     f"held to the end")

    lines.append(f"\n{result.survivorship}")
    if not result.survivorship.passed:
        lines += ["",
                  "  Every number below is suspect until this is fixed. A universe",
                  "  filtered by survival cannot lose money on the things that died."]

    lines.append(f"\n  mean AUC {np.nanmean(aucs):.4f}   "
                 f"folds above 0.5: {(aucs > 0.5).sum()}/{len(aucs)}")
    lines.append(f"  out-of-sample window: {result.oos_times[0]:%Y-%m-%d} .. "
                 f"{result.oos_times[-1]:%Y-%m-%d} "
                 f"({len(result.oos_times):,} bars)")

    lines += _banner("RESULTS")
    lines.append(format_table(result.books))

    # A dollar-neutral book hides its leverage: net is 0 and `expo` reads 100%
    # whether it carries 1x or 5x.
    name, book = result.heaviest_book
    if book.get("gross_exposure", 0.0) > 1.01:
        lines.append(f"\nGross exposure: {name} carries "
                     f"{book['gross_exposure']:.2f}x notional "
                     f"(peak {book['peak_exposure']:.2f}x) against zero net. "
                     f"`sizing.max_leverage`\nis a single-asset clip and does not "
                     f"apply here — a long/short book is 100% each\nway by "
                     f"construction. It still needs margin, and the engine "
                     f"models none.")

    lines += _deflated_block(
        result.books, result.n_trials, result.trials_caveat,
        f"Deflated Sharpe (adjusted for {result.n_trials:,} variants across the "
        f"run ledger):")

    if result.stability is not None and not result.stability.empty:
        lines.append(f"\nper-year (k={result.stability_k}):")
        lines.append(f"  {'year':<8}{'total':>10}{'sharpe':>9}{'trades':>8}")
        for _, row in result.stability.iterrows():
            lines.append(f"  {row['period']:<8}{row['total_return']:>10.1%}"
                         f"{row['sharpe']:>9.2f}{row['n_trades']:>8,}")

    lines += ["", "--- cost sensitivity " + "-" * 55,
              f"The config charges "
              f"{cfg.cost.fee_bps + cfg.cost.slippage_bps:.0f}bps/side. That is "
              f"about right for BTC perps and\nfiction for thin alts — which is "
              f"exactly what this book shorts.", ""]
    lines.append(_cost_table(result))
    lines += ["",
              "  (Sharpe. If the model's column decays fast while static holds up,",
              "   the durable part of the result needs no model.)"]

    if result.verification is not None:
        lines.append(format_verification(result.verification, panel))

    lines += ["",
              "Read this against static_vs_alts, not equal_weight. A model that",
              "only learned 'the lowest-volatility member outperforms' has learned",
              "long-BTC/short-alts under another name — and that book needs no model",
              "and three trades. Beating equal_weight is not evidence of anything."]
    lines += ledger_line(result.record)
    return "\n".join(lines)


def _cost_table(result: XsecResult) -> str:
    table = result.cost_sensitivity
    ks = result.ks
    header = (f"  {'slip bps':>9}" + "".join(f"{f'k={k}':>12}" for k in ks)
              + (f"{'static':>12}" if "static" in table.columns else ""))
    lines = [header, "  " + "-" * (len(header) - 2)]
    for _, row in table.iterrows():
        cells = "".join(f"{row[f'k={k}']:>12.2f}" for k in ks)
        tail = f"{row['static']:>12.2f}" if "static" in table.columns else ""
        lines.append(f"  {row['slippage_bps']:>9.0f}" + cells + tail)
    return "\n".join(lines)


def format_verification(v: PanelVerification, panel) -> str:
    """The controls that decide whether cross-sectional skill is real."""
    lines = ["", "--- verification " + "-" * 59, ""]

    verdict = "clean" if v.shuffle_is_clean else "SUSPECT"
    lines.append(f"  shuffled labels        AUC {v.shuffled_auc:.4f}   "
                 f"vs {v.mean_auc:.4f} real — {verdict}")

    if v.survivors_auc is None:
        lines.append("  survivors only         n/a — no delisted symbols to remove")
    else:
        drop = v.mean_auc - v.survivors_auc
        reading = "death detection" if v.detects_death else "not death detection"
        lines.append(f"  survivors only         AUC {v.survivors_auc:.4f}   "
                     f"({drop:+.4f}) — {reading}")

    # Symbols scored on a handful of bars carry accuracies that swing wildly by
    # chance. Quoting a spread across them measures the screen, not the model.
    stats = v.per_symbol
    thick = stats[(stats["n"] >= v.min_obs) & stats["auc"].notna()]
    if len(thick) >= 2:
        thin = len(stats) - len(thick)
        note = f", {thin} thinner excluded" if thin else ""
        spread = float(thick["auc"].iloc[0] - thick["auc"].iloc[-1])
        lines.append(f"  per-symbol skill       AUC spread {spread:.3f} over "
                     f"{len(thick)} symbols with >={v.min_obs} scored bars{note}")
        for label, name in (("best ", thick.index[0]), ("worst", thick.index[-1])):
            row = thick.loc[name]
            lines.append(f"                         {label} {name} "
                         f"AUC {row['auc']:.3f}  (accuracy {row['accuracy']:.3f} "
                         f"vs base rate {row['base_rate']:.3f}, "
                         f"lift {row['lift']:+.3f}, n={int(row['n']):,})")
        above = int((thick["auc"] > 0.5).sum())
        lines.append(f"                         {above} of {len(thick)} symbols "
                     f"score above 0.5; median {thick['auc'].median():.3f}")
        lines.append(f"                         raw accuracy spread is "
                     f"{float(thick['accuracy'].max() - thick['accuracy'].min()):.3f}"
                     f", but that is mostly each symbol's own base rate — AUC is "
                     f"the base-rate-free read")
    elif len(stats) >= 2:
        lines.append(f"  per-symbol skill       n/a — no symbol reached "
                     f"{v.min_obs} scored bars")

    lines.append(f"  delisted contribution  {v.delisted_share:.1%} of ABSOLUTE "
                 f"P&L (not netted) from {len(panel.delisted)} symbol(s) that "
                 f"stopped trading")

    top = ", ".join(f"{s}" for s in v.contribution.index[:4])
    bottom = ", ".join(f"{s}" for s in v.contribution.index[-4:])
    lines.append(f"  contributors           + {top}")
    lines.append(f"                         - {bottom}")

    curve = v.tail_curve
    short_bars = curve.attrs["short_name_bars"]
    weight = curve.attrs["worst_short_weight"]
    lines += [
        "",
        f"  tail exposure — book held {short_bars:,} short-name-bars across "
        f"{curve.attrs['observed_bars']:,} observed,",
        f"  at a largest short weight of {weight:.2f} per name. Worst bar "
        f"actually suffered: {v.tail_census['worst_bar_return']:.1%}",
    ]

    if v.nominal_weight and v.concentration:
        lines += _concentration_block(v.concentration)

    lines += ["",
              f"    {'move':>7}{'occurred':>10}{'1 in':>12}{'expected':>10}"
              f"{'actual':>8}{'costs':>9}"]
    for _, row in curve.iterrows():
        bound = "" if row["estimated"] else "<"
        one_in = (f"{row['one_in']:,.0f}" if row["estimated"]
                  else f">{row['one_in']:,.0f}")
        flag = "  <- RUIN" if row["ruinous"] else ""
        lines.append(f"    {row['move']:>6.0%}{int(row['occurrences']):>10,}"
                     f"{one_in:>12}"
                     f"{bound + format(row['expected_hits'], '.2f'):>10}"
                     f"{int(row['actual_hits']):>8}{row['cost_of_one']:>8.0%}{flag}")
    if not curve["estimated"].all():
        lines += ["",
                  "    '<' marks a move size never observed here. Its rate is "
                  "not zero — it is unknown,",
                  f"    bounded above by the rule of three (3 events in "
                  f"{curve.attrs['observed_bars']:,} observations)."]

    lines.append("")
    ruin = curve[curve["ruinous"]]
    if len(ruin):
        smallest = ruin.iloc[0]
        lines.append(f"    A single +{smallest['move']:.0%} move against the "
                     f"largest short would cost {smallest['cost_of_one']:.0%} of "
                     f"capital — more than all of it.")
        if smallest["estimated"]:
            lines.append(f"    Chance predicted {smallest['expected_hits']:.2f} "
                         f"such hits and {int(smallest['actual_hits'])} occurred, "
                         f"so surviving is what the exposure predicts,\n    not "
                         f"evidence the risk was absent.")
        else:
            lines.append(f"    No move that large occurred here, so its rate is "
                         f"not measured at all — only bounded\n    ABOVE, at most "
                         f"{smallest['expected_hits']:.2f} expected hits. This "
                         f"sample cannot show you this risk; it can\n    only fail "
                         f"to. UNFI did +274% in a single 4h bar in 2021.")
    lines += ["    The tail is UNTESTED, not absent. The engine models no margin, "
              "so a ruinous",
              "    bar would show as a large negative return rather than a "
              "liquidation."]
    return "\n".join(lines)


def _concentration_block(conc: dict) -> list[str]:
    if conc["max_short"] <= conc["nominal"] * 1.000001:
        return ["", f"    Concentration: never exceeded the nominal "
                    f"{conc['nominal']:.2f} per name."]
    return [
        "",
        f"    CONCENTRATION: nominal weight is {conc['nominal']:.2f} per name, "
        f"but a delisted leg leaves the",
        f"    survivor rescaled to keep the book dollar-neutral — peaking at "
        f"{conc['max_short']:.2f} short ({conc['max_long']:.2f} long).",
        f"    Concentrated on {conc['concentrated_bars']:,} of "
        f"{conc['bars_held']:,} bars held ({conc['share']:.1%}), "
        f"{conc['share_at_peak']:.1%} of them at the peak,",
        f"    longest unbroken stretch {conc['longest_run']:,} bars.",
        f"    Gross exposure never changes; the move that ruins the book halves, "
        f"from +{1 / conc['nominal'] * 100:.0f}% to "
        f"+{1 / conc['max_short'] * 100:.0f}%.",
    ]


def format_xsec(result: XsecResult) -> str:
    """The complete `nullres xsec` report."""
    return (xsec_header(result.cfg, len(result.requested_symbols),
                        result.universe_month) + "\n" + xsec_body(result))


__all__ = [
    "ledger_line",
    "run_header", "run_body", "format_run",
    "audit_header", "audit_body", "format_audit",
    "budget_header", "budget_body", "format_budget",
    "sweep_header", "sweep_body", "format_sweep",
    "robust_header", "robust_body", "format_robust",
    "ablate_header", "ablate_body", "format_ablate",
    "features_header", "features_body", "format_features",
    "xsec_header", "xsec_body", "format_xsec", "format_verification",
    "format_fetch", "format_ledger",
]
