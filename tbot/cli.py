"""Command line interface.

    tbot fetch    --config configs/btc_1h.toml     download and cache bars
    tbot budget   --config configs/btc_1h.toml     accuracy needed to beat costs
    tbot audit    --config configs/btc_1h.toml     leakage + null-data checks
    tbot run      --config configs/btc_1h.toml     backtest every strategy
    tbot sweep    --config configs/btc_1h.toml     threshold sensitivity surface
    tbot features --config configs/btc_1h.toml     out-of-sample importances
    tbot robust   --config configs/btc_4h.toml -s donchian
                                                   three falsification tests:
                                                   parameter neighbourhood,
                                                   sub-period stability vs hold,
                                                   cross-symbol transfer

Any option can be overridden without editing the file:

    tbot run -c configs/btc_1h.toml --set sizing.min_hold=168
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tbot import audit as audit_mod
from tbot.backtest.metrics import format_table
from tbot.config import load_config
from tbot.data import load_bars
from tbot.pipeline import prepare, run_pipeline


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def cmd_fetch(cfg, args) -> int:
    bars = load_bars(cfg.data)
    print(f"\n{len(bars):,} bars cached in {cfg.data.cache_dir}/")
    print(bars.head(3).to_string())
    return 0


def cmd_run(cfg, args) -> int:
    _banner(f"{cfg.name}: {cfg.data.symbol} {cfg.data.interval} "
            f"{cfg.data.start}..{cfg.data.end}")
    print(f"label={cfg.label.kind}(h={cfg.label.horizon}) "
          f"model={cfg.model.kind} "
          f"cost={cfg.cost.fee_bps + cfg.cost.slippage_bps:.0f}bps/side "
          f"min_hold={cfg.sizing.min_hold}")

    results = run_pipeline(cfg)

    _banner("RESULTS (out-of-sample only)")
    print(format_table(results))

    bh = results.get("buy_hold", {})
    print(f"\nDeflated Sharpe (adjusted for {len(results)} variants tried):")
    for name, m in results.items():
        verdict = "" if m["deflated_sharpe"] > 0 else "   <- indistinguishable from luck"
        print(f"  {name:<18}{m['deflated_sharpe']:>7.2f}{verdict}")

    print("\nRead this before celebrating:")
    print(f"  - buy & hold returned {bh.get('total_return', 0):.1%} "
          f"at sharpe {bh.get('sharpe', 0):.2f} for zero effort.")
    print("  - a strategy is only interesting if it beats that RISK-ADJUSTED,")
    print("    survives `tbot audit`, and has a t-stat above ~3.")
    print("  - these are in-sample-of-the-research-process results. The only")
    print("    honest test left is forward paper trading on bars you have never seen.")

    if args.save:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{cfg.name}.json"
        path.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {path}")
    return 0


def cmd_audit(cfg, args) -> int:
    _banner(f"AUDIT: {cfg.name}")
    ctx = prepare(cfg)
    checks = []

    print("\n1/4 point-in-time feature check (recomputing on truncated history)")
    checks.append(audit_mod.check_point_in_time(ctx.bars))

    print("2/4 single-feature AUC against the label")
    checks.append(audit_mod.check_label_leakage(ctx.features, ctx.label["y"]))

    print("3/4 shuffled-label control (retraining on permuted targets)")
    checks.append(audit_mod.check_shuffled_label(
        ctx.features, ctx.label["y"],
        ctx.label["t_end"].to_numpy(dtype=np.int64),
        cfg.split, cfg.model,
    ))

    print("4/4 null data (running the full pipeline on a random walk)")
    checks.append(audit_mod.check_null_data(run_pipeline, cfg))

    _banner("AUDIT RESULTS")
    for check in checks:
        print(check)

    failed = [c for c in checks if not c.passed]
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

    # The hold range spans the break-even table from `tbot budget`, because
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

    print("\nIf the best cell is isolated, you found noise. If a contiguous")
    print("region is positive, you may have found something — but you have now")
    print(f"tried {len(entries) * len(holds)} variants, so deflate accordingly.")
    return 0


def cmd_budget(cfg, args) -> int:
    """What accuracy would this instrument and cost structure actually require?

    Run this FIRST, before building features. It is arithmetic, it takes two
    seconds, and it will tell you whether the thing you are about to attempt is
    possible at all.
    """
    from tbot.costs import budget_table

    _banner(f"COST BUDGET: {cfg.data.symbol} {cfg.data.interval}")
    bars = load_bars(cfg.data)
    sigma = float(np.log(bars["close"]).diff().std())

    print()
    print(budget_table(sigma, cfg.cost.fee_bps, cfg.cost.slippage_bps))
    print(f"\nYour config holds a position for at least {cfg.sizing.min_hold} bars.")
    print("If that is far below the break-even row for the accuracy you actually")
    print("achieve, no amount of feature engineering will save the strategy — the")
    print("costs are structurally larger than the signal.")
    return 0


def cmd_robust(cfg, args) -> int:
    """Try three times to kill a strategy that looked good once."""
    from tbot.robustness import (
        DEFAULT_GRIDS, cross_symbol, parameter_neighbourhood,
        period_stability, pivot_grid, verdict,
    )

    strategy = args.strategy
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    _banner(f"ROBUSTNESS: {strategy} on {cfg.data.symbol} {cfg.data.interval}")
    ctx = prepare(cfg, verbose=False)
    params = cfg.params.get(strategy, {})
    print(f"\nreported parameters: {params or 'defaults'}")

    print("\n--- 1/3 parameter neighbourhood " + "-" * 44)
    print("Does the result survive a change to its own parameters?\n")
    grid = parameter_neighbourhood(cfg, strategy, ctx=ctx)
    keys = [k for k in DEFAULT_GRIDS[strategy]]
    print(pivot_grid(grid, keys))

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
    transfer = cross_symbol(cfg, strategy, symbols)
    print(f"  {'symbol':<12}{'total':>10}{'sharpe':>9}{'vs hold':>9}{'trades':>8}  note")
    for _, row in transfer.iterrows():
        if pd.isna(row["sharpe"]):
            print(f"  {row['symbol']:<12}{'—':>10}{'—':>9}{'—':>9}{'—':>8}  {row['note']}")
        else:
            print(f"  {row['symbol']:<12}{row['total_return']:>10.1%}"
                  f"{row['sharpe']:>9.2f}{row['vs_hold']:>9.2f}{row['n_trades']:>8,}")

    bench = float(stability["sharpe_hold"].mean()) if not stability.empty else None
    ok, notes = verdict(grid, stability, transfer, benchmark_sharpe=bench)
    _banner("VERDICT: " + ("SURVIVED" if ok else "KILLED"))
    for note in notes:
        print(f"  {note}")
    if ok:
        print("\nThis strategy survived three attempts to falsify it. That earns it")
        print("a forward paper-trading run — not capital, and not confidence.")
    else:
        print("\nThis strategy is not worth further work in its current form.")
        print("That is a cheap answer to have obtained today rather than in six months.")
    return 0


def cmd_features(cfg, args) -> int:
    from tbot.models.classifier import feature_importance

    _banner(f"FEATURE IMPORTANCE: {cfg.name}")
    ctx = prepare(cfg)
    print("\npermutation importance (AUC drop) on the final fold's test window\n")
    imp = feature_importance(
        ctx.features, ctx.label["y"],
        ctx.label["t_end"].to_numpy(dtype=np.int64),
        cfg.split, cfg.model,
    )
    for name, value in imp.items():
        bar = "#" * max(0, int(value * 400))
        print(f"  {name:<14}{value:>8.4f}  {bar}")
    print("\nValues near zero mean the feature carried nothing out of sample.")
    print("Most of them will be near zero. That is the normal result.")
    return 0


COMMANDS = {
    "fetch": cmd_fetch,
    "budget": cmd_budget,
    "run": cmd_run,
    "audit": cmd_audit,
    "sweep": cmd_sweep,
    "robust": cmd_robust,
    "features": cmd_features,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", "-c", default="configs/btc_1h.toml")
    parser.add_argument("--strategy", "-s", default="ml_meta",
                        help="sweep: which strategy to sweep")
    parser.add_argument("--save", action="store_true", help="run: write metrics JSON")
    parser.add_argument("--symbols", default="ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT",
                        help="robust: symbols for the cross-symbol transfer test")
    parser.add_argument("--set", action="append", default=[], metavar="a.b=v",
                        help="override a config value, e.g. --set sizing.min_hold=12")
    args = parser.parse_args(argv)

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
    target = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    key = parts[-1]
    if not hasattr(target, key):
        raise SystemExit(f"unknown config path {path!r}")

    current = getattr(target, key)
    if isinstance(current, bool):
        value = raw.lower() in ("1", "true", "yes")
    elif isinstance(current, int):
        value = int(raw)
    elif isinstance(current, float):
        value = float(raw)
    elif isinstance(current, list):
        value = [v.strip() for v in raw.split(",")]
    else:
        value = raw
    setattr(target, key, value)


if __name__ == "__main__":
    sys.exit(main())
