"""End-to-end orchestration: bars -> features -> labels -> positions -> metrics.

One rule governs the ordering here. Features and labels are built on the FULL
frame first, and only then are rows dropped and positions renumbered. Building
them per-fold would be slower and no safer; building them after dropping rows
would silently shorten every rolling window across the gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nullres.backtest import backtest, summarize
from nullres.data import load_auxiliary, load_bars
from nullres.features import build_features
from nullres.labels import build_label
from nullres.strategies import Context, build as build_strategy
from nullres.validation import describe_folds, purged_walk_forward, remap_t_end


def prepare(cfg, verbose: bool = True) -> Context:
    """Load data, build features and labels, align them, and mark the OOS window."""
    bars = load_bars(cfg.data)
    funding, metrics = load_auxiliary(cfg.data, verbose=verbose, bars=bars)
    features = build_features(bars, funding=funding, metrics=metrics)
    label = build_label(bars, cfg.label)

    # Drop the warmup period where rolling windows have not filled, plus any bar
    # with no volatility estimate. Rows with a NaN target are KEPT: the model
    # predicts on them, it just does not train on them.
    keep = features.notna().all(axis=1) & label["sigma"].notna() & label["ret"].notna()
    keep_arr = keep.to_numpy()
    if keep_arr.sum() < 1_000:
        raise SystemExit(
            f"only {int(keep_arr.sum())} usable bars after alignment — "
            f"widen the date range or shorten the feature windows"
        )

    t_end = remap_t_end(label["t_end"].to_numpy(dtype=np.int64), keep_arr)
    bars, features, label = bars[keep], features[keep], label[keep].copy()
    label["t_end"] = t_end

    # The out-of-sample window is the union of every walk-forward test fold.
    # Every strategy, rules included, is judged only here.
    oos = np.zeros(len(bars), dtype=bool)
    for _, test in purged_walk_forward(t_end, cfg.split):
        oos[test] = True
    oos_mask = pd.Series(oos, index=bars.index)

    if verbose:
        labelled = label["y"].notna()
        print(f"\n{len(bars):,} usable bars | {features.shape[1]} features "
              f"| {int(labelled.sum()):,} labelled | base rate {label['y'].mean():.3f}")
        print(f"out-of-sample: {int(oos.sum()):,} bars "
              f"({bars.index[oos][0]:%Y-%m-%d} .. {bars.index[oos][-1]:%Y-%m-%d})")
        for row in describe_folds(t_end, cfg.split, bars.index):
            print(f"  fold {row['fold']}: train {row['train']:>7,} "
                  f"(purged {row['purged']:>4,})  test {row['test']:>6,}  "
                  f"[{row['test_from']}..{row['test_to']}]")

    if verbose:
        for warning in coherence_warnings(cfg, bars):
            print(f"  WARNING: {warning}")

    return Context(bars=bars, features=features, label=label, cfg=cfg,
                   oos_mask=oos_mask, verbose=verbose)


def ablate(ctx: Context, group: str) -> Context:
    """Drop a feature group AFTER row alignment, for a matched-sample ablation.

    Turning the data off in the config is not a controlled comparison: the
    derivative features carry their own warmup (`oi_z` needs 168 bars), so
    disabling them changes which rows survive the NaN mask, which changes the
    fold boundaries and the out-of-sample window. The two runs then differ in
    their samples as well as their features, and even buy & hold moves.

    This drops the columns from an already-prepared context, so the rows, the
    splits and the benchmark are byte-identical and the only variable is the
    feature set.
    """
    from nullres.features import DERIVATIVE_DOC

    groups = {"derivatives": set(DERIVATIVE_DOC)}
    if group not in groups:
        raise ValueError(f"unknown feature group {group!r}; choose from {sorted(groups)}")

    drop = [c for c in ctx.features.columns if c in groups[group]]
    if not drop:
        raise ValueError(f"no {group} features present to ablate")
    ctx.features = ctx.features.drop(columns=drop)
    ctx.diagnostics.clear()          # cached predictions are now stale
    return ctx


def coherence_warnings(cfg, bars: pd.DataFrame) -> list[str]:
    """Catch configurations that cannot work, before spending compute on them.

    These are not style notes. Each one describes a setup where the backtest
    will produce a number that means nothing.
    """
    from nullres.costs import breakeven_hold, required_accuracy

    out = []
    horizon = cfg.label.horizon
    hold = max(cfg.sizing.min_hold, 1)

    # A model trained to predict 24 bars ahead tells you nothing about whether
    # to keep a position for 500 — and vice versa.
    if hold > 4 * horizon or horizon > 4 * hold:
        out.append(
            f"label.horizon={horizon} but sizing.min_hold={hold}. The model "
            f"predicts a {horizon}-bar outcome while the strategy holds for "
            f"{hold} bars; these should be within a factor of ~2."
        )

    sigma = float(np.log(bars["close"]).diff().std())
    need = required_accuracy(sigma, hold, cfg.cost.fee_bps, cfg.cost.slippage_bps)
    if need > 0.56:
        be = breakeven_hold(sigma, 0.52, cfg.cost.fee_bps, cfg.cost.slippage_bps)
        target = "impossible at any accuracy" if need > 1.0 else f"{need:.1%} accuracy"
        out.append(
            f"at min_hold={hold} this strategy needs {target} just to break even. "
            f"A realistic 52% model would need to hold ~{be:,.0f} bars. "
            f"Run `nullres budget` for the full table."
        )
    return out


def trials_so_far(cfg, extra: int = 0) -> int:
    """Multiple-testing exposure: everything looked at before reporting this.

    Reads the run ledger rather than counting strategies in the current run.
    Counting only the current run is the mistake this replaces — it reported
    `n_trials=6` for a project that had explored well over a hundred parameter
    combinations, which made every deflated Sharpe too generous.
    """
    from nullres.runlog import count_trials, load_runs

    try:
        history = load_runs()
    except OSError:
        history = []
    return max(count_trials(history, prior=getattr(cfg, "prior_trials", 0)) + extra, 1)


def run_pipeline(cfg, verbose: bool = True, ctx: Context | None = None,
                 n_trials: int | None = None) -> dict[str, dict]:
    """Run every configured strategy and return {name: metrics}.

    A caller may pass a prepared `ctx` to avoid recomputing features when only
    sizing or cost parameters change (see `nullres sweep`). The context's cfg is
    repointed at `cfg` so those overrides actually take effect — strategies read
    their parameters from ctx.cfg, not from the closure.
    """
    if ctx is None:
        ctx = prepare(cfg, verbose=verbose)
    else:
        ctx.cfg = cfg
        ctx.verbose = verbose

    names = list(dict.fromkeys(["buy_hold", *cfg.strategies]))
    if n_trials is None:
        n_trials = trials_so_far(cfg, extra=len(names))
    results: dict[str, dict] = {}

    for name in names:
        if verbose:
            print(f"\n-> {name}")
        strategy = build_strategy(name, cfg.params.get(name))
        positions = strategy.positions(ctx)
        result = backtest(ctx.bars, positions, cfg.cost)
        metrics = summarize(result, cfg.data.bars_per_year, n_trials=n_trials)
        results[name] = metrics
        ctx.diagnostics.setdefault(name, {})["result"] = result

    return results
