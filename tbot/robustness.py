"""Falsification tests for a strategy that looked good once.

A single backtest number is a hypothesis, not a finding. Before it earns any
more of your time it has to survive three attempts to kill it:

  NEIGHBOURHOOD  Do nearby parameter values also work? A real effect degrades
                 smoothly. If only one cell in a grid is positive, you did not
                 find an edge — you found the cell that happened to fit.

  STABILITY      Does it work in every year, or did one spectacular period
                 carry the average? A Sharpe of 0.5 built from 2021 and nothing
                 else is a bet that 2021 recurs.

  TRANSFER       Does it work on other symbols? A rule that describes market
                 structure should generalise. A rule that only works on the
                 asset you developed it on describes that asset's history.

Passing all three does not make a strategy real — the only test that does is
forward paper trading. Failing any one of them makes it dead, cheaply, today.
"""

from __future__ import annotations

import copy
from itertools import product

import numpy as np
import pandas as pd

from tbot.backtest import backtest, by_period, summarize
from tbot.pipeline import prepare, run_pipeline
from tbot.strategies import build as build_strategy

# Parameter grids centred on each rule's default. Deliberately coarse: the point
# is to map the neighbourhood, not to find the best cell in it.
DEFAULT_GRIDS = {
    "donchian": {"entry": [48, 72, 96, 120, 168], "exit": [24, 36, 48, 72]},
    "sma_cross": {"fast": [12, 24, 50, 100], "slow": [100, 150, 200, 300]},
    "mean_reversion": {"window": [24, 48, 72, 120], "entry": [1.5, 2.0, 2.5]},
    "vol_target": {"target": [0.35, 0.40, 0.50, 0.60, 0.75],
                   "vol_window": [15, 30, 60, 120]},
}


def _valid(name: str, combo: dict) -> bool:
    """Reject parameter combinations that are structurally meaningless."""
    if name == "sma_cross":
        return combo["fast"] < combo["slow"]
    if name == "donchian":
        return combo["exit"] <= combo["entry"]
    return True


def parameter_neighbourhood(cfg, strategy: str, ctx=None, grid=None) -> pd.DataFrame:
    """Sharpe across a grid of parameter values, on one prepared context.

    Rule strategies read only the bars, so the context (features, labels,
    splits) is identical across the grid and is computed once.
    """
    grid = grid or DEFAULT_GRIDS.get(strategy)
    if not grid:
        raise ValueError(f"no parameter grid defined for {strategy!r}")
    ctx = ctx or prepare(cfg, verbose=False)

    keys = list(grid)
    rows = []
    for values in product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        if not _valid(strategy, combo):
            continue
        positions = build_strategy(strategy, combo).positions(ctx)
        result = backtest(ctx.bars, positions, cfg.cost)
        metrics = summarize(result, cfg.data.bars_per_year)
        rows.append({**combo,
                     "sharpe": metrics["sharpe"],
                     "total_return": metrics["total_return"],
                     "n_trades": metrics["n_trades"]})
    return pd.DataFrame(rows)


def period_stability(cfg, strategy: str, ctx=None, freq: str = "YE") -> pd.DataFrame:
    """Per-year performance, alongside buy & hold over the same periods.

    The benchmark column is not decoration. "Lost money in 2022" and "lost less
    than half what holding lost in 2022" are opposite findings, and the bare
    per-year Sharpe cannot distinguish them. For a long-only trend filter
    evaluated over a historic bull market, this column is the whole argument.
    """
    ctx = ctx or prepare(cfg, verbose=False)

    positions = build_strategy(strategy, cfg.params.get(strategy)).positions(ctx)
    strat = by_period(backtest(ctx.bars, positions, cfg.cost),
                      cfg.data.bars_per_year, mask=ctx.oos_mask, freq=freq)

    hold_pos = build_strategy("buy_hold").positions(ctx)
    hold = by_period(backtest(ctx.bars, hold_pos, cfg.cost),
                     cfg.data.bars_per_year, mask=ctx.oos_mask, freq=freq)

    merged = strat.merge(
        hold[["period", "total_return", "sharpe"]],
        on="period", how="left", suffixes=("", "_hold"),
    )
    merged["excess_sharpe"] = merged["sharpe"] - merged["sharpe_hold"]
    return merged


def cross_symbol(cfg, strategy: str, symbols: list[str]) -> pd.DataFrame:
    """The same strategy and parameters, on other instruments.

    Each symbol needs its own data, features and splits, so this is the
    expensive test — and the most informative one.
    """
    rows = []
    for symbol in symbols:
        trial = copy.deepcopy(cfg)
        trial.data.symbol = symbol
        trial.strategies = [strategy]
        try:
            results = run_pipeline(trial, verbose=False)
        except (SystemExit, ValueError) as exc:
            rows.append({"symbol": symbol, "sharpe": np.nan,
                         "total_return": np.nan, "n_trades": 0,
                         "note": str(exc)[:60]})
            continue
        m = results[strategy]
        bh = results["buy_hold"]
        rows.append({
            "symbol": symbol,
            "sharpe": m["sharpe"],
            "total_return": m["total_return"],
            "n_trades": m["n_trades"],
            "vs_hold": m["sharpe"] - bh["sharpe"],
            "note": "",
        })
    return pd.DataFrame(rows)


def verdict(neighbourhood: pd.DataFrame, stability: pd.DataFrame,
            transfer: pd.DataFrame,
            benchmark_sharpe: float | None = None) -> tuple[bool, list[str]]:
    """Turn the three tables into a pass/fail with reasons.

    The thresholds are deliberately lenient. They are meant to catch strategies
    that are obviously fitted, not to certify the survivors.
    """
    notes, passed = [], True

    frac = float((neighbourhood["sharpe"] > 0).mean())
    median = float(neighbourhood["sharpe"].median())
    best = float(neighbourhood["sharpe"].max())
    if frac < 0.6 or median <= 0:
        passed = False
        notes.append(
            f"NEIGHBOURHOOD FAIL: only {frac:.0%} of {len(neighbourhood)} parameter "
            f"combinations are positive (median sharpe {median:.2f}, best {best:.2f}). "
            f"A real effect is not this sensitive to its own parameters."
        )
    else:
        # "Positive" is a low bar. Without this clause the note reads as a pass
        # even when the entire grid sits below the thing you'd have done anyway.
        context = ""
        if benchmark_sharpe is not None:
            beat = float((neighbourhood["sharpe"] > benchmark_sharpe).mean())
            context = (f", but only {beat:.0%} beat buy & hold "
                       f"({benchmark_sharpe:.2f})")
        notes.append(
            f"neighbourhood ok: {frac:.0%} of combinations positive, "
            f"median sharpe {median:.2f}{context}"
        )

    if stability.empty:
        passed = False
        notes.append("STABILITY FAIL: no periods with activity to evaluate")
    else:
        # The test that matters is beating the benchmark, not being positive.
        # A long-only filter over a bull market is positive in most years by
        # construction; that says nothing about whether the rule adds anything.
        beat = float((stability["excess_sharpe"] > 0).mean())
        worst = float(stability["excess_sharpe"].min())
        if beat < 0.6:
            passed = False
            notes.append(
                f"STABILITY FAIL: beat buy & hold in only {beat:.0%} of years "
                f"(worst excess sharpe {worst:.2f}). The edge is concentrated in "
                f"a minority of periods."
            )
        else:
            notes.append(
                f"stability ok: beat buy & hold in {beat:.0%} of years "
                f"(worst excess sharpe {worst:.2f})"
            )

    scored = transfer.dropna(subset=["sharpe"])
    if scored.empty:
        passed = False
        notes.append("TRANSFER FAIL: no other symbol produced a result")
    else:
        pos_sym = float((scored["sharpe"] > 0).mean())
        if pos_sym < 0.6:
            passed = False
            notes.append(
                f"TRANSFER FAIL: positive on only {pos_sym:.0%} of "
                f"{len(scored)} other symbols. The rule may describe this asset's "
                f"history rather than market structure."
            )
        else:
            notes.append(f"transfer ok: positive on {pos_sym:.0%} of other symbols")

    return passed, notes


def pivot_grid(df: pd.DataFrame, keys: list[str]) -> str:
    """Render a two-parameter grid as a Sharpe matrix."""
    if len(keys) != 2:
        return df.to_string(index=False)
    table = df.pivot(index=keys[0], columns=keys[1], values="sharpe")
    head = f"{keys[0]:>6} \\ {keys[1]:<4}" + "".join(f"{c:>9}" for c in table.columns)
    lines = [head, "-" * len(head)]
    for idx, row in table.iterrows():
        cells = "".join("      —  " if pd.isna(v) else f"{v:>9.2f}" for v in row)
        lines.append(f"{idx:>13}" + cells)
    return "\n".join(lines)
