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
forward paper trading.

**What these tests can and cannot resolve.** Two of the three rest on very few
observations: five years, four symbols. A gate reading "beat the benchmark in
60% of periods" sounds demanding, but at n=5 it means three of five, and a
strategy that is genuinely a coin flip against buy & hold clears it half the
time. At n=4 symbols a coin flip clears it 31% of the time. That is not a
detail — it means a bare count cannot tell "this is worse than holding" from
"there is not enough evidence here", and for a long time this module reported
both as KILLED.

So each count gate is now read alongside the magnitude of the shortfall, and
`verdict` has three outcomes rather than two. A gate FAILS only when the count
goes against it AND the mean excess is distinguishable from zero. A count that
fails alone yields INCONCLUSIVE, and every note prints how often the gate would
have fired by chance.

The decision rule stays aggressive on purpose — a strategy earns SURVIVED only
by clearing all three — because a false kill costs one idea and a false survival
costs months. That is a decision about which error to prefer, not a claim that
the thresholds are statistically strong. They are not.
"""

from __future__ import annotations

import copy
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sps

from nullres.backtest import backtest, by_period, summarize
from nullres.pipeline import prepare, run_pipeline
from nullres.strategies import build as build_strategy

# Parameter grids centred on each rule's default. Deliberately coarse: the point
# is to map the neighbourhood, not to find the best cell in it.
DEFAULT_GRIDS = {
    "donchian": {"entry": [48, 72, 96, 120, 168], "exit": [24, 36, 48, 72]},
    "sma_cross": {"fast": [12, 24, 50, 100], "slow": [100, 150, 200, 300]},
    "mean_reversion": {"window": [24, 48, 72, 120], "entry": [1.5, 2.0, 2.5]},
    "vol_target": {"target": [0.35, 0.40, 0.50, 0.60, 0.75],
                   "vol_window": [15, 30, 60, 120]},
}

# ML strategies take no rule parameters — their knobs live in [sizing]. Varying
# those does not change the model's predictions, so the walk-forward fit is
# reused across the whole grid via the context's prediction cache.
SIZING_GRIDS = {
    "ml_direction": {"long_entry": [0.52, 0.54, 0.56, 0.58, 0.60],
                     "min_hold": [42, 84, 126, 168]},
    "ml_meta": {"long_entry": [0.52, 0.54, 0.56, 0.58, 0.60],
                "min_hold": [42, 84, 126, 168]},
}


def grid_for(strategy: str) -> tuple[dict, str]:
    """Return (grid, kind) where kind is 'params' or 'sizing'."""
    if strategy in SIZING_GRIDS:
        return SIZING_GRIDS[strategy], "sizing"
    if strategy in DEFAULT_GRIDS:
        return DEFAULT_GRIDS[strategy], "params"
    raise ValueError(f"no parameter grid defined for {strategy!r}")


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
    default_grid, kind = grid_for(strategy)
    grid = grid or default_grid
    ctx = ctx or prepare(cfg, verbose=False)
    original_cfg = ctx.cfg

    keys = list(grid)
    rows = []
    try:
        for values in product(*(grid[k] for k in keys)):
            combo = dict(zip(keys, values))
            if not _valid(strategy, combo):
                continue

            if kind == "sizing":
                trial = copy.deepcopy(cfg)
                for key, value in combo.items():
                    setattr(trial.sizing, key, value)
                # Keep the short band symmetric with the long one, so the grid
                # varies conviction rather than quietly introducing a long bias.
                if "long_entry" in combo:
                    trial.sizing.short_entry = round(1.0 - combo["long_entry"], 4)
                ctx.cfg = trial
                strategy_obj = build_strategy(strategy)
            else:
                trial = cfg
                strategy_obj = build_strategy(strategy, combo)

            positions = strategy_obj.positions(ctx)
            result = backtest(ctx.bars, positions, trial.cost)
            # Same out-of-sample restriction the pipeline applies. Without it
            # the grid is measured over the full frame while `benchmark_sharpe`
            # comes from `period_stability`, which masks — so `verdict` compared
            # a deflated grid against an undeflated bar.
            metrics = summarize(result, trial.data.bars_per_year,
                                mask=ctx.oos_mask)
            rows.append({**combo,
                         "sharpe": metrics["sharpe"],
                         "total_return": metrics["total_return"],
                         "n_trades": metrics["n_trades"]})
    finally:
        ctx.cfg = original_cfg
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

    # Join onto the BENCHMARK's periods, not the strategy's. `by_period` drops a
    # period with no variance, which is precisely a year the strategy sat flat —
    # so a left join silently removed it from the stability test rather than
    # scoring it. Live on the 1d config: sma_cross and donchian are flat through
    # 2022 and that year simply disappeared, taking with it the year a flat book
    # most obviously beats a -65% benchmark. The bias is not consistently in the
    # strategy's favour, which is worse than if it were: whether dropping a year
    # helps or hurts depends on what the benchmark happened to do in it.
    #
    # A flat period is a real observation. The strategy returned nothing, so its
    # Sharpe is 0 and its excess is minus the benchmark's.
    merged = strat.merge(
        hold[["period", "total_return", "sharpe"]],
        on="period", how="right", suffixes=("", "_hold"),
    ).sort_values("period", ignore_index=True)
    for column, filler in (("sharpe", 0.0), ("total_return", 0.0),
                           ("n_trades", 0), ("bars", 0)):
        if column in merged:
            merged[column] = merged[column].fillna(filler)

    merged["excess_sharpe"] = merged["sharpe"] - merged["sharpe_hold"]
    return merged


def hold_sharpe(cfg, ctx) -> float:
    """Buy & hold's Sharpe, on the same window AND the same statistic as the grid.

    `verdict` reports what fraction of the neighbourhood beats buy & hold, and
    that comparison is only meaningful if both sides are the same measurement.
    They were not: the grid reports a full-window Sharpe while the benchmark was
    taken as the mean of `period_stability`'s per-year Sharpes. Averaging annual
    Sharpes is a different statistic — on the 4h config it gives 0.53 against a
    full-window 0.38, overstating the bar by 40%.
    """
    positions = build_strategy("buy_hold").positions(ctx)
    result = backtest(ctx.bars, positions, cfg.cost)
    return float(summarize(result, cfg.data.bars_per_year,
                           mask=ctx.oos_mask)["sharpe"])


def cross_symbol(cfg, strategy: str, symbols: list[str],
                 start: str | None = None) -> pd.DataFrame:
    """The same strategy and parameters, on other instruments.

    Each symbol needs its own data, features and splits, so this is the
    expensive test — and the most informative one. It is what killed both
    previous candidates.

    `start` overrides the config's start date for EVERY symbol including the
    reference one. Auxiliary archives begin at different dates per symbol
    (BTCUSDT open-interest metrics start 2020-09, everything else 2021-12), and
    letting each symbol use its own maximum range would compare different eras
    and call the difference "transfer".
    """
    rows = []
    for symbol in symbols:
        trial = copy.deepcopy(cfg)
        trial.data.symbol = symbol
        trial.strategies = [strategy]
        if start:
            trial.data.start = start
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


def sign_flip_rate(grid: pd.DataFrame, keys: list[str]) -> float:
    """Fraction of adjacent grid cells whose Sharpe changes sign.

    This is the measure the docs have always claimed to care about — "a real
    effect degrades SMOOTHLY; an isolated spike is a fitting artefact" — and
    which nothing actually computed. Counting positive cells cannot see the
    difference between a coherent positive region and a checkerboard: a grid
    reading +0.8 / -0.2 / +0.9 / -0.7 scores 50% positive either way.

    Neighbours are cells one step apart along a single axis. A rate near 0
    means a smooth surface; near 0.5 means the sign carries no information.
    """
    if len(keys) != 2 or grid.empty:
        return float("nan")
    table = grid.pivot(index=keys[0], columns=keys[1], values="sharpe")
    values = table.to_numpy(dtype="float64")

    flips = total = 0
    for axis in (0, 1):
        a = np.moveaxis(values, axis, 0)
        for i in range(len(a) - 1):
            pair = np.vstack([a[i], a[i + 1]])
            valid = np.isfinite(pair).all(axis=0)
            flips += int((np.sign(pair[0][valid]) != np.sign(pair[1][valid])).sum())
            total += int(valid.sum())
    return flips / total if total else float("nan")


def _is_noise_field(frac_positive: float, flip_rate: float | None,
                    tolerance: float = 0.8) -> bool:
    """Is the grid's sign pattern indistinguishable from random placement?

    If a fraction p of cells are positive and they are scattered at random,
    adjacent cells differ in sign with probability 2p(1-p). A genuine effect
    clusters, giving a materially lower rate. Measuring against this baseline
    rather than a fixed threshold matters: a grid that is 95% positive can
    never flip more than ~10% of the time, so a flat cutoff would wave it
    through regardless of arrangement.

    When almost every cell shares one sign the test is vacuous — there is
    nothing for an arrangement to be informative about — so it is skipped.
    """
    if flip_rate is None or not np.isfinite(flip_rate):
        return False
    expected = 2 * frac_positive * (1 - frac_positive)
    if expected < 0.05:
        return False
    return flip_rate >= tolerance * expected


KILLED, SURVIVED, INCONCLUSIVE = "KILLED", "SURVIVED", "INCONCLUSIVE"


def count_gate_power(n: int, threshold: float = 0.6, p_null: float = 0.5) -> float:
    """How often a strategy exactly as good as the benchmark clears a count gate.

    "Beat the benchmark in 60% of periods" sounds demanding until you count the
    periods. With five years it means three of five, and a strategy that is
    genuinely a coin flip against buy & hold clears that **half the time**. With
    four symbols it is three of four, which a coin flip clears 31% of the time.

    A gate this noisy cannot carry a verdict by itself, so the number is printed
    beside every count so the reader knows what the count is worth.
    """
    if n < 1:
        return float("nan")
    need = int(np.ceil(threshold * n))
    return float(1 - sps.binom.cdf(need - 1, n, p_null))


def excess_magnitude(values) -> tuple[float, float]:
    """(mean, p) for "is the mean excess distinguishable from zero".

    The count gates throw away magnitude, and that loses real information: on
    the 4h config `donchian` and `mean_reversion` both beat hold in 40% of years
    and score identically, while their mean excess Sharpes are -0.04 and -1.13.
    One is indistinguishable from holding; the other is far worse. A test on the
    magnitude separates them; counting signs cannot.

    It is not a replacement for the count, because it is less decisive on noisy
    series — `mean_reversion`'s -1.13 carries p=0.31 across five volatile years.
    Neither statistic dominates, so both are reported.
    """
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2 or np.allclose(clean, clean[0]):
        return (float(clean.mean()) if len(clean) else float("nan"), float("nan"))
    _, p = sps.ttest_1samp(clean, 0.0)
    return float(clean.mean()), float(p)


def _decisively_worse(mean: float, p: float, alpha: float = 0.05) -> bool:
    return bool(np.isfinite(p) and p < alpha and mean < 0)


def _count_gate(values, label: str, unit: str,
                threshold: float = 0.6) -> tuple[str, str]:
    """Score one count-plus-magnitude gate. Returns (outcome, note).

    Outcomes are FAIL (decisive evidence against), PASS, or WEAK — the tests
    ran but cannot separate this strategy from one exactly as good as the
    benchmark. WEAK is not a pass; it is the honest description of an
    underpowered result, and it is why the overall verdict has three states.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return "FAIL", f"{label} FAIL: no {unit} produced a result"

    beat = int((values > 0).sum())
    frac = beat / n
    mean, p = excess_magnitude(values)
    power = count_gate_power(n, threshold)
    p_text = "n/a" if not np.isfinite(p) else f"{p:.3f}"

    if _decisively_worse(mean, p):
        return "FAIL", (
            f"{label} FAIL: beat the benchmark in {beat} of {n} {unit} "
            f"({frac:.0%}); mean excess {mean:+.2f} (p={p_text}) — decisively "
            f"worse than the thing you would have done anyway."
        )
    if frac >= threshold:
        return "PASS", (
            f"{label.lower()} ok: beat the benchmark in {beat} of {n} {unit} "
            f"({frac:.0%}); mean excess {mean:+.2f} (p={p_text})."
        )
    return "WEAK", (
        f"{label} INCONCLUSIVE: beat the benchmark in {beat} of {n} {unit} "
        f"({frac:.0%}), but mean excess {mean:+.2f} (p={p_text}) is not "
        f"distinguishable from zero. At n={n} this count gate passes "
        f"{power:.0%} of the time even for a strategy exactly as good as the "
        f"benchmark, so failing it on its own decides nothing."
    )


def verdict(neighbourhood: pd.DataFrame, stability: pd.DataFrame,
            transfer: pd.DataFrame,
            benchmark_sharpe: float | None = None,
            flip_rate: float | None = None) -> tuple[str, list[str]]:
    """Turn the three tables into KILLED, SURVIVED or INCONCLUSIVE, with reasons.

    Three states, not two, because two states forced a claim the evidence does
    not support. The count gates rest on four or five Bernoulli draws: at n=5 a
    strategy genuinely equal to buy & hold fails the stability gate half the
    time. Reporting that as KILLED dressed a coin flip as a finding, and the
    verdict then propagated into the ledger and warned future runs off the
    config.

    So a gate now FAILS only on decisive evidence — the count went against it
    AND the magnitude of the shortfall is distinguishable from zero. A count
    that fails on its own returns WEAK, and the run comes out INCONCLUSIVE.

    The decision rule stays deliberately aggressive: for research triage a false
    kill is cheap and a false survival is expensive, so a strategy has to earn
    SURVIVED by clearing every gate. That is a decision-theoretic stance, not a
    claim that the thresholds are statistically demanding. They are not, and
    each note now says so.
    """
    notes, outcomes = [], []

    frac = float((neighbourhood["sharpe"] > 0).mean())
    median = float(neighbourhood["sharpe"].median())
    best = float(neighbourhood["sharpe"].max())
    # The neighbourhood gate keeps a hard pass/fail, but not because counting
    # ~20 cells is decisive on its own — at a 60% threshold even 20 independent
    # draws fire a quarter of the time by chance, and adjacent grid cells are
    # heavily correlated, so the effective count is smaller still. What carries
    # this gate is the two conditions the count does not supply: the median must
    # be positive, and `_is_noise_field` judges the ARRANGEMENT of signs against
    # what random placement would give. Those do not degrade with sample size
    # the way a bare count does.
    if frac < 0.6 or median <= 0:
        outcomes.append("FAIL")
        notes.append(
            f"NEIGHBOURHOOD FAIL: only {frac:.0%} of {len(neighbourhood)} parameter "
            f"combinations are positive (median sharpe {median:.2f}, best {best:.2f}). "
            f"A real effect is not this sensitive to its own parameters."
        )
    elif _is_noise_field(frac, flip_rate):
        # Counting positive cells cannot distinguish a coherent region from a
        # checkerboard. Compare the observed flip rate against what random
        # placement of the SAME number of positive cells would produce:
        # 2p(1-p). Matching that means the arrangement carries no information.
        expected = 2 * frac * (1 - frac)
        outcomes.append("FAIL")
        notes.append(
            f"NEIGHBOURHOOD FAIL: {frac:.0%} of combinations are positive, but the "
            f"sign flips across {flip_rate:.0%} of adjacent cells versus "
            f"{expected:.0%} expected from random placement. The grid is no "
            f"smoother than chance — a noise field with a lucky maximum "
            f"({best:.2f}), not a region of edge."
        )
    else:
        # "Positive" is a low bar. Without this clause the note reads as a pass
        # even when the entire grid sits below the thing you'd have done anyway.
        context = ""
        if benchmark_sharpe is not None:
            beat = float((neighbourhood["sharpe"] > benchmark_sharpe).mean())
            context = (f", but only {beat:.0%} beat buy & hold "
                       f"({benchmark_sharpe:.2f})")
        smooth = f", sign flips across {flip_rate:.0%} of adjacent cells" \
            if flip_rate is not None else ""
        outcomes.append("PASS")
        notes.append(
            f"neighbourhood ok: {frac:.0%} of combinations positive, "
            f"median sharpe {median:.2f}{context}{smooth}"
        )

    # The test that matters is beating the benchmark, not being positive. A
    # long-only filter over a bull market is positive in most years by
    # construction; that says nothing about whether the rule adds anything.
    if stability.empty:
        outcomes.append("FAIL")
        notes.append("STABILITY FAIL: no periods with activity to evaluate")
    else:
        outcome, note = _count_gate(
            stability["excess_sharpe"].to_numpy(dtype=float), "STABILITY", "years"
        )
        outcomes.append(outcome)
        notes.append(note)

    # Same correction: "positive" is a low bar when every asset in the sample
    # rose. BNBUSDT scored Sharpe 0.03 — positive, and 0.71 WORSE than simply
    # holding it. Judge against the alternative.
    scored = transfer.dropna(subset=["sharpe"])
    if scored.empty:
        outcomes.append("FAIL")
        notes.append("TRANSFER FAIL: no other symbol produced a result")
    else:
        column = "vs_hold" if "vs_hold" in scored.columns else "sharpe"
        outcome, note = _count_gate(
            scored[column].to_numpy(dtype=float), "TRANSFER", "symbols"
        )
        outcomes.append(outcome)
        notes.append(note)

    if "FAIL" in outcomes:
        result = KILLED
    elif "WEAK" in outcomes:
        result = INCONCLUSIVE
        notes.append(
            "INCONCLUSIVE means the battery ran and could not separate this "
            "strategy from one exactly as good as the benchmark — not that it "
            "looks promising. Deciding what an underpowered result means is "
            "the human's job; see docs/05-graveyard.md."
        )
    else:
        result = SURVIVED
    return result, notes


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
