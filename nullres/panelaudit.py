"""Controls that decide whether cross-sectional skill is real.

A panel AUC above 0.5 has more ways of being an artefact than a single-asset
one, and the interesting failures are not leaks — they are the model learning
something true but useless:

  SHUFFLED LABEL   Permute the target within each timestamp and refit. The
                   permutation preserves the balanced structure, so anything
                   left is a side channel, not signal.

  SURVIVORS ONLY   Refit with the delisted symbols removed. If the AUC
                   collapses, the model was detecting death, not ranking
                   assets — a real regularity you cannot trade, because by the
                   time a coin is dying its borrow has vanished.

  PER-SYMBOL       Cross-sectional ranks never name a symbol, but if BTC is
                   permanently the lowest-volatility member then "rank 1 by low
                   vol" and "BTC" are the same column. A wide spread in
                   per-symbol accuracy is that tell.

  CONTRIBUTION     Which symbols actually produced the P&L, and how much of it
                   came from the ones that delisted.

  TAIL CENSUS      A short book that never got hit is not a book with no tail
                   risk. Count how often the moves that would hurt occur, and
                   how many short-name-bars were exposed to them; the product
                   is the number of hits chance predicts.

These were run once by hand and quoted in RESEARCH.md, which meant the numbers
underneath the project's strongest result were the only ones no command could
regenerate. That is exactly backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shuffled_label_auc(panel, cfg, seed: int = 0) -> float:
    """Mean fold AUC after permuting the label within each timestamp.

    Permuting *within* a timestamp rather than globally keeps the label balanced
    in every regime, so the control isolates the ranking signal instead of also
    destroying the panel's structure.

    Each timestamp draws from a stream seeded by `(seed, that timestamp)`, so a
    given bar receives the same permutation no matter where it falls in the
    iteration. Sharing one generator across groups would have made the result
    depend on processing order — reproducible only so long as nothing upstream
    changed how the panel is grouped, which is not a property worth relying on
    in a control whose whole job is to be trustworthy.
    """
    from nullres.crosssec import Panel, fit_predict_panel

    def permute(group: pd.Series) -> pd.Series:
        ts = group.index.get_level_values("ts")[0]
        rng = np.random.default_rng([seed, int(pd.Timestamp(ts).value)])
        # Sort by symbol before permuting. Seeding per timestamp fixes which
        # stream a bar draws from, but the permutation still lands on whatever
        # order the values arrive in — so row order would otherwise decide which
        # symbol got which label. Sorting first makes the symbol -> label
        # assignment a function of the timestamp alone.
        ordered = group.sort_index(level="symbol")
        drawn = pd.Series(rng.permutation(ordered.to_numpy()), index=ordered.index)
        return drawn.reindex(group.index)

    shuffled = panel.y.groupby(level="ts", group_keys=False).apply(permute)

    control = Panel(features=panel.features, y=shuffled, ret_next=panel.ret_next,
                    funding=panel.funding, times=panel.times,
                    horizon=panel.horizon, symbols=panel.symbols,
                    delisted=panel.delisted)
    _, reports = fit_predict_panel(control, cfg, verbose=False)
    return float(np.nanmean([r["auc"] for r in reports]))


def survivors_only_auc(panel, cfg) -> float | None:
    """Mean fold AUC with every delisted symbol dropped.

    A model that only knows which coins are dying has found something real and
    untradable. None when the universe contains no corpses to remove.
    """
    from nullres.crosssec import Panel, fit_predict_panel

    if not panel.delisted:
        return None
    dead = set(panel.delisted)
    keep = [s for s in panel.symbols if s not in dead]
    alive = panel.features.index.get_level_values("symbol").isin(keep)

    control = Panel(features=panel.features[alive], y=panel.y[alive],
                    ret_next=panel.ret_next[keep], funding=panel.funding[keep],
                    times=panel.times, horizon=panel.horizon,
                    symbols=keep, delisted={})
    _, reports = fit_predict_panel(control, cfg, verbose=False)
    return float(np.nanmean([r["auc"] for r in reports]))


def per_symbol_accuracy(proba: pd.Series, panel) -> pd.DataFrame:
    """Directional accuracy per symbol, with the count it rests on.

    Skill concentrated in one or two names is skill that has learned those
    names, whatever the ranks pretend. But the spread only means that if every
    symbol has enough scored bars to have an accuracy worth reading.

    **The count is not decoration.** On a screened wide universe, symbols drift
    in and out of the tradable set and some are scored on a few dozen bars. The
    first version of this returned bare accuracies, and the 136-symbol panel
    duly produced a spread of 0.685 — seven times the narrow panel's, and
    entirely an artefact of thin symbols. At n=30 a coin flip reaches 0.86
    without trying. Callers filter on `n` before quoting a spread.

    **Accuracy alone is the wrong statistic here, and this is subtle.** The
    label is "beats the cross-sectional median", so a coin that persistently
    underperformed has a lopsided base rate of its own — say 0.85 zeros. A model
    that learned nothing but "this one usually lags" scores 0.85 on it. High
    per-symbol accuracy can therefore be pure unconditional drift, and reading
    the raw spread as "skill is concentrated in these names" overstates it.

    Lift over the symbol's own majority class removes that, but introduces its
    own bias in exactly this setting: a cross-sectional model MUST rank, so at
    every timestamp roughly half the universe is predicted low. It structurally
    cannot predict the majority class for a symbol that beats the median 79% of
    the time, and gets charged a large negative lift for a constraint rather
    than a mistake.

    **So the spread worth reading is in per-symbol AUC.** It is threshold-free
    and base-rate invariant: 0.5 means the model cannot tell this symbol's good
    bars from its bad ones, whatever its unconditional tendency and whatever the
    ranking forced. Accuracy, base rate and lift are kept alongside because they
    are what a reader expects to see, but the AUC column is the one that answers
    "is the skill concentrated in particular names".
    """
    from sklearn.metrics import roc_auc_score

    scored = proba.notna() & panel.y.notna()
    p, y = proba[scored], panel.y[scored]

    hit = ((p > 0.5) == (y > 0.5)).astype(float).groupby(level="symbol")
    mean_y = y.groupby(level="symbol").mean()
    base = pd.concat([mean_y, 1.0 - mean_y], axis=1).max(axis=1)

    def symbol_auc(group: pd.Series) -> float:
        truth = y.loc[group.index]
        if truth.nunique() < 2 or len(truth) < 2:
            return float("nan")
        return float(roc_auc_score(truth.astype(int), group))

    auc = p.groupby(level="symbol").apply(symbol_auc)

    out = pd.DataFrame({"auc": auc, "accuracy": hit.mean(), "base_rate": base,
                        "n": hit.size()})
    out["lift"] = out["accuracy"] - out["base_rate"]
    return out.sort_values("auc", ascending=False)


def pnl_contribution(positions: pd.DataFrame, panel) -> pd.Series:
    """Gross log P&L attributable to each symbol."""
    ret = panel.ret_next.reindex(positions.index)[positions.columns].fillna(0.0)
    return (positions * ret).sum().sort_values(ascending=False)


def delisted_share(positions: pd.DataFrame, panel) -> float:
    """Share of P&L ACTIVITY, in absolute terms, from symbols that later delisted.

    **This is a share of absolute P&L, not of net profit, and the distinction
    changes what the number means.** Each symbol contributes `|its P&L|`, so a
    +30% winner and a -30% loser both count as 30 rather than cancelling to
    zero. The question being asked is "how much of what this book did happened
    in coins that were dying" — exposure, not profitability.

    Netting would answer a different and weaker question. A book that made a
    fortune on one delisting and lost it on another would net to ~0% and look
    untouched by delisting, when in fact its entire outcome hinged on dying
    coins. For a survivorship control that is the wrong answer, so the metric
    deliberately does not net.

    The consequence to keep in mind when reading it: this number cannot be
    compared against a return, and it can be large while the delisted names
    contributed nothing to the bottom line.
    """
    contribution = pnl_contribution(positions, panel).abs()
    total = float(contribution.sum())
    if total <= 0:
        return 0.0
    dead = [s for s in panel.delisted if s in contribution.index]
    return float(contribution[dead].sum() / total)


def tail_census(positions: pd.DataFrame, panel, threshold: float = 0.65) -> dict:
    """How many moves big enough to matter occurred, and how exposed the book was.

    Observing no blow-up means nothing until you know how many blow-ups chance
    predicted. If the expected count is a fraction of one, zero hits is what
    chance produces and the tail is untested rather than absent.
    """
    ret = panel.ret_next.reindex(positions.index)[positions.columns]
    simple = np.expm1(ret)

    extreme = int((simple > threshold).sum().sum())
    observed_bars = int(simple.notna().sum().sum())
    rate = extreme / observed_bars if observed_bars else 0.0

    short_bars = int((positions < 0).sum().sum())
    worst_bar = float(np.expm1((positions * ret.fillna(0.0)).sum(axis=1)).min())

    hit = simple.where(positions < 0) > threshold
    return {
        "threshold": threshold,
        "extreme_moves": extreme,
        "observed_bars": observed_bars,
        "rate": rate,
        "short_name_bars": short_bars,
        "expected_hits": rate * short_bars,
        "actual_hits": int(hit.sum().sum()),
        "worst_bar_return": worst_bar,
    }


def format_report(panel, cfg, proba, positions, mean_auc: float,
                  min_obs: int = 200) -> str:
    """Run every control and render it. The order is cheapest-first."""
    lines = ["", "--- verification " + "-" * 59, ""]

    shuffled = shuffled_label_auc(panel, cfg)
    verdict = "clean" if abs(shuffled - 0.5) < 0.02 else "SUSPECT"
    lines.append(f"  shuffled labels        AUC {shuffled:.4f}   "
                 f"vs {mean_auc:.4f} real — {verdict}")

    survivors = survivors_only_auc(panel, cfg)
    if survivors is None:
        lines.append("  survivors only         n/a — no delisted symbols to remove")
    else:
        drop = mean_auc - survivors
        reading = ("death detection" if drop > 0.02 else
                   "not death detection")
        lines.append(f"  survivors only         AUC {survivors:.4f}   "
                     f"({drop:+.4f}) — {reading}")

    # Symbols scored on a handful of bars carry accuracies that swing wildly by
    # chance, and on a screened universe there are always some. Quoting a spread
    # across them measures the screen, not the model.
    stats = per_symbol_accuracy(proba, panel)
    thick = stats[(stats["n"] >= min_obs) & stats["auc"].notna()]
    if len(thick) >= 2:
        thin = len(stats) - len(thick)
        note = f", {thin} thinner excluded" if thin else ""
        auc_spread = float(thick["auc"].iloc[0] - thick["auc"].iloc[-1])
        lines.append(
            f"  per-symbol skill       AUC spread {auc_spread:.3f} over "
            f"{len(thick)} symbols with >={min_obs} scored bars{note}"
        )
        for label, name in (("best ", thick.index[0]), ("worst", thick.index[-1])):
            row = thick.loc[name]
            lines.append(
                f"                         {label} {name} AUC {row['auc']:.3f}"
                f"  (accuracy {row['accuracy']:.3f} vs base rate "
                f"{row['base_rate']:.3f}, lift {row['lift']:+.3f}, "
                f"n={int(row['n']):,})"
            )
        above = int((thick["auc"] > 0.5).sum())
        lines.append(
            f"                         {above} of {len(thick)} symbols score "
            f"above 0.5; median {thick['auc'].median():.3f}"
        )
        lines.append(
            f"                         raw accuracy spread is "
            f"{float(thick['accuracy'].max() - thick['accuracy'].min()):.3f}, "
            f"but that is mostly each symbol's own base rate — AUC is the "
            f"base-rate-free read"
        )
    elif len(stats) >= 2:
        lines.append(f"  per-symbol skill       n/a — no symbol reached "
                     f"{min_obs} scored bars")

    share = delisted_share(positions, panel)
    lines.append(f"  delisted contribution  {share:.1%} of ABSOLUTE P&L (not "
                 f"netted) from {len(panel.delisted)} symbol(s) that stopped "
                 f"trading")

    contribution = pnl_contribution(positions, panel)
    top = ", ".join(f"{s}" for s in contribution.index[:4])
    bottom = ", ".join(f"{s}" for s in contribution.index[-4:])
    lines.append(f"  contributors           + {top}")
    lines.append(f"                         - {bottom}")

    census = tail_census(positions, panel)
    lines += [
        "",
        f"  tail census (moves above +{census['threshold']:.0%} in one bar)",
        f"    {census['extreme_moves']:,} such moves in "
        f"{census['observed_bars']:,} symbol-bars "
        f"(1 in {1 / census['rate']:,.0f})" if census["rate"] else
        "    none observed",
        f"    book held {census['short_name_bars']:,} short-name-bars, so chance "
        f"predicts {census['expected_hits']:.2f} hits",
        f"    actually hit {census['actual_hits']}; worst single bar "
        f"{census['worst_bar_return']:.1%}",
    ]
    if census["expected_hits"] < 1:
        lines.append("    Expected count is below one, so observing none is what "
                     "chance predicts.")
        lines.append("    The tail is UNTESTED, not absent — and the engine "
                     "models no margin.")
    return "\n".join(lines)
