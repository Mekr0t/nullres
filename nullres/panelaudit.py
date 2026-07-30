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

    `threshold` is one point on a curve, and a single point is a constant doing
    analytical work — the thing this repo keeps having to remove. Prefer
    `tail_curve`, which sweeps it and reports the capital each level would cost,
    so no one number carries the argument.
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


def concentration(positions: pd.DataFrame, nominal: float) -> dict:
    """How often a dead leg left the book concentrated, and for how long.

    `_neutralise` keeps the book dollar-neutral when a symbol delists by
    rescaling the surviving side, so a k=2 book whose short partner dies holds
    -1.0 in one name instead of -0.5 in two. Gross exposure does not change and
    net stays at zero — what changes is that the move which ruins the book
    halves, from +200% to +100%. Nine symbols delist in the wide universe, so
    this is not hypothetical.

    Keeping that behaviour is a choice: dollar neutrality is the book's defining
    constraint, and the alternatives (halve the long side, go flat) change the
    strategy rather than make it safer. What was missing is visibility. A
    maximum weight on its own cannot distinguish one bar from three thousand —
    the difference between a curiosity and the dominant risk in the book — so
    this reports the share of bars spent concentrated and the longest unbroken
    stretch of it.
    """
    longs = positions.where(positions > 0)
    shorts = -positions.where(positions < 0)

    per_bar = pd.concat([longs.max(axis=1), shorts.max(axis=1)], axis=1).max(axis=1)
    tol = nominal * 1e-6
    over = per_bar > nominal + tol
    held = positions.abs().sum(axis=1) > tol          # bars with a position at all

    max_short = float(shorts.max().max()) if shorts.notna().any().any() else 0.0
    max_long = float(longs.max().max()) if longs.notna().any().any() else 0.0
    peak = max(max_short, max_long)

    # Longest unbroken stretch of concentration, in bars.
    longest = run = 0
    for flag in over.to_numpy():
        run = run + 1 if flag else 0
        longest = max(longest, run)

    active = int(held.sum())
    at_peak = int((per_bar >= peak - tol).sum()) if peak > nominal + tol else 0
    return {
        "nominal": nominal,
        "max_short": max_short,
        "max_long": max_long,
        "bars_held": active,
        "concentrated_bars": int((over & held).sum()),
        "share": float((over & held).sum() / active) if active else 0.0,
        "bars_at_peak": at_peak,
        "share_at_peak": float(at_peak / active) if active else 0.0,
        "longest_run": longest,
    }


def tail_curve(positions: pd.DataFrame, panel,
               thresholds=(0.25, 0.50, 0.65, 1.00, 2.00)) -> pd.DataFrame:
    """Expected tail hits across move sizes, with what each would cost.

    Two questions the single-threshold census could not answer.

    **How often?** A rate estimated at one move size is one point on a steeply
    falling curve, and which point you pick decides whether the answer sounds
    reassuring. Sweeping removes the choice — the same reason `nullres sweep`
    prints a surface instead of its maximum.

    **How bad?** The graveyard works this out by hand: "one UNFI-type event
    (+274% in 4h) against a -0.5 weight is -137% of capital". That arithmetic is
    mechanical and belongs in code. `cost_of_one` multiplies each move by the
    largest short weight the book actually held, so the loss is the book's own
    rather than an illustration, and `ruinous` marks the levels that would take
    more than all of it.

    This is as far as tail risk can honestly be taken here. It says how exposed
    the book was and what one hit would have cost — it does not model margin,
    liquidation price, or auto-deleveraging, none of which are in the archive.
    """
    ret = panel.ret_next.reindex(positions.index)[positions.columns]
    simple = np.expm1(ret)
    observed = int(simple.notna().sum().sum())
    short_bars = int((positions < 0).sum().sum())
    worst_short = float(-positions[positions < 0].min().min()) if short_bars else 0.0

    rows = []
    for level in thresholds:
        moves = int((simple > level).sum().sum())
        hits = int(((simple.where(positions < 0)) > level).sum().sum())
        loss = level * worst_short

        # A move size never observed does NOT have probability zero, and
        # reporting 0.00 expected hits for the level that would ruin the book is
        # the error this whole file exists to catch — concluding absence from
        # non-observation. With no events in `observed` trials the rate is
        # unknown but bounded: the rule of three puts its 95% upper limit at
        # 3/observed. Rows with no occurrences therefore carry that bound, and
        # `estimated` marks which is which.
        estimated = moves > 0
        rate = (moves / observed) if estimated else (3.0 / observed if observed else 0.0)
        rows.append({
            "move": level,
            "occurrences": moves,
            "one_in": (observed / moves) if moves else (observed / 3.0),
            "expected_hits": rate * short_bars,
            "actual_hits": hits,
            "cost_of_one": loss,
            "ruinous": loss >= 1.0,
            "estimated": estimated,
        })
    out = pd.DataFrame(rows)
    out.attrs["short_name_bars"] = short_bars
    out.attrs["worst_short_weight"] = worst_short
    out.attrs["observed_bars"] = observed
    return out


def format_report(panel, cfg, proba, positions, mean_auc: float,
                  min_obs: int = 200, nominal_weight: float | None = None) -> str:
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

    curve = tail_curve(positions, panel)
    census = tail_census(positions, panel)
    short_bars = curve.attrs["short_name_bars"]
    weight = curve.attrs["worst_short_weight"]

    lines += [
        "",
        f"  tail exposure — book held {short_bars:,} short-name-bars across "
        f"{curve.attrs['observed_bars']:,} observed,",
        f"  at a largest short weight of {weight:.2f} per name. Worst bar "
        f"actually suffered: {census['worst_bar_return']:.1%}",
    ]

    if nominal_weight:
        conc = concentration(positions, nominal_weight)
        if conc["max_short"] > conc["nominal"] * 1.000001:
            lines += [
                "",
                f"    CONCENTRATION: nominal weight is {conc['nominal']:.2f} per "
                f"name, but a delisted leg leaves the",
                f"    survivor rescaled to keep the book dollar-neutral — peaking "
                f"at {conc['max_short']:.2f} short "
                f"({conc['max_long']:.2f} long).",
                f"    Concentrated on {conc['concentrated_bars']:,} of "
                f"{conc['bars_held']:,} bars held ({conc['share']:.1%}), "
                f"{conc['share_at_peak']:.1%} of them at the peak,",
                f"    longest unbroken stretch {conc['longest_run']:,} bars.",
                f"    Gross exposure never changes; the move that ruins the book "
                f"halves, from "
                f"+{1 / conc['nominal'] * 100:.0f}% to "
                f"+{1 / conc['max_short'] * 100:.0f}%.",
            ]
        else:
            lines.append(f"\n    Concentration: never exceeded the nominal "
                         f"{conc['nominal']:.2f} per name.")

    lines += [
        "",
        f"    {'move':>7}{'occurred':>10}{'1 in':>12}{'expected':>10}"
        f"{'actual':>8}{'costs':>9}",
    ]
    for _, row in curve.iterrows():
        bound = "" if row["estimated"] else "<"
        one_in = (f"{row['one_in']:,.0f}" if row["estimated"]
                  else f">{row['one_in']:,.0f}")
        flag = "  <- RUIN" if row["ruinous"] else ""
        lines.append(
            f"    {row['move']:>6.0%}{int(row['occurrences']):>10,}{one_in:>12}"
            f"{bound + format(row['expected_hits'], '.2f'):>10}"
            f"{int(row['actual_hits']):>8}{row['cost_of_one']:>8.0%}{flag}"
        )
    if not curve["estimated"].all():
        lines.append("")
        lines.append("    '<' marks a move size never observed here. Its rate is "
                     "not zero — it is unknown,")
        lines.append(f"    bounded above by the rule of three (3 events in "
                     f"{curve.attrs['observed_bars']:,} observations).")

    lines.append("")
    ruin = curve[curve["ruinous"]]
    if len(ruin):
        smallest = ruin.iloc[0]
        lines.append(
            f"    A single +{smallest['move']:.0%} move against the largest short "
            f"would cost {smallest['cost_of_one']:.0%} of capital — more than all "
            f"of it."
        )
        if smallest["estimated"]:
            lines.append(
                f"    Chance predicted {smallest['expected_hits']:.2f} such hits "
                f"and {int(smallest['actual_hits'])} occurred, so surviving is "
                f"what the exposure predicts,\n    not evidence the risk was "
                f"absent."
            )
        else:
            lines.append(
                f"    No move that large occurred here, so its rate is not "
                f"measured at all — only bounded\n    ABOVE, at most "
                f"{smallest['expected_hits']:.2f} expected hits. This sample "
                f"cannot show you this risk; it can\n    only fail to. UNFI did "
                f"+274% in a single 4h bar in 2021."
            )
    lines.append("    The tail is UNTESTED, not absent. The engine models no "
                 "margin, so a ruinous")
    lines.append("    bar would show as a large negative return rather than a "
                 "liquidation.")
    return "\n".join(lines)
