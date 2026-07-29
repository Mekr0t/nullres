"""The cost budget: what accuracy does a strategy actually need to survive?

This is the arithmetic that decides whether a research direction is worth
pursuing, and almost nobody does it before spending a month on features.

For a directional strategy:

    edge            = 2 * accuracy - 1        (fraction of moves called right)
    E|move| over h  ~ sigma * sqrt(h) * sqrt(2/pi)   for a driftless walk
    gross per trade = edge * E|move|
    cost per trade  = 2 * (fee + slippage)    (round trip, both sides)

Break-even requires gross >= cost. Rearranged, that gives the minimum accuracy
a strategy needs at a given holding period — or the minimum holding period it
needs at a given accuracy.

The result for hourly BTC is bracing. At 51% accuracy and 12bps a side, no
holding period under several weeks breaks even. That is not pessimism, it is
division, and it is why the honest baseline lost 100%: it was not a modelling
failure, it was an arithmetic one.
"""

from __future__ import annotations

import numpy as np

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def expected_abs_move(sigma_per_bar: float, hold_bars: float) -> float:
    """E|return| over `hold_bars` for a driftless GAUSSIAN random walk.

    Crypto returns are neither Gaussian nor driftless, and the error is not a
    constant you can divide out — it changes sign with the horizon. Measured on
    BTC, this formula overstates the typical move by ~25% at one bar and
    understates it by ~13% at 336 bars: fat tails inflate sigma relative to a
    typical short move, while drift and trending make long moves larger than a
    driftless walk predicts. Aggregation pulls the middle toward normal.

    Prefer `empirical_abs_move` when you have the return series. This is kept
    because it is invertible in closed form, which `breakeven_hold` relies on,
    and because it is the right null when you are reasoning rather than
    measuring.
    """
    return sigma_per_bar * np.sqrt(hold_bars) * SQRT_2_OVER_PI


def empirical_abs_move(logret: np.ndarray, hold_bars: int) -> float:
    """Measured mean |return| over `hold_bars`, from the returns themselves.

    No distributional assumption: it sums the actual overlapping windows. The
    windows overlap, so this is an estimate of the mean and not an independent
    sample — fine for the purpose, which is a break-even threshold rather than
    an inference.
    """
    r = np.asarray(logret, dtype="float64")
    r = r[np.isfinite(r)]
    hold_bars = int(hold_bars)
    if hold_bars < 1 or len(r) <= hold_bars:
        return float("nan")
    cumulative = np.concatenate([[0.0], np.cumsum(r)])
    windows = cumulative[hold_bars:] - cumulative[:-hold_bars]
    return float(np.abs(windows).mean())


def breakeven_hold_empirical(logret: np.ndarray, accuracy: float,
                             fee_bps: float, slippage_bps: float,
                             max_hold: int = 8_760) -> float:
    """`breakeven_hold` against measured moves instead of the Gaussian one.

    No closed form is available once the move is measured rather than modelled,
    so this bisects. `empirical_abs_move` rises monotonically with the horizon,
    which is what makes that valid.
    """
    edge = 2.0 * accuracy - 1.0
    if edge <= 0:
        return float("inf")
    cost = round_trip_cost(fee_bps, slippage_bps)

    def gross(h: int) -> float:
        move = empirical_abs_move(logret, h)
        return edge * move if np.isfinite(move) else float("inf")

    if gross(1) >= cost:
        return 1.0
    if gross(max_hold) < cost:
        return float("inf")

    lo, hi = 1, max_hold
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gross(mid) < cost:
            lo = mid
        else:
            hi = mid
    return float(hi)


def round_trip_cost(fee_bps: float, slippage_bps: float) -> float:
    """Cost of getting in and back out again, as a fraction."""
    return 2.0 * (fee_bps + slippage_bps) / 10_000.0


def required_accuracy(sigma_per_bar: float, hold_bars: float,
                      fee_bps: float, slippage_bps: float) -> float:
    """Directional accuracy needed to break even. Returns >1.0 when impossible."""
    move = expected_abs_move(sigma_per_bar, hold_bars)
    if move <= 0:
        return float("inf")
    edge = round_trip_cost(fee_bps, slippage_bps) / move
    return 0.5 * (1.0 + edge)


def breakeven_hold(sigma_per_bar: float, accuracy: float,
                   fee_bps: float, slippage_bps: float) -> float:
    """Bars a position must be held for a given accuracy to break even."""
    edge = 2.0 * accuracy - 1.0
    if edge <= 0:
        return float("inf")
    # edge * sigma * sqrt(h) * sqrt(2/pi) = cost  ->  solve for h
    return (round_trip_cost(fee_bps, slippage_bps)
            / (edge * sigma_per_bar * SQRT_2_OVER_PI)) ** 2


def format_duration(hours: float) -> str:
    """Wall-clock rendering of a holding period, in units a human can act on."""
    if not np.isfinite(hours):
        return "never"
    if hours < 48:
        return f"{hours:.1f} hours"
    days = hours / 24.0
    return f"{days:.1f} days" if days < 90 else f"{days / 30.4:.1f} months"


def budget_table(sigma_per_bar: float, fee_bps: float, slippage_bps: float,
                 hours_per_bar: float = 1.0,
                 holds=(1, 6, 12, 24, 72, 168, 336, 720),
                 accuracies=(0.51, 0.52, 0.55, 0.60),
                 logret=None) -> str:
    """Render the two tables that should precede any modelling work.

    `hours_per_bar` converts break-even BARS into wall-clock duration, and it is
    the whole point of the second table — duration is what is invariant across
    timeframes, bars are not. Hardcoding 24 bars-to-a-day (i.e. assuming hourly)
    made `nullres budget` claim a 1d config broke even in 0.9 days when the
    honest answer is the same ~21 days it is at every other timeframe. Callers
    pass `8760 / bars_per_year`.

    `logret` is the actual per-bar log return series. Given it, the table adds a
    measured column beside every modelled one — because the Gaussian assumption
    is wrong in a direction that flatters the strategy at exactly the holding
    periods people are tempted by. See `expected_abs_move`.
    """
    cost = round_trip_cost(fee_bps, slippage_bps)
    measured = logret is not None and len(np.asarray(logret)) > max(holds)

    lines = [
        f"per-bar volatility      {sigma_per_bar:.4%}",
        f"round-trip cost         {cost:.4%}   ({fee_bps + slippage_bps:.0f}bps/side)",
        "",
        "Accuracy needed to break even, by holding period:",
    ]
    if measured:
        lines.append(f"  {'hold (bars)':<14}{'E|move|':>10}{'measured':>11}"
                     f"{'accuracy':>12}{'measured':>11}")
    else:
        lines.append(f"  {'hold (bars)':<14}{'E|move|':>10}{'accuracy':>12}")

    for h in holds:
        move = expected_abs_move(sigma_per_bar, h)
        acc = required_accuracy(sigma_per_bar, h, fee_bps, slippage_bps)
        cell = f"{acc:>11.1%}" if acc <= 1.0 else "  impossible"
        if not measured:
            lines.append(f"  {h:<14,}{move:>10.2%}{cell}")
            continue
        real = empirical_abs_move(logret, h)
        real_acc = 0.5 * (1.0 + cost / real) if real > 0 else float("inf")
        real_cell = f"{real_acc:>10.1%}" if real_acc <= 1.0 else " impossible"
        lines.append(f"  {h:<14,}{move:>10.2%}{real:>11.2%}{cell}{real_cell}")

    lines += ["", "Holding period needed to break even, by accuracy:"]
    if measured:
        lines.append(f"  {'accuracy':<14}{'hold (bars)':>14}{'measured':>11}"
                     f"{'~duration':>16}")
    else:
        lines.append(f"  {'accuracy':<14}{'hold (bars)':>14}{'~duration':>14}")

    for acc in accuracies:
        h = breakeven_hold(sigma_per_bar, acc, fee_bps, slippage_bps)
        if not measured:
            lines.append(f"  {acc:<14.0%}{h:>14,.0f}"
                         f"{format_duration(h * hours_per_bar):>14}")
            continue
        real_h = breakeven_hold_empirical(logret, acc, fee_bps, slippage_bps)
        duration = (format_duration(real_h * hours_per_bar)
                    if np.isfinite(real_h) else "never")
        real_cell = f"{real_h:>11,.0f}" if np.isfinite(real_h) else "      never"
        lines.append(f"  {acc:<14.0%}{h:>14,.0f}{real_cell}{duration:>16}")

    if measured:
        lines += [
            "",
            "  'measured' uses the actual distribution of moves rather than a",
            "  Gaussian. It is the column to trust: fat tails make short moves",
            "  SMALLER than sigma implies, so the modelled accuracy is too",
            "  forgiving at short holds, while drift makes long moves larger.",
            "  ~duration is the measured hold at this config's bar size.",
        ]
    return "\n".join(lines)
