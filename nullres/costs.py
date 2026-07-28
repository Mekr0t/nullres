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
    """E|return| over `hold_bars` for a driftless random walk."""
    return sigma_per_bar * np.sqrt(hold_bars) * SQRT_2_OVER_PI


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
                 accuracies=(0.51, 0.52, 0.55, 0.60)) -> str:
    """Render the two tables that should precede any modelling work.

    `hours_per_bar` converts break-even BARS into wall-clock duration, and it is
    the whole point of the second table — duration is what is invariant across
    timeframes, bars are not. Hardcoding 24 bars-to-a-day (i.e. assuming hourly)
    made `nullres budget` claim a 1d config broke even in 0.9 days when the
    honest answer is the same ~21 days it is at every other timeframe. Callers
    pass `8760 / bars_per_year`.
    """
    cost = round_trip_cost(fee_bps, slippage_bps)
    lines = [
        f"per-bar volatility      {sigma_per_bar:.4%}",
        f"round-trip cost         {cost:.4%}   ({fee_bps + slippage_bps:.0f}bps/side)",
        "",
        "Accuracy needed to break even, by holding period:",
        f"  {'hold (bars)':<14}{'E|move|':>10}{'accuracy':>12}",
    ]
    for h in holds:
        move = expected_abs_move(sigma_per_bar, h)
        acc = required_accuracy(sigma_per_bar, h, fee_bps, slippage_bps)
        verdict = f"{acc:>11.1%}" if acc <= 1.0 else "  impossible"
        lines.append(f"  {h:<14,}{move:>10.2%}{verdict}")

    lines += ["", "Holding period needed to break even, by accuracy:",
              f"  {'accuracy':<14}{'hold (bars)':>14}{'~duration':>14}"]
    for acc in accuracies:
        h = breakeven_hold(sigma_per_bar, acc, fee_bps, slippage_bps)
        lines.append(f"  {acc:<14.0%}{h:>14,.0f}"
                     f"{format_duration(h * hours_per_bar):>14}")
    return "\n".join(lines)
