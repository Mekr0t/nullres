"""Performance statistics, including the ones that are inconvenient.

Total return and Sharpe are the numbers people quote. The ones that decide
whether a strategy is real are further down this list: the t-statistic on the
mean return, how much of gross profit the costs ate, and how the Sharpe holds
up once you account for how many variants you tried before this one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summarize(result, bars_per_year: int, n_trials: int = 1) -> dict:
    """Return a flat dict of performance statistics.

    Args:
        n_trials: how many strategy variants were evaluated before reporting
            this one. Used for the deflated Sharpe ratio. Be honest here — the
            count includes every threshold, horizon, and feature set you tried,
            not just the ones you kept.
    """
    r = result.returns.astype("float64")
    n = len(r)
    years = n / bars_per_year
    equity = result.equity

    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(r.std() * np.sqrt(bars_per_year))
    sharpe = float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else 0.0

    downside = r[r < 0]
    sortino = (
        float(r.mean() / downside.std() * np.sqrt(bars_per_year))
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )

    dd = _drawdown(equity)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Is the mean return distinguishable from zero at all?
    t_stat = float(r.mean() / (r.std() / np.sqrt(n))) if r.std() > 0 and n > 1 else 0.0
    p_value = float(2 * (1 - sps.norm.cdf(abs(t_stat)))) if t_stat else 1.0

    gross_total = float(np.exp(result.gross.sum()) - 1.0)
    cost_log = result.total_cost
    exposure = float((result.position.abs() > 1e-12).mean())
    active = r[result.position.shift(1).fillna(0.0).abs() > 1e-12]
    hit_rate = float((active > 0).mean()) if len(active) else 0.0

    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "deflated_sharpe": deflated_sharpe(sharpe, n, bars_per_year, n_trials),
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "t_stat": t_stat,
        "p_value": p_value,
        "gross_return": gross_total,
        "cost_drag": float(1 - np.exp(-cost_log)),
        "n_trades": result.n_trades,
        "turnover_per_year": float(result.turnover.sum() / years) if years > 0 else 0.0,
        "exposure": exposure,
        "hit_rate": hit_rate,
        "bars": n,
        "years": years,
    }


def deflated_sharpe(sharpe: float, n_obs: int, bars_per_year: int,
                    n_trials: int = 1) -> float:
    """Sharpe adjusted for the number of variants tried (Bailey & López de Prado).

    Searching 100 strategy variants on pure noise yields a best-of-100 Sharpe
    around 0.6 by luck alone. This subtracts the Sharpe you would expect to
    reach by chance given `n_trials`, so the remainder is what needs explaining.
    A deflated Sharpe at or below zero means: you found nothing, you just looked
    a lot of times.
    """
    if n_trials <= 1 or n_obs < 2:
        return float(sharpe)
    euler = 0.5772156649
    # Expected maximum of n_trials standard normals.
    e_max = (
        (1 - euler) * sps.norm.ppf(1 - 1 / n_trials)
        + euler * sps.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    # Convert to the same annualised units as `sharpe`.
    return float(sharpe - e_max * np.sqrt(bars_per_year / n_obs))


def by_period(result, bars_per_year: int, mask=None, freq: str = "YE") -> pd.DataFrame:
    """Break performance down by calendar period.

    A strategy with a Sharpe of 0.5 built from one spectacular year and four
    flat ones is not the same object as one that earned 0.5 every year, and the
    headline number cannot tell them apart. This is the cheapest test for
    "did I fit a regime that has since ended".

    Args:
        mask: restrict to these bars (pass the out-of-sample mask; bars outside
            it contribute structural zeros that deflate the volatility estimate
            and inflate Sharpe).
    """
    r = result.returns
    turnover = result.turnover
    if mask is not None:
        r, turnover = r[mask], turnover[mask]

    rows = []
    for period, chunk in r.groupby(pd.Grouper(freq=freq)):
        if len(chunk) < 2 or chunk.std() == 0:
            continue
        trades = int((turnover.loc[chunk.index] > 1e-12).sum())
        rows.append({
            "period": str(period)[:4] if freq.startswith("Y") else str(period)[:10],
            "bars": len(chunk),
            "total_return": float(np.exp(chunk.sum()) - 1.0),
            "sharpe": float(chunk.mean() / chunk.std() * np.sqrt(bars_per_year)),
            "n_trades": trades,
        })
    return pd.DataFrame(rows)


# (metric key, column label, column width, formatter)
COLUMNS = [
    ("total_return", "total",  11, lambda v: _pct_or_sci(v)),
    ("cagr",         "cagr",    9, lambda v: f"{v:.1%}"),
    ("sharpe",       "sharpe",  8, lambda v: f"{v:.2f}"),
    ("max_dd",       "max dd",  9, lambda v: f"{v:.1%}"),
    ("calmar",       "calmar",  8, lambda v: f"{v:.2f}"),
    ("t_stat",       "t-stat",  8, lambda v: f"{v:.2f}"),
    ("n_trades",     "trades",  9, lambda v: f"{v:,.0f}"),
    ("cost_drag",    "cost",    8, lambda v: f"{v:.1%}"),
    ("exposure",     "expo",    7, lambda v: f"{v:.0%}"),
]


def _pct_or_sci(v: float) -> str:
    """Percentages are unreadable past a few thousand percent."""
    return f"{v:.1%}" if abs(v) < 100 else f"{v + 1:.2e}x"


def format_table(rows: dict[str, dict]) -> str:
    """Render {strategy_name: metrics} as a fixed-width comparison table."""
    name_w = max([len(k) for k in rows] + [8]) + 2
    head = f"{'strategy':<{name_w}}" + "".join(
        f"{label:>{w}}" for _, label, w, _ in COLUMNS
    )
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        cells = "".join(
            f"{fmt(m.get(key, 0.0)):>{w}}" for key, _, w, fmt in COLUMNS
        )
        lines.append(f"{name:<{name_w}}" + cells)
    return "\n".join(lines)
