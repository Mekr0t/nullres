"""Cross-sectional (relative-value) research on a panel of symbols.

Everything else in this repo asks "will BTC go up?". That question has to clear
a 24bps round trip out of a single asset's own move, and six lines of attack
have now died against that wall (docs/05-graveyard.md).

This asks a different question: **"which of these assets will outperform the
others?"** It is easier in three specific ways.

  1. The market move cancels. A dollar-neutral book is not betting on crypto
     going up, so it does not need to out-predict a 60% annualised drift.
  2. The label is balanced by construction — half the universe beats the
     median at every timestamp, in every regime.
  3. Errors are relative. Being wrong about BTC and wrong about ETH in the same
     direction costs nothing; only the ranking matters.

What it costs you is a new failure mode: **survivorship bias**. Choosing a
universe by looking at what is liquid today is a test of "assets that survived",
and it will produce a beautiful, meaningless equity curve. The universe here is
fixed as of 2021-12 and includes LUNAUSDT, which went to zero in May 2022 and
was delisted. If your cross-sectional backtest cannot lose money on LUNA, it is
not measuring anything.

Long/short also requires PERPETUAL FUTURES — you cannot short spot — so this
module uses USD-M perp bars and charges the 8-hourly funding rate on every
position held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nullres.data import load_auxiliary
from nullres.data.binance import load_binance
from nullres.features import build_features

# The universe as it looked in December 2021, NOT today. LUNAUSDT is in it
# because it was, and because a survivorship-honest test needs the corpses.
UNIVERSE_2021_12 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "AVAXUSDT", "DOTUSDT",
    "LUNAUSDT",
]

_INTERVAL_HOURS = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}


@dataclass
class Panel:
    """A tidy panel: MultiIndex (ts, symbol) features, plus per-bar returns."""

    features: pd.DataFrame          # MultiIndex (ts, symbol)
    y: pd.Series                    # 1 if the symbol beats the cross-sectional median
    ret_next: pd.DataFrame          # ts x symbol, open[t+1] -> open[t+2] log return
    funding: pd.DataFrame           # ts x symbol, funding cost per bar for a long
    times: pd.DatetimeIndex
    horizon: int
    symbols: list[str] = field(default_factory=list)
    delisted: dict[str, pd.Timestamp] = field(default_factory=dict)


def load_panel(cfg, symbols: list[str] | None = None, verbose: bool = True,
               top_n: int | None = None, screen_window: int = 180) -> Panel:
    """Load symbols, screen for liquidity, build features, stack into a panel.

    Args:
        top_n: keep only the top-N symbols by TRAILING dollar volume at each
            bar. Without this a wide universe ranks BTC against coins that
            traded $50k a day, which is a ranking you could not act on.
            Screening on full-sample volume instead would be lookahead — it
            selects the coins that went on to matter.
    """
    symbols = symbols or UNIVERSE_2021_12
    d = cfg.data
    interval_hours = _INTERVAL_HOURS.get(d.interval, 4)

    # Pass 1: bars only. Cheap enough to hold the whole universe in memory,
    # which is what lets the screen be computed before features are built.
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    delisted: dict[str, pd.Timestamp] = {}
    for symbol in symbols:
        bars = load_binance(symbol, d.interval, d.start, d.end, d.cache_dir,
                            verbose=False, market="um", required=False)
        if bars is None or len(bars) < 500:
            continue
        bars_by_symbol[symbol] = bars
        if bars.index[-1] < pd.Timestamp(d.end) - pd.Timedelta(days=60):
            delisted[symbol] = bars.index[-1]

    if len(bars_by_symbol) < 3:
        raise SystemExit("need at least 3 symbols with data for a cross-section")

    times = pd.DatetimeIndex(
        sorted(set().union(*(b.index for b in bars_by_symbol.values())))
    )
    log_open = pd.DataFrame(
        {s: np.log(b["open"]) for s, b in bars_by_symbol.items()}
    ).reindex(times)

    screen = None
    if top_n:
        from nullres.data.universe import liquidity_screen

        dollar_volume = pd.DataFrame(
            {s: b["volume"] * b["close"] for s, b in bars_by_symbol.items()}
        ).reindex(times)
        screen = liquidity_screen(dollar_volume, top_n=top_n, window=screen_window)
        # Symbols never selected cannot influence any decision, so skip the
        # cost of featurising them. This is an optimisation, not a filter:
        # the time-varying screen below is what actually gates membership.
        keep = [s for s in bars_by_symbol if screen[s].any()]
        if verbose:
            print(f"  liquidity screen: top {top_n} by {screen_window}-bar "
                  f"trailing volume; {len(keep)} of {len(bars_by_symbol)} "
                  f"symbols ever qualify")
        bars_by_symbol = {s: bars_by_symbol[s] for s in keep}
        screen = screen[keep]
        log_open = log_open[keep]

    _guard_metrics_fetch(d, list(bars_by_symbol))

    # Pass 2: features, only for symbols that can ever be traded.
    per_symbol: dict[str, pd.DataFrame] = {}
    funding_cols: dict[str, pd.Series] = {}
    for symbol, bars in bars_by_symbol.items():
        sym_cfg = _symbol_cfg(d, symbol)
        fund, metrics = load_auxiliary(sym_cfg, verbose=False, bars=bars)
        per_symbol[symbol] = build_features(bars, funding=fund, metrics=metrics)

        if fund is not None and len(fund):
            # Funding settles every 8h; a bar of `interval_hours` carries that
            # fraction of one settlement. A long pays when the rate is positive.
            from nullres.features.derivatives import _asof

            rate = _asof(bars, fund, ["funding_rate"])["funding_rate"]
            funding_cols[symbol] = rate * (interval_hours / 8.0)
        else:
            funding_cols[symbol] = pd.Series(0.0, index=bars.index)

        if verbose and len(bars_by_symbol) <= 15:
            print(f"  {symbol:<12} {len(bars):>6,} bars  "
                  f"{bars.index[0]:%Y-%m-%d}..{bars.index[-1]:%Y-%m-%d}"
                  + ("   DELISTED" if symbol in delisted else ""))

    # open[t+1] -> open[t+2], per symbol. Same convention as the single-asset
    # engine: decided at close of t, filled at open of t+1.
    ret_next = log_open.shift(-2) - log_open.shift(-1)
    if screen is not None:
        # Un-screened symbols are not tradable, so they must not contribute
        # returns, ranks, or peer-group medians.
        ret_next = ret_next.where(screen)

    funding = pd.DataFrame(funding_cols).reindex(times).fillna(0.0)

    frame = pd.concat(
        {s: f.reindex(times) for s, f in per_symbol.items()},
        names=["symbol", "ts"],
    ).swaplevel().sort_index()

    ranked = _cross_sectional_rank(frame, screen)
    y = _relative_label(log_open.where(screen) if screen is not None else log_open,
                        cfg.label.horizon)

    common = ranked.index.intersection(y.index)
    return Panel(
        features=ranked.loc[common],
        y=y.loc[common],
        ret_next=ret_next,
        funding=funding,
        times=times,
        horizon=cfg.label.horizon,
        symbols=list(per_symbol),
        delisted={s: t for s, t in delisted.items() if s in per_symbol},
    )


def _guard_metrics_fetch(data_cfg, symbols: list[str],
                         free_limit: int = 15) -> None:
    """Refuse to silently start a multi-hour download.

    Open-interest metrics are published ONE FILE PER DAY, so a wide panel needs
    roughly `symbols x months x 30` requests — about 100,000 for 70 symbols over
    four years, which is four hours of quiet network traffic with no output.

    Funding and klines are monthly and cheap; only metrics have this cliff. The
    check counts what is already cached, so re-runs and gradual backfills are
    unaffected.
    """
    from pathlib import Path

    if not getattr(data_cfg, "metrics", False):
        return

    cache = Path(data_cfg.cache_dir)
    cached = {p.name.split("-metrics-")[0] for p in cache.glob("*-metrics-*.parquet")}
    missing = [s for s in symbols if s not in cached]
    if len(missing) <= free_limit:
        return

    months = len(pd.date_range(data_cfg.start, data_cfg.end, freq="MS"))
    requests_needed = len(missing) * months * 30
    raise SystemExit(
        f"\n{len(missing)} of {len(symbols)} symbols have no cached open-interest "
        f"metrics.\nFetching them needs roughly {requests_needed:,} requests "
        f"(~{requests_needed / 25_000:.0f} hours), because Binance publishes "
        f"metrics one file per day.\n\n"
        f"Either run without them:\n"
        f"    --set data.metrics=false\n"
        f"or pre-cache deliberately in the background first. Refusing rather "
        f"than starting a silent multi-hour download."
    )


def _symbol_cfg(data_cfg, symbol: str):
    import copy

    out = copy.deepcopy(data_cfg)
    out.symbol = symbol
    return out


def _cross_sectional_rank(frame: pd.DataFrame,
                          screen: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replace each feature with its percentile rank WITHIN each timestamp.

    This is the transformation that makes the panel comparable. A raw RSI of 70
    means different things in different regimes; "the 3rd most overbought of 11
    right now" means the same thing always. It also removes the market-wide
    component of every feature for free — when everything sells off together,
    every symbol's return z-score drops, but their ranks do not move.

    Uses only data at time t across symbols, so it is point-in-time safe.

    When a liquidity `screen` is supplied, ranks are computed only among the
    tradable symbols at that bar. Ranking against illiquid names would produce
    an ordering you cannot act on, and would shift every percentile as coins
    drift in and out of the archive.
    """
    if screen is not None:
        mask = screen.stack(future_stack=True)
        mask.index.names = ["ts", "symbol"]
        frame = frame.where(mask.reindex(frame.index).fillna(False), other=np.nan)

    ranked = frame.groupby(level="ts").rank(pct=True)
    # Timestamps with too few live symbols cannot support a ranking.
    live = frame.notna().any(axis=1).groupby(level="ts").sum()
    usable = live[live >= 3].index
    return ranked[ranked.index.get_level_values("ts").isin(usable)]


def _relative_label(log_open: pd.DataFrame, horizon: int) -> pd.Series:
    """1 if the symbol's forward return beats the cross-sectional median.

    Balanced by construction: at every timestamp roughly half the universe is
    above its own median, in a bull market and a crash alike. Compare with the
    single-asset labels, whose base rate drifts with the regime.

    The return runs open[t+1] -> open[t+1+horizon], matching the execution
    convention: you cannot trade at the close you are predicting from.
    """
    fwd = log_open.shift(-(1 + horizon)) - log_open.shift(-1)
    median = fwd.median(axis=1)
    relative = fwd.sub(median, axis=0)

    y = (relative > 0).astype("float64")
    y = y.where(relative.notna())
    stacked = y.stack(future_stack=True)
    stacked.index.names = ["ts", "symbol"]
    return stacked.dropna()


# ---------------------------------------------------------------------------
# Splitting: folds are over TIME, and every symbol at a timestamp moves together
# ---------------------------------------------------------------------------

def time_folds(times: pd.DatetimeIndex, cfg_split, horizon: int):
    """Yield (train_times, test_times). Purged by the label horizon.

    Splitting on row position would be wrong here: rows are (ts, symbol) pairs,
    so a positional split would put BTC's Tuesday in training and ETH's Tuesday
    in test. The model would learn the day from ten correlated siblings.
    """
    n = len(times)
    if n <= cfg_split.min_train:
        raise ValueError(
            f"only {n:,} timestamps but min_train is {cfg_split.min_train:,}"
        )
    fold = (n - cfg_split.min_train) // cfg_split.n_folds
    if fold < 50:
        raise ValueError("folds too small; reduce n_folds or widen the range")

    for k in range(cfg_split.n_folds):
        test_start = cfg_split.min_train + k * fold
        test_end = min(test_start + fold, n)
        if test_end - test_start < 50:
            break
        train_hi = test_start - horizon - cfg_split.embargo
        if train_hi < 50:
            continue
        lo = max(0, train_hi - cfg_split.train_window) if cfg_split.scheme == "rolling" else 0
        yield times[lo:train_hi], times[test_start:test_end]


def fit_predict_panel(panel: Panel, cfg, verbose: bool = True):
    """Walk-forward P(outperforms) for every (ts, symbol) in a test fold."""
    from nullres.models.classifier import make_model
    from sklearn.metrics import roc_auc_score

    X, y = panel.features, panel.y
    ts_level = X.index.get_level_values("ts")
    proba = pd.Series(np.nan, index=X.index, dtype="float64")
    reports = []

    for k, (train_times, test_times) in enumerate(
        time_folds(panel.times, cfg.split, panel.horizon), start=1
    ):
        train_mask = ts_level.isin(train_times)
        test_mask = ts_level.isin(test_times)
        if train_mask.sum() < 500 or test_mask.sum() < 100:
            continue

        model = make_model(cfg.model)
        model.fit(X[train_mask], y[train_mask].astype(int))
        p = model.predict_proba(X[test_mask])[:, 1]
        proba[test_mask] = p

        y_test = y[test_mask].astype(int)
        auc = roc_auc_score(y_test, p) if y_test.nunique() == 2 else float("nan")
        reports.append({"fold": k, "train": int(train_mask.sum()),
                        "test": int(test_mask.sum()), "auc": float(auc),
                        "test_from": str(test_times[0])[:10],
                        "test_to": str(test_times[-1])[:10]})
        if verbose:
            print(f"  fold {k}: train {train_mask.sum():>7,}  test {test_mask.sum():>6,}"
                  f"  [{reports[-1]['test_from']}..{reports[-1]['test_to']}]"
                  f"  auc {auc:.4f}")

    if not reports:
        raise RuntimeError("no fold produced predictions")
    return proba, reports


# ---------------------------------------------------------------------------
# Long/short book construction and backtest
# ---------------------------------------------------------------------------

def panel_positions(proba: pd.Series, panel: Panel, top_k: int = 3,
                    rebalance: int = 42, allow_short: bool = True) -> pd.DataFrame:
    """Long the top-k ranked symbols, short the bottom-k. Dollar neutral.

    Weights are +1/k and -1/k so gross exposure is 2 and net is 0 — the book
    makes no bet on the market's direction, which is the entire point.

    **That gross of 2.0 is deliberate, and `sizing.max_leverage` does not
    constrain it.** That field clips a single-asset position to +/-1 and is
    never read here; a dollar-neutral book is 100% long and 100% short by
    definition, and capping gross at 1.0 would mean holding half of each side,
    which is a different strategy rather than a safer version of this one. But
    2x gross notional is 2x notional whatever the net is: it requires margin,
    and it is why the tail census in `panelaudit` matters. `summarize` reports
    `gross_exposure` so this is visible in the results table instead of being
    inferable only from reading this function.

    `rebalance` throttles turnover exactly as `min_hold` does in the
    single-asset engine. Reshuffling an 11-symbol book every bar is the
    cross-sectional version of the mistake that cost the original baseline 100%.
    """
    wide = proba.unstack("symbol").reindex(panel.times)
    positions = pd.DataFrame(0.0, index=panel.times, columns=wide.columns)

    current = pd.Series(0.0, index=wide.columns)
    for i, ts in enumerate(panel.times):
        if i % rebalance == 0:
            row = wide.loc[ts].dropna()
            if len(row) >= 2 * top_k:
                ranked = row.sort_values(ascending=False)
                current = pd.Series(0.0, index=wide.columns)
                current[ranked.index[:top_k]] = 1.0 / top_k
                if allow_short:
                    current[ranked.index[-top_k:]] = -1.0 / top_k
        # A delisted symbol cannot be held: force it flat so the book does not
        # silently carry a position in an instrument that stopped existing.
        alive = panel.ret_next.loc[ts].notna()
        positions.loc[ts] = _neutralise(current.where(alive, 0.0), allow_short)

    return positions


def _neutralise(weights: pd.Series, allow_short: bool = True) -> pd.Series:
    """Rescale each side to unit gross so the book stays dollar neutral.

    Needed because symbols die mid-book. If a shorted symbol delists, zeroing
    its leg leaves the remainder NET LONG — and it stays that way until the next
    rebalance, quietly turning a market-neutral strategy into a directional one
    at exactly the moment the market is disorderly. That is not a rounding
    error: it is how a "market-neutral" book ends up holding all of the beta and
    none of the intent.
    """
    long_gross = float(weights[weights > 0].sum())
    short_gross = float(-weights[weights < 0].sum())

    if not allow_short:
        if long_gross <= 0:
            return pd.Series(0.0, index=weights.index)
        return weights / long_gross

    # A long/short book that has lost an entire side cannot be neutralised.
    # Going flat is the only honest option — the alternative is running naked
    # directional risk under a market-neutral label.
    if long_gross <= 0 or short_gross <= 0:
        return pd.Series(0.0, index=weights.index)

    out = weights.copy()
    out[weights > 0] = weights[weights > 0] / long_gross
    out[weights < 0] = weights[weights < 0] / short_gross
    return out


def backtest_panel(positions: pd.DataFrame, panel: Panel, cost_cfg,
                   charge_funding: bool = True):
    """Per-bar net log return of the book, with per-symbol costs and funding."""
    from nullres.backtest.engine import BacktestResult

    ret = panel.ret_next.reindex(positions.index)[positions.columns].fillna(0.0)
    gross = (positions * ret).sum(axis=1)

    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0].abs()
    turnover_total = turnover.sum(axis=1)

    rate = (cost_cfg.fee_bps + cost_cfg.slippage_bps) / 10_000.0
    cost = turnover_total * rate

    if charge_funding:
        # Longs pay funding when the rate is positive, shorts receive it. On a
        # dollar-neutral book these largely offset — but "largely" is not
        # "exactly", and the residual is a real cost of carrying perps.
        fund = panel.funding.reindex(positions.index)[positions.columns].fillna(0.0)
        cost = cost + (positions * fund).sum(axis=1)

    net = gross - cost
    return BacktestResult(
        equity=np.exp(net.cumsum()),
        returns=net,
        gross=gross,
        position=positions.abs().sum(axis=1),   # gross exposure
        turnover=turnover_total,
        cost=cost,
    )


def benchmarks(panel: Panel, cost_cfg, oos_times: pd.DatetimeIndex | None = None,
               rebalance: int = 42, reference: str = "BTCUSDT") -> dict:
    """The books a cross-sectional model has to beat to be worth anything.

    All are restricted to `oos_times`. That restriction is not cosmetic: the ML
    book only trades inside its test folds, so an unmasked benchmark carries the
    entire pre-OOS period. Here that meant equal-weight absorbing the 2022 bear
    market the model never touched, reporting -83% and making a mediocre
    strategy look excellent by comparison.

    `static_vs_alts` is the important one. It is long the reference asset and
    short everything else, rebalanced never — no model, three trades. BTC
    outperformed alts massively over 2022-2025, so any model that has learned
    "the lowest-volatility member outperforms" has learned this bet under
    another name, and must be measured against it.

    Note that `static_vs_alts` is itself hindsight-selected: BTC is the long leg
    because we know how the period ended. It is a lower bound on what a model
    must beat, not a strategy.
    """
    cols = panel.ret_next.columns
    alive = panel.ret_next.notna()

    def mask(pos: pd.DataFrame) -> pd.DataFrame:
        if oos_times is None:
            return pos
        out = pos.copy()
        out.loc[~pos.index.isin(oos_times)] = 0.0
        return out

    out = {}

    weights = alive.div(alive.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    held = weights.copy()
    for i in range(1, len(held)):
        if i % rebalance:
            held.iloc[i] = held.iloc[i - 1].where(alive.iloc[i], 0.0)
    out["equal_weight"] = backtest_panel(mask(held), panel, cost_cfg)

    if reference in cols:
        long_only = pd.DataFrame(0.0, index=panel.times, columns=cols)
        long_only[reference] = 1.0
        out[f"{reference[:3].lower()}_only"] = backtest_panel(
            mask(long_only.where(alive, 0.0)), panel, cost_cfg
        )

        alts = [c for c in cols if c != reference]
        static = pd.DataFrame(0.0, index=panel.times, columns=cols)
        static[reference] = 1.0
        alt_alive = alive[alts]
        static[alts] = -alt_alive.div(
            alt_alive.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0.0)
        out["static_vs_alts"] = backtest_panel(
            mask(static.where(alive, 0.0)), panel, cost_cfg
        )

    return out


def equal_weight_benchmark(panel: Panel, cost_cfg, rebalance: int = 42):
    """Backwards-compatible single benchmark; prefer `benchmarks`."""
    return benchmarks(panel, cost_cfg, rebalance=rebalance)["equal_weight"]
