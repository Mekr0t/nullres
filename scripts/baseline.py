"""
KEPT AS A TEACHING ARTIFACT. This is the original single-file baseline that the
`tbot` package grew out of. It is deliberately NOT maintained or imported by the
package — it exists so the leak it demonstrates stays runnable and visible.

    python scripts/baseline.py --synthetic

Its closing lesson ("walk-forward validation did not catch the leak, only
reading the label definition would have") is now enforced mechanically:

    python -m tbot audit --config configs/btc_1h.toml

See docs/02-leakage.md for what replaced each part of this file.

---

Baseline ML trading pipeline — runs two experiments side by side:

  (A) LEAKY   : the label is derivable from a feature. Walk-forward validation
                does NOT save you. The equity curve goes to the moon. It is a lie.
  (B) HONEST  : purged walk-forward, signal at close of t executed at open of
                t+1, real Binance taker fees + slippage.

The gap between them is the most important thing to internalise before you
write another line of trading code.

Usage:
    python baseline.py --symbol BTCUSDT --interval 1h --start 2020-01 --end 2025-12
    python baseline.py --synthetic     # random-walk data, no network. Sanity test:
                                       # (B) MUST show no edge. If it does, you have a bug.

Deps: pandas numpy scikit-learn requests pyarrow
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = Path("data")
BINANCE_VISION = "https://data.binance.vision/data/spot/monthly/klines"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

# =============================================================================
# 1. DATA
# =============================================================================

def fetch_month(symbol, interval, month):
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{symbol}-{interval}-{month}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    url = f"{BINANCE_VISION}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  miss {month} ({r.status_code})")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode()

    # Archives from ~2025 onward ship a header row; older ones do not.
    header = 0 if raw.lstrip().lower().startswith("open_time") else None
    df = pd.read_csv(io.StringIO(raw), header=header, names=KLINE_COLS)
    df.to_parquet(cached)
    print(f"  ok   {month}  ({len(df)} bars)")
    return df


def load_data(symbol, interval, start, end):
    print(f"Fetching {symbol} {interval} {start}..{end}")
    months = [d.strftime("%Y-%m") for d in pd.date_range(start, end, freq="MS")]
    parts = [p for p in (fetch_month(symbol, interval, m) for m in months) if p is not None]
    if not parts:
        raise SystemExit("No data downloaded — check symbol / date range.")

    df = pd.concat(parts, ignore_index=True)
    # Binance switched open_time from ms to us during 2025. Detect by magnitude.
    unit = "us" if df["open_time"].max() > 1e15 else "ms"
    df["ts"] = pd.to_datetime(df["open_time"], unit=unit)

    cols = ["open", "high", "low", "close", "volume", "trades"]
    df = df[["ts"] + cols].astype({c: float for c in cols})
    return df.drop_duplicates("ts").sort_values("ts").set_index("ts")


def synthetic_data(n=40_000, seed=0):
    """Geometric random walk — by construction there is NO edge here."""
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0005, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": abs(rng.normal(500, 150, n)),
            "trades": abs(rng.normal(3000, 800, n)),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )

# =============================================================================
# 2. FEATURES — every column must be knowable at the CLOSE of bar t
# =============================================================================

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(df):
    f = pd.DataFrame(index=df.index)
    logret = np.log(df["close"]).diff()

    for lag in (1, 2, 3, 6, 12, 24):
        f[f"ret_{lag}"] = logret.rolling(lag).sum()
    for w in (12, 24, 72):
        f[f"vol_{w}"] = logret.rolling(w).std()
        f[f"z_{w}"] = (df["close"] - df["close"].rolling(w).mean()) / df["close"].rolling(w).std()

    f["rsi_14"] = rsi(df["close"])
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]
    f["vol_z"] = (df["volume"] - df["volume"].rolling(72).mean()) / df["volume"].rolling(72).std()
    f["trade_z"] = (df["trades"] - df["trades"].rolling(72).mean()) / df["trades"].rolling(72).std()
    f["hour"] = df.index.hour
    f["dow"] = df.index.dayofweek
    return f


def label_honest(df):
    """1 if the NEXT bar rises. Knowable only at close of t+1 -> must purge."""
    return (np.log(df["close"]).diff().shift(-1) > 0).astype(int)


def label_leaky(df):
    """1 if THIS bar rose. Looks innocent — but `ret_1` IS this quantity, so the
    model just reads the answer off its own input. The single most common bug."""
    return (np.log(df["close"]).diff() > 0).astype(int)

# =============================================================================
# 3. PURGED WALK-FORWARD
# =============================================================================

def walk_forward(n, n_folds=6, purge=24, min_train=5_000):
    """Expanding window. `purge` bars dropped from the tail of each training set
    so no training label overlaps the test window."""
    fold = (n - min_train) // n_folds
    for k in range(n_folds):
        tr_end = min_train + k * fold
        te_end = min(tr_end + fold, n)
        if te_end - tr_end < 100:
            break
        yield np.arange(0, tr_end - purge), np.arange(tr_end, te_end)


def fit_predict(X, y, purge):
    """Out-of-sample P(up), aligned to the original index."""
    proba = pd.Series(np.nan, index=X.index)
    for tr, te in walk_forward(len(X), purge=purge):
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=0,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        proba.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return proba

# =============================================================================
# 4. THE TWO BACKTESTS
# =============================================================================

def backtest_leaky(df, proba, thresh):
    """WRONG twice over: the label leaked into the features, and we assume a free
    instant fill at the close of the very bar being predicted."""
    pos = (proba > thresh).astype(float)
    return np.exp((pos * np.log(df["close"]).diff()).fillna(0).cumsum())


def backtest_honest(df, proba, thresh, fee_bps=10.0, slip_bps=2.0):
    """RIGHT. Decide at close of t, enter at OPEN of t+1, exit at OPEN of t+2.
    Charge fee + slippage on every position change."""
    pos = (proba > thresh).astype(float)
    lo = np.log(df["open"])
    oo = (lo.shift(-2) - lo.shift(-1)).fillna(0)      # open[t+1] -> open[t+2]
    turnover = pos.diff().abs().fillna(0)
    cost = turnover * (fee_bps + slip_bps) / 10_000
    return np.exp((pos * oo - cost).cumsum()), cost.sum(), turnover.sum()


def stats(eq, ppy):
    r = eq.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(total=0.0, sharpe=0.0, maxdd=0.0)
    return dict(
        total=eq.iloc[-1] / eq.iloc[0] - 1,
        sharpe=r.mean() / r.std() * np.sqrt(ppy),
        maxdd=(eq / eq.cummax() - 1).min(),
    )

# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--thresh", type=float, default=0.52)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    df = synthetic_data() if a.synthetic else load_data(a.symbol, a.interval, a.start, a.end)
    ppy = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}.get(a.interval, 24 * 365)

    X = build_features(df)
    y_h, y_l = label_honest(df), label_leaky(df)
    keep = X.notna().all(axis=1) & y_h.notna() & y_l.notna()
    df, X, y_h, y_l = df[keep], X[keep], y_h[keep], y_l[keep]
    print(f"\n{len(df):,} bars | {X.shape[1]} features | base rate {y_h.mean():.3f}")

    print("\n(A) training on the LEAKY label ...")
    pa = fit_predict(X, y_l, purge=0)
    print("(B) training on the HONEST label ...")
    pb = fit_predict(X, y_h, purge=24)

    oos = pa.notna() & pb.notna()
    df, pa, pb = df[oos], pa[oos], pb[oos]
    acc_a = ((pa > 0.5).astype(int) == y_l[oos]).mean()
    acc_b = ((pb > 0.5).astype(int) == y_h[oos]).mean()

    leaky = backtest_leaky(df, pa, a.thresh)
    honest, cost, turns = backtest_honest(df, pb, a.thresh)
    hold = df["close"] / df["close"].iloc[0]

    print(f"\n{len(df):,} out-of-sample bars")
    print(f"{'':<24}{'OOS acc':>9}{'total':>14}{'sharpe':>9}{'max dd':>9}")
    for name, eq, acc in [
        ("(A) leaky", leaky, acc_a),
        ("(B) honest", honest, acc_b),
        ("    buy & hold", hold, None),
    ]:
        s = stats(eq, ppy)
        av = f"{acc:.4f}" if acc is not None else "—"
        tot = f"{s['total']:.1%}" if abs(s["total"]) < 100 else f"{s['total']:.2e}x"
        print(f"{name:<24}{av:>9}{tot:>13}{s['sharpe']:>9.2f}{s['maxdd']:>9.1%}")

    print(f"\n{turns:,.0f} position changes | fees+slippage alone consumed "
          f"{1 - np.exp(-cost):.1%} of equity")
    print("\nNote (A)'s out-of-sample accuracy. Walk-forward validation did not")
    print("catch the leak — only reading the label definition would have.")


if __name__ == "__main__":
    main()