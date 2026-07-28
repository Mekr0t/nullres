# 00 — The rules

Three rules. Everything else in this repo is machinery for enforcing them.

---

## Rule 1: a feature at bar `t` uses only information available at the close of bar `t`

Violations are called *lookahead* and they are the reason backtests lie. They
are also much harder to spot than people expect, because the code that contains
them looks completely ordinary.

```python
# All four of these are lookahead. None of them look wrong.
f["a"] = returns.shift(-1)                       # obvious in isolation
f["b"] = (close - close.mean()) / close.std()    # mean over the WHOLE sample
f["c"] = daily.reindex(hourly.index, method="bfill")   # tomorrow's daily close
f["d"] = scaler.fit_transform(X)                 # fitted before the split
```

`b`, `c` and `d` are the ones that get people. They contain no negative shift
and no obvious reference to the future — they just quietly average over data
that hadn't happened yet.

**How the repo enforces it.** `audit.check_point_in_time` recomputes the whole
feature matrix on truncated history and compares:

```
build_features(bars[:t+1]).iloc[-1]   ==   build_features(bars).iloc[t]
```

If a feature used anything after bar `t`, these differ and the check fails,
naming the column. It is mechanical, it needs no understanding of what the
feature means, and it catches all four cases above. `tests/test_audit.py`
injects each one and asserts it gets caught.

## Rule 2: a position decided at bar `t` is filled at the open of bar `t+1`

You cannot observe a bar's close and also trade at that close. The close *is*
the last trade of the bar; by the time you know it, it is gone.

```
decide at close[t]  ->  fill at open[t+1]  ->  exit at open[t+2]
so position[t] earns  log(open[t+2]) - log(open[t+1])
```

The original baseline's leaky backtest used `pos * log(close).diff()` — earning
the return of the very bar being predicted. Combined with the leaked label, that
produced `2.92e+41x`.

`tests/test_engine.py::test_cannot_trade_the_bar_it_predicts` builds a price
series with a single 2x jump and asserts a position opened at the jump bar
captures **nothing**.

## Rule 3: every position change pays fees and slippage

Costs are not a haircut you subtract at the end. They compound, and they scale
with how often you change your mind.

The baseline's honest run had a genuine 54.25% accuracy over 47,502 bars —
**18.5σ above coin-flip** if you treat the bars as independent, and still
comfortably significant if you don't. (They are not independent: overlapping
labels mean the effective sample is a fraction of the bar count. Even deflating
by the full 24-bar label overlap leaves ~3.8σ.) It still lost 100%, because it
changed position 15,527 times:

```
15,527 changes x 12bps  =  18.6 in log space  ->  e^-18.6  ~  0
```

The model was never the problem. See [03 — Costs](03-costs.md).

---

## The corollary: prove the harness works before trusting it

A backtest is a measuring instrument. Calibrate it against known answers:

```bash
python -m nullres run --config configs/null.toml
```

That runs the entire pipeline on a geometric random walk, where there is no
edge **by construction**.

What the check actually requires is narrower than "no strategy makes money",
and the difference matters. A random walk still has a *realised* drift over any
finite sample, so buy & hold posts a positive Sharpe on it, and any strategy
that spends time long inherits some of that. Demanding zero would fail a
correct engine. So `audit.check_null_data` exempts `buy_hold` and fails only if
some other strategy clears **Sharpe 0.5** — comfortably above what riding a
random drift can produce, comfortably below a real edge. A representative run:

```
buy_hold  +0.23    donchian  +0.19    ml_direction  +0.30    PASS
```

If a strategy does clear 0.5 here, there is a bug in the engine, the split, or
the labels — and every other result in the repo is void until you find it.

The reverse test matters too. `synthetic_bars(edge=0.05)` generates a walk with
a real, known AR(1) autocorrelation. If your pipeline *cannot* find that, it is
too weak to find anything, and a null result on real data tells you nothing.

A harness that passes neither test is not measuring what you think it is.
