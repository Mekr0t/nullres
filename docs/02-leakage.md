# 02 — Leakage catalogue

Every leak below has been shipped to production by someone competent. They are
ordered roughly by how often they occur.

Run `python -m tbot audit --config <cfg>` to check the first four mechanically.

---

## 1. The label is derivable from a feature

**The original baseline's bug.** The label was "did this bar rise?" and the
feature set contained `ret_1`, which *is* that quantity. The model read the
answer off its own input and scored 0.9986 out of sample.

```python
def label_leaky(df):
    return (np.log(df["close"]).diff() > 0).astype(int)   # same bar

f["ret_1"] = np.log(df["close"]).diff()                   # ...is this
```

Walk-forward validation cannot catch this. The leak is inside every fold
equally, so every fold agrees, and agreement reads as robustness.

**Detection:** `audit.check_label_leakage` scores each feature's AUC against the
label individually. Real financial features land at 0.50–0.55. Anything above
0.65 is a leak, not a discovery. On the baseline's label, `ret_1` scores ~1.0.

## 2. Normalising with statistics from the whole sample

```python
f["z"] = (close - close.mean()) / close.std()      # mean of the FUTURE too
scaler.fit(X)                                      # before splitting
```

Small leak, real leak. The 2024 mean tells the 2021 model where prices were
headed. In `tbot`, scaling for linear models lives inside a `sklearn.Pipeline`
so it is refitted per fold — see `models/classifier.py`.

**Detection:** point-in-time check. Truncating the history changes the mean,
which changes the feature, which fails the comparison.

## 3. Backfilling, or joining a lower frequency badly

```python
daily = close.resample("D").last()
f["x"] = daily.reindex(hourly.index, method="bfill")   # every hour of Monday
                                                        # gets Tuesday's close
```

`ffill` is usually right; `bfill` is essentially always wrong. And even `ffill`
needs a `.shift(1)` on the daily series — at 09:00 Monday, Monday's daily close
has not happened yet.

**Detection:** point-in-time check. `tests/test_audit.py` contains both the
broken and the correct version of this join.

## 4. Overlapping labels crossing the train/test boundary

If a label at bar `t` resolves at `t+500`, and the test window starts at
`t+100`, that training row already contains 400 bars of test-period outcome.

A fixed purge constant only works when every label has the same horizon — which
stops being true the moment you use barriers, since a label resolves when a
barrier is *touched*, not on a schedule. So `labels/` returns a `t_end` column
recording each label's actual resolution bar, and `validation/splits.py` purges
on that:

```python
train = candidates[t_end[candidates] < test_start - embargo]
```

The `embargo` drops training rows that merely *end* near the boundary, because
serial correlation makes a row from five bars before the split nearly identical
to one inside it.

**Detection:** `tests/test_splits.py` asserts `t_end[train].max() < test.min()`
for every fold, and that longer horizons purge strictly more rows.

## 5. Survivorship and delisting

Backtesting today's top 100 coins over 2020–2025 tests the strategy "buy things
that will still exist and be popular in 2025". The losers were deleted from your
universe, so your strategy cannot lose on them.

**Not detectable mechanically.** It is a property of how you chose the universe,
not of your code. This repo sidesteps it by trading a single fixed symbol. The
moment you add a universe selector, you own this problem.

## 6. Choosing hyperparameters on the test set

Running 25 threshold combinations and reporting the best one is not
out-of-sample, no matter how correct your splits are. The test set became your
training set the second you used it to choose.

```bash
python -m tbot sweep --config configs/btc_1h.toml
```

prints the whole surface rather than the maximum, and tells you what to look
for: a real edge degrades *smoothly* as parameters move. A lone positive cell
surrounded by negatives is a fitting artefact.

If you must select, `metrics.deflated_sharpe` subtracts the Sharpe you would
expect to reach by luck given `n_trials`. Searching 100 variants on pure noise
yields a best-of-100 Sharpe around 0.6 for free.

## 7. Hindsight in the research process itself

The subtlest one, and the only one with no code fix. You know BTC went up
2020–2025. You know March 2020 and FTX happened. Every choice you make — the
symbol, the date range, which result made you keep going — is informed by
outcomes you already know.

There is exactly one honest test left, and it is the one nobody wants to run:
**forward paper trading on bars that did not exist when you wrote the code.**
Everything before that is a hypothesis.

---

## What the audit actually covers

| check | catches |
|---|---|
| `check_point_in_time` | leaks 2, 3 — and any future reference in features |
| `check_label_leakage` | leak 1 |
| `check_shuffled_label` | side channels: permuted labels must be unlearnable |
| `check_null_data` | engine/split bugs: a random walk must yield no edge |

Leaks 5, 6 and 7 are yours to avoid. No test suite can do it for you.
