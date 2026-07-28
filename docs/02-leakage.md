# 02 — Leakage catalogue

Every leak below has been shipped to production by someone competent. They are
ordered roughly by how often they occur.

Run `python -m nullres audit --config <cfg>` to check leaks 1, 2, 3, 3b and 5
mechanically. Leak 4 is prevented by construction rather than checked, leak 6 is
handled by the run ledger, and leak 7 cannot be automated at all. The table at
the bottom of this file maps each check to what it catches.

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
headed. In `nullres`, scaling for linear models lives inside a `sklearn.Pipeline`
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

### 3b. Joining an EXTERNAL series on the wrong instant

The same class of bug, but sharper, and it arrived with funding rates and open
interest. Every other feature in this repo derives from the bars themselves, so
a lookahead had to be a visible `.shift(-1)`. Joining a separate time series
fails differently — a join *direction* or a one-interval misalignment, neither
of which reads as wrong.

Three things have to be right at once:

1. **The bar's close, not its open.** A bar indexed at `T` covers `[T, T+interval)`
   and closes at `T+interval`. Joining on `T` silently discards a full bar of
   information; joining on anything past `T+interval` is lookahead.
2. **Strictly before.** A funding settlement stamped exactly at the close is
   *simultaneous* with it. `allow_exact_matches=False`.
3. **The auxiliary frame must be clipped to the bar range.** This is the
   non-obvious one. `check_point_in_time` truncates the bars and recomputes —
   but if the funding frame still holds future rows, a `direction="forward"`
   join returns the same value either way and **the audit passes while the
   feature reads tomorrow's funding**. Clipping makes truncating the bars also
   truncate the auxiliary data, which restores the check.

`tests/test_derivatives.py::test_forward_join_is_caught_by_the_audit` injects
exactly that bug and asserts it is caught. Without point 3 it is not.

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

**Partly detectable, contrary to what this file used to claim.** The dominant
failure mode is mechanical, and it is this:

> A multi-symbol universe, spanning a period that killed assets, which contains
> none of them, was filtered by survival.

`audit.check_survivorship` tests exactly that. It fails a universe with no
delisted members, and fails one that drops most of the symbols which were
actually trading at the sample start — because "they are not around any more"
is not a selection criterion.

Two supporting pieces make it possible:

- `data.universe.universe_as_of(month)` enumerates the archive and asks each
  symbol "did you have data that month". A coin that listed in 2023 fails; a
  coin that died in 2022 passes, and belongs in the sample. Writing the list by
  hand instead is hindsight — you will recall the survivors.
- `data.universe.delisted_from_cache` identifies symbols whose archive stops
  early, offline, from local files.

The repo's own universe is built this way. The archive lists **138** USDT perps
trading in 2021-12; two of them (`BTCDOMUSDT`, `DEFIUSDT`) are index products
rather than assets and are excluded, leaving **136**. Of those, **16** have
archives that stop before the sample ends — they were delisted, and they stay in
the universe. Fewer than that reach the traded book, because the top-40
liquidity screen is applied afterwards; `nullres xsec` prints which ones and the
date each stopped trading.

**What it still cannot see** is whether you picked the *winners among the
survivors*. That is hindsight, entry 7 below, and no check will ever catch it.

**A note on the `n/a` verdict.** For a single-symbol backtest this check
reports "not applicable" rather than PASS. A green tick would claim the risk
was ruled out when it was never examined, which is the same species of false
assurance the audit exists to prevent.

## 6. Choosing hyperparameters on the test set

Running 25 threshold combinations and reporting the best one is not
out-of-sample, no matter how correct your splits are. The test set became your
training set the second you used it to choose.

```bash
python -m nullres sweep --config configs/btc_1h.toml
```

prints the whole surface rather than the maximum, and tells you what to look
for: a real edge degrades *smoothly* as parameters move. A lone positive cell
surrounded by negatives is a fitting artefact.

`nullres robust` makes that judgement quantitative. It measures how often the sign
changes between adjacent grid cells and compares it against `2p(1-p)` — the
rate you would get by scattering the same number of positive cells at random.
Matching the baseline means the arrangement carries no information, however
good the best cell looks.

If you must select, `metrics.deflated_sharpe` subtracts the Sharpe you would
expect to reach by luck given `n_trials`. Searching 100 variants on pure noise
yields a best-of-100 Sharpe around 0.6 for free.

**The hard part is counting honestly.** `n_trials` was originally the number of
strategies in the current run — six — for a project that had explored 214
parameter combinations. That understated the penalty on every result published
here. Deflating the best one (Sharpe 1.61) against six trials leaves 0.88;
against the true count it leaves **0.05**.

The count now comes from the run ledger (`runs/`), which records how many
variants each run evaluated — a 25-cell sweep is 25 trials, not one — plus a
`prior_trials` field for exposure predating the ledger. It is deliberately
global rather than per-config: a researcher who would have published whichever
of six configs worked has tried six, not one.

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
| `check_point_in_time` | leaks 2, 3, 3b — any future reference in features |
| `check_label_leakage` | leak 1 |
| `check_shuffled_label` | side channels: permuted labels must be unlearnable |
| `check_null_data` | engine/split bugs: a random walk must yield no edge |
| `check_survivorship` | leak 5 — a universe with no corpses in it |
| purged splits + `t_end` | leak 4, enforced by construction rather than checked |
| ledger-derived `n_trials` | leak 6 — deflates against every variant tried |

**Six of the seven are now mechanical.** Only entry 7 — hindsight in the
research process — is left entirely to you, and it is the one that no test
suite will ever reach. Everything above tells you whether your measurements are
sound. None of it tells you whether the question was chosen after you already
knew the answer.
