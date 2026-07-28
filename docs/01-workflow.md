# 01 — Research workflow

The order matters. Each step is cheap and can kill the idea before you spend
effort on the next one.

---

## 0a. Check whether you already killed this

```bash
python -m nullres log
```

Eight approaches are already dead. `run` and `robust` also check automatically
and warn when a config sits within a few parameters of something marked KILLED —
because nobody re-reads a 300-line markdown file before every experiment, and
eighteen months from now you will not remember why you stopped.

## 0b. Write down the hypothesis first

One sentence, before any code:

> *"BTC hourly returns mean-revert after volume spikes, over a 6–24 bar horizon."*

If you cannot write it, you are not doing research, you are searching for a
number that looks good — and with enough features you will always find one.

The point of writing it down is that it constrains what counts as success
**before** you see the result.

## 1. Check the cost budget — 2 seconds

```bash
python -m nullres budget --config configs/btc_1h.toml
```

Can any plausible accuracy clear the costs at the holding period you have in
mind? If break-even needs 60% and realistic models get 52%, stop. Change the
timeframe or the instrument. Do not proceed hoping features will save it.

Most ideas should die here. That is the step working.

## 2. Verify the harness on data with known answers — 1 minute

```bash
python -m nullres run --config configs/null.toml
```

No strategy may find an edge in a random walk. The bar is Sharpe 0.5, not zero,
and `buy_hold` is exempt — a random walk has a realised drift, so holding it
scores positive on any finite sample and demanding zero would fail a correct
engine. Anything else clearing 0.5 is a bug, and every other result is void
until you find it. See [00 — The rules](00-the-rules.md).

## 3. Audit for leakage — 1 minute

```bash
python -m nullres audit --config configs/btc_1h.toml
```

Five checks: point-in-time features, single-feature AUC, shuffled-label control,
null data, and survivorship. Run this **every time you add a feature**, not once
at the start. Each new feature is a fresh chance to introduce lookahead.

Survivorship reports `n/a` on a single-symbol config rather than PASS — it only
has something to test on a multi-asset universe, and a green tick would claim a
risk was ruled out when it was never examined.

## 4. Run the baselines before the model

```bash
python -m nullres run --config configs/btc_1h.toml
```

Look at `buy_hold`, `sma_cross`, `donchian` **first**. That is the bar. A model
that returns 40% where buy & hold returned 355% has not found an edge, it has
found an expensive way to underperform.

Compare risk-adjusted, not absolute: Sharpe, max drawdown and Calmar, not total
return. On the 4h config, `donchian` returns slightly less than buy & hold but
at Sharpe 0.52 vs 0.33 and a -41% drawdown vs -77%. That is a better strategy
by every measure that matters.

## 5. Read AUC, not accuracy

Accuracy at a fixed 0.5 cut hides a model that ranks well but is poorly
calibrated. AUC is the better read on whether *any* signal exists.

```
AUC 0.50        nothing
AUC 0.52-0.53   a real but tiny ranking signal — check it against the budget
AUC 0.55+       genuinely interesting on financial data
AUC 0.65+       you have a leak; go back to step 3
```

Check consistency across folds too. One fold at 0.58 and five at 0.50 is a
regime artefact, not an edge.

## 6. Sweep for sensitivity, not for a winner

```bash
python -m nullres sweep --config configs/btc_1h.toml --strategy ml_meta
```

Read the **shape**. A real edge degrades smoothly as parameters move. An
isolated positive cell in a field of negatives is noise, and picking it is how
a backtest becomes fiction.

Then deflate: 25 cells means 25 trials, and `deflated_sharpe` subtracts what
you would expect to reach by luck at that count.

## 7. Check which features actually carried

```bash
python -m nullres features --config configs/btc_1h.toml
```

Permutation importance on the final fold's *test* window. In-sample importance
tells you what the model memorised; this tells you what survived, which is a
much shorter list. Most values will be ~0. That is the normal result.

## 8. Try to kill anything that looks good

```bash
python -m nullres robust --config configs/btc_4h.toml --strategy donchian
```

Three attempts to falsify, all of which a real effect should survive:

- **Parameter neighbourhood** — do nearby parameter values also work? A real
  effect degrades smoothly; if one cell in a grid is positive and its
  neighbours are not, you found the cell that fit, not an edge.

  Counting positive cells cannot see this, which is why the check reads the
  **arrangement** of signs rather than their number. If a fraction `p` of cells
  are positive and scattered at random, adjacent cells differ in sign with
  probability `2p(1-p)`; a real effect clusters and flips far less often. Over
  `n` adjacent pairs the flip count is approximately `Binomial(n, 2p(1-p))`, so
  "is this grid smoother than chance" is a significance test, not a threshold.
  The gate fails when the observed count is **not** significantly below the
  random rate at 5%.

  It also checks whether the test can conclude anything at all. A grid that is
  95% positive can only flip ~10% of the time however it is arranged, so even a
  perfectly smooth one would not be significant — there the test abstains
  instead of condemning a strong result for having too few signs to shuffle.

  Two things this gate deliberately does *not* do. It runs no magnitude test:
  grid cells are neighbouring parameters evaluated on the same bars, so they are
  heavily correlated and a t-test over them would invent an effective sample
  size nobody knows. And it no longer fails on the positive count alone — at
  ~20 correlated cells a 60% threshold fires on roughly a quarter of grids with
  no edge at all, so a low count downgrades to INCONCLUSIVE rather than killing.
  What can still be said without assuming independence is that the *median*
  must be positive (the typical parameter choice must not lose money) and that
  the best cell must not tower over its own neighbours.

  This is what killed the derivatives ML strategies. `ml_direction` scored 75%
  of cells positive with a median Sharpe of 0.33 — and flipped sign across 39%
  of adjacent cell pairs against a 37.5% random baseline. Indistinguishable
  from chance.
- **Sub-period stability** — profitable *relative to buy & hold* in most years,
  or carried by one? Note the benchmark clause. A long-only filter over a bull
  market is profitable in most years by construction, which is why the absolute
  per-year Sharpe cannot answer this and the excess can.
- **Cross-symbol transfer** — does it work on other instruments? A rule that
  describes market structure should generalise. A rule that only works on the
  asset you developed it on describes that asset's history.

This is where the repo's one promising result died. `donchian` on 4h passed the
first and third tests comfortably and failed the second: its whole five-year
Sharpe advantage came from 2022, and it underperformed holding in every
trending year. The table is in [the graveyard](05-graveyard.md#donchian-breakout-on-4h).

### How much these tests can actually settle

The stability gate reads five years and the transfer gate four symbols, and
"beat the benchmark in 60% of them" is a much weaker demand than it sounds:

```
5 years,   need 3:  a strategy exactly as good as hold clears this 50% of the time
4 symbols, need 3:  ...31% of the time
```

So a bare count cannot tell *"worse than holding"* from *"not enough evidence"*,
and reporting both as KILLED published coin flips as findings. A gate now fails
only on **decisive** evidence — the count went against it and the mean excess is
distinguishable from zero — and a count that fails alone yields **INCONCLUSIVE**.
Each note prints its own false-kill rate so you can weigh it.

Counting also discards magnitude, which matters more than the threshold does. On
the 4h config `donchian` and `mean_reversion` both beat hold in 40% of years and
used to score identically; their mean excess Sharpes are **−0.04** and **−1.13**.
One is indistinguishable from holding, the other is far worse. Both numbers are
now reported.

The decision rule stays aggressive — SURVIVED requires clearing all three gates —
because a false kill costs one idea and a false survival costs months. That is a
choice about which error to prefer, not a claim that the gates are demanding.

Surviving all three does not certify a strategy. It only means it has not yet
been cheaply disproved. INCONCLUSIVE means even less: the battery ran and could
not separate the strategy from the thing you would have done anyway.

## 9. Forward paper trade

Everything above is in-sample with respect to the research process, because you
chose the symbol, the dates and the direction knowing how the market turned out.

The only remaining honest test is bars that did not exist when you wrote the
code. Run it forward, with costs, for long enough to be meaningful — then
compare against what the backtest predicted for that same window. If they
disagree, the backtest was wrong.

---

## Adding a feature

1. Add it to `build_features` in `nullres/features/technical.py`.
2. Document it in `FEATURE_DOC` — if you cannot say what it measures in one
   line, you do not know why you added it.
3. Keep it **stationary**: a ratio, a z-score, or a bounded oscillator. No raw
   price levels. BTC ran 4k → 100k over this sample; a tree that learned
   `close > 60000` learned the calendar.
4. Run `python -m nullres audit` — the point-in-time check must still pass.
5. Re-run and compare against the *previous* result, not against zero.

## Adding a strategy

1. Implement `positions(self, ctx) -> pd.Series` in `nullres/strategies/`.
2. Register it in `REGISTRY` in `nullres/strategies/__init__.py`.
3. Mask to out-of-sample with `mask_to_oos(pos, ctx)` so it is judged on the
   same window as everything else.
4. Add it to `strategies` in your config.
5. Verify it earns nothing on `configs/null.toml`.

## Adding a data source

Implement a loader returning the `OHLCV` contract from `nullres/data/__init__.py`:
a UTC-indexed frame with float `[open, high, low, close, volume, trades]`,
strictly increasing, no duplicates. Then dispatch on it in `load_bars`. The
rest of the pipeline is source-agnostic.
