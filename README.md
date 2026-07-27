# tbot — an honest research harness for systematic trading

[![CI](https://github.com/Mekr0t/tbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Mekr0t/tbot/actions/workflows/ci.yml)

A repo for asking "does this strategy have an edge?" and getting an answer you
can trust — including when the answer is no, which it usually is.

It grew out of [`scripts/baseline.py`](scripts/baseline.py), which demonstrated
a label leak producing a `2.92e+41x` return. That script's closing line was:

> Walk-forward validation did not catch the leak — only reading the label
> definition would have.

That is now a command:

```bash
python -m tbot audit --config configs/btc_1h.toml
```

---

## Quick start

```bash
python -m tbot budget --config configs/btc_1h.toml
```

Run that **first**. It tells you what accuracy your cost structure demands
before you write a single feature. For hourly BTC at 12bps a side:

```
per-bar volatility      0.6715%
round-trip cost         0.2400%   (12bps/side)

Accuracy needed to break even, by holding period:
  hold (bars)      E|move|    accuracy
  1                  0.54%      72.4%
  12                 1.86%      56.5%
  168                6.94%      51.7%
  720               14.38%      50.8%
```

Flip your position every bar and you need to be right 72% of the time. This is
arithmetic, not pessimism, and it explains the original `-100%` result better
than any amount of model tuning would.

Then:

```bash
python -m tbot fetch --config configs/btc_1h.toml    # cache bars (offline after this)
python -m tbot audit --config configs/btc_1h.toml    # prove the harness isn't lying
python -m tbot run   --config configs/btc_1h.toml    # backtest every strategy
python -m tbot sweep --config configs/btc_1h.toml    # parameter sensitivity
```

When something finally looks good, try to kill it before you believe it:

```bash
python -m tbot robust --config configs/btc_4h.toml --strategy donchian
```

Three falsification tests — parameter neighbourhood, sub-period stability
against buy & hold, and cross-symbol transfer. A strategy has to survive all
three to earn a forward paper-trading run.

One thing the budget table will *not* tell you, and which is easy to get
backwards: **moving to a slower timeframe does not reduce costs.** Volatility
scales as `sqrt(bar length)`, so break-even holding *duration* is invariant —
~21 days for BTC whether you use 1h, 4h or 1d bars. Pick the timeframe for
signal quality and sample size; pick the holding period to beat costs. See
[03 — Costs](docs/03-costs.md#changing-timeframe-is-not-a-cost-lever).

Override anything without editing files:

```bash
python -m tbot run -c configs/btc_1h.toml --set sizing.min_hold=168
```

---

## The three rules everything else follows from

1. **A feature at bar `t` may only use information available at the close of
   bar `t`.** Enforced by `audit.check_point_in_time`, which recomputes features
   on truncated history and asserts nothing changed.
2. **A position decided at bar `t` is filled at the open of bar `t+1`.** There
   is no mechanism by which you observe a close and also trade at it.
3. **Every position change pays fees and slippage.** Costs are not a haircut
   applied at the end; they compound against you on every single trade.

---

## What's here

```
tbot/
  config.py        typed TOML config — one file fully describes an experiment
  costs.py         break-even arithmetic: run this before modelling
  audit.py         four mechanical leak detectors
  pipeline.py      bars -> features -> labels -> positions -> metrics
  data/            Binance archives (cached) + synthetic null data
  features/        32 stationary, point-in-time technical features
  labels/          triple-barrier, vol-scaled forward return, next-bar sign
  validation/      purged + embargoed walk-forward, uniqueness weights
  models/          the only module allowed to call .fit()
  backtest/        execution engine, position sizing, metrics
  strategies/      rule baselines + ML direction + ML meta-labelling
configs/           btc_1h, btc_4h, btc_1d, null (the control experiment)
docs/              the reasoning behind each of the above
tests/             48 tests, including leaks that MUST be caught
```

---

## Results as of the last run

Out-of-sample, BTCUSDT, purged walk-forward, 12bps/side, 2020-12 to 2025-12.

**1h** (44,190 OOS bars):

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 355.3% | 0.46 | -77.2% | 2 | 0.2% |
| sma_cross | 173.4% | 0.45 | -57.6% | 308 | 30.9% |
| donchian | 107.7% | 0.38 | -54.7% | 304 | 30.6% |
| ml_direction | -15.2% | -0.06 | -69.6% | 132 | 17.7% |
| ml_meta | -35.4% | -0.18 | -72.4% | 127 | 14.9% |

**4h** (9,980 OOS bars):

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| donchian | 144.4% | **0.52** | -40.7% | 66 | 7.6% |
| buy & hold | 148.6% | 0.33 | -77.0% | 2 | 0.2% |
| sma_cross | 74.8% | 0.30 | -56.2% | 62 | 7.2% |
| vol_target | 63.6% | 0.21 | -73.1% | 274 | 5.2% |
| ml_direction | -29.2% | -0.15 | -49.4% | 112 | 16.5% |
| ml_meta | -73.3% | -0.66 | -79.7% | 113 | 13.8% |

**1d** (1,424 OOS bars):

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 101.6% | 0.29 | -66.7% | 1 | 0.1% |
| sma_cross | 85.8% | **0.36** | -37.3% | 8 | 1.0% |
| donchian | 48.9% | 0.30 | -25.7% | 12 | 1.4% |
| ml_meta | 20.8% | 0.13 | -40.1% | 67 | 7.8% |
| ml_direction | 9.6% | 0.04 | -64.3% | 65 | 10.2% |

### The honest reading

**No machine-learned edge was found, on any timeframe.** Fold AUCs sit at
0.50–0.53 on 1h and 4h — a real but tiny ranking signal that does not survive
costs. Every deflated Sharpe in all three tables is at or below zero once you
account for how many variants were tried.

Daily deserves a specific warning, because its ML rows are the only positive
ones and they are the *least* trustworthy numbers here. The per-fold AUCs decay
monotonically and end up **below** coin-flip:

```
ml_direction   fold 1  0.548   fold 2  0.541   fold 3  0.422   fold 4  0.422
```

That is the signature of a model fitting a regime that then ended — not an
edge. The positive total return comes from two good folds in 2022–23 being
larger than two bad ones in 2024–25. On 2,200 bars and four folds, that is
noise with a direction.

**`donchian` on 4h was the one hypothesis worth testing, and it has been
falsified.** Over the full period it looked like the best thing here: Sharpe
0.52 against buy & hold's 0.33, on 66 trades, with a much smaller drawdown
(-41% vs -77%). It survived a parameter-neighbourhood test (100% of 19 grid
cells positive, median 0.52) and a cross-symbol test (positive on ETH, BNB,
SOL, XRP).

It died on sub-period stability, once buy & hold was put next to it year by year:

```
  year     total  sharpe |     hold  hold sh |  excess
  2021     30.3%    0.93 |    32.5%    0.72  |   +0.21
  2022    -26.8%   -1.13 |   -64.7%   -1.70  |   +0.56
  2023     73.8%    1.75 |   156.0%    2.24  |   -0.50
  2024     60.9%    1.32 |   121.7%    1.53  |   -0.21
  2025     -8.4%   -0.41 |    -6.3%   -0.15  |   -0.27
```

The entire full-period advantage comes from **2022**, where it cut a -64.7%
year to -26.8%. In every trending year it gave up roughly half the rally. That
is not an alpha source — it is a drawdown-reduction mechanism, correctly priced,
and its five-year Sharpe advantage is one bear market wearing a trend-following
costume. You cannot know in advance whether the next five years look like 2022
or like 2023.

Reproduce with:

```bash
python -m tbot robust --config configs/btc_4h.toml --strategy donchian
```

Two caveats on that verdict, in fairness to it. Five annual observations is a
very small sample to judge stability on, and the 60% pass threshold in
`robustness.verdict` is a judgement call, not a statistic. And "reduces max
drawdown from -77% to -41%" is a real property that some people would pay for —
it is just not the property the Sharpe number was advertising.

What improved between the original baseline and now was not the model. It was:

| | baseline | now |
|---|---|---|
| label | sign of next bar | triple barrier, vol-scaled, ~3 week horizon |
| purge | fixed 24 bars | actual per-label resolution time + embargo |
| position changes | 15,527 | 132 |
| cost drag | **100.0%** | 17.7% |
| result | -100% | -15.2% |

The strategy still doesn't make money. But it now loses for the honest reason
(there is no signal) rather than the structural one (it was being asked an
impossible question and charged for the privilege 15,527 times).

---

## What this repo will not do for you

It will not find an edge. That is not what a harness is for. It will tell you,
quickly and with a straight face, when you haven't found one — which is the
expensive thing to learn late.

Nothing here is investment advice, and none of it is connected to an exchange.
Passing every check in this repo means your measurements are sound; it says
nothing about whether a strategy will make money in the future. Before risking
anything real, forward paper-trade on bars the model has never seen.

## Docs

- [00 — The rules](docs/00-the-rules.md)
- [01 — Research workflow](docs/01-workflow.md)
- [02 — Leakage catalogue](docs/02-leakage.md)
- [03 — Costs and execution](docs/03-costs.md)
- [04 — Labels and strategies](docs/04-labels-and-strategies.md)
- [05 — The graveyard](docs/05-graveyard.md) — every hypothesis killed so far,
  and why. Read this before proposing a new one.
