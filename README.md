# nullres

[![CI](https://github.com/Mekr0t/nullres/actions/workflows/ci.yml/badge.svg)](https://github.com/Mekr0t/nullres/actions/workflows/ci.yml)

**A quantitative research framework built to disprove trading strategies.**

Most backtesting tools help you find an edge. This one is built on the
assumption that you probably haven't, and its job is to prove that quickly and
cheaply — before you risk money on a number that was never real.

The name is the usual output: a null result.

---

## Why it exists

It started as a single script that produced a `2.92e+41x` return from a label
leak. Walk-forward validation passed. Six folds agreed. The equity curve was
beautiful and the strategy did not exist.

That script's closing line was:

> Walk-forward validation did not catch the leak — only reading the label
> definition would have.

Relying on a human to re-read a label definition is not a control. So that
became a command:

```bash
python -m nullres audit --config configs/btc_1h.toml
```

Five mechanical checks — point-in-time features, single-feature AUC, a
shuffled-label control, a random-walk null run, and survivorship. The audit
catches the leak that walk-forward missed, and the test suite injects realistic
leaks and asserts every one is caught.

Of the [seven leaks catalogued](docs/02-leakage.md), **six are now mechanical**.
The seventh is hindsight in the research process, and no test suite will ever
reach it.

---

## Quick start

```bash
pip install -r requirements.txt

python -m nullres budget --config configs/btc_1h.toml   # can this idea pay for itself?
python -m nullres fetch  --config configs/btc_1h.toml   # cache bars (offline after this)
python -m nullres audit  --config configs/btc_1h.toml   # is the harness lying to me?
python -m nullres run    --config configs/btc_1h.toml   # backtest, with baselines
python -m nullres robust --config configs/btc_4h.toml --strategy donchian
python -m nullres log                                   # what have I already killed?
```

Run `budget` **first**. It is arithmetic, it takes two seconds, and it kills
most ideas before you write a feature:

```
per-bar volatility  0.6715%     round-trip cost  0.2400%  (12bps/side)

hold (bars)   E|move|   measured   modelled   measured accuracy
  1            0.54%      0.40%      72.4%        79.8%
  12           1.86%      1.45%      56.5%        58.3%
  720         14.38%     14.29%      50.8%        50.8%
```

Flip your position every bar and you need to be right **80%** of the time. Read
the measured column: returns are not Gaussian, and the modelled one is too
forgiving at exactly the short holds people are tempted by.

---

## Architecture

The linear pipeline is the boring part. The loop around it is the product.

```
        hypothesis (written down BEFORE the code)
                        │
  Data → Features → Labels → Validation → Models → Execution → Metrics
    └────────┴─────────┴──────────┴──────────┴──────────┴─────────┘
                        │
                  AUDIT ── point-in-time · leakage · shuffle · null control
                        │
                  ROBUSTNESS ── neighbourhood · stability · transfer · cost
                        │
        ┌───────────────┼───────────────┐
     KILLED       INCONCLUSIVE       SURVIVED
        │               │               │
   graveyard      your judgement   forward paper trade
```

Three outcomes, not two, because two forced a claim the evidence could not
support. The stability and transfer gates read five years and four symbols; at
that size a strategy exactly as good as buy & hold fails them 50% and 31% of the
time. Those two now return KILLED only when the count goes against a strategy
**and** the magnitude of the shortfall is distinguishable from zero.

The neighbourhood gate cannot use that test — grid cells are neighbouring
parameters on the same bars, so they are correlated and their effective sample
size is unknowable. It relies instead on what survives that: the median must be
positive, and the *arrangement* of signs must be significantly smoother than
random placement. It too can return INCONCLUSIVE, when the surface is smooth but
the headline cell towers over its neighbours.

Anything short of decisive is INCONCLUSIVE, and deciding what that means is
yours. Every note prints how often its own gate would have fired by chance.

| stage | guarantees |
|---|---|
| **Data** | validated OHLCV contract; point-in-time universes; delisted assets kept |
| **Features** | stationary only; every value knowable at that bar's close |
| **Labels** | triple-barrier with real resolution times, not a fixed horizon |
| **Validation** | purged + embargoed walk-forward; uniqueness-weighted samples |
| **Models** | three audited `.fit()` sites, pinned by a test — a fourth fails CI |
| **Execution** | decide at close `t`, fill at open `t+1`, pay fees every change |
| **Metrics** | deflated Sharpe, t-stats, cost drag, per-period breakdowns |
| **Robustness** | four independent attempts to falsify anything that looks good |

Every run appends to `runs/` — config, git SHA, metrics, verdict — so results
are reproducible rather than remembered, and a config that sits close to
something already KILLED says so before it costs you an afternoon:

```
  WARNING: this config is within 1 parameter(s) of 1 run(s) already marked KILLED.
    2026-07-28  btc_4h  robust  [7ecf45b3]  differs by: strategies
    See docs/05-graveyard.md before spending time on this.
```

The ledger is generated; [the graveyard](docs/05-graveyard.md) that interprets
it is written by hand. Deciding what a result *means* is the actual work.

---

## Example: the tool doing its job

`donchian` on 4h looked like the best result in the project — Sharpe 0.59
against buy & hold's 0.38, on 66 untuned trades, with half the drawdown.

```
$ python -m nullres robust --config configs/btc_4h.toml --strategy donchian

  year     total  sharpe |     hold  hold sh |  excess
  2021     30.3%    0.93 |    32.5%    0.72  |   +0.21
  2022    -26.8%   -1.13 |   -64.7%   -1.70  |   +0.56
  2023     73.8%    1.75 |   156.0%    2.24  |   -0.50
  2024     60.9%    1.32 |   121.7%    1.53  |   -0.21
  2025     -8.4%   -0.41 |    -6.3%   -0.15  |   -0.27

VERDICT: INCONCLUSIVE
  STABILITY INCONCLUSIVE: beat the benchmark in 2 of 5 years (40%), but mean
  excess -0.04 (p=0.840) is not distinguishable from zero. At n=5 this count
  gate passes 50% of the time even for a strategy exactly as good as the
  benchmark, so failing it on its own decides nothing.
```

The entire five-year advantage was **2022**. It is a drawdown-reduction
mechanism, not an alpha source — one bear market wearing a trend-following
costume. Six lines of output, one afternoon, instead of six months.

Note what the tool does *not* claim. It reports that five years cannot separate
this from buy & hold, and stops. The kill is in [the graveyard](docs/05-graveyard.md),
argued from the concentration in 2022 and a mean excess of zero. Machine reports
evidence; human decides meaning.

---

## Results

**Eight approaches tested. Eight abandoned — seven killed on the evidence, one
because the evidence could not settle it and the reasoning could.** Full numbers
in [RESEARCH.md](RESEARCH.md); the argument for each in
[the graveyard](docs/05-graveyard.md).

That distinction is not pedantry. `donchian` is the one, and its battery now
returns INCONCLUSIVE: five years cannot separate it from buy & hold. It is still
abandoned, on the grounds that its entire advantage sits in 2022 and its mean
excess is zero — but that is a judgement, and the tool no longer pretends to
have proved it.

| approach | how it died |
|---|---|
| next-bar direction | 15,527 trades × 12bps |
| direction, any horizon | AUC 0.50–0.53 |
| slower timeframes | break-even *duration* is timeframe-invariant |
| donchian breakout | one bear market, four bad years |
| volatility targeting | crypto has no leverage effect to exploit |
| funding / open interest | +0.012 AUC — real, doesn't clear 24bps |
| cross-sectional (11) | loses to a 3-trade static bet |
| cross-sectional (136) | **AUC 0.5496, t-stat 3.59 — dies to slippage** |

That last row is the one to read. The signal is real and it is *distributed*:
shuffled labels give 0.5017, delisted names contribute 2.6% of absolute P&L, and
per-symbol AUC has a median of 0.521 with 67 of 105 symbols above chance — so
unlike the 11-symbol book, this one is not a static ordering wearing a model. It
survives five independent checks. It dies anyway — to costs (Sharpe 2.02 at
3bps, **0.24** at 100bps), and it does not clear the arithmetic nobody enjoys as
well as it appears to:

```
  trials      1      6     25    100    220    500   1000
  deflated  2.02   1.29   0.90   0.60   0.45   0.30   0.19
```

The ledger holds **220 distinct trials**, and `nullres` reads the count from it,
so the correction applies itself rather than depending on anyone remembering to.
Deflating the best result against 220 leaves **0.45** — the largest surviving
figure this project has produced.

**And it should not be believed.** That 2.02 came from adding open-interest
features which *lowered* mean AUC from 0.5575 to 0.5496. Discrimination got worse
while the equity curve got luckier, and at ~190 trades the curve is decided by
which handful of positions landed. The 37-feature panel is the honest version:
Sharpe 1.80, deflating to **0.23**. Deflation corrects for how often you looked,
not for whether what you found was real — which is a limit of the method worth
knowing before you lean on it.

**Does the framework work?** It told me eight times that I had nothing, then cut
my own headline number to near-zero — and when I made its verdicts honest about
their own sample size, it took back one of the eight kills and told me it had
never had the evidence. That is the product.

---

## Docs

- [00 — The three rules](docs/00-the-rules.md) everything else follows from
- [01 — Research workflow](docs/01-workflow.md), in the order that kills ideas cheapest
- [02 — Leakage catalogue](docs/02-leakage.md) — every leak, and which check catches it
- [03 — Costs and execution](docs/03-costs.md) — the arithmetic that decides most questions
- [04 — Labels and strategies](docs/04-labels-and-strategies.md)
- [05 — The graveyard](docs/05-graveyard.md) — read before proposing a hypothesis

## Status and scope

Research tooling, not a trading bot: nothing here connects to an exchange or
places an order, and none of it is investment advice. Results are crypto-only so
far and every one of them is in-sample with respect to the research process —
the symbols, dates and threads were all chosen knowing how the period ended.
Passing every check here means your measurements are sound. It says nothing
about the future.
