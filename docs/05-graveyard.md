# 05 — The graveyard

**This file is written by hand, on purpose.** Its value is in sentences like
"one bear market wearing a trend-following costume" — judgements about what a
result *means*. No amount of structured logging produces those, and a
generated version of this file would be a table of statuses that teaches
nobody anything.

What *is* generated is the evidence underneath it. Every run appends a record to
`runs/` — config, git SHA, metrics, verdict — so the numbers quoted here are
reproducible rather than remembered, and so a future experiment can be warned
that it is re-treading a dead end:

```
$ nullres log                    # the ledger
$ nullres run --config ...       # warns if this config is near a KILLED run
```

Prose without evidence rots. Evidence without prose is a spreadsheet. Deciding
what a result means stays the human's job; remembering it exactly stays the
machine's.

*For the measurements themselves, see [RESEARCH.md](../RESEARCH.md).*

Every hypothesis this repo has killed, why it died, and what it cost to find
out. Kept because the most expensive mistake in research is re-running a dead
end you already disproved and forgot about.

Add to it every time something dies. A negative result you can point at is
worth more than a positive one you can't reproduce.

---

## Predict next-bar direction with a gradient-boosted classifier

**Status:** dead. **Cost to disprove:** the original `baseline.py`.

Out-of-sample accuracy 54.25% over 47,502 bars — 18.5σ above coin-flip on the
iid reading, ~3.8σ after deflating for the 24-bar label overlap — and it still
lost 100%. It changed position 15,527 times in 47,502 bars; at 12bps that is
18.6 in log cost. The model was never the problem.

**Lesson:** run `nullres budget` first. At one-bar holds this needed **79.8%**
accuracy to break even — measured from the actual distribution of moves; the
Gaussian model flatters it to 72.4%. Nothing was ever going to close that gap.

---

## Direction prediction at any horizon, 1h / 4h / 1d

**Status:** dead. **Cost:** the whole `ml_direction` and `ml_meta` line of work.

Fold AUCs across every timeframe and label: **0.50–0.53**. On daily the AUCs
decay monotonically across folds and end *below* coin-flip (0.548 → 0.541 →
0.422 → 0.422) — a model fitting a regime that then ended.

Permutation importance on 32 features: only `vol_72` clears 0.01 (0.0244, next
is 0.0084), and 13 of the 32 score negative.

**Lesson:** 32 technical indicators derived from OHLCV are 32 views of the same
four numbers. The information is not there to be extracted.

---

## Moving to a slower timeframe to escape costs

**Status:** dead — and it was never alive; the reasoning was simply wrong.

The intuition: daily bars move ~3.3% against the same fixed 24bps round trip,
versus 0.67% for hourly, so costs should matter less. But volatility scales as
`sqrt(bar length)`, so break-even bars scale as `1/bar_length` and break-even
**duration is invariant**:

```
1h   557 bars = 23.2 days      4h   140 bars = 23.3 days      1d   24 bars = 24.0 days
```

**Lesson:** choose the timeframe for signal quality and sample size; choose the
holding period to beat costs. Pinned by
`test_breakeven_duration_is_invariant_to_timeframe`.

---

## Donchian breakout on 4h

**Status:** dead as an alpha source — killed by judgement, not by the machine.
Alive as a drawdown tool, with caveats.

Looked like the best thing in the repo: Sharpe 0.59 vs buy & hold's 0.38,
-41% max drawdown vs -77%, on 66 untuned trades. Survived the parameter
neighbourhood test (100% of 19 cells positive, no sign flips) and cross-symbol
transfer (beat hold on 3 of 4: ETH, SOL, XRP; lost on BNB).

**The battery now returns INCONCLUSIVE, not KILLED, and that is the honest
answer.** Its mean excess Sharpe across the five years is **-0.04 with p=0.84** —
statistically indistinguishable from simply holding. The old KILLED verdict came
from a count gate ("beat hold in only 40% of years") that fires on a strategy
exactly as good as hold half the time at n=5. That was never evidence.

The kill stands anyway, on the reasoning below rather than on the gate — which
is the division of labour this repo claims: the machine reports what the
evidence supports, the graveyard decides what it means.

The case for killing it, once buy & hold is placed beside it:

```
  year     excess sharpe
  2021          +0.21
  2022          +0.56     <- the entire result
  2023          -0.50
  2024          -0.21
  2025          -0.27
```

The whole five-year advantage is 2022, where it cut a -64.7% year to -26.8%.
In every trending year it surrendered roughly half the rally. A mechanism that
pays off in exactly one of five years, and whose five-year mean excess is zero,
is a drawdown tool — you would hold it for the -41% instead of -77%, not for
return.

**Lesson:** a long-only filter over a bull market is profitable most years by
construction. Judge against the benchmark per period, not against zero. This
finding is what turned `verdict()` from an absolute test into an excess one.

**Second lesson, learned later and more expensively.** The original entry said
this "died on sub-period stability", and it had not — a 2-of-5 count is what a
coin flip produces half the time. The number that actually supports the kill is
the mean excess of -0.04, which says the rule adds nothing on average, and the
2022 concentration, which says what little it does is one regime. Counting
periods felt like evidence and was not.

---

## Volatility targeting

**Status:** dead on crypto. The premise was right; the mechanism is absent.

The premise held up beautifully — volatility really is an order of magnitude
more predictable than direction:

```
lag      ret     |ret|   vol_30
  1   -0.029    0.227    0.992
  6   -0.063    0.176    0.934
```

So the plan was to stop guessing direction and size by `1/sigma` instead. It
failed, for two independent reasons.

**1. No leverage effect in crypto.** Vol targeting works on equities because
volatility spikes accompany crashes — cutting exposure when vol rises avoids
losses. BTC has no such asymmetry. `corr(vol, forward 30-bar return) = +0.059`,
and the *highest*-vol quintile has the *highest* mean forward return:

```
  vol bucket    ann vol   mean fwd ret
  lowest          25.6%        +1.121%
  highest         97.6%        +1.395%
```

Up bars average 0.813%, down bars 0.824% — symmetric. De-risking on volatility
cuts the best periods as hard as the worst.

**2. Capped at 1.0, it is dominated by construction.** With `max_leverage=1.0`
the position sits at the cap on 54% of bars and can never exceed it, so it only
ever holds *less* than buy & hold — and pays fees to do so.

Battery verdict: **KILLED**, and decisively — this one needs no interpretation.
It beat buy & hold in **0 of 5 years**, and the mean excess Sharpe is **-0.22
with p=0.008**. That is evidence, not an unlucky count: the rule is reliably
worse than the thing you would have done anyway. It is also negative against
hold on all four other symbols, and negative in absolute terms on three.

Worth contrasting with `donchian` above, which the old battery killed on the
same "0% / 40% of years" style of count. `vol_target` survives the stricter test
and donchian does not — the two were never the same finding.

**Lesson:** a real statistical regularity (vol clustering) is not the same as
an exploitable one. The regularity has to connect to returns through some
mechanism, and here it doesn't.

*Unresolved footnote:* uncapped, the same rule showed gross Sharpe 0.65 → 0.81
as leverage went 1.0 → 2.0, versus buy & hold's 0.68. That is gross, unmasked,
and un-costed, and part of it is simply carrying 12% more average exposure
through a bull market. It is not evidence of anything yet, and it would need
the full battery plus a margin account to become so. Recorded so nobody
mistakes it for a finding.

---

## Funding rates and open interest

**Status:** the DATA carries information. The STRATEGIES built on it are dead.
Those are two different claims and only the second was tested by the battery.

The one direction that brings information the price series does not contain.
Binance publishes 8-hourly funding rates (from 2020-01) and open-interest /
positioning metrics (daily archives, BTCUSDT from 2020-09).

**What the evidence says.** A matched-sample ablation — identical rows, folds
and benchmark, only the feature set differing:

```
  with derivatives      0.5455  0.5303  0.5368  0.5115  0.5343
                        mean 0.5317   std 0.0126   folds above 0.5: 5/5
  without derivatives   0.5014  0.4973  0.5224  0.5164  0.5619
                        mean 0.5199   std 0.0257   folds above 0.5: 4/5

  per-fold difference   +0.0441  +0.0330  +0.0144  -0.0049  -0.0276
  paired t-test: t = 0.92, p = 0.412
```

Reproduce: `python -m nullres ablate --config configs/btc_4h_deriv.toml`

Three observations, in descending order of confidence:

1. **`ls_accounts` — the long/short account ratio — is the single most
   important feature in the model**, ahead of every OHLCV transform, and four
   of the top ten are derivatives. That is new; nothing OHLCV-derived has ever
   led that list except `vol_72`.
2. **Consistency improved more than accuracy.** All five folds clear 0.5 with
   the data versus four without, at half the variance. Mean AUC rises 0.0118.
3. **It is not statistically significant** (p = 0.41 on five folds), and the
   sign flips in two of them.

**Funding rate itself carries almost nothing.** `funding`, `funding_ma_3` and
`funding_ma_21` all rank at or below zero importance. The signal, such as it
is, comes from *positioning* — who is long and how much — not from the price
of carry. Worth knowing before spending more effort on funding transforms.

**Do not read the Sharpe numbers.** The same ablation moves `ml_direction` from
-0.68 to +0.43, which is not evidence of anything. At ~80 trades the equity
curve is decided by which handful of positions landed. The AUCs above differ by
one percentage point and are computed over 6,660 bars; the Sharpes differ by
1.11 and are computed over 80 trades. Believe the first pair.

### The battery verdict: both strategies KILLED

```
python -m nullres robust --config configs/btc_4h_deriv.toml \
    --strategy ml_direction --transfer-start 2021-12
```

| test | ml_direction | ml_meta |
|---|---|---|
| neighbourhood | **FAIL** — 39% sign flips vs 38% expected | **FAIL** — 48% vs 48% expected |
| stability | ok — beat hold in 2 of 3 years | **FAIL** — 0 of 3 years |
| transfer | **FAIL** — beat hold on 2 of 4 | **FAIL** — 2 of 4 |

**The single most damning number is not in that table.** It is the spread of
the transfer results, and the fact that the two strategies disagree about which
assets work:

```
                 ETH     BNB     SOL     XRP      (Sharpe vs buy & hold)
  ml_direction  +0.02   -0.71   +0.81   -1.62
  ml_meta       -1.13   -0.13   +0.59   +0.52
```

`ml_direction` loves SOL and is destroyed by XRP. `ml_meta` is destroyed by ETH
and likes XRP. If positioning data carried a real, tradable signal about crypto
market structure, two models reading the same features would not disagree about
which assets it works on. A 2.4-Sharpe spread across four correlated majors is
a random draw per symbol, not an edge with exceptions.

### Reconciling this with the ablation

The ablation result above still stands: adding derivatives raises mean AUC from
0.5199 to 0.5317 and `ls_accounts` is the model's most important feature. That
is a statement about **information**, measured over 6,660 bars.

The battery is a statement about **tradability**, measured over ~25 trades per
year per symbol. Both can be true, and here both are: the data contains perhaps
one percentage point of extra discrimination, and one point of AUC does not
survive contact with 24bps round trips and a 3-week holding period.

That is the same wall everything else in this file hit. It is not a modelling
failure — it is `nullres budget` again, from a different direction.

### What this does not rule out

The transfer test judged **strategies**, not the data. Untested: a
cross-sectional formulation (rank the five symbols by positioning extremity
rather than trade each independently), which is the natural use of a signal
that says "the crowd is offside on X relative to Y" and does not require any
single asset's signal to clear costs alone.

## Cross-sectional long/short

**Status:** dead — and the most instructive death here. The model has real,
verified ranking skill and still loses to a book with no model and three trades.

Rank an 11-symbol crypto universe and go long the top-k, short the bottom-k,
dollar neutral. Universe frozen as of 2021-12 including LUNAUSDT and MATICUSDT,
both later delisted. USD-M perps (you cannot short spot), funding charged.

```bash
python -m nullres xsec --config configs/xsec_4h.toml
```

**The skill is real.** Mean AUC **0.5443**, the highest in this project, with
folds 3-5 at 0.573 / 0.590 / 0.551. Three checks, all passed:

| check | result | reading |
|---|---|---|
| shuffled labels | AUC 0.4986 | not a leak |
| survivors only | AUC 0.5480 | not just death-detection |
| per-symbol AUC | median **0.501**, 5 of 10 below 0.5 | **there is no per-symbol skill** |

That last row is the crack, and it is worse than the original version of this
entry claimed. Panel AUC is 0.5443 while the median symbol scores 0.501 — the
model cannot tell a given coin's good bars from its bad ones. Pooled AUC counts
every (timestamp, symbol) pair together, so knowing only which symbols are
persistently better already scores above chance. That is what it learned.

BTC is permanently the lowest-volatility, highest-open-interest member here, so
"rank 1 by low vol" and "BTC" are the same column — but the model did not even
need the proxy. Its 0.585 accuracy on BTC is essentially BTC's base rate: the
share of bars BTC beat the median at all. Take the base rate away and nothing
remains.

*This supersedes a reading based on raw per-symbol accuracy (BTC 0.585, SOL
0.491, "skill is concentrated"). In a beats-the-median panel, accuracy is
inflated by each symbol's own base rate, so a coin that persistently lagged
scores high for a model that learned only the lag. AUC is base-rate free. The
conclusion did not soften — a static ordering is exactly what `static_vs_alts`
is, and it wins.*

**And the trivial version wins.** Out-of-sample 2022-10 to 2025-12:

```
  book                 total   sharpe   max dd  t-stat  trades
  static_vs_alts      220.1%     0.89   -41.8%    1.57       2
  btc_only            228.2%     0.81   -34.7%    1.44       1
  longshort_k2        170.5%     0.59   -78.6%    1.05     165
  equal_weight         15.1%     0.07   -57.7%    0.12       3
```

Long BTC, short everything else, rebalance never. Three trades. It beats the
model on return, Sharpe, drawdown and turnover simultaneously. The ML book's
per-year record is 2023 **-69.7%**, 2024 **+258.8%** — and a -78.6% drawdown on
a book labelled "market neutral".

**Lesson:** a model can have genuine, verifiable skill and still be worthless.
0.544 AUC is real; 166 trades at 8bps and a two-name book turn it into
underperformance. Always construct the dumbest strategy that could explain your
result and check you beat it. Here the dumbest version was better.

And note what `static_vs_alts` is *not*: at t = 1.57 it is not significant
either, and BTC is its long leg only because we know how 2022-2025 ended. It is
a floor a model must clear, not a strategy.

### Two measurement bugs this exposed

1. **Benchmarks were not restricted to the model's out-of-sample window.**
   `equal_weight` ran from 2021-12 while the book only traded from 2022-10, so
   it absorbed the entire bear market and reported **-83.4%**. Correctly masked
   it is **+15.0%**. The strategy looked good for reasons that had nothing to
   do with the strategy.
2. **Delisting silently broke dollar-neutrality.** Zeroing a dead short leg
   leaves the remainder net long — for up to a full rebalance period, at exactly
   the moment the market is disorderly. A "market-neutral" book quietly becoming
   directional is worse than a directional one, because nothing in the metrics
   says so. `_neutralise` now rescales each side, or goes flat if a side dies.

*Footnote on survivorship:* the machinery is correct but barely bound here.
LUNAUSDT delisted 2022-05-13, before the first test fold opened, so it never
entered the out-of-sample result at all. MATICUSDT (delisted 2024-09-11) did.
A universe fixed in 2021-12 still cannot capture assets that were never listed
on Binance perps, so this is *less* survivorship-biased, not unbiased.

## Wide cross-sectional (136 symbols)

**Status:** dead as a tradable strategy. It contains the strongest verified
signal in this project, and it still does not survive realistic costs.

The narrow version above forced a two-name book, so the obvious next move was
width. Universe enumerated mechanically from the archive — every USDT perp
trading in 2021-12 (**136** symbols, including 9 that later delisted), screened
to the top 40 by *trailing* dollar volume.

```bash
python -m nullres xsec --config configs/xsec_4h.toml --universe 2021-12 --top-n 40 \
    --set data.metrics=false
```

**The original width comparison was confounded, and I nearly published it.**
The narrow panel carries 46 features; this one carries 37, because fetching
open-interest metrics for 136 symbols is a multi-hour download and the run sets
`--set data.metrics=false`. So "AUC rose 0.5443 -> 0.5575 because the universe
got wider" was crediting width with a change that also stripped out nine
features — including `ls_accounts`, which the derivatives section above calls
the single most important feature in the model. The derivatives ablation was
careful to match samples exactly; this comparison was not.

Matched at 37 features on both sides, the narrow panel scores **0.5460**. So
width is worth **+0.0115**, not the +0.0132 first claimed. The conclusion
survives, slightly smaller. The lesson is that a confound can sit inside a
result for months when the result agrees with what you expected.

**Width worked, on the matched comparison.** AUC rose 0.5460 -> **0.5575**, all
five folds above 0.5 and *rising* over time (0.552 -> 0.581). Max drawdown at
k=15 is -43.8% against the narrow book's -67.9%. Diversification did its job.

**And the signal is real.** Every check passed:

| check | result |
|---|---|
| shuffled labels | AUC 0.5015 — not a leak |
| survivors only | AUC 0.5612 — not death-detection |
| delisted contribution | 3.7% of absolute P&L — not a bet on dying coins |
| contributor spread | STORJ, BNB, BAKE, BEL long; ETH, DOGE, TRB, REEF short |
| per-symbol AUC | median **0.519**, 68 of 105 above 0.5 — **skill is distributed** |

**That last row is what the narrow book never had.** Its per-symbol median was
0.501; this one is 0.519 with two thirds of the universe above chance. The
11-symbol result was a static ordering wearing a model. This one genuinely knows
something about individual coins, across most of them.

So the kill below is the cleanest in this file. Everything else here died of a
signal that was not there, or was there and was really something else. This one
is real, distributed, verified from five directions — and still loses to its own
slippage.

At the configured 8bps the k=2 book returns **212x, Sharpe 1.80, t-stat 3.19** —
the only t-stat above 3 this project has produced.

### Why it is still dead

**Costs.** The book makes its money shorting thin alts, and 8bps all-in is
fiction for them:

```
  slippage   k=2    k=5   static
     3bps   1.80   1.72    1.60
    25bps   1.42   1.19    1.55
    50bps   0.99   0.59    1.49
   100bps   0.13  -0.57    1.38      <- Sharpe
```

The model's column collapses. `static_vs_alts` barely moves, because it spreads
turnover across 40 small adjustments while the ML book flips whole ±0.5 weights
in and out of illiquid names.

**So the durable part needs no model** — it is long BTC, short a broad alt
basket. And that is hindsight: BTC is the long leg because we know alts fell
~90% against it over 2022-2025. In December 2021, after alt season, the
consensus was the opposite trade. This is
[02 — Leakage](02-leakage.md#7-hindsight-in-the-research-process-itself) wearing
a cross-sectional costume.

**Tail risk is untested, not absent.** A liquidation check found no bar worse
than -11.8% and no case of being short a name that then exploded. That is not
reassurance: >+65% four-hour moves occur ~12 times in 1.1M bars, and the book
held ~13,800 short-name-bars, so the expected number of hits is **0.15**.
Observing zero is what chance predicts. One UNFI-type event (+274% in 4h) against
a -0.5 weight is -137% of capital, and the engine models no margin at all.

**Lesson:** the strongest signal in the project, verified clean from four
directions, and it still loses to arithmetic. Cost sensitivity is not a
footnote to a backtest — for anything trading thin instruments it *is* the
result, which is why `nullres xsec` now prints the sweep by default.

*Two suspicions I raised during this investigation were wrong and are recorded
so nobody re-chases them: the extreme bars are genuine market events (LUNA's
+346% is the real death-spiral bounce), not gap artifacts; and interior archive
gaps do NOT produce spanning returns, because reindexing before differencing
already yields NaN across holes.*

## Still open

- **A better primary rule for `ml_meta`.** The meta-labeller currently filters a
  rule that is right 49.9% of the time. Filtering a coin flip yields a filtered
  coin flip.
- **Order-book imbalance and on-chain flows.** The remaining sources of
  genuinely new information, now that funding and open interest are in.
- **Lower-cost execution.** The clear front-runner, and the only lever never
  pulled. Eight approaches have now died on turnover, and the wide
  cross-sectional sweep put a number on it: the same book is Sharpe **1.72 at
  3bps and -0.57 at 100bps**. Nothing else tried here — features, labels,
  models, timeframes, universes — moved a result by remotely that much. Maker-
  only execution, or a venue where these names are not 50bps wide, is the one
  change that could revive a dead result rather than produce a new one.

*(Removed: "a wider cross-section" — done, and it died. See the 136-symbol
entry above. It confirmed the diversification hypothesis, raising AUC to 0.5575
and cutting drawdown to -43.8% at k=15, and still lost to slippage.)*
