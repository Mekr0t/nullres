# 05 — The graveyard

Every hypothesis this repo has killed, why it died, and what it cost to find
out. Kept because the most expensive mistake in research is re-running a dead
end you already disproved and forgot about.

Add to it every time something dies. A negative result you can point at is
worth more than a positive one you can't reproduce.

---

## Predict next-bar direction with a gradient-boosted classifier

**Status:** dead. **Cost to disprove:** the original `baseline.py`.

Out-of-sample accuracy 54.25%, genuinely ~2.4σ above coin-flip, and it still
lost 100%. It changed position 15,527 times in 47,502 bars; at 12bps that is
18.6 in log cost. The model was never the problem.

**Lesson:** run `tbot budget` first. At one-bar holds this needed 72.4%
accuracy to break even. Nothing was ever going to close that gap.

---

## Direction prediction at any horizon, 1h / 4h / 1d

**Status:** dead. **Cost:** the whole `ml_direction` and `ml_meta` line of work.

Fold AUCs across every timeframe and label: **0.50–0.53**. On daily the AUCs
decay monotonically across folds and end *below* coin-flip (0.548 → 0.541 →
0.422 → 0.422) — a model fitting a regime that then ended.

Permutation importance on 32 features: only `vol_72` clears 0.01, and a third
score negative.

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
1h   502 bars = 20.9 days      4h   133 bars = 22.2 days      1d   21 bars = 21.0 days
```

**Lesson:** choose the timeframe for signal quality and sample size; choose the
holding period to beat costs. Pinned by
`test_breakeven_duration_is_invariant_to_timeframe`.

---

## Donchian breakout on 4h

**Status:** dead as an alpha source. Alive as a drawdown tool, with caveats.

Looked like the best thing in the repo: Sharpe 0.52 vs buy & hold's 0.33,
-41% max drawdown vs -77%, on 66 untuned trades. Survived the parameter
neighbourhood test (100% of 19 cells positive) and cross-symbol transfer
(positive on ETH, BNB, SOL, XRP).

Died on sub-period stability once buy & hold was placed beside it:

```
  year     excess sharpe
  2021          +0.21
  2022          +0.56     <- the entire result
  2023          -0.50
  2024          -0.21
  2025          -0.27
```

The whole five-year advantage is 2022, where it cut a -64.7% year to -26.8%.
In every trending year it surrendered roughly half the rally.

**Lesson:** a long-only filter over a bull market is profitable most years by
construction. Judge against the benchmark per period, not against zero. This
finding is what turned `verdict()` from an absolute test into an excess one.

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

Battery verdict: **KILLED**. Beat buy & hold in **0% of years**; negative on 3
of 4 other symbols.

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

## Still open

- **A better primary rule for `ml_meta`.** The meta-labeller currently filters a
  rule that is right 49.7% of the time. Filtering a coin flip yields a filtered
  coin flip.
- **Data outside the price series.** Funding rates, open interest, order-book
  imbalance, on-chain flows. The one direction not yet ruled out, and the only
  one that brings genuinely new information.
- **Cross-sectional ranking.** "Which of these 50 assets outperforms the rest"
  is an easier question than "does this one go up", and it hedges the market
  move. Introduces survivorship bias as a live concern.
