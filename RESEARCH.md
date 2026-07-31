# Research report

The numbers. [docs/05-graveyard.md](docs/05-graveyard.md) explains what each
one means and why the hypothesis died; this file is the measurements.

Every result below is **out-of-sample** in the walk-forward sense: purged and
embargoed splits, positions decided at a bar's close and filled at the next
bar's open, fees and slippage charged on every change. Every result is also
**in-sample with respect to the research process** — the symbols, date ranges
and directions were chosen by someone who knew how the period ended. That
caveat is not boilerplate; see [02 — Leakage](docs/02-leakage.md#7-hindsight-in-the-research-process-itself).

---

## Read this before any table below: the multiple-testing correction

**220 distinct parameter combinations** produced this document, and the run
ledger holds every one of them:

```
  robust  130      sweep  50      run  24      xsec  14      ablate  2
```

`prior_trials` used to declare an estimated 214 combinations explored before the
ledger existed. It is now **0**, because `scripts/reproduce_all.py` re-ran that
work and the ledger records it — the two 25-cell sweeps match the old estimate
at exactly 50, and the batteries over-cover it at 130 against ~80. Keeping both
counted the same experiments twice, which made a result deflate harder for
having been verified.

`deflated_sharpe` subtracts the Sharpe you would expect to reach by luck given
the number of trials. Re-deflating the headline results at 220:

| result | sharpe | deflated @ 6 | deflated @ 220 |
|---|---|---|---|
| xsec wide k=2 | 2.02 | 1.29 | **0.45** |
| xsec wide k=10 | 1.74 | 1.01 | 0.17 |
| xsec wide static_vs_alts | 1.60 | 0.87 | 0.03 |
| xsec wide k=15 | 1.43 | 0.70 | -0.14 |
| xsec wide k=5 | 1.40 | 0.67 | -0.17 |
| xsec 11-symbol static | 0.89 | 0.16 | -0.68 |
| donchian 4h | 0.59 | -0.14 | -0.98 |
| BTC buy & hold, 1h | 0.50 | -0.23 | -1.07 |

### The honest reading of that 0.45

An earlier version of this file reported the best figure as **0.05** and
concluded that nothing here survived its own correction. That is no longer true
as stated, and the history of the number is worth more than the number:

| | sharpe | basis | deflated |
|---|---|---|---|
| as first published | 1.61 | 214 | 0.05 |
| metrics measured on the out-of-sample window only | 1.80 | 214 | 0.23 |
| trial count deduplicated, `prior_trials` retired | 1.80 | 220 | 0.23 |
| open-interest features added to the panel | **2.02** | 220 | **0.45** |

The first two steps were corrections: Sharpe was being diluted across bars the
strategy never traded, and the same experiments were being counted twice. Both
were errors in the conservative direction, and fixing them raised the figure.

**The last step is not a correction, and it should not be believed.** Adding the
nine open-interest features lifted the k=2 Sharpe from 1.80 to 2.02 while
*lowering* mean AUC from 0.5575 to 0.5496 — the model discriminates worse and
the equity curve got luckier. This repository already has a rule for that
situation, in the derivatives entry of the graveyard: at these trade counts the
curve is decided by which handful of positions landed, so believe the AUC. On
that rule the honest figure is the **0.23** of the 37-feature panel, and 0.45 is
what deflating a lucky Sharpe looks like.

Deflation cannot see this. It corrects for how many times you looked; it cannot
tell you that the thing you are correcting was noise to begin with.

Sensitivity, so no single number is taken as precise:

```
  trials      1      6     25    100    220    500   1000
  deflated  2.02   1.29   0.90   0.60   0.45   0.30   0.19
```

And none of it rescues the book. The cost sweep in §3.2 takes k=2 from 2.02 to
**0.24** at realistic alt slippage, an argument that never depended on the
deflation at all.

## 1. Single-asset direction — BTCUSDT

Purged walk-forward, 12bps/side, out-of-sample 2020-12 to 2025-12.

### 1h (44,190 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 355.3% | 0.50 | -77.2% | 2 | 0.2% |
| sma_cross | 173.4% | 0.49 | -57.6% | 308 | 30.9% |
| donchian | 107.7% | 0.41 | -54.7% | 304 | 30.6% |
| ml_direction | -15.2% | -0.07 | -69.6% | 132 | 17.7% |
| ml_meta | -35.4% | -0.20 | -72.4% | 127 | 14.9% |
| mean_reversion | -92.8% | -1.16 | -94.0% | 1,330 | 79.7% |

### 4h (9,980 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| donchian | 144.4% | **0.59** | -40.7% | 66 | 7.6% |
| buy & hold | 148.6% | 0.38 | -77.0% | 2 | 0.2% |
| sma_cross | 74.8% | 0.35 | -56.2% | 62 | 7.2% |
| vol_target | 63.6% | 0.24 | -73.1% | 274 | 5.2% |
| ml_direction | -29.2% | -0.17 | -49.4% | 112 | 16.5% |
| mean_reversion | -69.3% | -0.66 | -86.0% | 290 | 29.4% |
| ml_meta | -73.3% | -0.75 | -79.7% | 113 | 13.8% |

### 1d (1,424 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 101.6% | 0.35 | -66.7% | 1 | 0.1% |
| sma_cross | 85.8% | **0.43** | -37.3% | 8 | 1.0% |
| donchian | 48.9% | 0.36 | -25.7% | 12 | 1.4% |
| ml_meta | 20.8% | 0.15 | -40.1% | 67 | 7.8% |
| ml_direction | 9.6% | 0.05 | -64.3% | 65 | 10.2% |
| mean_reversion | -58.8% | -0.59 | -66.1% | 37 | 4.3% |

Totals, trade counts and cost drags here match this file's first publication
exactly; only Sharpe and CAGR moved. Two independent fixes account for that.
Metrics are now measured on the out-of-sample window rather than diluted across
bars the strategy never traded, which raised every Sharpe by `1/sqrt(oos
fraction)`. And the trade that closes a position still open at the window's edge
is charged again — masking had been dropping it, which is why buy & hold briefly
reported one trade for a round trip.

Daily is the only config where ML returns are positive, and they are the least
trustworthy numbers here: per-fold AUC decays monotonically and ends *below*
coin-flip (0.548 → 0.541 → 0.422 → 0.422). Two good folds in 2022-23 outweighing
two bad ones in 2024-25, on 2,200 bars.

### The cost budget, measured

Break-even holding **duration** is invariant to bar size, because volatility
scales as `sqrt(bar length)`:

| timeframe | sigma/bar | modelled bars @51% | measured bars | measured duration |
|---|---|---|---|---|
| 1h | 0.6715% | 502 | **557** | 23.2 days |
| 4h | 1.3054% | 133 | **140** | 23.3 days |
| 1d | 3.2905% | 21 | **24** | 24.0 days |

Each sigma is the realised standard deviation of that timeframe's log returns.
The 4h row previously read 1.3430%, which is `0.6715% x 2` — the sqrt-scaling
rule, not a measurement, quoted in a table whose entire purpose is to check that
rule against reality.

The **modelled** column assumes Gaussian returns; the **measured** one sums the
actual overlapping windows. The gap is not a constant that could be divided
out — it changes sign with the horizon. On 1h bars the Gaussian overstates the
typical move by 25% at one bar and understates it by nothing at 720, because
aggregation pulls the distribution toward normal while drift pushes long moves
past it. So the modelled figures are **too forgiving at short holds**, which is
the wrong direction for a tool built to kill ideas early.

The conclusion is unchanged and slightly stronger: break-even is **~23 days**
regardless of timeframe, not ~21. Pinned by
`tests/test_costs.py::test_breakeven_duration_is_invariant_to_timeframe`.

The same correction moves the headline single-bar number. At one-bar holds the
Gaussian says 72.4% accuracy is needed; measured, it is **79.8%**. The baseline
was even further from viable than this document used to claim.

---

## 2. Derivatives data — funding rates and open interest

BTCUSDT 4h, 2020-09 to 2025-12, matched-sample ablation (identical rows, folds
and benchmark; only the feature set differs).

```
  with derivatives      0.5455  0.5303  0.5368  0.5115  0.5343
                        mean 0.5317   std 0.0126   folds above 0.5: 5/5
  without derivatives   0.5014  0.4973  0.5224  0.5164  0.5619
                        mean 0.5199   std 0.0257   folds above 0.5: 4/5

  per-fold difference   +0.0441  +0.0330  +0.0144  -0.0049  -0.0276
  paired t-test over 5 folds: t = 0.92, p = 0.412
```

Reproduce: `python -m nullres ablate --config configs/btc_4h_deriv.toml`

**Feature importance.** `ls_accounts` — the long/short account ratio — is the
most important single feature in the model, ahead of every price-derived one.
Four of the top ten are derivatives. Funding *rate* itself carries nothing:
`funding`, `funding_ma_3` and `funding_ma_21` all rank at or below zero.

**Robustness battery.** Both strategies KILLED:

| test | ml_direction | ml_meta |
|---|---|---|
| neighbourhood | **FAIL** — 39% sign flips vs 38% expected | **FAIL** — 48% vs 48% |
| stability | ok — 2 of 3 years, mean excess -0.07 (p=0.96) | inconclusive — 0 of 3, mean -0.99 (p=0.18) |
| transfer | inconclusive — 2 of 4, mean -0.48 (p=0.52) | inconclusive — 2 of 4, mean -0.04 (p=0.94) |

Both are **KILLED**, and on the neighbourhood test alone: their grids are no
smoother than randomly scattered signs. The stability and transfer counts point
the same way but cannot carry a verdict at three years and four symbols — see
the note on gate power in [01 — Workflow](docs/01-workflow.md).

The models disagree about which assets they work on, which is the tell:

```
                 ETH     BNB     SOL     XRP    (Sharpe vs buy & hold)
  ml_direction  +0.02   -0.90   +1.03   -2.07
  ml_meta       -1.43   -0.16   +0.75   +0.67
```

---

## 3. Cross-sectional long/short

### 11-symbol universe (fixed at 2021-12)

Out-of-sample 2022-10-30 to 2025-12-24, 6,909 bars. Mean AUC **0.5443**,
4/5 folds above 0.5.

| book | total | sharpe | max dd | t-stat | trades |
|---|---|---|---|---|---|
| static_vs_alts | 219.6% | **0.89** | -41.8% | 1.57 | 3 |
| btc_only | 227.9% | 0.81 | -34.7% | 1.44 | 2 |
| longshort_k2 | 170.5% | 0.59 | -78.6% | 1.05 | 165 |
| equal_weight | 15.0% | 0.07 | -57.7% | 0.12 | 4 |

Verification (`nullres xsec --verify`): shuffled labels **0.4986**;
survivors-only **0.5480**, so not death-detection; 7.1% of absolute P&L from the
two symbols that delisted; tail census 0.16 expected hits against 0 observed.

**Per-symbol skill, and the number that explains the rest.** Panel AUC is
0.5443. Broken down per symbol it is **0.501 at the median, with 5 of 10 symbols
below 0.5** — best XRPUSDT 0.546, worst MATICUSDT 0.458, spread 0.088.

Pooled AUC counts every (timestamp, symbol) pair together, so a model that has
only learned *which symbols are persistently better* scores above chance from
the cross-sectional ordering alone, without knowing when any individual symbol
will do well. That is what these numbers say happened: there is almost no
within-symbol timing skill anywhere in this universe.

And a stable cross-sectional ordering is a static bet — which is why
`static_vs_alts`, with no model and two trades, beats the book.

This replaces an earlier reading based on raw per-symbol *accuracy* (BTC 0.585
vs SOL 0.491). Accuracy is inflated by each symbol's own base rate: the label is
"beats the cross-sectional median", so a coin that persistently lagged scores
high for a model that learned nothing but the lag. BTC's 0.585 is essentially
BTC's base rate. Per-symbol AUC is base-rate free and says the skill was never
there to be concentrated.

**The 46-vs-37 feature comparison, which matters more than it looks.** Re-run
without open-interest features (`--set data.metrics=false`), this panel scores
mean AUC **0.5460** — marginally *higher* than the 0.5443 above — while the k=2
book's Sharpe falls from **0.59 to 0.20** and its total return from 170.5% to
38.1%. Same bars, same folds, nine features different. Discrimination did not
move; the equity curve moved by two thirds. At ~160 trades the Sharpe is decided
by which handful of positions landed, and this is the cleanest demonstration of
it in the repository.

### 136-symbol universe (enumerated from the archive at 2021-12)

Top-40 by trailing dollar volume, 9 delisted symbols retained. Mean AUC
**0.5496**, 5/5 folds above 0.5 (0.532 / 0.535 / 0.570 / 0.557 / 0.555).

| book | total | sharpe | max dd | t-stat | trades |
|---|---|---|---|---|---|
| longshort_k2 | 3.78e+02x | **2.02** | -73.5% | **3.59** | 187 |
| longshort_k10 | 764.1% | 1.74 | -49.6% | 3.08 | 235 |
| static_vs_alts | 1078.9% | 1.60 | -40.3% | 2.84 | 628 |
| longshort_k15 | 299.1% | 1.43 | -42.0% | 2.54 | 269 |
| longshort_k5 | 1231.1% | 1.40 | -64.3% | 2.49 | 199 |
| btc_only | 227.9% | 0.81 | -34.7% | 1.44 | 2 |
| equal_weight | -65.5% | -0.47 | -78.5% | -0.83 | 569 |

The k=2 total fell from a previously reported 2.39e+02x because the book used to
keep trading past the last bar it had a prediction for. `panel_positions` holds
its last weights to the end of the panel, and only the *benchmarks* were
restricted to the out-of-sample window — so the model's books collected ~42 bars
that `static_vs_alts` and `btc_only` were zeroed out of. Both sides are now
measured on the same bars.

**Every long/short book here carries 2x gross notional** — 100% long and 100%
short — against zero net. That is inherent to dollar neutrality, not a leverage
choice, and `sizing.max_leverage` does not apply to the panel (it clips a
single-asset position and `crosssec.py` never reads it). It is reported as
`gross` in the results table because net-zero hides it: `expo` reads 100%
whether a book carries 1x or 5x.

It applies to `static_vs_alts` too, which matters when reading it as the
"simpler alternative" — it is simpler in turnover and in needing no model, but
it requires the same margin. `btc_only` and `equal_weight` are the only 1x books
in the comparison.

**The open-interest features make this panel WORSE, not better.** Matched on
everything but the feature set:

| | 37 features | 46 features |
|---|---|---|
| narrow (11) | 0.5460 | 0.5443 |
| wide (136) | **0.5575** | 0.5496 |

Adding the nine positioning features lowers mean AUC in both panels, and width is
worth less at 46 features (+0.0053) than at 37 (+0.0115). That does not
contradict §2, which found the same data *helps* single-asset BTC: positioning is
far noisier for small alts, and rank-transforming it across a heterogeneous
universe adds noise rather than signal. The feature §2 calls the most important
in the model is a liability in the cross-section.

It is also why the k=2 Sharpe of 2.02 should not be believed — see the note on
the headline figure at the top of this file. Discrimination fell; the curve got
luckier.

**A k=2 book can hold 100% of capital short a single name.** `_neutralise`
preserves dollar-neutrality when a symbol delists by rescaling the surviving
side, so when one of two shorts dies the other goes from -0.5 to **-1.0**. Gross
exposure and net are unchanged; the move that ruins the book halves, from +200%
to +100%. It happened on 88 of 6,934 bars held (1.3%), longest unbroken stretch
14 bars. The behaviour is deliberate — dollar-neutrality is the book's defining
constraint and the alternatives change the strategy rather than make it safer —
but nothing reported it until `--verify` did.

**Cost sensitivity — the decisive measurement.** Fee held at 5bps:

| slippage/side | k=2 | k=5 | k=10 | k=15 | static |
|---|---|---|---|---|---|
| 3bps | 2.02 | 1.40 | 1.74 | 1.43 | 1.60 |
| 25bps | 1.62 | 0.85 | 1.03 | 0.67 | 1.54 |
| 50bps | 1.15 | 0.22 | 0.24 | -0.18 | 1.49 |
| 100bps | **0.24** | **-0.97** | -1.23 | -1.71 | 1.37 |

The model's column collapses; the static book barely moves. What survives
realistic alt slippage needs no model — and is hindsight, since BTC is its long
leg only because we know how 2022-2025 ended.

**Verification** (`nullres xsec --universe 2021-12 --top-n 40 --verify`):
shuffled labels **0.5017**; survivors-only **0.5504**, so not death-detection;
**2.6%** of absolute P&L from the nine symbols that delisted; contributors
spread across STORJ, BNB, BAKE, BEL long and ETH, DOGE, TRB, REEF short. Extreme
bars are genuine market events (LUNA +346% in one 4h bar is the real
death-spiral bounce), not gap artefacts.

**The skill here is distributed, and that is what separates this from the
11-symbol book.** Per-symbol AUC over the 105 symbols with at least 200 scored
bars has a **median of 0.521, with 67 of 105 above 0.5** — best ATAUSDT 0.797,
worst OMGUSDT 0.193.

Read that against §3.1, where the same decomposition gives a median of 0.501 and
5 of 10 symbols below 0.5. The narrow book's pooled AUC was almost entirely
cross-sectional ordering: it had learned which symbols were persistently better,
which is a static bet wearing a model. This one has genuine within-symbol timing
skill spread across most of the universe. Width did not just add names, it added
a different kind of signal.

Which makes the cost result below the more damning, not the less. This is the
one book in the project whose edge is real, distributed and verified from five
directions — and it still does not survive its own slippage.

**Tail risk is untested, not absent.** No bar lost more than 11.8% and the book
was never short a name that then exploded — but >+65% four-hour moves occur ~12
times in 1.1M bars against ~13,800 short-name-bars, so the expected count is
**0.15**. Zero is what chance predicts. One such event against a -0.5 weight is
-137% of capital, and the engine models no margin.

---

## Reproducing

Every result above is described entirely by a file in `configs/`. The data
cache is not committed; `nullres fetch` rebuilds it from Binance's public
archives.

```bash
python -m nullres run    --config configs/btc_4h.toml
python -m nullres ablate --config configs/btc_4h_deriv.toml
python -m nullres xsec   --config configs/xsec_4h.toml --universe 2021-12 \
                         --top-n 40 --set data.metrics=false
```

Pinned dependency versions are in `requirements.txt`. Backtest numbers are only
comparable across runs when the numerics are identical — a changed default in a
pandas rolling operation can shift every feature without failing a test.

The verification figures quoted above are regenerated by `--verify`:

```bash
python -m nullres xsec --config configs/xsec_4h.toml --verify
```

which runs the shuffled-label control, the survivors-only refit, per-symbol
accuracy, P&L attribution and the tail census, and prints them together. For a
while these were the only numbers in this file that no command reproduced —
quoted from one-off analyses whose code was never committed, sitting underneath
the strongest result in the project. That was exactly backwards, and
`nullres/panelaudit.py` now holds them.

Some regenerated values differ from the hand-computed originals, because the
method is now pinned rather than remembered. The shuffled-label control permutes
**within each timestamp** so the label stays balanced in every regime, and draws
from a stream seeded by the timestamp so the answer does not depend on
processing order: 0.4986 against the 0.4970 once quoted. Survivors-only gives
0.5480 against 0.5518. Both still say the same thing. The tail census and the
contributor lists reproduce exactly.

The per-symbol figure changed for a substantive reason rather than a procedural
one — it is now AUC rather than accuracy, because accuracy in a
beats-the-median panel is inflated by each symbol's own base rate. See §3.1.
