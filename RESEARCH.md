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

Roughly **214 parameter combinations** were explored to produce this document —
six configs across ~six strategies, two 25-cell threshold sweeps, four
robustness batteries of ~20 grid cells each, ablations, k-sweeps and cost
sweeps. Searching that many variants finds good-looking results in pure noise.

`deflated_sharpe` subtracts the Sharpe you would expect to reach by luck given
the number of trials. It was originally wired to the count of strategies in a
single run — six — which badly understated the exposure. Re-deflating the
headline results honestly:

| result | sharpe | deflated @ 6 | deflated @ 214 | deflated @ 343 |
|---|---|---|---|---|
| xsec wide k=2 | 1.80 | 1.07 | **0.23** | **0.15** |
| xsec wide k=5 | 1.72 | 0.99 | 0.15 | 0.07 |
| xsec wide static_vs_alts | 1.60 | 0.87 | 0.03 | -0.05 |
| xsec wide k=10 | 1.50 | 0.77 | -0.07 | -0.15 |
| xsec wide k=15 | 1.20 | 0.47 | -0.37 | -0.45 |
| xsec 11-symbol static | 0.89 | 0.16 | -0.68 | -0.76 |
| donchian 4h | 0.59 | -0.14 | -0.98 | -1.06 |
| BTC buy & hold, 1h | 0.50 | -0.23 | -1.07 | -1.15 |

**Almost nothing in this repository survives its own multiple-testing
correction, and what does survive is too small to trade.** The best result — the
wide cross-sectional book, Sharpe 1.80, t-stat 3.19 — deflates to **0.23** at
the 214 trials estimated before the ledger existed, and to **0.15** once the
ledger's own 129 are added. Positive, and far too small to act on. Every other
result in the table is at or below zero.

An earlier version of this file reported that best figure as **0.05** and
claimed nothing survived at all. Two corrections moved it: metrics are now
measured on the out-of-sample window rather than diluted across bars the
strategy never traded, which raised every Sharpe by `1/sqrt(oos fraction)`; and
the trial count now collapses repeated runs of the same experiment instead of
summing them. The conclusion is weaker than it was, and it should be — the old
number was the product of two measurement errors that happened to point the same
way.

It does not rescue the strategy. The cost sweep below still takes k=2 from 1.80
to **0.13** at realistic alt slippage, and that argument never depended on the
deflation.

Sensitivity, so the number is not taken as precise:

```
  trials      1      6     25    100    214    343    500   1000
  deflated  1.80   1.07   0.68   0.38   0.23   0.15   0.08  -0.03
```

The trial count is read from the run ledger rather than assumed, and
`prior_trials` in each config declares the exposure that predates it.
`nullres run` prints the count it used.

**A caveat on the count itself.** `prior_trials = 214` is an estimate of work
done before the ledger existed. Reproducing those same experiments now also
records them in the ledger, so they are counted twice — verifying a result makes
its deflation harsher without any new hypothesis being tested. The 343 column is
therefore an upper bound and 214 a lower one. The honest reading is that the
best result sits somewhere in **0.15 to 0.23**, and that the distinction does
not matter, because 100bps of slippage takes it to 0.13 regardless.

---

## 1. Single-asset direction — BTCUSDT

Purged walk-forward, 12bps/side, out-of-sample 2020-12 to 2025-12.

### 1h (44,190 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 355.8% | 0.50 | -77.2% | 1 | 0.1% |
| sma_cross | 173.7% | 0.49 | -57.6% | 307 | 30.8% |
| donchian | 108.0% | 0.41 | -54.7% | 303 | 30.5% |
| ml_direction | -15.1% | -0.07 | -69.6% | 131 | 17.6% |
| ml_meta | -35.4% | -0.20 | -72.4% | 127 | 14.9% |
| mean_reversion | -92.8% | -1.16 | -94.0% | 1,330 | 79.7% |

### 4h (9,980 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| donchian | 144.4% | **0.59** | -40.7% | 66 | 7.6% |
| buy & hold | 148.9% | 0.38 | -77.0% | 1 | 0.1% |
| sma_cross | 74.8% | 0.35 | -56.2% | 62 | 7.2% |
| vol_target | 63.8% | 0.24 | -73.1% | 273 | 5.0% |
| ml_direction | -29.2% | -0.17 | -49.4% | 112 | 16.5% |
| mean_reversion | -69.3% | -0.66 | -86.0% | 290 | 29.4% |
| ml_meta | -73.3% | -0.75 | -79.7% | 112 | 13.7% |

### 1d (1,424 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 101.6% | 0.35 | -66.7% | 1 | 0.1% |
| sma_cross | 85.8% | **0.43** | -37.3% | 8 | 1.0% |
| donchian | 48.9% | 0.36 | -25.7% | 12 | 1.4% |
| ml_meta | 20.8% | 0.15 | -40.1% | 67 | 7.8% |
| ml_direction | 9.6% | 0.05 | -64.3% | 65 | 10.2% |
| mean_reversion | -58.8% | -0.59 | -66.1% | 37 | 4.3% |

Daily is the only config where ML returns are positive, and they are the least
trustworthy numbers here: per-fold AUC decays monotonically and ends *below*
coin-flip (0.548 → 0.541 → 0.422 → 0.422). Two good folds in 2022-23 outweighing
two bad ones in 2024-25, on 2,200 bars.

### The cost budget, measured

Break-even holding **duration** is invariant to bar size, because volatility
scales as `sqrt(bar length)`:

| timeframe | sigma/bar | break-even bars @51% | duration |
|---|---|---|---|
| 1h | 0.6715% | 502 | 20.9 days |
| 4h | 1.3054% | 133 | 22.1 days |
| 1d | 3.2912% | 21 | 20.9 days |

Each sigma is the realised standard deviation of that timeframe's log returns.
The 4h row previously read 1.3430%, which is `0.6715% x 2` — the sqrt-scaling
rule, not a measurement, quoted in a table whose entire purpose is to check that
rule against reality.

Pinned by `tests/test_costs.py::test_breakeven_duration_is_invariant_to_timeframe`.

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
| transfer | inconclusive — 2 of 4 symbols | inconclusive — 2 of 4, mean -0.04 (p=0.94) |

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
| static_vs_alts | 220.1% | **0.89** | -41.8% | 1.57 | 2 |
| btc_only | 228.2% | 0.81 | -34.7% | 1.44 | 1 |
| longshort_k2 | 170.5% | 0.59 | -78.6% | 1.05 | 165 |
| equal_weight | 15.1% | 0.07 | -57.7% | 0.12 | 3 |

Verification (`nullres xsec --verify`): shuffled labels **0.5042**;
survivors-only **0.5480**, so not death-detection; per-symbol accuracy spread
**0.093**, BTC 0.585 vs SOL 0.491; 7.1% of gross P&L from the two symbols that
delisted; tail census 0.16 expected hits against 0 observed.

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
**0.5575**, 5/5 folds above 0.5 and rising (0.552 → 0.581).

| book | total | sharpe | max dd | t-stat | trades |
|---|---|---|---|---|---|
| longshort_k2 | 2.12e+02x | **1.80** | -67.9% | **3.19** | 191 |
| longshort_k5 | 2375.9% | 1.72 | -58.8% | 3.06 | 208 |
| static_vs_alts | 1080.8% | 1.60 | -40.3% | 2.84 | 627 |
| longshort_k10 | 654.0% | 1.50 | -53.1% | 2.67 | 234 |
| longshort_k15 | 253.9% | 1.20 | -43.8% | 2.12 | 268 |
| btc_only | 228.2% | 0.81 | -34.7% | 1.44 | 1 |
| equal_weight | -65.5% | -0.47 | -78.5% | -0.83 | 568 |

The k=2 total fell from a previously reported 2.39e+02x because the book used to
keep trading past the last bar it had a prediction for. `panel_positions` holds
its last weights to the end of the panel, and only the *benchmarks* were
restricted to the out-of-sample window — so the model's books collected ~42 bars
that `static_vs_alts` and `btc_only` were zeroed out of. Both sides are now
measured on the same bars.

**Cost sensitivity — the decisive measurement.** Fee held at 5bps:

| slippage/side | k=2 | k=5 | k=10 | k=15 | static |
|---|---|---|---|---|---|
| 3bps | 1.80 | 1.72 | 1.50 | 1.20 | 1.60 |
| 25bps | 1.42 | 1.19 | 0.91 | 0.58 | 1.55 |
| 50bps | 0.99 | 0.59 | 0.23 | -0.11 | 1.49 |
| 100bps | **0.13** | **-0.57** | -1.03 | -1.39 | 1.38 |

The model's column collapses; the static book barely moves. What survives
realistic alt slippage needs no model — and is hindsight, since BTC is its long
leg only because we know how 2022-2025 ended.

**Verification.** Shuffled labels 0.5018. Delisted symbols contribute 2% of
gross P&L. Contributors spread across STORJ, BNB, BAKE, BEL long and ETH, DOGE,
TRB, REEF short. Extreme bars are genuine market events (LUNA +346% in one 4h
bar is the real death-spiral bounce), not gap artefacts.

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

Two of the regenerated values differ slightly from the hand-computed originals,
because the method is now pinned rather than remembered: the shuffled-label
control permutes **within each timestamp** so the label stays balanced in every
regime, giving 0.5042 rather than the 0.4970 once quoted, and survivors-only
gives 0.5480 against 0.5518. Both still say the same thing. The per-symbol
spread (0.093, BTC 0.585 vs SOL 0.491) and the tail census reproduce exactly.
