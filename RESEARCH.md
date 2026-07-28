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

## 1. Single-asset direction — BTCUSDT

Purged walk-forward, 12bps/side, out-of-sample 2020-12 to 2025-12.

### 1h (44,190 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 355.3% | 0.46 | -77.2% | 2 | 0.2% |
| sma_cross | 173.4% | 0.45 | -57.6% | 308 | 30.9% |
| donchian | 107.7% | 0.38 | -54.7% | 304 | 30.6% |
| ml_direction | -15.2% | -0.06 | -69.6% | 132 | 17.7% |
| ml_meta | -35.4% | -0.18 | -72.4% | 127 | 14.9% |

### 4h (9,980 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| donchian | 144.4% | **0.52** | -40.7% | 66 | 7.6% |
| buy & hold | 148.6% | 0.33 | -77.0% | 2 | 0.2% |
| sma_cross | 74.8% | 0.30 | -56.2% | 62 | 7.2% |
| vol_target | 63.6% | 0.21 | -73.1% | 274 | 5.2% |
| ml_direction | -29.2% | -0.15 | -49.4% | 112 | 16.5% |
| ml_meta | -73.3% | -0.66 | -79.7% | 113 | 13.8% |

### 1d (1,424 OOS bars)

| strategy | total | sharpe | max dd | trades | cost drag |
|---|---|---|---|---|---|
| buy & hold | 101.6% | 0.29 | -66.7% | 1 | 0.1% |
| sma_cross | 85.8% | **0.36** | -37.3% | 8 | 1.0% |
| donchian | 48.9% | 0.30 | -25.7% | 12 | 1.4% |
| ml_meta | 20.8% | 0.13 | -40.1% | 67 | 7.8% |
| ml_direction | 9.6% | 0.04 | -64.3% | 65 | 10.2% |

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
| 4h | 1.3430% | 133 | 22.2 days |
| 1d | 3.2912% | 21 | 21.0 days |

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
| neighbourhood | FAIL — 39% sign flips vs 38% expected | FAIL — 48% vs 48% |
| stability | ok — 2 of 3 years | FAIL — 0 of 3 |
| transfer | FAIL — 2 of 4 symbols | FAIL — 2 of 4 |

The models disagree about which assets they work on, which is the tell:

```
                 ETH     BNB     SOL     XRP    (Sharpe vs buy & hold)
  ml_direction  +0.02   -0.71   +0.81   -1.62
  ml_meta       -1.13   -0.13   +0.59   +0.52
```

---

## 3. Cross-sectional long/short

### 11-symbol universe (fixed at 2021-12)

Out-of-sample 2022-10-30 to 2025-12-24, 6,909 bars. Mean AUC **0.5443**,
4/5 folds above 0.5.

| book | total | sharpe | max dd | t-stat | trades |
|---|---|---|---|---|---|
| static_vs_alts | 219.6% | **0.78** | -41.8% | 1.57 | 3 |
| btc_only | 227.9% | 0.71 | -34.7% | 1.44 | 2 |
| longshort_k2 | 161.5% | 0.50 | -78.6% | 1.01 | 166 |
| equal_weight | 15.0% | 0.06 | -57.7% | 0.12 | 4 |

Verification: shuffled labels 0.4970; survivors-only 0.5518 (so not
death-detection); per-symbol accuracy spread 0.093, BTC 0.585 vs SOL 0.491.

### 136-symbol universe (enumerated from the archive at 2021-12)

Top-40 by trailing dollar volume, 9 delisted symbols retained. Mean AUC
**0.5575**, 5/5 folds above 0.5 and rising (0.552 → 0.581).

| book | total | sharpe | max dd | t-stat | trades |
|---|---|---|---|---|---|
| longshort_k2 | 2.39e+02x | **1.61** | -67.9% | **3.26** | 192 |
| longshort_k5 | 2481.0% | 1.53 | -58.8% | 3.09 | 211 |
| static_vs_alts | 1078.9% | 1.40 | -40.3% | 2.83 | 628 |
| longshort_k10 | 649.0% | 1.32 | -53.1% | 2.66 | 237 |
| longshort_k15 | 252.9% | 1.05 | -43.8% | 2.11 | 271 |
| btc_only | 227.9% | 0.71 | -34.7% | 1.44 | 2 |
| equal_weight | -65.5% | -0.41 | -78.5% | -0.83 | 569 |

**Cost sensitivity — the decisive measurement.** Fee held at 5bps:

| slippage/side | k=2 | k=5 | static |
|---|---|---|---|
| 3bps | 1.61 | 1.53 | 1.40 |
| 25bps | 1.28 | 1.06 | 1.36 |
| 50bps | 0.90 | 0.53 | 1.31 |
| 100bps | **0.15** | **-0.49** | 1.20 |

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
