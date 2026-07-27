# 04 — Labels and strategies

## Labels: what question are you asking?

The label choice matters more than the model choice. It is the difference
between a hard question and an impossible one.

### `next_bar_sign` — the baseline's label

"Will the next bar close higher?" Available for comparison; not recommended.

A 1h BTC bar moves ~0.67% on average and a round trip costs 0.24%. You are
asking a model to call a coin flip well enough to clear 36% of the typical
move, on the noisiest timescale available. Even at a genuine 54% accuracy, the
economics do not work — see [03 — Costs](03-costs.md).

### `fwd_return` — vol-scaled return over a horizon, with a dead band

Sign of the `horizon`-bar return, scaled by volatility, with small moves left
**unlabelled**:

```python
scaled = fwd_return / (sigma * sqrt(horizon))
y = 1 if scaled > deadband, 0 if scaled < -deadband, else NaN
```

The dead band matters. Without it, roughly half your training set is noise the
model dutifully tries to fit, and what it finds is spurious. Unlabelled rows
are still *predicted* — they just don't train.

### `triple_barrier` — the default

From the close of bar `t`, place three barriers: a profit target at
`+upper * sigma`, a stop at `-lower * sigma`, and a vertical barrier `horizon`
bars out. Label by whichever is touched first.

Two properties make it better than a fixed-percent target:

1. **Volatility scaling.** A 1% target is a routine move in March 2020 and a
   big one in July 2023. Mixing those trains a model that only works in whichever
   regime dominated your sample. Scaling by sigma makes the label mean the same
   thing throughout.
2. **Path dependence.** It asks whether price *travels* 1.5σ up before 1.5σ
   down — a question with real autocorrelation structure, and one that maps
   directly onto how a stop-loss actually behaves.

It also yields an honest `t_end` per row, which is what makes correct purging
possible. See [02 — Leakage](02-leakage.md#4-overlapping-labels-crossing-the-traintest-boundary).

**One deliberate pessimism:** when both barriers fall inside the same bar, OHLC
cannot tell which came first. The implementation assumes the **stop** hit first.
The alternative silently inflates every result the repo will ever produce.

### Overlapping labels need sample weights

With a 504-bar horizon, 504 consecutive rows describe almost the same stretch of
price. Treating them as 504 independent observations tells the model it has far
more evidence than it does. `validation/weights.py` computes each row's average
uniqueness — a row overlapped by 23 others carries ~1/24 the weight — and
`fit_predict_walk_forward` passes it as `sample_weight` by default.

---

## Strategies

### The rule baselines exist to be beaten

`sma_cross`, `donchian`, `mean_reversion`. Deliberately simple and deliberately
**not tuned** — a tuned rule isn't a benchmark, it's just another overfit
strategy with fewer parameters.

If a gradient-boosted model on 32 features cannot beat a 50/200 moving-average
cross after costs, the model has not learned anything worth the complexity. Say
so and move on. On this data, it cannot.

All strategies — rules included — are masked to the same out-of-sample window as
the ML strategies. A rule evaluated over six years against a model evaluated
over five is not a comparison.

### `ml_direction` — predict the direction

The obvious formulation, and the weaker one. The model must answer "which way",
which on liquid intraday crypto is close to unanswerable. Measured AUC here:
0.50–0.53.

### `ml_meta` — meta-labelling

The structural idea worth taking from this repo.

A simple rule decides **which way** to trade. The model only decides **whether
to take** the trade:

```
primary rule  ->  side ∈ {-1, 0, +1}
meta-label    ->  "was the rule right?"     (binary, balanced)
model         ->  P(rule is right)
position      ->  side  if  P > threshold  else  flat
```

Why it's easier:

- **The model may decline.** Declining is free; being wrong is not. A direction
  model must commit on every bar.
- **Clean binary target.** "Was the rule right" is naturally balanced, rather
  than an unbalanced 3-class up/down/flat problem.
- **Output maps to size.** `P(correct)` is exactly the quantity you want to
  scale a bet by.
- **Costs stay bounded.** The model can only reduce trading relative to the
  primary rule, never increase it.

It did not rescue the strategy here — the primary rule is right 49.9% of the
time and the model cannot reliably tell the good calls from the bad. But it is
the right shape for the problem, and it is where to attach a better primary
signal.

---

## Where to look next

The ranked list, given what this repo has already ruled out:

1. **Longer timeframes.** The cost arithmetic gets easier roughly as `sqrt(h)`.
   Daily bars need ~50.3% accuracy at 500-bar holds; hourly needs 72% at one bar.
2. **A better primary rule for `ml_meta`.** The model is a filter. Filtering a
   coin flip yields a filtered coin flip.
3. **Data the price series doesn't contain.** Funding rates, open interest,
   order-book imbalance, on-chain flows. Thirty-two technical indicators derived
   from OHLCV are thirty-two views of the same four numbers, and `tbot features`
   will show you most of them carry nothing out of sample — on the 4h config
   only `vol_72` clears 0.01 AUC, and a third of them score *negative*.
4. **Cross-sectional instead of time-series.** "Which of these 50 assets will
   outperform the others" is a genuinely easier question than "will this one go
   up", and it hedges out the market move. It also introduces survivorship bias
   as a live concern — see [02 — Leakage](02-leakage.md#5-survivorship-and-delisting).

Model architecture is not on this list. It is nearly always the least valuable
lever, and it is the one people reach for first.
