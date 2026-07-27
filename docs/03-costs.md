# 03 — Costs and execution

The arithmetic that decides whether a research direction is worth pursuing.
Almost nobody runs it before spending a month on features.

```bash
python -m tbot budget --config configs/btc_1h.toml
```

---

## The equation

For a directional strategy:

```
edge             = 2 * accuracy - 1
E|move| over h   = sigma * sqrt(h) * sqrt(2/pi)     driftless walk
gross per trade  = edge * E|move|
cost per trade   = 2 * (fee + slippage)             round trip, both sides
```

Break-even needs `gross >= cost`. Rearranged either way:

```
required_accuracy(h)  = 0.5 * (1 + cost / E|move|(h))
breakeven_hold(acc)   = (cost / (edge * sigma * sqrt(2/pi)))^2
```

Both live in `tbot/costs.py`. `tests/test_costs.py` asserts they are exact
inverses of each other, because they are the same equation rearranged and it
would be embarrassing if they disagreed.

## What it says about hourly BTC

Measured sigma is 0.6715% per bar. At Binance spot taker (10bps) plus 2bps
slippage, a round trip costs 0.24%.

```
hold (bars)      E|move|    accuracy needed
  1                0.54%      72.4%
  6                1.31%      59.1%
  12               1.86%      56.5%
  24               2.62%      54.6%
  168              6.94%      51.7%
  720             14.38%      50.8%
```

And inverted — how long you must hold, given the accuracy you actually have:

```
accuracy      hold (bars)     ~duration
51%                   502     20.9 days
52%                   125      5.2 days
55%                    20      0.8 days
60%                     5      0.2 days
```

**The measured out-of-sample accuracy of the gradient-boosted model on this
data is 50.2%–52.2%, AUC 0.50–0.53.** Read that against the table. A 51% model
needs to hold for three weeks. The original baseline held for one bar.

This is why the baseline lost 100%, and why no amount of feature engineering
would have rescued it. The costs were structurally larger than the signal.

## Changing timeframe is NOT a cost lever

This one is worth getting right, because the intuition points the wrong way.

Moving from 1h to 1d bars *looks* like it should help: daily bars move ~3.3%
against the same fixed 24bps round trip, versus 0.67% for hourly. The cost is a
much smaller fraction of the move.

It does not help, because volatility scales as `sqrt(bar length)`:

```
sigma      ∝ sqrt(bar_length)
breakeven bars ∝ 1 / sigma^2  ∝  1 / bar_length
breakeven DURATION = bars x bar_length = constant
```

Measured on BTCUSDT at 51% accuracy and 12bps/side:

```
1h    sigma 0.6715%   ->  502 bars   =  20.9 days
4h    sigma 1.3430%   ->  133 bars   =  22.2 days
1d    sigma 3.2912%   ->   21 bars   =  21.0 days
```

**Break-even is ~21 days regardless of timeframe.** `tests/test_costs.py::
test_breakeven_duration_is_invariant_to_timeframe` pins this.

So what does timeframe change? Signal-to-noise (fewer, larger bars carry less
microstructure noise), sample size (52,000 hourly bars vs 2,200 daily), and how
precisely you can time an entry. Those are real considerations. Costs are not
among them.

Choose the **timeframe** for signal quality and sample size. Choose the
**holding period** to beat costs. They are separate decisions, and conflating
them will send you chasing a saving that does not exist.

## The three levers, in order of effect

### 1. Hold longer (in wall-clock time, not bars)

Turnover is the dominant term, and `sizing.min_hold` is a hard cap:
`n_bars / min_hold` state changes, maximum. Going from `min_hold=12` to `336`
on the 1h config took cost drag from 97.5% to 17.7% and total return from
-97.7% to -15.2%. Nothing about the model changed.

### 2. Match the label horizon to the holding period

A model trained to predict 24 bars ahead tells you nothing about whether to
hold for 500. If `tbot budget` says you need 500-bar holds, the label has to
ask a 500-bar question.

`pipeline.coherence_warnings` prints a warning when `label.horizon` and
`sizing.min_hold` differ by more than 4x, because that configuration produces
a number that means nothing.

### 3. Use hysteresis, not a single threshold

A single cutoff makes the position chatter every time the signal grazes it.
Separate entry and exit bands stop that:

```toml
long_entry = 0.56    # enter above this
long_exit  = 0.50    # but don't leave until below THIS
```

`tests/test_sizing.py` shows a signal oscillating `0.57, 0.55, 0.57, 0.55...`
producing **one** trade with hysteresis and **99** without.

## Execution timing

```
decide at close[t]  ->  fill at open[t+1]  ->  exit at open[t+2]
position[t] earns log(open[t+2]) - log(open[t+1])
cost charged on |position[t] - position[t-1]| at the fill
```

## What the engine does not model

Be aware of these before trusting any number it produces:

- **Partial fills and depth.** It assumes your size never moves the market.
  True for $1k, false for $1M.
- **Slippage correlated with signal.** Modelled as a constant. In reality it is
  worst exactly when your signal is strongest, because everyone else's signal
  fired too.
- **Funding rates.** Irrelevant for spot, material for perpetual futures.
- **Exchange downtime.** Binance has halted during precisely the moves a
  strategy most wants to trade.
- **Maker fees.** The configs assume taker (10bps). Resting limit orders are
  cheaper but do not always fill — and modelling that honestly requires
  order-book data this repo does not load.

Every one of these makes live results worse than the backtest. None make them
better. When in doubt, raise `cost.slippage_bps` and see whether your conclusion
survives.
