# API reference

Everything the command line does is reachable from Python, because the command
line is one of two front ends over the same functions rather than the only way
in.

The package splits three ways, and the split is enforced by tests rather than
by convention:

| module | job | does not |
|---|---|---|
| [`nullres.api`](commands.md) | compute, return a result object | print, format |
| [`nullres.results`](results.md) | hold what was measured | compute, format |
| [`nullres.report`](report.md) | turn a result into text | compute, read files |

```python
from nullres import load_config, run
from nullres import report

result = run(load_config("configs/btc_4h.toml"))

result.metrics["donchian"]["sharpe"]     # the number
result.survivors                         # deflated Sharpe still above zero
print(report.format_run(result))         # the text the terminal shows
```

## Progress output

Importing `nullres` is silent and configures no logging. Library modules log to
the `nullres` logger, which has a `NullHandler` attached and nothing else — so a
consumer sees nothing until they ask:

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
```

Per-module granularity works, because each module logs under its own name:

```python
logging.getLogger("nullres.crosssec").setLevel(logging.WARNING)
```

## The ledger

The command functions append to the run ledger in `runs/` by default. That is
not incidental: `deflated_sharpe` reads the ledger to find out how many variants
have been tried, and a run that goes unrecorded undercounts the exposure and
flatters every result that follows it.

Pass `record=False` when you are genuinely not testing a hypothesis — re-deriving
a number for a plot, say — and be honest about which case you are in.

## Errors

Every deliberate failure is a [`NullresError`](errors.md) subclass carrying the
exit code the CLI returns for it. Nothing in the library raises `SystemExit`,
so importing `nullres` cannot terminate the process that imported it; a test
fails if that changes.
