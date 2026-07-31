"""nullres — an honest research harness for systematic trading strategies.

The design rule that governs every module in this package:

    A feature at bar t may only use information available at the CLOSE of bar t.
    A position decided at bar t is filled at the OPEN of bar t+1.
    Every position change pays fees and slippage.

Anything that violates those three lines produces a backtest that cannot be
traded. `nullres.audit` exists to catch violations mechanically rather than by
hoping someone re-reads the label definition.

Importing this package is side-effect free and prints nothing. Progress goes to
the `nullres` logger, which has no handler of its own — so by default a library
consumer sees silence, and `nullres.cli` attaches a stdout handler when the
command line is the caller. To see it from your own code:

    logging.basicConfig(level=logging.INFO)

The public API is one function per command, each returning a result object
rather than printing:

    from nullres import load_config, run
    result = run(load_config("configs/btc_4h.toml"))
    result.metrics["donchian"]["sharpe"]
    result.survivors                     # deflated Sharpe still above zero

`nullres.report` turns any of those result objects back into the text the
terminal shows, so the CLI is a thin layer over the two and nothing is
reachable from the command line that is not reachable from Python.
"""

import logging

from nullres.api import (
    ablate,
    audit,
    budget,
    fetch,
    feature_importance,
    killed_warning,
    ledger,
    resolve_universe,
    robust,
    run,
    sweep,
    verify_panel,
    xsec,
)
from nullres.config import RunConfig, load_config
from nullres.errors import (
    ConfigError,
    DataUnavailableError,
    InsufficientDataError,
    NullresError,
)
from nullres.results import (
    AblationResult,
    AuditResult,
    BudgetResult,
    FeatureImportanceResult,
    FetchResult,
    LedgerView,
    PanelVerification,
    RobustnessResult,
    RunResult,
    SweepResult,
    XsecResult,
)

__version__ = "0.1.0"

# No handler, and no "no handlers could be found" warning either. The consumer
# owns logging configuration; a library that attaches its own handler at import
# time takes that decision away from them.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    # config
    "load_config", "RunConfig",
    # commands
    "fetch", "ledger", "budget", "run", "audit", "sweep", "robust", "ablate",
    "feature_importance", "xsec", "resolve_universe", "verify_panel",
    "killed_warning",
    # results
    "RunResult", "AuditResult", "BudgetResult", "SweepResult",
    "RobustnessResult", "AblationResult", "PanelVerification", "XsecResult",
    "FeatureImportanceResult", "FetchResult", "LedgerView",
    # errors
    "NullresError", "ConfigError", "DataUnavailableError",
    "InsufficientDataError",
]
