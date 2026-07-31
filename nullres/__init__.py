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
"""

import logging

from nullres.errors import (
    ConfigError,
    DataUnavailableError,
    InsufficientDataError,
    NullresError,
)

__version__ = "0.1.0"

# No handler, and no "no handlers could be found" warning either. The consumer
# owns logging configuration; a library that attaches its own handler at import
# time takes that decision away from them.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "NullresError",
    "ConfigError",
    "DataUnavailableError",
    "InsufficientDataError",
]
