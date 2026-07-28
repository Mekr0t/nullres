"""nullres — an honest research harness for systematic trading strategies.

The design rule that governs every module in this package:

    A feature at bar t may only use information available at the CLOSE of bar t.
    A position decided at bar t is filled at the OPEN of bar t+1.
    Every position change pays fees and slippage.

Anything that violates those three lines produces a backtest that cannot be
traded. `nullres.audit` exists to catch violations mechanically rather than by
hoping someone re-reads the label definition.
"""

__version__ = "0.1.0"
