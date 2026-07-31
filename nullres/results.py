"""What each command computed, as data rather than as text.

Every command in this repo used to end in a wall of `print`. That made three
things impossible at once: calling a command from Python and getting its
numbers, testing what a command reports without capturing stdout, and
documenting the output without copying a terminal transcript into markdown.

The split is: `nullres.api` computes and returns one of these; `nullres.report`
turns one of these into the text a terminal shows. Nothing here computes and
nothing here formats. A field is either something a command measured or
something it needs in order to explain what it measured.

These types ARE the public API. Adding a field is a compatible change; removing
or renaming one is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                    # keeps importing this module cheap
    import numpy as np
    import pandas as pd

    from nullres.audit import Check
    from nullres.config import RunConfig
    from nullres.crosssec import Panel
    from nullres.runlog import RunRecord


@dataclass
class RunResult:
    """`nullres run` — every configured strategy, measured out of sample."""

    cfg: RunConfig
    metrics: dict[str, dict]         # {strategy name: metric dict}
    n_trials: int
    #: Non-empty when some ledger records predate variant counting, which makes
    #: `n_trials` a floor and every deflated Sharpe below it generous.
    trials_caveat: str = ""
    #: Non-empty when this config sits near something already marked KILLED.
    killed_warning: str = ""
    #: Feature group dropped for a matched-sample ablation, if any.
    ablated: str | None = None
    features_before: int = 0
    features_after: int = 0
    record: RunRecord | None = None

    @property
    def benchmark(self) -> dict:
        """Buy & hold's metrics — the bar everything else has to clear."""
        return self.metrics.get("buy_hold", {})

    @property
    def survivors(self) -> list[str]:
        """Strategies whose deflated Sharpe is still above zero.

        Deflated, not raw: a positive Sharpe that does not survive the trial
        count is a strategy that looked, not a strategy that found.
        """
        return [n for n, m in self.metrics.items()
                if m.get("deflated_sharpe", 0.0) > 0 and n != "buy_hold"]


@dataclass
class AuditResult:
    """`nullres audit` — the five mechanical leak checks."""

    cfg: RunConfig
    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.applicable and not c.passed]

    @property
    def skipped(self) -> list[Check]:
        """Checks that could not apply. Not passes — nothing was ruled out."""
        return [c for c in self.checks if not c.applicable]

    @property
    def passed(self) -> bool:
        return not self.failed

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


@dataclass
class BudgetResult:
    """`nullres budget` — the arithmetic that decides most questions."""

    cfg: RunConfig
    sigma: float                     # per-bar stdev of log returns
    logret: np.ndarray               # the measured returns behind it
    hours_per_bar: float
    min_hold: int


@dataclass
class SweepResult:
    """`nullres sweep` — out-of-sample Sharpe across entry x hold."""

    cfg: RunConfig
    strategy: str
    entries: list[float]
    holds: list[int]
    #: Long form, one row per cell: entry, hold, sharpe.
    cells: pd.DataFrame
    record: RunRecord | None = None

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def grid(self) -> pd.DataFrame:
        """Sharpe as an entry x hold matrix."""
        return self.cells.pivot(index="entry", columns="hold", values="sharpe")


@dataclass
class RobustnessResult:
    """`nullres robust` — three independent attempts to falsify a strategy."""

    cfg: RunConfig
    strategy: str
    params: dict[str, Any]
    #: One row per parameter combination, with its Sharpe.
    grid: pd.DataFrame
    grid_keys: list[str]
    grid_kind: str                   # "params" | "sizing"
    flip_rate: float
    flip_pairs: int
    stability: pd.DataFrame          # per-year, against buy & hold
    transfer: pd.DataFrame           # per-symbol, against buy & hold
    benchmark_sharpe: float
    verdict: str                     # KILLED | SURVIVED | INCONCLUSIVE
    notes: list[str]
    symbols: list[str]
    transfer_start: str | None = None
    killed_warning: str = ""
    record: RunRecord | None = None

    @property
    def killed(self) -> bool:
        return self.verdict == "KILLED"

    @property
    def survived(self) -> bool:
        return self.verdict == "SURVIVED"


@dataclass
class AblationResult:
    """`nullres ablate` — does a feature group improve DISCRIMINATION?

    Measured on AUC across folds, not Sharpe. With ~80 trades an equity curve
    is decided by which handful of positions landed; AUC uses every labelled
    bar, so it can tell "the model knows more" from "the model got luckier".
    """

    cfg: RunConfig
    group: str
    n_rows: int
    features_with: int
    features_without: int
    auc_with: np.ndarray             # per fold
    auc_without: np.ndarray          # per fold
    t_stat: float
    p_value: float
    record: RunRecord | None = None

    @property
    def delta(self) -> np.ndarray:
        """Per-fold AUC difference, with the group minus without it."""
        return self.auc_with - self.auc_without

    @property
    def significant(self) -> bool:
        return bool(self.p_value <= 0.05)


@dataclass
class PanelVerification:
    """The controls that decide whether cross-sectional skill is real.

    Expensive — the first two refit the whole walk-forward — so this is only
    populated when `verify=True`.
    """

    mean_auc: float
    shuffled_auc: float
    survivors_auc: float | None      # None when no symbol delisted
    per_symbol: pd.DataFrame         # auc, accuracy, base_rate, n, lift
    delisted_share: float            # of ABSOLUTE P&L, deliberately not netted
    contribution: pd.Series          # gross log P&L per symbol
    tail_curve: pd.DataFrame
    tail_census: dict
    concentration: dict | None = None
    min_obs: int = 200
    nominal_weight: float | None = None

    @property
    def shuffle_is_clean(self) -> bool:
        return abs(self.shuffled_auc - 0.5) < 0.02

    @property
    def detects_death(self) -> bool:
        """True when dropping delisted names collapses the AUC.

        A model that only knows which coins are dying has found something real
        and untradable: by the time a coin is dying its borrow has vanished.
        """
        if self.survivors_auc is None:
            return False
        return (self.mean_auc - self.survivors_auc) > 0.02


@dataclass
class XsecResult:
    """`nullres xsec` — cross-sectional long/short on a panel."""

    cfg: RunConfig
    panel: Panel
    proba: pd.Series                 # P(outperforms), MultiIndex (ts, symbol)
    fold_reports: list[dict]
    #: The universe as requested, BEFORE liquidity screening and before
    #: symbols with no archive were dropped. `panel.symbols` is what survived.
    #: The gap between the two is what `check_survivorship` reports on.
    requested_symbols: list[str]
    survivorship: Check
    oos_times: pd.DatetimeIndex
    oos_mask: pd.Series
    #: {book name: metrics}, benchmarks and longshort_k* together.
    books: dict[str, dict]
    ks: tuple[int, ...]
    positions: dict[int, pd.DataFrame] = field(default_factory=dict)
    stability: pd.DataFrame | None = None
    stability_k: int | None = None
    #: Sharpe by slippage level; one row per bps, one column per book.
    cost_sensitivity: pd.DataFrame | None = None
    n_trials: int = 1
    trials_caveat: str = ""
    universe_month: str | None = None
    verification: PanelVerification | None = None
    record: RunRecord | None = None

    @property
    def mean_auc(self) -> float:
        import numpy as np

        return float(np.nanmean([r["auc"] for r in self.fold_reports]))

    @property
    def heaviest_book(self) -> tuple[str, dict]:
        """The book carrying the most gross notional.

        A dollar-neutral book hides its leverage: net is 0 and `exposure` reads
        100% whether it carries 1x or 5x.
        """
        return max(self.books.items(),
                   key=lambda kv: kv[1].get("gross_exposure", 0.0))


@dataclass
class FeatureImportanceResult:
    """`nullres features` — permutation importance on the last fold's test window."""

    cfg: RunConfig
    importances: pd.Series           # descending, indexed by feature name
    derivative_names: set[str]       # which came from funding / open interest

    @property
    def derivatives_in_top_10(self) -> list[str]:
        return [n for n in self.importances.index[:10]
                if n in self.derivative_names]


@dataclass
class FetchResult:
    """`nullres fetch` — what is now in the cache."""

    cfg: RunConfig
    bars: pd.DataFrame
    funding: pd.DataFrame | None = None
    metrics: pd.DataFrame | None = None


@dataclass
class LedgerView:
    """`nullres log` — the evidence ledger, summarised."""

    runs: list[RunRecord]            # after any verdict filter
    n_total: int
    n_configs: int
    n_trials: int
    killed: int
    survived: int
    inconclusive: int
    unrecorded_variants: int
    verdict_filter: str | None = None
    limit: int = 25


__all__ = [
    "RunResult", "AuditResult", "BudgetResult", "SweepResult",
    "RobustnessResult", "AblationResult", "PanelVerification", "XsecResult",
    "FeatureImportanceResult", "FetchResult", "LedgerView",
]
