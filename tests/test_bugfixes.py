"""Regressions for the defects found in the July 2026 audit.

Each test names the wrong behaviour it pins against, because a test whose
assertion looks arbitrary gets "fixed" by the next person to touch the code.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from nullres.backtest.engine import backtest
from nullres.backtest.metrics import summarize
from nullres.config import CostConfig, DataConfig
from nullres.errors import ConfigError, InsufficientDataError


@pytest.fixture
def bars():
    idx = pd.date_range("2021-01-01", periods=500, freq="h")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 500)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1.0, "trades": 10.0}, index=idx)


# ---------------------------------------------------------------------------
# An empty measurement window
# ---------------------------------------------------------------------------

def test_summarize_refuses_an_empty_window_instead_of_crashing(bars):
    """It raised `IndexError: single positional indexer is out-of-bounds`.

    That came from `equity.iloc[-1]` on an empty frame, five lines into a
    function whose caller — `robustness.cross_symbol` — catches per-symbol
    failures so one bad symbol cannot take down the transfer test. IndexError
    was not in what it caught, so it did.
    """
    result = backtest(bars, pd.Series(1.0, index=bars.index), CostConfig())
    with pytest.raises(InsufficientDataError, match="empty"):
        summarize(result, 8_760, mask=pd.Series(False, index=bars.index))


def test_cross_symbol_records_an_empty_window_as_a_note(monkeypatch, bars):
    """The point of the exception type: the battery survives one bad symbol."""
    from nullres import robustness

    def explode(cfg, verbose=True):
        raise InsufficientDataError("no usable bars")

    monkeypatch.setattr(robustness, "run_pipeline", explode)
    from nullres.config import RunConfig

    frame = robustness.cross_symbol(RunConfig(), "donchian", ["AAA", "BBB"])
    assert len(frame) == 2
    assert frame["sharpe"].isna().all()
    assert all("no usable bars" in note for note in frame["note"])


def test_summarize_still_works_on_a_one_bar_window(bars):
    """The guard must not have widened into rejecting small-but-real windows."""
    mask = pd.Series(False, index=bars.index)
    mask.iloc[10] = True
    metrics = summarize(
        backtest(bars, pd.Series(1.0, index=bars.index), CostConfig()),
        8_760, mask=mask)
    assert metrics["bars"] == 1


# ---------------------------------------------------------------------------
# RSI at zero loss
# ---------------------------------------------------------------------------

def test_rsi_is_100_when_nothing_fell_not_nan():
    """`up / dn.replace(0, nan)` made a pure uptrend NaN.

    RSI is 100 there by definition. The NaN mattered more than it looks:
    `pipeline.prepare` keeps only rows where every feature is present, so one
    NaN RSI evicted the entire bar and the other 45 features with it —
    silent sample loss concentrated in exactly the strong trends a momentum
    feature is meant to describe.
    """
    from nullres.features.technical import rsi

    out = rsi(pd.Series(np.arange(100, 200, dtype=float)))
    assert out.iloc[-1] == pytest.approx(100.0)
    assert out.notna().sum() == 99, "only the first diff should be NaN"


def test_rsi_is_50_on_a_flat_series():
    from nullres.features.technical import rsi

    out = rsi(pd.Series([100.0] * 60))
    assert out.iloc[-1] == pytest.approx(50.0)


def test_rsi_warmup_stays_nan():
    """The fix must not manufacture values where there is genuinely no data."""
    from nullres.features.technical import rsi

    assert pd.isna(rsi(pd.Series([100.0, 101.0])).iloc[0])


def test_rsi_is_unchanged_on_an_ordinary_series():
    """A two-sided series must produce exactly what it always did."""
    from nullres.features.technical import rsi

    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    out = rsi(close)
    assert out.dropna().between(0, 100).all()
    assert pd.isna(out.iloc[0]), "the first diff has no history"
    assert 20 < out.iloc[-1] < 80


# ---------------------------------------------------------------------------
# The funding-scale unit table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval,hours", [
    ("1h", 1), ("2h", 2), ("4h", 4), ("6h", 6), ("12h", 12), ("1d", 24),
    ("30m", 0.5), ("15m", 0.25),
])
def test_bar_hours_are_derived_for_every_supported_interval(interval, hours):
    """A lookup table held six intervals and defaulted to 4 for the rest.

    `BARS_PER_YEAR` accepts 1m through 1w, so a 30m panel scaled its funding
    charge as though the bars were 4h — an 8x error, applied silently. Deriving
    from `bars_per_year` gives one answer per interval and a ConfigError for
    anything unknown.
    """
    assert 8_760 / DataConfig(interval=interval).bars_per_year == pytest.approx(hours)


def test_an_unknown_interval_raises_rather_than_defaulting():
    with pytest.raises(ConfigError, match="unknown interval"):
        _ = DataConfig(interval="3h").bars_per_year


def test_crosssec_no_longer_carries_its_own_interval_table():
    from nullres import crosssec

    assert not hasattr(crosssec, "_INTERVAL_HOURS"), (
        "the duplicate interval table is back; derive hours from "
        "cfg.data.bars_per_year so there is one source of truth"
    )


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------

def _result_from(returns: np.ndarray):
    """A BacktestResult carrying exactly these per-bar net returns."""
    from nullres.backtest.engine import BacktestResult

    idx = pd.date_range("2021-01-01", periods=len(returns), freq="h")
    r = pd.Series(returns, index=idx)
    zeros = pd.Series(0.0, index=idx)
    return BacktestResult(equity=np.exp(r.cumsum()), returns=r, gross=r,
                          position=pd.Series(1.0, index=idx),
                          turnover=zeros, cost=zeros)


def test_sortino_uses_downside_deviation_not_the_std_of_losses():
    """The old denominator was `r[r < 0].std()`.

    That is computed about the mean loss rather than about zero, and divides by
    the count of losing bars rather than by all of them — so it is smaller and
    the Sortino it produces is larger. Measured at 13% too high.
    """
    r = np.random.default_rng(0).normal(0.001, 0.01, 5_000)
    metrics = summarize(_result_from(r), 8_760)

    correct = float(r.mean() / np.sqrt(np.minimum(r, 0) ** 2 @ np.ones(len(r))
                                       / len(r)) * np.sqrt(8_760))
    inflated = float(r.mean() / r[r < 0].std(ddof=1) * np.sqrt(8_760))

    assert metrics["sortino"] == pytest.approx(correct, rel=1e-9)
    assert metrics["sortino"] < inflated, "the old formula flattered the ratio"
    assert inflated / metrics["sortino"] > 1.10, (
        "the two formulas should differ materially on this series"
    )


def test_sortino_is_zero_when_nothing_ever_fell():
    assert summarize(_result_from(np.full(100, 0.001)), 8_760)["sortino"] == 0.0


# ---------------------------------------------------------------------------
# The prediction cache
# ---------------------------------------------------------------------------

def test_cache_key_covers_the_primary_rules_parameters():
    """`MLMeta` puts `primary_side` into the feature matrix.

    The fingerprint was (label, split, model) only, so two MLMeta strategies
    with different primary rules sharing one prepared context would have taken
    each other's predictions. A cache hit is indistinguishable from a fast
    computation, so nothing would have shown.
    """
    from nullres.strategies.base import strategy_fingerprint
    from nullres.strategies.ml import MLMeta
    from nullres.strategies.rules import SMACross

    a = strategy_fingerprint(MLMeta(SMACross(fast=24, slow=120)).primary)
    b = strategy_fingerprint(MLMeta(SMACross(fast=50, slow=200)).primary)
    assert a != b, "differing primary parameters must produce different keys"
    assert a == strategy_fingerprint(SMACross(fast=24, slow=120)), (
        "the fingerprint must be a function of the parameters, not of id()"
    )


def test_cached_proba_misses_when_the_extra_key_changes():
    from nullres.config import RunConfig
    from nullres.strategies.base import Context, cached_proba

    calls = []

    def compute():
        calls.append(1)
        return pd.Series([0.5]), []

    idx = pd.date_range("2021-01-01", periods=1, freq="h")
    ctx = Context(bars=pd.DataFrame(index=idx), features=pd.DataFrame(index=idx),
                  label=pd.DataFrame(index=idx), cfg=RunConfig(),
                  oos_mask=pd.Series([True], index=idx))

    cached_proba(ctx, "ml_meta", compute, extra="primary=A")
    cached_proba(ctx, "ml_meta", compute, extra="primary=A")
    assert len(calls) == 1, "identical inputs must hit the cache"

    cached_proba(ctx, "ml_meta", compute, extra="primary=B")
    assert len(calls) == 2, "a changed primary rule must miss the cache"


# ---------------------------------------------------------------------------
# Cache writes
# ---------------------------------------------------------------------------

def test_parquet_writes_are_atomic(tmp_path):
    """`to_parquet` writes in place, so an interrupted multi-hour download left
    a truncated file at the path every later run treats as authoritative."""
    from nullres.data.cache import write_parquet_atomic

    target = tmp_path / "x.parquet"
    write_parquet_atomic(pd.DataFrame({"a": [1, 2, 3]}), target)
    assert pd.read_parquet(target)["a"].tolist() == [1, 2, 3]
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"


def test_an_interrupted_write_leaves_no_partial_file_and_no_litter(tmp_path,
                                                                   monkeypatch):
    from nullres.data import cache

    target = tmp_path / "x.parquet"
    cache.write_parquet_atomic(pd.DataFrame({"a": [1]}), target)

    def die(self, *a, **k):
        raise KeyboardInterrupt("lid closed mid-download")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", die)
    with pytest.raises(KeyboardInterrupt):
        cache.write_parquet_atomic(pd.DataFrame({"a": [9, 9]}), target)

    # The previous good file survives untouched, and nothing is left over.
    assert pd.read_parquet(target)["a"].tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_corrupt_cache_file_is_discarded_not_fatal(tmp_path, caplog):
    """Files written before atomic writes may already be truncated. A corrupt
    entry should cost one re-download, not an afternoon reading tracebacks."""
    import logging

    from nullres.data.cache import read_parquet_or_discard

    bad = tmp_path / "truncated.parquet"
    bad.write_bytes(b"PAR1 this is not a parquet file")
    with caplog.at_level(logging.WARNING, logger="nullres.data.cache"):
        assert read_parquet_or_discard(bad) is None
    assert not bad.exists(), "the unreadable file must be removed, not left"
    assert "discarding unreadable cache file" in caplog.text


def test_fetch_month_treats_a_corrupt_cache_entry_as_a_miss(tmp_path, monkeypatch):
    """End to end: a bad cached month must not poison every future run."""
    from nullres.data import binance

    (tmp_path / "BTCUSDT-1h-2021-01.parquet").write_bytes(b"not parquet")
    monkeypatch.setattr(binance.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            binance.requests.RequestException("offline")))
    # Falls through to the network (which fails here) instead of raising a
    # parquet decode error from inside the cache read.
    assert binance.fetch_month("BTCUSDT", "1h", "2021-01", str(tmp_path)) is None


# ---------------------------------------------------------------------------
# CLI edge cases
# ---------------------------------------------------------------------------

def test_trials_zero_is_rejected_rather_than_silently_ignored():
    """`args.trials or trials_so_far(...)` treated 0 as "not given"."""
    from nullres import api
    from nullres.config import RunConfig

    with pytest.raises(ConfigError, match="at least 1"):
        api.run(RunConfig(), n_trials=0)


def test_log_runs_without_a_config_file(tmp_path, monkeypatch, capsys):
    """`nullres log` reads the ledger and nothing else.

    It used to load configs/btc_1h.toml first, so the one command that should
    work in any checkout failed when that file was absent. Since the CLI moved
    to subcommands the guarantee is stronger: `log` has no `--config` to load.
    """
    from nullres.cli import main

    monkeypatch.chdir(tmp_path)          # no configs/ directory here at all
    assert main(["log"]) == 0
    assert "No runs recorded yet" in capsys.readouterr().out


def test_log_does_not_accept_a_config_at_all():
    """The flat parser took --config for every command, needed or not."""
    from nullres.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["log", "--config", "anything.toml"])
    assert caught.value.code == 2
