"""Survivorship detection.

The leakage catalogue used to call this "not detectable mechanically". That was
true before `universe_as_of` and delisting tracking existed; the dominant
failure mode is now checkable, and the residue that isn't is stated plainly
rather than left implied.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nullres.audit import check_survivorship
from nullres.data.universe import delisted_from_cache


def test_universe_with_no_corpses_fails():
    """The dominant failure mode: eleven coins, all still trading."""
    survivors = [f"COIN{i}USDT" for i in range(11)]
    check = check_survivorship(survivors, delisted={})
    assert check.applicable
    assert not check.passed
    assert "survivors" in check.detail


def test_hardcoded_universe_says_so_in_the_failure():
    check = check_survivorship([f"C{i}" for i in range(11)], delisted={},
                               hardcoded=True)
    assert not check.passed
    assert "hardcoded" in check.detail


def test_universe_holding_delisted_assets_passes():
    symbols = [f"COIN{i}USDT" for i in range(11)]
    check = check_survivorship(symbols, delisted={"LUNAUSDT": "2022-05",
                                                  "MATICUSDT": "2024-09"})
    assert check.passed
    assert "LUNAUSDT" in check.detail


def test_single_symbol_is_not_applicable_rather_than_a_pass():
    """A vacuous green tick is worse than silence.

    Reporting PASS here would claim survivorship was ruled out for a backtest
    where it was never examined — which is the exact species of false assurance
    this whole module exists to prevent.
    """
    check = check_survivorship(["BTCUSDT"], delisted={})
    assert not check.applicable
    assert "does not apply" in check.detail
    assert "NOT been ruled out" in check.detail
    assert "n/a" in str(check)


def test_dropping_most_of_the_point_in_time_universe_fails():
    """Trading 5 of 100 symbols that existed needs a reason."""
    traded = [f"COIN{i}USDT" for i in range(5)]
    existed = [f"COIN{i}USDT" for i in range(100)]
    check = check_survivorship(traded, delisted={"COIN1USDT": "2022-05"},
                               point_in_time=existed)
    assert not check.passed
    assert "excluded" in check.detail


def test_keeping_most_of_the_point_in_time_universe_passes():
    traded = [f"COIN{i}USDT" for i in range(90)]
    existed = [f"COIN{i}USDT" for i in range(100)]
    check = check_survivorship(traded, delisted={"COIN1USDT": "2022-05"},
                               point_in_time=existed)
    assert check.passed


def test_delisted_detection_reads_the_local_cache(tmp_path):
    """Offline: a symbol whose archive stops early is delisted."""
    for month in pd.period_range("2022-01", "2025-12", freq="M"):
        (tmp_path / f"um-ALIVEUSDT-4h-{month}.parquet").touch()
    for month in pd.period_range("2022-01", "2022-05", freq="M"):
        (tmp_path / f"um-DEADUSDT-4h-{month}.parquet").touch()

    found = delisted_from_cache(["ALIVEUSDT", "DEADUSDT"], "4h", "2025-12",
                                cache_dir=str(tmp_path))
    assert set(found) == {"DEADUSDT"}
    assert found["DEADUSDT"] == "2022-05"


def test_recent_months_are_not_mistaken_for_delisting(tmp_path):
    """The archive lags the present; that is not an asset dying."""
    for month in pd.period_range("2022-01", "2025-11", freq="M"):
        (tmp_path / f"um-LAGGYUSDT-4h-{month}.parquet").touch()
    assert delisted_from_cache(["LAGGYUSDT"], "4h", "2025-12",
                               cache_dir=str(tmp_path)) == {}


def test_the_repos_own_universe_would_pass():
    """The 2021-12 universe contains LUNA and MATIC, both of which died."""
    from nullres.crosssec import UNIVERSE_2021_12

    check = check_survivorship(
        UNIVERSE_2021_12,
        delisted={"LUNAUSDT": "2022-05", "MATICUSDT": "2024-09"},
    )
    assert check.passed
