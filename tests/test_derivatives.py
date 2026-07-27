"""Funding / open-interest joins — where the next leak would come from.

Every previous feature in this repo derives from the bars themselves, so a
lookahead had to be an obvious `.shift(-1)`. Joining an EXTERNAL time series is
different: the bug is a join direction or an off-by-one-interval alignment, and
neither looks wrong when you read it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tbot.audit import check_point_in_time
from tbot.data.synthetic import synthetic_bars
from tbot.features import build_features
from tbot.features.derivatives import _asof, build_derivative_features


@pytest.fixture
def bars_4h():
    return synthetic_bars(n=1_200, seed=5, interval="4h", start="2020-01-01")


def _spanning(bars: pd.DataFrame, freq: str, **columns) -> pd.DataFrame:
    """Auxiliary frame covering the WHOLE bar range.

    Generating a fixed number of rows instead is how the first draft of these
    tests broke: the aux data ran out halfway through the bars, the as-of join
    held its last value flat, and every change-based feature went to zero for
    the second half of the sample.
    """
    idx = pd.date_range(bars.index[0], bars.index[-1] + pd.Timedelta(freq), freq=freq)
    return pd.DataFrame({k: fn(len(idx)) for k, fn in columns.items()}, index=idx)


@pytest.fixture
def funding_8h(bars_4h):
    """Settlements at 00:00, 08:00, 16:00 — the real Binance schedule."""
    return _spanning(bars_4h, "8h",
                     funding_rate=lambda n: np.arange(n, dtype="float64"))


@pytest.fixture
def metrics_1h(bars_4h):
    return _spanning(
        bars_4h, "1h",
        open_interest=lambda n: np.linspace(1e5, 2e5, n),
        all_accounts_ls=lambda n: np.linspace(0.9, 1.4, n),
        top_trader_positions_ls=lambda n: np.linspace(1.0, 1.3, n),
        taker_buy_sell_ratio=lambda n: np.linspace(0.8, 1.2, n),
    )


def test_join_takes_the_last_value_strictly_before_the_bar_close(bars_4h, funding_8h):
    """A 4h bar at T closes at T+4h and may use funding stamped before that.

    bar 00:00 closes 04:00 -> the 00:00 settlement (value 0)
    bar 04:00 closes 08:00 -> STILL the 00:00 settlement; 08:00 is simultaneous
                              with the close, not available before it
    bar 08:00 closes 12:00 -> the 08:00 settlement (value 1)
    """
    joined = _asof(bars_4h, funding_8h, ["funding_rate"])
    assert joined["funding_rate"].iloc[0] == 0.0
    assert joined["funding_rate"].iloc[1] == 0.0, "used a simultaneous settlement"
    assert joined["funding_rate"].iloc[2] == 1.0


def test_join_never_reaches_forward(bars_4h, funding_8h):
    """For every bar, the joined value must exist at a timestamp before its close."""
    joined = _asof(bars_4h, funding_8h, ["funding_rate"])
    interval = pd.Timedelta("4h")
    for bar_ts, value in joined["funding_rate"].dropna().items():
        source_ts = funding_8h.index[funding_8h["funding_rate"] == value]
        assert (source_ts < bar_ts + interval).all(), (
            f"bar {bar_ts} used funding from {list(source_ts)}, at or after its close"
        )


def test_clean_derivative_features_pass_the_audit(bars_4h, funding_8h, metrics_1h):
    check = check_point_in_time(
        bars_4h,
        builder=lambda d: build_features(d, funding=funding_8h, metrics=metrics_1h),
        probes=4,
    )
    assert check.passed, check.detail


def test_forward_join_is_caught_by_the_audit(bars_4h, funding_8h):
    """The realistic bug: `direction="forward"` instead of `"backward"`.

    This is why the auxiliary frames are clipped to the bar range before
    joining. Without that clipping, truncating the bars would leave the funding
    frame untouched, the forward join would return the same value either way,
    and this check would pass while the feature read tomorrow's funding.
    """
    def leaky(df):
        f = build_features(df)
        interval = pd.Timedelta("4h")
        aux = funding_8h[funding_8h.index < df.index[-1] + interval]
        joined = pd.merge_asof(
            pd.DataFrame({"close_time": df.index + interval}, index=df.index)
              .reset_index(names="bar_ts").sort_values("close_time"),
            aux.reset_index(names="aux_ts").sort_values("aux_ts"),
            left_on="close_time", right_on="aux_ts",
            direction="forward",                       # <-- the bug
        ).set_index("bar_ts")
        f["funding_ahead"] = joined["funding_rate"].reindex(df.index)
        return f

    check = check_point_in_time(bars_4h, builder=leaky, probes=4)
    assert not check.passed
    assert "funding_ahead" in check.detail


def test_open_interest_enters_only_as_changes_and_zscores(bars_4h):
    """OI grew ~10x over the sample. A level feature would encode the calendar."""
    oi = _spanning(
        bars_4h, "1h",
        open_interest=lambda n: np.exp(np.linspace(np.log(1e5), np.log(1e6), n)),
    )
    f = build_derivative_features(bars_4h, metrics=oi)

    assert "open_interest" not in f.columns, "raw OI level must not be a feature"
    # OI grows at a constant rate in log space, so a correct log-difference
    # feature is exactly constant — a level or a raw difference would trend.
    #
    # The final bar is excluded deliberately: auxiliary data is clipped to
    # `< last bar close`, so that one bar sees one fewer 1h reading and its
    # change is proportionally smaller. That is the point-in-time rule working,
    # not drift, and it is worth pinning rather than smoothing away.
    for col in ("oi_chg_6", "oi_chg_24"):
        series = f[col].dropna()
        assert len(series) > 100
        body = series.iloc[:-1]
        assert body.std() < 1e-9, f"{col} is not stationary under an exponential ramp"
        assert body.iloc[0] > 0
        assert series.iloc[-1] < body.iloc[-1], "expected the clipped final bar"


def test_missing_auxiliary_data_yields_nan_not_a_crash(bars_4h):
    empty = pd.DataFrame({"funding_rate": []},
                         index=pd.DatetimeIndex([], name="ts"))
    f = build_derivative_features(bars_4h, funding=empty)
    assert f.empty or f.isna().all().all()


def test_features_without_derivatives_are_unchanged(bars_4h, funding_8h):
    """Adding the option must not perturb the existing 32-feature baseline."""
    base = build_features(bars_4h)
    with_deriv = build_features(bars_4h, funding=funding_8h)
    pd.testing.assert_frame_equal(base, with_deriv[base.columns])
    assert with_deriv.shape[1] > base.shape[1]
