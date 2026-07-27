"""The volatility findings that killed the vol-targeting hypothesis.

These assert properties of real BTC data, so they skip when the parquet cache
is absent (CI does not commit `data/`). Reproduce the cache with:

    python -m tbot fetch --config configs/btc_4h.toml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tbot.config import load_config

pytestmark = pytest.mark.slow

CACHED = Path("data").glob("BTCUSDT-4h-*.parquet")
needs_data = pytest.mark.skipif(
    not any(CACHED), reason="BTCUSDT 4h cache absent; run `tbot fetch`"
)


@pytest.fixture(scope="module")
def btc():
    from tbot.data import load_bars

    cfg = load_config("configs/btc_4h.toml")
    bars = load_bars(cfg.data)
    return bars, np.log(bars["close"]).diff(), cfg.data.bars_per_year


@needs_data
def test_volatility_is_far_more_predictable_than_direction(btc):
    """The premise behind vol targeting, and it holds.

    This asymmetry is why "predict direction" is the wrong question to ask of
    this data, and it is worth keeping pinned: if a future refactor makes it
    stop holding, something upstream has broken.
    """
    _, logret, _ = btc
    vol = logret.rolling(30).std()

    ret_ac = abs(logret.autocorr(1))
    vol_ac = vol.autocorr(1)

    assert vol_ac > 0.9, f"30-bar vol should be near-deterministic, got {vol_ac:.3f}"
    assert ret_ac < 0.05, f"returns should be ~unpredictable, got {ret_ac:.3f}"
    assert vol_ac > 10 * ret_ac


@needs_data
def test_high_volatility_does_not_predict_losses_in_btc(btc):
    """Why vol targeting works on equities and fails here.

    Equities show a leverage effect: volatility spikes accompany crashes, so
    cutting exposure when vol rises avoids losses. BTC has no such asymmetry —
    its biggest rallies are as violent as its crashes. De-risking on volatility
    therefore cuts the upside just as hard as the downside, and pays fees to
    do it.

    Measured: corr(vol, forward 30-bar return) = +0.059, and the HIGHEST vol
    quintile has the highest mean forward return (+1.40% vs +1.12% lowest).
    """
    _, logret, _ = btc
    vol = logret.rolling(30).std()
    fwd = logret.rolling(30).sum().shift(-30)
    ok = vol.notna() & fwd.notna()

    corr = float(vol[ok].corr(fwd[ok]))
    assert corr > -0.05, (
        f"corr(vol, forward return) = {corr:.4f}. A strongly negative value "
        f"would mean vol targeting has a mechanism to exploit here; it does not."
    )


@needs_data
def test_updown_moves_are_symmetric_in_magnitude(btc):
    """No leverage effect: up bars are about as large as down bars."""
    _, logret, _ = btc
    up = logret[logret > 0].abs().mean()
    down = logret[logret < 0].abs().mean()
    assert abs(up - down) / down < 0.10, (
        f"up {up:.4%} vs down {down:.4%} — asymmetry this small leaves vol "
        f"targeting nothing directional to exploit"
    )


@needs_data
def test_capped_vol_target_is_dominated_by_construction(btc):
    """With max_leverage=1.0 the strategy can only ever hold LESS than buy & hold.

    It is capped at full exposure on ~54% of bars, so its position is <= 1.0
    everywhere. In a rising market that guarantees lower gross return, and it
    pays fees for the privilege. Any benefit has to come purely from the
    volatility reduction — which the test above shows is not there.
    """
    _, logret, bpy = btc
    ann = logret.rolling(30).std() * np.sqrt(bpy)
    pos = (0.50 / ann).clip(0, 1.0).dropna()

    assert (pos <= 1.0 + 1e-12).all()
    assert pos.mean() < 1.0
    assert (pos >= 0.999).mean() > 0.4, "should sit at the cap much of the time"
