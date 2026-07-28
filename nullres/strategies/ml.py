"""Machine-learning strategies.

Two formulations, and the difference between them matters more than the model:

  MLDirection  Predict the direction. The model must answer "which way", which
               on liquid intraday crypto is close to unanswerable.

  MLMeta       Meta-labelling. A simple rule decides WHICH WAY to trade; the
               model only decides WHETHER TO TAKE the trade. This is a far
               easier question — the model is allowed to say "I don't know" by
               declining, and declining is free. It also turns an unbalanced
               3-class problem into a clean binary one, and the model's output
               maps naturally onto position size.

If you only take one structural idea from this repo, take the second one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nullres.backtest.sizing import apply_min_hold, apply_vol_target, signal_to_position
from nullres.models.classifier import fit_predict_walk_forward
from nullres.strategies.base import Context, cached_proba, mask_to_oos
from nullres.strategies.rules import SMACross


class MLDirection:
    name = "ml_direction"

    def positions(self, ctx: Context) -> pd.Series:
        cfg = ctx.cfg
        proba, _ = cached_proba(ctx, self.name, lambda: fit_predict_walk_forward(
            ctx.features,
            ctx.label["y"],
            ctx.label["t_end"].to_numpy(dtype=np.int64),
            cfg.split,
            cfg.model,
            verbose=ctx.verbose,
        ))

        pos = signal_to_position(
            proba, cfg.sizing, sigma=ctx.sigma,
            bars_per_year=cfg.data.bars_per_year,
        )
        return mask_to_oos(pos, ctx)


class MLMeta:
    """Meta-labelling on top of a moving-average trend filter.

    The primary rule supplies the side. The label becomes "was the rule right?",
    which is trained only on bars where the rule actually had a position — the
    model never wastes capacity on bars it will not trade.
    """

    name = "ml_meta"

    def __init__(self, primary=None):
        self.primary = primary or SMACross(fast=24, slow=120, allow_short=True)

    def positions(self, ctx: Context) -> pd.Series:
        cfg = ctx.cfg

        # 1. Primary rule -> side. Evaluated on the full index; it uses no labels.
        side = self.primary.positions(
            Context(ctx.bars, ctx.features, ctx.label, cfg,
                    pd.Series(True, index=ctx.bars.index))
        )

        # 2. Meta-label: did the move over the label's horizon go the rule's way?
        ret = ctx.label["ret"]
        y_meta = pd.Series(np.nan, index=ctx.bars.index, dtype="float64")
        active = (side != 0) & ret.notna() & ctx.label["y"].notna()
        y_meta[active] = (side[active] * ret[active] > 0).astype("float64")

        # 3. The side is a legitimate input — it is known at the bar's close.
        X = ctx.features.copy()
        X["primary_side"] = side

        if ctx.verbose:
            print(f"  primary rule active on {active.mean():.0%} of bars, "
                  f"right {y_meta.mean():.1%} of the time — the model's job is to "
                  f"tell those apart")

        proba, _ = cached_proba(ctx, self.name, lambda: fit_predict_walk_forward(
            X, y_meta, ctx.label["t_end"].to_numpy(dtype=np.int64),
            cfg.split, cfg.model, verbose=ctx.verbose,
        ))

        # 4. Size the bet. `proba` is P(the rule is right), so it is one-sided:
        #    a low value means "skip", never "reverse". Shorting on low
        #    confidence would be trading against our own primary signal.
        take = proba >= cfg.sizing.long_entry
        desired = side.where(take, 0.0)
        desired[proba.isna()] = np.nan          # outside the OOS window

        if not cfg.sizing.allow_short:
            desired = desired.clip(lower=0.0)

        pos = apply_min_hold(desired, cfg.sizing.min_hold)
        pos = apply_vol_target(pos, ctx.sigma, cfg.sizing, cfg.data.bars_per_year)
        return mask_to_oos(pos, ctx)
