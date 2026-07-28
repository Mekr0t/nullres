"""Mechanical leakage detection.

The baseline script's closing line was right: walk-forward validation did not
catch the leak, only reading the label definition would have. That is an
unacceptable place to leave things — humans re-read code badly, and every new
feature is a fresh chance to introduce lookahead.

These five checks catch the overwhelming majority of leaks automatically:

  1. POINT-IN-TIME    Recompute features using only bars <= t and assert row t
                      is unchanged. Catches any use of future data in feature
                      construction, including the subtle ones (a global mean, a
                      backfill, an accidental negative shift).

  2. LABEL/FEATURE    Check whether any single feature predicts the label far
     CORRELATION      too well on its own. `ret_1` versus a same-bar label
                      scores ~1.0 here, which is precisely the baseline's bug.

  3. NULL DATA        Run the whole pipeline on a random walk. There is no edge
                      by construction, so a positive result after costs proves
                      a bug — in the engine, the split, or the labels.

  4. SHUFFLED LABEL   Retrain with labels randomly permuted. Out-of-sample
                      accuracy must collapse to the base rate. If it does not,
                      information is reaching the model through a side channel.

  5. SURVIVORSHIP     A multi-symbol universe spanning a period that killed
                      assets, containing none of them, was filtered by survival.
                      Reports `n/a` rather than PASS on a single-symbol config —
                      it has nothing to test there, and a green tick would claim
                      a risk was ruled out when it was never examined.

Run `nullres audit` before believing any result. It takes a minute and it has a
much better record than intuition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nullres.features import build_features


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    # Some checks cannot apply to some configs — survivorship is meaningless for
    # a single-symbol backtest. Reporting that as PASS would be a vacuous green
    # tick, which is worse than saying nothing: it implies a risk was ruled out
    # when it was never examined.
    applicable: bool = True

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        if not self.applicable:
            mark = "n/a "
        return f"  [{mark}] {self.name}\n         {self.detail}"


def check_point_in_time(bars: pd.DataFrame, builder=build_features,
                        probes: int = 5, tol: float = 1e-9,
                        seed: int = 0) -> Check:
    """Recompute features on a truncated history and compare the final row.

    If `build_features(bars[:t+1]).iloc[t]` differs from
    `build_features(bars).iloc[t]`, then the full-history version used data from
    after bar t. There is no way for that to be legitimate.
    """
    if len(bars) < 50:
        return Check("point-in-time features", False,
                     f"only {len(bars)} bars — too few to probe meaningfully")

    full = builder(bars)
    rng = np.random.default_rng(seed)
    # Probe past the warmup so rolling windows have filled, but never past the
    # end of a short series: a fixed 500-bar floor crashed on anything smaller.
    warmup = min(500, len(bars) // 2)
    lo = min(max(warmup, len(bars) // 10), len(bars) - 1)
    offenders: dict[str, float] = {}

    for t in rng.integers(lo, len(bars), size=probes):
        t = int(t)
        truncated = builder(bars.iloc[: t + 1])
        a = truncated.iloc[-1]
        b = full.iloc[t]
        for col in full.columns:
            av, bv = a[col], b[col]
            if pd.isna(av) and pd.isna(bv):
                continue
            denom = max(abs(bv), 1.0) if pd.notna(bv) else 1.0
            diff = abs(av - bv) / denom if pd.notna(av) and pd.notna(bv) else np.inf
            if diff > tol:
                offenders[col] = max(offenders.get(col, 0.0), float(diff))

    if offenders:
        worst = sorted(offenders.items(), key=lambda kv: -kv[1])[:5]
        listing = ", ".join(f"{c} ({d:.2e})" for c, d in worst)
        return Check(
            "point-in-time features", False,
            f"{len(offenders)} feature(s) change when future bars are removed: {listing}",
        )
    return Check(
        "point-in-time features", True,
        f"all {full.shape[1]} features identical across {probes} truncation probes",
    )


def check_label_leakage(features: pd.DataFrame, y: pd.Series,
                        auc_limit: float = 0.65) -> Check:
    """Flag any single feature that separates the label suspiciously well.

    A lone technical indicator with AUC > 0.65 on a directional label is not a
    discovery. On real financial data, single-feature AUCs live in 0.50-0.55.
    """
    from sklearn.metrics import roc_auc_score

    ok = y.notna()
    y_ok = y[ok].astype(int)
    if y_ok.nunique() < 2:
        return Check("single-feature AUC", False, "label has only one class")

    scores: dict[str, float] = {}
    for col in features.columns:
        v = features.loc[ok, col]
        valid = v.notna()
        if valid.sum() < 100 or y_ok[valid].nunique() < 2:
            continue
        auc = roc_auc_score(y_ok[valid], v[valid])
        scores[col] = max(auc, 1 - auc)      # direction-agnostic

    if not scores:
        return Check("single-feature AUC", False, "no feature had enough valid rows")

    worst = sorted(scores.items(), key=lambda kv: -kv[1])
    top = ", ".join(f"{c}={a:.3f}" for c, a in worst[:3])
    flagged = [c for c, a in worst if a > auc_limit]
    if flagged:
        return Check(
            "single-feature AUC", False,
            f"{len(flagged)} feature(s) exceed AUC {auc_limit}: {top}  "
            f"-- this is what a leaked label looks like",
        )
    return Check("single-feature AUC", True, f"max AUC {worst[0][1]:.3f} ({top})")


def check_null_data(run_pipeline, cfg, sharpe_limit: float = 0.5) -> Check:
    """The pipeline must find nothing on a random walk.

    `run_pipeline(cfg)` is injected to avoid a circular import.
    """
    import copy

    null_cfg = copy.deepcopy(cfg)
    null_cfg.data.source = "synthetic"
    null_cfg.name = f"{cfg.name}-null"

    results = run_pipeline(null_cfg, verbose=False)
    offenders = {
        name: m["sharpe"] for name, m in results.items()
        if name != "buy_hold" and m["sharpe"] > sharpe_limit
    }
    if offenders:
        listing = ", ".join(f"{n} sharpe {s:.2f}" for n, s in offenders.items())
        return Check(
            "null (random-walk) data", False,
            f"found 'edge' where none exists: {listing} -- there is a bug in the "
            f"engine, the split, or the labels",
        )

    best = max((m["sharpe"] for n, m in results.items() if n != "buy_hold"), default=0.0)
    return Check(
        "null (random-walk) data", True,
        f"no strategy beat sharpe {sharpe_limit} on synthetic data (best {best:.2f})",
    )


def check_survivorship(symbols, delisted, point_in_time=None,
                       hardcoded: bool = False) -> Check:
    """Does this universe contain assets that died?

    Backtesting a universe chosen from what is liquid today is a test of
    "things that survived", and it will produce a beautiful, meaningless
    result. The catalogue used to call this undetectable. It is not — not
    fully, but the dominant failure mode is mechanical:

      A multi-symbol universe spanning a period that killed assets, which
      contains none of them, was filtered by survival.

    What this CANNOT see is whether you picked the winners among the survivors.
    That is hindsight, and it stays yours to avoid — see the catalogue's entry 7.

    Args:
        symbols: the universe actually traded.
        delisted: symbols whose data stops before the sample ends.
        point_in_time: optionally, the universe as enumerated at the sample
            start. Lets the check measure how much of the graveyard was dropped.
        hardcoded: True when the universe was a literal list rather than
            enumerated from the archive as of a date.
    """
    symbols = list(symbols)
    dead = set(delisted or ())

    if len(symbols) < 2:
        return Check(
            "survivorship", True,
            f"single-symbol backtest ({symbols[0] if symbols else 'none'}) — "
            f"survivorship does not apply, and has NOT been ruled out for any "
            f"multi-asset extension of this work",
            applicable=False,
        )

    if not dead:
        detail = (
            f"none of the {len(symbols)} symbols stopped trading during the "
            f"sample. Either the period genuinely killed nothing, or the "
            f"universe was chosen from survivors"
        )
        if hardcoded:
            detail += " — and this universe is a hardcoded list, which is how "\
                      "that happens"
        return Check("survivorship", False, detail)

    share = len(dead) / len(symbols)
    detail = (f"{len(dead)} of {len(symbols)} symbols ({share:.0%}) delisted "
              f"during the sample and were held to the end: "
              f"{', '.join(sorted(dead)[:5])}"
              f"{'...' if len(dead) > 5 else ''}")

    if point_in_time:
        missing = set(point_in_time) - set(symbols)
        detail += (f". Universe covers {len(symbols)}/{len(point_in_time)} of the "
                   f"symbols trading at the sample start")
        if missing and len(missing) > len(point_in_time) * 0.5:
            return Check(
                "survivorship", False,
                detail + f" — {len(missing)} were excluded, which needs a reason "
                         f"that is not 'they are not around any more'",
            )
    return Check("survivorship", True, detail)


def check_shuffled_label(X: pd.DataFrame, y: pd.Series, t_end: np.ndarray,
                         split_cfg, model_cfg, tol: float = 0.02,
                         seed: int = 0) -> Check:
    """Permuted labels must be unlearnable.

    Any accuracy above the base rate here means the model is reaching the
    target through something other than the features it was given.
    """
    from nullres.models.classifier import fit_predict_walk_forward

    rng = np.random.default_rng(seed)
    shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)

    proba, reports = fit_predict_walk_forward(
        X, shuffled, t_end, split_cfg, model_cfg, verbose=False
    )
    scored = proba.notna() & shuffled.notna()
    acc = float(((proba[scored] > 0.5) == (shuffled[scored] > 0.5)).mean())
    base = float(max(shuffled[scored].mean(), 1 - shuffled[scored].mean()))

    if acc > base + tol:
        return Check(
            "shuffled-label control", False,
            f"accuracy {acc:.4f} exceeds base rate {base:.4f} on RANDOM labels",
        )
    return Check(
        "shuffled-label control", True,
        f"accuracy {acc:.4f} vs base rate {base:.4f} — no learnable signal, as expected",
    )
