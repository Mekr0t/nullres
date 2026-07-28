"""Model construction and out-of-sample prediction.

Every `.fit()` in this repository is a place that could accidentally train on
the future, so the set of them is kept small, deliberate, and pinned by
`tests/test_packaging.py::test_no_unaudited_fit_sites`. There are three:

    classifier.fit_predict_walk_forward   the single-asset walk-forward
    classifier.feature_importance         refits the last fold to permute it
    crosssec.fit_predict_panel            the panel walk-forward, split on TIME

The third is easy to miss and long went unmentioned — the docs claimed a single
call site while the cross-sectional path, which produced the strongest result in
the project, had its own. Each is purged independently, so nothing leaks; the
risk was that a fourth could appear without anyone noticing. The test now fails
if one does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nullres.validation import purged_walk_forward, uniqueness_weights


def make_model(cfg):
    """Build an unfitted estimator from a ModelConfig."""
    if cfg.kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            l2_regularization=cfg.l2,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            early_stopping=False,
        )
    if cfg.kind == "logistic":
        # Scaling must be fitted inside the fold, hence the pipeline: fitting a
        # scaler on the whole sample leaks test-period mean and variance into
        # training. It is a small leak, and it is still a leak.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0 / max(cfg.l2, 1e-6), max_iter=1_000,
                               random_state=cfg.seed),
        )
    raise ValueError(f"unknown model kind {cfg.kind!r}; choose hgb or logistic")


def fit_predict_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    t_end: np.ndarray,
    split_cfg,
    model_cfg,
    use_uniqueness: bool = True,
    verbose: bool = True,
) -> tuple[pd.Series, list[dict]]:
    """Out-of-sample P(class 1) for every bar in a test fold.

    Bars outside every test window stay NaN — they are training-only and must
    never appear in a backtest. Rows with a NaN label are predicted but not
    trained on, which is how the deadband in `fwd_return` works.

    Returns (proba, fold_reports).
    """
    proba = pd.Series(np.nan, index=X.index, dtype="float64")
    y_arr = y.to_numpy(dtype="float64")
    weights = uniqueness_weights(t_end, len(X)) if use_uniqueness else np.ones(len(X))
    reports: list[dict] = []

    for k, (train, test) in enumerate(purged_walk_forward(t_end, split_cfg), start=1):
        labelled = train[np.isfinite(y_arr[train])]
        if labelled.size < 100:
            continue
        classes = np.unique(y_arr[labelled])
        if classes.size < 2:
            if verbose:
                print(f"  fold {k}: only one class in training set, skipped")
            continue

        model = make_model(model_cfg)
        model.fit(
            X.iloc[labelled],
            y_arr[labelled].astype(int),
            **{"sample_weight": weights[labelled]} if use_uniqueness else {},
        )
        p = model.predict_proba(X.iloc[test])[:, 1]
        proba.iloc[test] = p

        y_test = y_arr[test]
        scored = np.isfinite(y_test)
        acc = float(((p[scored] > 0.5) == (y_test[scored] > 0.5)).mean()) if scored.any() else float("nan")

        # AUC is the better read on whether ANY signal exists: accuracy at a
        # fixed 0.5 cut hides a model that ranks well but is poorly calibrated.
        auc = float("nan")
        if scored.sum() > 10 and np.unique(y_test[scored]).size == 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_test[scored].astype(int), p[scored]))

        report = {
            "fold": k,
            "train": int(labelled.size),
            "test": int(test.size),
            "base_rate": float(np.nanmean(y_arr[labelled])),
            "acc": acc,
            "auc": auc,
            "test_from": str(X.index[test[0]])[:10],
            "test_to": str(X.index[test[-1]])[:10],
        }
        reports.append(report)
        if verbose:
            print(
                f"  fold {k}: train {labelled.size:>7,}  test {test.size:>6,}  "
                f"[{report['test_from']}..{report['test_to']}]  "
                f"acc {acc:.4f}  auc {auc:.4f}"
            )

    if not reports:
        raise RuntimeError(
            "no fold produced predictions — check split.min_train and label config"
        )
    return proba, reports


def feature_importance(X: pd.DataFrame, y: pd.Series, t_end: np.ndarray,
                       split_cfg, model_cfg, n_repeats: int = 3) -> pd.Series:
    """Permutation importance on the LAST fold's test window only.

    In-sample importances tell you what the model memorised. This tells you what
    actually carried out of sample, which is a much shorter list.
    """
    from sklearn.inspection import permutation_importance

    folds = list(purged_walk_forward(t_end, split_cfg))
    train, test = folds[-1]
    y_arr = y.to_numpy(dtype="float64")
    labelled = train[np.isfinite(y_arr[train])]
    scored = test[np.isfinite(y_arr[test])]

    model = make_model(model_cfg)
    model.fit(X.iloc[labelled], y_arr[labelled].astype(int))
    result = permutation_importance(
        model, X.iloc[scored], y_arr[scored].astype(int),
        n_repeats=n_repeats, random_state=model_cfg.seed, scoring="roc_auc",
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
