"""Model builders, Optuna tuning and the out-of-fold stacking ensemble.

Contents
--------
* :func:`build_xgboost`, :func:`build_lightgbm`, :func:`build_catboost` -
  tuned, imbalance-aware builders with sensible defaults.
* :func:`make_optuna_objective` / :func:`tune_model` - PR-AUC (average
  precision) optimisation over the validation period.
* :class:`StackingEnsemble` - a time-aware out-of-fold stacking ensemble that
  combines XGBoost + LightGBM + CatBoost with a regularised
  ``LogisticRegression`` meta-learner. (An optional ``backend="sklearn"`` path
  delegates to :class:`~sklearn.ensemble.StackingClassifier` for benchmarking,
  but it uses random K-fold and is therefore not leakage-safe.)
* :func:`cross_validate_time_series` - expanding-window CV over the training
  period, with resampling confined to each training fold.
* :func:`save_model` / :func:`load_model` - joblib persistence.

Leakage-safety
--------------
The Optuna objective always scores the *chronologically later* validation
period. Cross-validation uses :func:`src.preprocessing.time_series_folds`, so
no training row post-dates a validation row, and
:func:`src.preprocessing.apply_resampling` is applied inside each training
fold only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .preprocessing import apply_resampling, time_series_folds

__all__ = [
    "DEFAULT_MODEL_NAMES",
    "DEFAULT_PARAMS",
    "ModelSpec",
    "StackingEnsemble",
    "build_catboost",
    "build_lightgbm",
    "build_model",
    "build_xgboost",
    "chronological_holdout",
    "cross_validate_time_series",
    "fit_estimator",
    "load_model",
    "make_optuna_objective",
    "normalise_storage",
    "save_model",
    "tune_model",
]

LOGGER = logging.getLogger(__name__)

#: Models understood by :func:`build_model`.
DEFAULT_MODEL_NAMES: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")

#: Imbalance-aware defaults chosen from the IEEE-CIS public leaderboard and
#: re-validated on this pipeline's chronological hold-out.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.45,
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "max_bin": 256,
        "tree_method": "hist",
        "eval_metric": "aucpr",
    },
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.45,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "min_split_gain": 0.01,
    },
    "catboost": {
        "iterations": 900,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.7,
        "border_count": 128,
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
    },
}


def _scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio of negatives to positives, clipped to a sane range."""
    labels = np.asarray(y).astype(int).ravel()
    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0:
        return 1.0
    return float(np.clip(negatives / positives, 1.0, 100.0))


def _common_n_jobs(n_jobs: int) -> int:
    """Resolve ``n_jobs=-1`` to the physical core count (avoid oversubscription)."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        import os

        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def build_xgboost(
    *,
    y_train: pd.Series | np.ndarray | None = None,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    early_stopping_rounds: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Return an imbalance-aware XGBoost classifier.

    Parameters
    ----------
    y_train:
        Labels used to derive ``scale_pos_weight`` when it is not supplied.
    scale_pos_weight:
        Explicit class weight; takes precedence over ``y_train``.
    early_stopping_rounds:
        ``None`` disables early stopping (it then needs no eval set).
    overrides:
        Parameter overrides merged last (used by Optuna).

    Returns
    -------
    xgboost.XGBClassifier
    """
    from xgboost import XGBClassifier

    params = dict(DEFAULT_PARAMS["xgboost"])
    weight = scale_pos_weight
    if weight is None and y_train is not None:
        weight = _scale_pos_weight(y_train)
    if weight is not None:
        params["scale_pos_weight"] = float(weight)
    params.update(
        random_state=random_state,
        n_jobs=_common_n_jobs(n_jobs),
        enable_categorical=False,
    )
    if early_stopping_rounds:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    if overrides:
        params.update(dict(overrides))
    return XGBClassifier(**params)


def build_lightgbm(
    *,
    y_train: pd.Series | np.ndarray | None = None,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Return an imbalance-aware LightGBM classifier."""
    from lightgbm import LGBMClassifier

    params = dict(DEFAULT_PARAMS["lightgbm"])
    weight = scale_pos_weight
    if weight is None and y_train is not None:
        weight = _scale_pos_weight(y_train)
    if weight is not None:
        params["scale_pos_weight"] = float(weight)
    params.update(
        objective="binary",
        # Early stopping MUST use a ranking metric here. With a large
        # scale_pos_weight the weighted binary_logloss starts increasing after a
        # couple of iterations, which previously stopped training at 2 trees and
        # cost ~0.20 PR-AUC. Average precision is aligned with the tuning target.
        metric="average_precision",
        n_jobs=_common_n_jobs(n_jobs),
        random_state=random_state,
        verbosity=-1,
        force_col_wise=True,
    )
    if overrides:
        params.update(dict(overrides))
    return LGBMClassifier(**params)


def build_catboost(
    *,
    y_train: pd.Series | np.ndarray | None = None,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    cat_features: Sequence[int] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Return an imbalance-aware CatBoost classifier.

    ``scale_pos_weight`` is translated into CatBoost's native
    ``class_weights`` because the former is not a CatBoost parameter.
    """
    from catboost import CatBoostClassifier

    params = dict(DEFAULT_PARAMS["catboost"])
    weight = scale_pos_weight
    if weight is None and y_train is not None:
        weight = _scale_pos_weight(y_train)
    params.update(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_state,
        thread_count=_common_n_jobs(n_jobs),
        verbose=False,
        allow_writing_files=False,
    )
    if weight is not None:
        params["class_weights"] = [1.0, float(weight)]
    if overrides:
        params.update(dict(overrides))
    model = CatBoostClassifier(**params)
    if cat_features:
        model.set_params(cat_features=list(cat_features))
    return model


#: Registry used by :func:`build_model`.
_BUILDERS: dict[str, Callable[..., Any]] = {
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
}


def build_model(
    name: str,
    *,
    y_train: pd.Series | np.ndarray | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Build a model by its registry name (``xgboost``/``lightgbm``/``catboost``)."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(_BUILDERS)}")
    if key == "catboost":
        return build_catboost(
            y_train=y_train,
            random_state=random_state,
            n_jobs=n_jobs,
            overrides=overrides,
        )
    return _BUILDERS[key](
        y_train=y_train,
        random_state=random_state,
        n_jobs=n_jobs,
        overrides=overrides,
    )


def chronological_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
    """Split the chronologically last ``fraction`` off for early stopping.

    The returned hold-out is *inside the training period*, so it is never the
    evaluation set. This keeps the reported validation metrics honest while
    still allowing early stopping and greedy iteration selection.

    Parameters
    ----------
    X, y:
        Chronologically ordered training data.
    fraction:
        Share of rows reserved for early stopping (``0`` disables it).

    Returns
    -------
    tuple
        ``(X_fit, y_fit, X_stop, y_stop)`` where ``X_stop`` may be ``None``.
    """
    if not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must lie in [0, 0.5)")
    if fraction <= 0.0 or len(X) < 100:
        return X, y, None, None
    cut = int(len(X) * (1.0 - fraction))
    cut = max(1, min(cut, len(X) - 1))
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def fit_estimator(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    X_stop: pd.DataFrame | None = None,
    y_stop: pd.Series | None = None,
    early_stopping_rounds: int | None = 100,
    verbose: bool = False,
) -> Any:
    """Fit any supported booster, wiring early stopping per library.

    XGBoost, LightGBM and CatBoost each expect a different signature for
    validation data and early stopping. This helper normalises them and falls
    back to a plain ``fit`` when the eval set is rejected.

    Parameters
    ----------
    model:
        Unfitted estimator produced by :func:`build_model`.
    X, y:
        Training features and labels.
    X_stop, y_stop:
        Optional chronological hold-out used for early stopping only.
    early_stopping_rounds:
        Patience; ``None`` disables early stopping.

    Returns
    -------
    The fitted model (same object, for chaining).
    """
    use_stop = (
        X_stop is not None
        and y_stop is not None
        and early_stopping_rounds is not None
        and early_stopping_rounds > 0
    )
    import inspect

    module = type(model).__module__.split(".")[0]
    if not use_stop:
        model.fit(X, y)
        return model

    try:
        if module == "xgboost":
            model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            model.fit(X, y, eval_set=[(X_stop, y_stop)], verbose=verbose)
            return model
        if module == "lightgbm":
            from lightgbm import early_stopping, log_evaluation

            callbacks = [
                early_stopping(int(early_stopping_rounds), verbose=False),
                log_evaluation(period=0),
            ]
            # LightGBM 4.7 renamed eval_set -> eval_X/eval_y (eval_set still
            # works but is deprecated), so pick whichever the build accepts.
            parameters = inspect.signature(model.fit).parameters
            if "eval_X" in parameters:
                model.fit(
                    X,
                    y,
                    eval_X=X_stop,
                    eval_y=y_stop,
                    callbacks=callbacks,
                )
            else:  # pragma: no cover - older LightGBM
                model.fit(
                    X,
                    y,
                    eval_set=[(X_stop, y_stop)],
                    callbacks=callbacks,
                )
            return model
        if module == "catboost":
            parameters = inspect.signature(model.fit).parameters
            extra = (
                {"early_stopping_rounds": int(early_stopping_rounds)}
                if "early_stopping_rounds" in parameters
                else {"od_type": "Iter", "od_wait": int(early_stopping_rounds)}
            )
            model.fit(X, y, eval_set=(X_stop, y_stop), verbose=verbose, **extra)
            return model
    except (TypeError, ValueError) as exc:
        LOGGER.warning(
            "Early stopping unavailable for %s (%s); fitting without it",
            type(model).__name__,
            exc,
        )
    model.fit(X, y)
    return model


@dataclass
class ModelSpec:
    """Fitted model plus its tuning metadata.

    Retained as a stable public container for downstream consumers that need a
    model together with its tuning/validation results; the pipeline itself
    currently stores fitted estimators in plain dictionaries.
    """

    name: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)
    validation_pr_auc: float | None = None
    validation_roc_auc: float | None = None
    training_seconds: float = 0.0
    fold_scores: list[float] = field(default_factory=list)

    def predict_proba_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """Return the fraud-class probability as a 1-D array."""
        probabilities = self.estimator.predict_proba(X)
        return np.asarray(probabilities)[:, 1]


# --------------------------------------------------------------------------- #
# Time-aware cross-validation
# --------------------------------------------------------------------------- #
def cross_validate_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str = "xgboost",
    n_splits: int = 4,
    validation_fraction: float = 0.20,
    random_state: int = 42,
    n_jobs: int = -1,
    resample_strategy: str = "none",
    resample_ratio: float = 0.2,
    resample_k_neighbors: int = 5,
    params_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expanding-window CV that respects time ordering.

    The frame must be sorted chronologically. Resampling (when enabled) is
    applied *inside* each training fold, and only to the gradient-fit portion
    after the early-stopping slice has been carved - so synthetic samples never
    cross a time boundary.

    Returns
    -------
    dict
        ``fold_scores`` (PR-AUC per fold), ``mean_pr_auc``, ``std_pr_auc``,
        ``roc_auc`` per fold and ``n_features``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    folds = time_series_folds(
        len(X), n_splits=n_splits, validation_fraction=validation_fraction
    )
    if not folds:
        raise RuntimeError("no valid time folds could be constructed")

    pr_scores: list[float] = []
    roc_scores: list[float] = []
    for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
        # Carve the early-stopping slice first, then resample only the fit part.
        X_inner, y_inner, X_stop, y_stop = chronological_holdout(
            X_fold, y_fold, fraction=0.10
        )
        if resample_strategy != "none":
            X_inner, y_inner = apply_resampling(
                X_inner,
                y_inner,
                strategy=resample_strategy,
                ratio=resample_ratio,
                k_neighbors=resample_k_neighbors,
                random_state=random_state,
            )
        model = build_model(
            model_name,
            y_train=y_inner,
            random_state=random_state,
            n_jobs=n_jobs,
            overrides=params_overrides,
        )
        fit_estimator(model, X_inner, y_inner, X_stop=X_stop, y_stop=y_stop)
        scores = np.asarray(model.predict_proba(X_val))[:, 1]
        pr = float(average_precision_score(y_val, scores))
        roc = float(roc_auc_score(y_val, scores))
        pr_scores.append(pr)
        roc_scores.append(roc)
        LOGGER.info(
            "CV %s fold %d/%d: n_train=%s n_valid=%s PR-AUC=%.4f ROC-AUC=%.4f",
            model_name,
            fold_index,
            len(folds),
            f"{len(X_fold):,}",
            f"{len(X_val):,}",
            pr,
            roc,
        )

    return {
        "model": model_name,
        "fold_scores": pr_scores,
        "mean_pr_auc": float(np.mean(pr_scores)),
        "std_pr_auc": float(np.std(pr_scores)),
        "roc_auc": roc_scores,
        "n_features": int(X.shape[1]),
    }


# --------------------------------------------------------------------------- #
# Optuna tuning
# --------------------------------------------------------------------------- #
def normalise_storage(storage: str | Path | None) -> str | None:
    """Convert a study path into a SQLAlchemy URL Optuna can use.

    Optuna's RDB backend requires a URL (``sqlite:///...``), not a bare path.
    If SQLAlchemy is unavailable the study silently falls back to in-memory
    storage - tuning still works, it just cannot be resumed.

    Parameters
    ----------
    storage:
        ``None`` (in-memory), an existing URL, or a filesystem path.

    Returns
    -------
    str | None
        A usable storage URL, or ``None`` for in-memory.
    """
    if storage is None:
        return None
    text = str(storage)
    if "://" in text:
        return text
    try:
        import sqlalchemy  # noqa: F401  (probe only)
    except ImportError:
        LOGGER.warning(
            "SQLAlchemy is not installed; Optuna studies will not be resumable"
        )
        return None
    path = Path(text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Sample a hyper-parameter configuration for ``model_name``."""
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 400, 1800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    raise ValueError(f"no search space defined for {model_name!r}")


def make_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    model_name: str,
    random_state: int = 42,
    n_jobs: int = -1,
    resample_strategy: str = "none",
    resample_ratio: float = 0.2,
    cap_training_rows: int | None = None,
) -> Callable[[Any], float]:
    """Create an Optuna objective that maximises validation PR-AUC.

    PR-AUC (average precision) is the optimisation target because the positive
    class is ~3.5 % of rows; ROC-AUC is far less sensitive to ranking quality
    in the high-precision region that drives operational alerts.

    Parameters
    ----------
    X_train, y_train:
        Training-period data (chronologically earlier).
    X_valid, y_valid:
        Validation-period data (strictly later; never resampled).
    model_name:
        One of ``xgboost``, ``lightgbm``, ``catboost``.
    cap_training_rows:
        Optional subsample of the training fold to keep trials fast. Rows are
        taken from the *end* of the training period so the sample stays as
        close as possible to the validation distribution.

    Returns
    -------
    Callable[[optuna.Trial], float]
        A function returning the validation PR-AUC to maximise.
    """
    from sklearn.metrics import average_precision_score

    fit_X, fit_y = X_train, y_train
    if cap_training_rows is not None and cap_training_rows < len(X_train):
        fit_X = X_train.iloc[-cap_training_rows:]
        fit_y = y_train.iloc[-cap_training_rows:]
        LOGGER.info("Optuna trials train on the last %s rows", f"{len(fit_X):,}")

    # Carve the early-stopping slice from the chronological tail first, then
    # resample only the gradient-fit portion (synthetic samples must never enter
    # the stopping slice or the validation period).
    X_inner, y_inner, X_stop, y_stop = chronological_holdout(
        fit_X, fit_y, fraction=0.10
    )
    if resample_strategy != "none":
        X_inner, y_inner = apply_resampling(
            X_inner,
            y_inner,
            strategy=resample_strategy,
            ratio=resample_ratio,
            random_state=random_state,
        )

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, model_name)
        model = build_model(
            model_name,
            y_train=y_inner,
            random_state=random_state,
            n_jobs=n_jobs,
            overrides=params,
        )
        started = time.perf_counter()
        fit_estimator(model, X_inner, y_inner, X_stop=X_stop, y_stop=y_stop)
        scores = np.asarray(model.predict_proba(X_valid))[:, 1]
        pr_auc = float(average_precision_score(y_valid, scores))
        trial.set_user_attr("fit_seconds", round(time.perf_counter() - started, 2))
        LOGGER.info("trial %s %s -> PR-AUC=%.5f", trial.number, model_name, pr_auc)
        return pr_auc

    return objective


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    model_name: str = "xgboost",
    n_trials: int = 40,
    timeout: int | None = 5_400,
    random_state: int = 42,
    n_jobs: int = -1,
    resample_strategy: str = "none",
    resample_ratio: float = 0.2,
    cap_training_rows: int | None = None,
    study_name: str | None = None,
    storage: str | Path | None = None,
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    """Run Optuna and return the best configuration for ``model_name``.

    Returns
    -------
    dict
        ``best_params``, ``best_value`` (validation PR-AUC), ``n_trials``,
        ``best_trial_number`` and ``elapsed_seconds``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = make_optuna_objective(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_name=model_name,
        random_state=random_state,
        n_jobs=n_jobs,
        resample_strategy=resample_strategy,
        resample_ratio=resample_ratio,
        cap_training_rows=cap_training_rows,
    )
    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=8)
    resolved_storage = normalise_storage(storage)
    study = optuna.create_study(
        study_name=study_name or f"{model_name}_fraud",
        direction="maximize",
        sampler=sampler,
        storage=resolved_storage,
        load_if_exists=resolved_storage is not None,
    )
    started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Optuna finished %s: %d trials in %.1fs, best PR-AUC=%.5f",
        model_name,
        len(study.trials),
        elapsed,
        study.best_value,
    )
    return {
        "model": model_name,
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "best_trial_number": int(study.best_trial.number),
        "elapsed_seconds": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# Stacking ensemble
# --------------------------------------------------------------------------- #
class StackingEnsemble:
    """Leakage-free, time-aware out-of-fold stacking.

    ``sklearn.ensemble.StackingClassifier`` builds its meta-features with
    ``cross_val_predict``, which raises ``ValueError: cross_val_predict only
    works for partitions`` when handed a ``TimeSeriesSplit`` (the earliest
    samples are never in a test fold). Rather than fall back to random K-fold
    - which would let the meta-learner see future rows - this class
    implements the stacking loop explicitly:

    1. Expanding-window time folds are cut over the training period
       (:func:`src.preprocessing.time_series_folds`).
    2. Each base learner predicts the held-out block; those predictions form
       the out-of-fold meta-features ``Z``.
    3. The regularised logistic meta-learner is fitted on the covered rows of
       ``Z`` only (the first folds act as burn-in and have no prior history).
    4. Base learners are refitted on the full training period for inference.

    Because the fold boundaries are chronological, no meta-feature is produced
    by a model that has seen a later row.

    Parameters
    ----------
    model_names:
        Base learners to combine (``xgboost``, ``lightgbm``, ``catboost``).
    meta_c:
        Inverse regularisation strength of the meta logistic regression.
        Smaller values shrink the ensemble weights harder.
    n_splits:
        Number of expanding-window folds used to build ``Z``.
    backend:
        ``"time_oof"`` (default, leakage-safe) or ``"sklearn"`` which delegates
        to :class:`~sklearn.ensemble.StackingClassifier` with random K-fold
        partitions. The sklearn backend is retained for benchmarking only and
        is **not** leakage-safe on time-ordered data.
    """

    def __init__(
        self,
        model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
        *,
        meta_c: float = 1.0,
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        resample_strategy: str = "none",
        resample_ratio: float = 0.2,
        passthrough: bool = False,
        params_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        backend: str = "time_oof",
        meta_class_weight: str | dict[int, float] | None = None,
    ) -> None:
        if len(model_names) < 2:
            raise ValueError("stacking needs at least two base models")
        if backend not in {"time_oof", "sklearn"}:
            raise ValueError(f"unknown stacking backend {backend!r}")
        self.model_names = list(model_names)
        self.meta_c = float(meta_c)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.resample_strategy = resample_strategy
        self.resample_ratio = float(resample_ratio)
        self.passthrough = bool(passthrough)
        self.params_overrides = dict(params_overrides or {})
        self.backend = backend
        # Class weighting the meta-learner inflates its probabilities by the
        # imbalance ratio, which destroys Brier score / calibration. The
        # cost-sensitive threshold already handles the imbalance, so the default
        # keeps the meta probabilities as close to calibrated as possible.
        self.meta_class_weight = meta_class_weight

        self.estimators_: list[Any] = []
        self.meta_learner_: Any | None = None
        self.oof_mask_: np.ndarray | None = None
        self.oof_scores_: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    # -- helpers -----------------------------------------------------------
    def _build_base_models(self, y: pd.Series) -> list[Any]:
        """Instantiate one fresh estimator per configured base learner."""
        models: list[Any] = []
        for name in self.model_names:
            models.append(
                build_model(
                    name,
                    y_train=y,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    overrides=self.params_overrides.get(name),
                )
            )
        return models

    @staticmethod
    def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        """Fraud-class probability from any of the supported boosters."""
        return np.asarray(model.predict_proba(X))[:, 1].astype(np.float64)

    def _fit_fold_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit one base learner, optionally resampling the gradient-fit portion.

        The early-stopping slice is carved from the chronological tail *before*
        resampling, so synthetic samples never enter it or cross a time boundary.
        """
        X_inner, y_inner, X_stop, y_stop = chronological_holdout(X, y, fraction=0.10)
        if self.resample_strategy != "none":
            X_inner, y_inner = apply_resampling(
                X_inner,
                y_inner,
                strategy=self.resample_strategy,
                ratio=self.resample_ratio,
                random_state=self.random_state,
            )
        fit_estimator(model, X_inner, y_inner, X_stop=X_stop, y_stop=y_stop)
        return model

    def _build_meta_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, list[list[Any]]]:
        """Generate out-of-fold base-learner probabilities ``Z``.

        Returns
        -------
        tuple[np.ndarray, numpy.ndarray, list[list[Any]]]
            ``Z`` of shape ``(n_rows, n_models)`` (``NaN`` for burn-in rows),
            a boolean mask of rows with valid predictions, and the per-fold
            fitted estimators (kept only for diagnostics).
        """
        folds = time_series_folds(len(X), n_splits=self.n_splits)
        if not folds:
            raise RuntimeError("could not build time folds for stacking")
        Z = np.full((len(X), len(self.model_names)), np.nan, dtype=np.float64)
        fold_models: list[list[Any]] = []

        for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
            X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_holdout = X.iloc[valid_idx]
            fitted: list[Any] = []
            for position, name in enumerate(self.model_names):
                model = build_model(
                    name,
                    y_train=y_fold,
                    random_state=self.random_state + fold_index,
                    n_jobs=self.n_jobs,
                    overrides=self.params_overrides.get(name),
                )
                self._fit_fold_model(model, X_fold, y_fold)
                Z[valid_idx, position] = self._positive_proba(model, X_holdout)
                fitted.append(model)
            fold_models.append(fitted)
            LOGGER.info(
                "Stacking OOF fold %d/%d: %s train rows, %s holdout rows",
                fold_index,
                len(folds),
                f"{len(train_idx):,}",
                f"{len(valid_idx):,}",
            )
        mask = np.isfinite(Z).all(axis=1)
        return Z, mask, fold_models

    def _log_oof_quality(self, Z: np.ndarray, mask: np.ndarray, y: pd.Series) -> None:
        """Log each base learner's PR-AUC on its own OOF predictions."""
        from sklearn.metrics import average_precision_score

        for position, name in enumerate(self.model_names):
            try:
                score = float(
                    average_precision_score(y.to_numpy()[mask], Z[mask, position])
                )
            except ValueError:  # pragma: no cover - degenerate folds
                score = float("nan")
            self.oof_scores_[name] = score
            LOGGER.info("Stacking OOF PR-AUC [%s]: %.4f", name, score)

    # -- sklearn-compatible API -------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> StackingEnsemble:
        """Fit base learners and the meta-learner on out-of-fold predictions."""
        from sklearn.linear_model import LogisticRegression

        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y).astype(int)).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self._feature_names = [str(c) for c in X.columns]
        started = time.perf_counter()

        if self.backend == "sklearn":
            return self._fit_sklearn_backend(X, y, started)

        Z, mask, _ = self._build_meta_features(X, y)
        if mask.sum() < 10:
            raise RuntimeError(
                "not enough OOF rows to fit the meta-learner; "
                "disable stacking or increase the number of folds"
            )
        self._log_oof_quality(Z, mask, y)

        meta_features = Z[mask]
        if self.passthrough:
            meta_features = np.hstack(
                [meta_features, X.to_numpy()[mask].astype(np.float64)]
            )
        self.meta_learner_ = LogisticRegression(
            C=self.meta_c,
            solver="lbfgs",
            max_iter=2_000,
            class_weight=self.meta_class_weight,
            random_state=self.random_state,
        )
        self.meta_learner_.fit(meta_features, y.to_numpy()[mask])
        self.oof_mask_ = mask

        # Refit on the full training period for inference.
        self.estimators_ = []
        for name in self.model_names:
            model = build_model(
                name,
                y_train=y,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                overrides=self.params_overrides.get(name),
            )
            self._fit_fold_model(model, X, y)
            self.estimators_.append(model)

        LOGGER.info(
            "Stacking fitted: %d base learners, meta-learner on %s OOF rows, %.1fs",
            len(self.estimators_),
            f"{int(mask.sum()):,}",
            time.perf_counter() - started,
        )
        self._fitted = True
        return self

    def _fit_sklearn_backend(
        self, X: pd.DataFrame, y: pd.Series, started: float
    ) -> StackingEnsemble:
        """Delegate to sklearn's StackingClassifier with random K-fold.

        Retained for benchmarking only: random folds mix past and future rows,
        so the resulting meta-learner is optimistic on time-ordered data.
        """
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        LOGGER.warning(
            "Stacking backend='sklearn' uses random K-fold and is NOT "
            "leakage-safe for time-ordered data"
        )
        estimators = list(
            zip(self.model_names, self._build_base_models(y), strict=False)
        )
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=self.meta_c,
                solver="lbfgs",
                max_iter=2_000,
                class_weight=self.meta_class_weight,
            ),
            cv=5,
            stack_method="predict_proba",
            passthrough=self.passthrough,
            n_jobs=1,
        )
        stack.fit(X, y)
        self._sklearn_stack = stack
        self.estimators_ = [m for _, m in stack.estimators_]
        self.meta_learner_ = stack.final_estimator_
        LOGGER.info("Sklearn stacking fitted in %.1fs", time.perf_counter() - started)
        self._fitted = True
        return self

    def _base_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Stack the base learners' fraud probabilities into ``Z``."""
        if not self.estimators_:
            raise RuntimeError("no fitted base learners")
        return np.column_stack(
            [self._positive_proba(model, X) for model in self.estimators_]
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the stacked probability matrix (classes ``[0, 1]``)."""
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predicting")
        frame = pd.DataFrame(X)
        if getattr(self, "backend", "time_oof") == "sklearn":
            return np.asarray(self._sklearn_stack.predict_proba(frame))

        base = self._base_probabilities(frame)
        meta_features = base
        if self.passthrough:
            meta_features = np.hstack([base, frame.to_numpy().astype(np.float64)])
        fraud_probability = self.meta_learner_.predict_proba(meta_features)[:, 1]
        return np.column_stack([1.0 - fraud_probability, fraud_probability])

    def predict(self, X: pd.DataFrame, *, threshold: float = 0.5) -> np.ndarray:
        """Return hard labels at ``threshold``."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importances(self) -> pd.Series | None:
        """Average normalised feature importance across the base learners."""
        importances = []
        for model in self.estimators_:
            values = getattr(model, "feature_importances_", None)
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            total = array.sum()
            if total > 0:
                importances.append(array / total)
        if not importances:
            return None
        mean_importance = np.mean(importances, axis=0)
        columns = self._feature_names or [f"f{i}" for i in range(len(mean_importance))]
        return pd.Series(mean_importance, index=columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(model: Any, path: Path | str, *, compress: int = 3) -> Path:
    """Persist a fitted model (or any picklable object) with joblib."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=compress)
    LOGGER.info("Saved model artifact to %s", destination)
    return destination


def load_model(path: Path | str) -> Any:
    """Load a joblib artifact written by :func:`save_model`.

    .. warning::
        ``joblib.load`` deserialises Python pickle data. Only load artifacts
        you produced yourself or obtained from a trusted source; loading a
        malicious ``.joblib`` file executes arbitrary code.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"model artifact not found: {source}")
    return joblib.load(source)
