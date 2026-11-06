"""Baseline XGBoost + TreeSHAP feature ranking.

A deliberately *fast* baseline model is fitted on the training period only,
TreeSHAP values are computed for a representative background sample, and
features are ranked by their mean absolute SHAP contribution. The top-k
(k=30 by default) feature names are exported as JSON for downstream models.

Why this design
---------------
* TreeSHAP (``pred_contribs``) is exact for tree ensembles and scales linearly
  in the number of trees, so it is cheap enough for a 30-feature shortlist on
  ~470k rows.
* The baseline uses a modest tree budget (not the tuned model): feature ranking
  is a structural question and does not require converged hyper-parameters.
* Ranking uses **training rows and training labels only**; the validation set
  is never touched, so the selection cannot leak.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import humanise_feature, humanise_features

__all__ = [
    "FeatureSelectionResult",
    "compute_tree_shap",
    "load_selected_features",
    "rank_features_by_shap",
    "run_feature_selection",
    "save_selected_features",
]

LOGGER = logging.getLogger(__name__)

#: XGBoost parameters for the ranking baseline (deliberately light).
BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 3,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "enable_categorical": False,
    "n_jobs": 4,
    "random_state": 42,
}


@dataclass
class FeatureSelectionResult:
    """Container for the ranking and the artefacts produced by selection."""

    ranking: pd.DataFrame
    selected_features: list[str]
    mean_abs_shap: pd.Series
    model: Any
    output_dir: Path

    @property
    def selection_path(self) -> Path:
        """Path of the JSON file holding the selected feature list."""
        return self.output_dir / "selected_features.json"

    @property
    def ranking_path(self) -> Path:
        """Path of the CSV holding the full SHAP ranking."""
        return self.output_dir / "shap_feature_ranking.csv"


def _build_baseline_model(n_estimators: int, random_state: int, n_jobs: int) -> Any:
    """Instantiate the light XGBoost ranking model."""
    from xgboost import XGBClassifier

    params = dict(BASELINE_PARAMS)
    params.update(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return XGBClassifier(**params)


def _tree_contributions(model: Any, X: np.ndarray) -> np.ndarray:
    """Return TreeSHAP contributions with shape ``(n_rows, n_features)``."""
    import xgboost as xgb

    try:
        booster = model.get_booster()
    except AttributeError as exc:  # pragma: no cover - non-XGB model
        raise TypeError("compute_tree_shap expects a fitted XGBoost model") from exc
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    # The last column is the bias term.
    return np.asarray(contribs)[:, :-1]


def compute_tree_shap(
    model: Any,
    X_background: pd.DataFrame,
    *,
    max_samples: int = 20_000,
    random_state: int = 42,
) -> np.ndarray:
    """Compute TreeSHAP values on a representative background sample.

    Parameters
    ----------
    model:
        Fitted XGBoost classifier.
    X_background:
        Pool of candidate rows to explain. A seeded random subsample of at most
        ``max_samples`` rows is used for speed and stability.
    max_samples:
        Maximum number of rows explained.
    random_state:
        Seed for subsampling.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(min(n, max_samples), n_features)``.
    """
    if len(X_background) == 0:
        raise ValueError("background sample is empty")
    take = min(len(X_background), max_samples)
    # Only re-sample when strictly smaller; otherwise keep the original row
    # order so the returned SHAP matrix stays aligned with ``X_background``
    # (a ``.sample(n=len)`` call would shuffle, breaking that alignment).
    if take == len(X_background):
        sample = X_background
    else:
        sample = X_background.sample(n=take, random_state=random_state)
    started = time.perf_counter()
    values = _tree_contributions(
        model, np.ascontiguousarray(sample.to_numpy(dtype=np.float32))
    )
    LOGGER.info(
        "Computed TreeSHAP for %d x %d features in %.1fs",
        values.shape[0],
        values.shape[1],
        time.perf_counter() - started,
    )
    return values


def rank_features_by_shap(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
) -> pd.Series:
    """Rank features by mean absolute SHAP value (descending)."""
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("shape mismatch between SHAP values and feature names")
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(
        mean_abs, index=list(feature_names), name="mean_abs_shap"
    ).sort_values(ascending=False)


def save_selected_features(
    selected: Sequence[str],
    path: Path,
    *,
    top_n: int,
    baseline_auc: float | None = None,
    strategy: str = "tree_shap_mean_abs",
) -> Path:
    """Write the selected feature list as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_features": list(selected),
        "n_features": len(selected),
        "top_n": int(top_n),
        "strategy": strategy,
        "baseline_validation_auc": baseline_auc,
    }
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Selected features written to %s", path)
    return path


def load_selected_features(path: Path) -> list[str]:
    """Read the feature list written by :func:`save_selected_features`."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):  # tolerate a bare list
        return [str(x) for x in payload]
    features = payload.get("selected_features") or payload.get("features")
    if not features:
        raise KeyError(f"no 'selected_features' key in {path}")
    return [str(x) for x in features]


def _plot_ranking(
    mean_abs_shap: pd.Series,
    feature_contribs: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int,
) -> list[Path]:
    """Save an importance bar chart and a SHAP beeswarm summary plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    top = mean_abs_shap.head(top_n).sort_values()
    # Display labels only; the ranking/JSON keep the raw feature names.
    labelled = top.rename(index=humanise_feature)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.30 * len(top) + 2)))
    labelled.plot.barh(ax=ax, color="#c1121f")
    ax.set_xlabel("Mean |TreeSHAP value|")
    ax.set_title(f"Top {len(top)} features by SHAP importance (baseline model)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    bar_path = output_dir / "fs_baseline_importance_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    written.append(bar_path)

    try:
        import shap

        plt.figure()
        shap.summary_plot(
            feature_contribs,
            features=X_sample.to_numpy(dtype=np.float32),
            feature_names=humanise_features(X_sample.columns),
            max_display=min(top_n, 30),
            show=False,
            plot_size=(10, 9),
        )
        bee_path = output_dir / "fs_baseline_shap_beeswarm.png"
        plt.gcf().savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        written.append(bee_path)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort  # pragma: no cover
        LOGGER.warning("SHAP beeswarm plot skipped: %s", exc)
        plt.close("all")
    return written


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    top_n: int = 30,
    n_estimators: int = 250,
    shap_sample_size: int = 20_000,
    output_dir: Path | str = "outputs",
    random_state: int = 42,
    n_jobs: int = -1,
    make_plots: bool = True,
) -> FeatureSelectionResult:
    """Fit the baseline, rank by TreeSHAP and export the top-k feature list.

    Parameters
    ----------
    X_train, y_train:
        Training-period rows only (already encoded and imputed).
    X_valid, y_valid:
        Optional hold-out used *only* to report the baseline's validation AUC
        for context; it never influences the ranking.
    top_n:
        Number of features to keep.
    shap_sample_size:
        Rows used for TreeSHAP.
    output_dir:
        Directory receiving ``selected_features.json`` and the ranking CSV.

    Returns
    -------
    FeatureSelectionResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal length")
    if top_n > X_train.shape[1]:
        LOGGER.warning(
            "top_n=%d exceeds %d features; keeping all", top_n, X_train.shape[1]
        )
        top_n = X_train.shape[1]

    started = time.perf_counter()
    model = _build_baseline_model(n_estimators, random_state, n_jobs)
    LOGGER.info("Fitting baseline XGBoost on %s rows", f"{len(X_train):,}")
    model.fit(X_train, y_train)

    baseline_auc: float | None = None
    if X_valid is not None and y_valid is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        valid_scores = model.predict_proba(X_valid)[:, 1]
        baseline_auc = float(roc_auc_score(y_valid, valid_scores))
        LOGGER.info(
            "Baseline hold-out performance: ROC-AUC=%.4f, PR-AUC=%.4f",
            baseline_auc,
            average_precision_score(y_valid, valid_scores),
        )

    sample = X_train.sample(
        n=min(len(X_train), shap_sample_size), random_state=random_state
    )
    contribs = compute_tree_shap(
        model, sample, max_samples=shap_sample_size, random_state=random_state
    )
    explained = sample.loc[sample.index[: len(contribs)]]
    mean_abs = rank_features_by_shap(contribs, list(X_train.columns))
    selected = mean_abs.head(top_n).index.tolist()

    ranking = mean_abs.reset_index()
    ranking.columns = ["feature", "mean_abs_shap"]
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["display_name"] = humanise_features(ranking["feature"].tolist())
    ranking["selected"] = ranking["feature"].isin(selected)
    ranking.to_csv(output_dir / "shap_feature_ranking.csv", index=False)
    save_selected_features(
        selected,
        output_dir / "selected_features.json",
        top_n=top_n,
        baseline_auc=baseline_auc,
    )

    if make_plots:
        _plot_ranking(mean_abs, contribs, explained, output_dir, top_n=top_n)

    LOGGER.info(
        "Feature selection finished in %.1fs: %d -> %d features. Top-10: %s",
        time.perf_counter() - started,
        X_train.shape[1],
        len(selected),
        selected[:10],
    )
    return FeatureSelectionResult(
        ranking=ranking,
        selected_features=selected,
        mean_abs_shap=mean_abs,
        model=model,
        output_dir=output_dir,
    )
