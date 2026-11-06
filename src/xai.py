"""Explainability utilities: SHAP, permutation importance, LIME and PDP.

Everything here is **post-hoc** and read-only with respect to the model: no
explanation changes training, thresholds or feature selection.

Artifacts produced (written under ``outputs/figures``):

======================  ==================================================
Function                Output
======================  ==================================================
:func:`shap_global_explanations`  ``shap_summary_beeswarm.png``,
                        ``shap_feature_importance_bar.png``
:func:`permutation_importance_plot`  ``permutation_importance.png``
:func:`lime_explanations`  ``lime_<case>.html`` + ``lime_<case>.png``
:func:`partial_dependence_plots`  ``pdp_<feature>.png``, ``pdp_2d_*.png``
======================  ==================================================

Notes on model coverage
-----------------------
* SHAP uses ``TreeExplainer`` for XGBoost/LightGBM/CatBoost, and for the
  stacking ensemble it explains a nominated base learner (attributing a
  stacked score to individual trees is not meaningful because the meta-learner
  is linear in the base probabilities).
* PDP is computed with a small, model-agnostic permutation routine rather than
  ``sklearn.inspection`` so that CatBoost and the stacking wrapper work
  uniformly.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import humanise_feature, humanise_features

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "DEFAULT_PDP_FEATURES",
    "XAIReport",
    "compute_partial_dependence",
    "extract_estimator_for_shap",
    "lime_explanations",
    "partial_dependence_plots",
    "permutation_importance_plot",
    "positive_probabilities",
    "run_xai",
    "shap_global_explanations",
]

LOGGER = logging.getLogger(__name__)

#: Transaction features that practitioners always want to sanity-check.
DEFAULT_PDP_FEATURES: tuple[str, ...] = (
    "TransactionAmt",
    "TransactionAmt_log",
    "C13",
    "V12",
    "card1",
)


@dataclass
class XAIReport:
    """Paths of every artifact produced by a :func:`run_xai` call."""

    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    top_features: list[str] = field(default_factory=list)

    def extend(self, other: XAIReport) -> None:
        """Merge another report into this one."""
        self.figures.extend(other.figures)
        self.tables.extend(other.tables)
        for feature in other.top_features:
            if feature not in self.top_features:
                self.top_features.append(feature)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability for any supported model."""
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(X))
        if values.ndim == 2:
            return values[:, 1]
        return values.ravel()
    if hasattr(model, "predict"):  # CatBoost raw / regressors
        values = np.asarray(model.predict(X)).ravel()
        if values.min() >= 0.0 and values.max() <= 1.0:
            return values
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))
    raise TypeError("model exposes neither predict_proba nor predict")


def extract_estimator_for_shap(model: Any) -> tuple[Any, str]:
    """Pick the tree ensemble to explain with SHAP.

    For the stacking wrapper the XGBoost base learner is preferred (it is the
    most accurate single model in this pipeline); otherwise the model itself is
    returned.
    """
    is_ensemble = getattr(model, "meta_learner_", None) is not None
    if hasattr(model, "estimators_") and is_ensemble:
        estimators = list(getattr(model, "estimators_", []))
        names = list(getattr(model, "model_names", []))
        for preferred in ("xgboost", "lightgbm"):
            if preferred in names:
                index = names.index(preferred)
                if index < len(estimators):
                    LOGGER.info("SHAP will explain the %s base learner", preferred)
                    return estimators[index], preferred
        if estimators:  # pragma: no cover - defensive
            return estimators[0], names[0] if names else "base"
    return model, "model"


# --------------------------------------------------------------------------- #
# Global SHAP
# --------------------------------------------------------------------------- #
def shap_global_explanations(
    model: Any,
    X_sample: pd.DataFrame,
    *,
    output_dir: Path | str,
    max_display: int = 25,
    prefix: str = "shap",
) -> XAIReport:
    """Create a SHAP beeswarm summary and a global importance bar chart.

    Parameters
    ----------
    model:
        Fitted tree model (or a compatible stacked wrapper).
    X_sample:
        Representative rows to explain (subsampled internally for speed).
    output_dir:
        Directory receiving the figures.
    max_display:
        Number of features shown in each plot.

    Returns
    -------
    XAIReport
    """
    import shap

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = XAIReport()

    estimator, label = extract_estimator_for_shap(model)
    sample = X_sample.head(min(len(X_sample), 5_000)).copy()
    LOGGER.info("Computing SHAP values for %s rows", f"{len(sample):,}")
    explainer = shap.TreeExplainer(estimator)
    raw_values = explainer.shap_values(sample)
    if isinstance(raw_values, list):  # older SHAP: one matrix per class
        values = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])
    else:
        values = np.asarray(raw_values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, -1]

    mean_abs = pd.Series(
        np.abs(values).mean(axis=0), index=list(sample.columns), name="mean_abs_shap"
    ).sort_values(ascending=False)
    report.top_features = mean_abs.head(max_display).index.tolist()

    ranking_path = directory / f"{prefix}_feature_importance.csv"
    mean_abs.reset_index().rename(columns={"index": "feature"}).to_csv(
        ranking_path, index=False
    )
    report.tables.append(ranking_path)

    try:
        plt.figure()
        shap.summary_plot(
            values,
            features=sample.to_numpy(dtype=np.float32),
            feature_names=humanise_features(sample.columns),
            max_display=max_display,
            show=False,
            plot_size=(10, 8),
        )
        beeswarm_path = directory / f"{prefix}_summary_beeswarm.png"
        plt.gcf().savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        report.figures.append(beeswarm_path)
    except Exception as exc:  # noqa: BLE001 - plotting robustness  # pragma: no cover
        LOGGER.warning("SHAP beeswarm failed (%s); writing manual bar chart", exc)
        plt.close("all")

    top = mean_abs.head(max_display).sort_values()
    labelled = top.rename(index=humanise_feature)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(top) + 2)))
    labelled.plot.barh(ax=ax, color="#1f4e79")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title(f"Global feature importance - {label}")
    fig.tight_layout()
    bar_path = directory / f"{prefix}_feature_importance_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    report.figures.append(bar_path)

    LOGGER.info("Top features by SHAP: %s", report.top_features[:10])
    return report


# --------------------------------------------------------------------------- #
# Permutation importance
# --------------------------------------------------------------------------- #
def permutation_importance_plot(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    output_dir: Path | str,
    n_repeats: int = 5,
    max_samples: int = 20_000,
    top_n: int = 20,
    random_state: int = 42,
    scoring: str = "average_precision",
    n_jobs: int = 1,
) -> XAIReport:
    """Compute permutation importance on held-out rows and plot the ranking.

    Permutation importance measures the PR-AUC drop when a feature is shuffled,
    which complements SHAP (magnitude of contribution) with an explicit
    performance-cost view.
    """
    from sklearn.inspection import permutation_importance

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = XAIReport()

    if len(X) > max_samples:
        X_eval = X.sample(n=max_samples, random_state=random_state)
        y_eval = y.loc[X_eval.index]
    else:
        X_eval, y_eval = X, y

    started = __import__("time").perf_counter()
    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    LOGGER.info(
        "Permutation importance on %s rows completed in %.1fs",
        f"{len(X_eval):,}",
        __import__("time").perf_counter() - started,
    )

    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    csv_path = directory / "permutation_importance.csv"
    table.to_csv(csv_path, index=False)
    report.tables.append(csv_path)
    report.top_features = table["feature"].head(top_n).tolist()

    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(top) + 2)))
    ax.barh(
        humanise_features(top["feature"].tolist()),
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#2a9d8f",
        ecolor="grey",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(f"Mean {scoring} decrease when shuffled")
    ax.set_title(f"Permutation importance (top {top_n})")
    fig.tight_layout()
    png_path = directory / "permutation_importance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    report.figures.append(png_path)
    return report


# --------------------------------------------------------------------------- #
# Partial dependence
# --------------------------------------------------------------------------- #
def compute_partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_resolution: int = 20,
    max_rows: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-agnostic 1-D partial dependence for ``feature``.

    The feature is replaced by each quantile-grid value across a random row
    sample and the mean predicted fraud probability is recorded.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(grid_values, mean_prediction)``.
    """
    if feature not in X.columns:
        raise KeyError(f"feature {feature!r} is not present in the matrix")
    rows = X.sample(n=min(len(X), max_rows), random_state=random_state)
    values = rows[feature].to_numpy(dtype=np.float64)
    grid = np.unique(np.nanpercentile(values, np.linspace(1, 99, grid_resolution)))
    if grid.size < 2:  # constant feature
        grid = np.array([values.min(), values.max()], dtype=np.float64)

    averages = np.empty(grid.size, dtype=np.float64)
    for index, value in enumerate(grid):
        perturbed = rows.copy()
        perturbed[feature] = value
        averages[index] = float(positive_probabilities(model, perturbed).mean())
    return grid, averages


def partial_dependence_plots(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    output_dir: Path | str,
    grid_resolution: int = 20,
    max_rows: int = 500,
    random_state: int = 42,
    top_n: int = 4,
) -> XAIReport:
    """Plot 1-D PDPs for the requested (or auto-selected) features.

    When ``features`` is ``None`` the first ``top_n`` entries of
    :data:`DEFAULT_PDP_FEATURES` that exist in ``X`` are used, falling back to
    the highest-variance columns.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = XAIReport()

    if features is None:
        features = [f for f in DEFAULT_PDP_FEATURES if f in X.columns]
        if not features:
            variances = X.var(numeric_only=True).sort_values(ascending=False)
            features = variances.head(top_n).index.tolist()
    selected = [f for f in features if f in X.columns]
    if not selected:
        LOGGER.warning("No PDP features available in the matrix")
        return report

    figure, axes = plt.subplots(
        1, len(selected), figsize=(4.2 * len(selected), 3.8), squeeze=False
    )
    for axis, feature in zip(axes[0], selected, strict=False):
        try:
            grid, averages = compute_partial_dependence(
                model,
                X,
                feature,
                grid_resolution=grid_resolution,
                max_rows=max_rows,
                random_state=random_state,
            )
        except Exception as exc:  # noqa: BLE001 - robustness  # pragma: no cover
            LOGGER.warning("PDP failed for %s: %s", feature, exc)
            axis.set_visible(False)
            continue
        axis.plot(grid, averages, marker="o", color="#1f4e79")
        axis.set_title(f"PDP: {humanise_feature(feature)}", fontsize=9)
        axis.set_xlabel(humanise_feature(feature), fontsize=8)
        axis.set_ylabel("Mean P(fraud)")
        axis.grid(alpha=0.3)
        report.top_features.append(feature)
    figure.tight_layout()
    combined = directory / "pdp_selected_features.png"
    figure.savefig(combined, dpi=150)
    plt.close(figure)
    report.figures.append(combined)

    for feature in selected:
        try:
            grid, averages = compute_partial_dependence(
                model,
                X,
                feature,
                grid_resolution=grid_resolution,
                max_rows=max_rows,
                random_state=random_state,
            )
        except Exception:  # noqa: BLE001  # pragma: no cover
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, averages, marker="o", color="#c1121f")
        ax.set_xlabel(humanise_feature(feature))
        ax.set_ylabel("Mean P(fraud)")
        ax.set_title(f"Partial dependence - {humanise_feature(feature)}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = directory / f"pdp_{feature}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        report.figures.append(path)
    return report


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
def _lime_case_label(y_true: int, y_pred: int) -> str:
    """Human-readable case name for a (truth, prediction) pair."""
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    if y_true == 1 and y_pred == 0:
        return "false_negative"
    return "true_negative"


def lime_explanations(
    model: Any,
    X_background: pd.DataFrame,
    X_cases: pd.DataFrame,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    output_dir: Path | str,
    threshold: float = 0.5,
    cases: Sequence[str] = ("false_negative", "true_positive"),
    num_samples: int = 3_000,
    num_features: int = 10,
    background_size: int = 500,
    random_state: int = 42,
    id_column: pd.Series | None = None,
) -> XAIReport:
    """Explain individual transactions with LIME, saving HTML and PNG.

    Parameters
    ----------
    model:
        Any model exposing ``predict_proba``.
    X_background:
        Reference population used to build the LIME perturbation distribution
        (training rows, never validation/test).
    X_cases:
        Rows to explain.
    y_true, y_prob:
        Aligned labels and model probabilities for ``X_cases``.
    threshold:
        Cut-off used to label a prediction as fraud.
    cases:
        Which error/success cases to explain, e.g. ``"false_negative"`` or
        ``"true_positive"``.
    id_column:
        Optional transaction identifiers aligned with ``X_cases`` for naming.

    Returns
    -------
    XAIReport
    """
    from lime.lime_tabular import LimeTabularExplainer

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = XAIReport()

    if len(X_cases) != len(y_true) or len(X_cases) != len(y_prob):
        raise ValueError("X_cases, y_true and y_prob must be aligned")

    background = X_background.sample(
        n=min(len(X_background), background_size), random_state=random_state
    )
    explainer = LimeTabularExplainer(
        training_data=background.to_numpy(dtype=np.float64),
        feature_names=list(X_cases.columns),
        class_names=["genuine", "fraud"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)
    labels = np.array(
        [_lime_case_label(t, p) for t, p in zip(y_true_arr, y_pred_arr, strict=True)]
    )

    for case in cases:
        positions = np.flatnonzero(labels == case)
        if positions.size == 0:
            LOGGER.warning("No %s cases available for LIME", case)
            continue
        position = int(positions[0])
        row = X_cases.iloc[[position]]
        identifier = ""
        if id_column is not None:
            try:
                identifier = str(pd.Series(id_column).iloc[position])
            except (IndexError, KeyError):  # pragma: no cover
                identifier = ""
        LOGGER.info(
            "LIME explaining %s (p=%.3f, true=%d, pred=%d)",
            case,
            y_prob_arr[position],
            y_true_arr[position],
            y_pred_arr[position],
        )
        try:
            explanation = explainer.explain_instance(
                row.to_numpy(dtype=np.float64)[0],
                lambda matrix: np.asarray(
                    model.predict_proba(
                        pd.DataFrame(matrix, columns=list(X_cases.columns))
                    )
                ),
                num_features=num_features,
                num_samples=num_samples,
            )
        except Exception as exc:  # noqa: BLE001 - LIME robustness  # pragma: no cover
            LOGGER.error("LIME failed for %s: %s", case, exc)
            continue

        stem = f"lime_{case}" + (f"_{identifier}" if identifier else "")
        html_path = directory / f"{stem}.html"
        title = (
            f"LIME - {case} (transaction {identifier or 'n/a'}, "
            f"p_fraud={y_prob_arr[position]:.3f})"
        )
        html_path.write_text(
            f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            "</head><body>"
            f"<h2>{html.escape(title)}</h2>"
            f"{explanation.as_html()}"
            "</body></html>",
            encoding="utf-8",
        )
        report.figures.append(html_path)

        try:
            figure = explanation.as_pyplot_figure()
            figure.suptitle(title, fontsize=10)
            figure.tight_layout()
            png_path = directory / f"{stem}.png"
            figure.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            report.figures.append(png_path)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            LOGGER.warning("LIME figure failed for %s: %s", case, exc)
            plt.close("all")

        table = pd.DataFrame(explanation.as_list(), columns=["condition", "weight"])
        csv_path = directory / f"{stem}_weights.csv"
        table.to_csv(csv_path, index=False)
        report.tables.append(csv_path)
        report.top_features.extend(
            explanation.as_list()[i][0]
            for i in range(min(3, len(explanation.as_list())))
        )
    return report


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_xai(
    model: Any,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    probabilities: np.ndarray,
    *,
    output_dir: Path | str,
    threshold: float = 0.5,
    transaction_ids: pd.Series | None = None,
    pdp_features: Iterable[str] | None = None,
    xai_sample_size: int = 5_000,
    lime_sample_size: int = 500,
    lime_num_samples: int = 3_000,
    pdp_permutation_samples: int = 2_000,
    random_state: int = 42,
    n_jobs: int = 1,
    run_permutation: bool = True,
) -> XAIReport:
    """Produce the full explainability bundle for a fitted model.

    Steps: global SHAP, permutation importance, LIME on a false negative and a
    true positive, and PDPs for the canonical transactional features. Each step
    is individually guarded so a single failure cannot abort the run.

    Returns
    -------
    XAIReport
        Aggregated artifact paths.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = XAIReport()

    sample = X_valid.sample(
        n=min(len(X_valid), xai_sample_size), random_state=random_state
    )

    try:
        report.extend(shap_global_explanations(model, sample, output_dir=directory))
    except Exception as exc:  # pragma: no cover - robustness
        LOGGER.error("SHAP explanations failed: %s", exc, exc_info=True)

    if run_permutation:
        permutation_rows = sample.head(min(len(sample), pdp_permutation_samples))
        try:
            report.extend(
                permutation_importance_plot(
                    model,
                    permutation_rows,
                    y_valid.loc[permutation_rows.index],
                    output_dir=directory,
                    n_jobs=n_jobs,
                )
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Permutation importance failed: %s", exc, exc_info=True)

    try:
        case_frame = X_valid.head(min(len(X_valid), lime_sample_size))
        case_probabilities = np.asarray(probabilities)[: len(case_frame)]
        case_truth = y_valid.head(len(case_frame))
        case_ids = (
            pd.Series(transaction_ids).head(len(case_frame))
            if transaction_ids is not None
            else None
        )
        report.extend(
            lime_explanations(
                model,
                X_train,
                case_frame,
                case_truth,
                case_probabilities,
                output_dir=directory,
                threshold=threshold,
                num_samples=lime_num_samples,
                id_column=case_ids,
                random_state=random_state,
            )
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("LIME explanations failed: %s", exc, exc_info=True)

    try:
        report.extend(
            partial_dependence_plots(
                model,
                sample,
                features=None if pdp_features is None else list(pdp_features),
                output_dir=directory,
                random_state=random_state,
            )
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("PDP generation failed: %s", exc, exc_info=True)

    LOGGER.info(
        "XAI bundle complete: %d figures, %d tables",
        len(report.figures),
        len(report.tables),
    )
    return report
