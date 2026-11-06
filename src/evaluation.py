"""Evaluation metrics, cost-sensitive thresholding and diagnostic figures.

The fraud problem is severely imbalanced (3.5 % positives), so accuracy is
never reported as a primary metric. Instead this module exposes:

* :func:`compute_metrics` - ROC-AUC, PR-AUC (average precision), F1,
  balanced accuracy, Brier score plus confusion-count summaries.
* :func:`optimize_threshold` - grid search for the probability cut-off that
  minimises ``FN * fraud_cost + FP * friction_cost``.
* :func:`evaluate_predictions` - one call that returns metrics at both the
  default 0.5 cut-off and the cost-optimal cut-off, saving figures.
* :func:`plot_confusion_matrix` / :func:`plot_precision_recall_curve` -
  publication-quality artifacts for the report.

Decision-theoretic background
-----------------------------
For a calibrated score ``p`` and a booking decision ``d``:

.. math::

    \\text{Cost}(d) = FN \\cdot C_{fraud} + FP \\cdot C_{friction}

Minimising this expected cost is equivalent to booking fraud when
``p >= C_friction / (C_fraud + C_friction)`` under perfect calibration, but
the empirical threshold search is robust to miscalibration and to the heavy
tails of the fraud class, so it is used operationally.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "DEFAULT_FRAUD_COST",
    "DEFAULT_FRICTION_COST",
    "EvaluationReport",
    "ThresholdResult",
    "compute_metrics",
    "evaluate_predictions",
    "optimise_threshold",
    "optimize_threshold",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_threshold_cost_curve",
    "save_report",
]

LOGGER = logging.getLogger(__name__)

#: Default expected loss when a fraudulent transaction is approved.
DEFAULT_FRAUD_COST: float = 500.0

#: Default friction cost of a wrongly declined genuine transaction (customer
#: support, churn, reputational damage).
DEFAULT_FRICTION_COST: float = 5.0


@dataclass
class ThresholdResult:
    """Outcome of the cost-sensitive threshold search."""

    threshold: float
    cost: float
    fraud_cost: float
    friction_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    @property
    def expected_cost_per_transaction(self) -> float:
        """Cost normalised by the scored population."""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        return self.cost / total if total else 0.0


@dataclass
class EvaluationReport:
    """Container for every metric produced for one model on one dataset."""

    model_name: str
    threshold_default: float
    threshold_optimal: float
    metrics_at_default: dict[str, float]
    metrics_at_optimal: dict[str, float]
    cost_at_default: float
    cost_optimal: float
    fraud_cost: float
    friction_cost: float
    n_samples: int
    fraud_rate: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return asdict(self)

    def summary_line(self) -> str:
        """Single-line, log-friendly summary."""
        return (
            f"{self.model_name}: ROC-AUC={self.metrics_at_optimal['roc_auc']:.4f} "
            f"PR-AUC={self.metrics_at_optimal['pr_auc']:.4f} "
            f"F1@opt={self.metrics_at_optimal['f1']:.4f} "
            f"thr={self.threshold_optimal:.3f} cost={self.cost_optimal:,.0f}"
        )


def _sigmoid_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Return probabilities whether ``scores`` are probabilities or margins."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if values.min() >= 0.0 and values.max() <= 1.0:
        return values
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination, calibration and classification metrics.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_prob:
        Predicted probability of fraud.
    threshold:
        Probability cut-off used for the classification metrics.

    Returns
    -------
    dict[str, float]
        Keys: ``roc_auc``, ``pr_auc``, ``f1``, ``balanced_accuracy``, ``brier``,
        ``precision``, ``recall``, ``accuracy``, ``mcc``,
        ``fpr_at_tpr_80``, ``threshold``, and the confusion-matrix counts
        ``true_positives`` / ``false_positives`` / ``true_negatives`` /
        ``false_negatives`` plus ``n_samples`` and ``n_positives``.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_prob {y_prob_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("cannot evaluate an empty array")

    has_both_classes = np.unique(y_true_arr).size == 2
    predictions = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, float] = {
        "roc_auc": (
            float(roc_auc_score(y_true_arr, y_prob_arr))
            if has_both_classes
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true_arr, y_prob_arr))
            if y_true_arr.sum() > 0
            else float("nan")
        ),
        "f1": float(f1_score(y_true_arr, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, predictions))
        if has_both_classes
        else float("nan"),
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "precision": float(precision_score(y_true_arr, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_arr, predictions, zero_division=0)),
        "accuracy": float((tp + tn) / max(1, y_true_arr.size)),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
        "n_samples": float(y_true_arr.size),
        "n_positives": float(y_true_arr.sum()),
        "threshold": float(threshold),
    }
    # Matthews correlation - robust to class imbalance.
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics["mcc"] = (
        float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0
    )
    metrics["fpr_at_tpr_80"] = _false_positive_rate_at_recall(
        y_true_arr, y_prob_arr, target_recall=0.80
    )
    return metrics


def _false_positive_rate_at_recall(
    y_true: np.ndarray, y_prob: np.ndarray, *, target_recall: float
) -> float:
    """FPR at the smallest threshold achieving ``target_recall``."""
    if y_true.sum() == 0 or np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    eligible = np.flatnonzero(tpr >= target_recall)
    return float(fpr[eligible[0]]) if eligible.size else float("nan")


def optimise_threshold(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    fraud_cost: float = DEFAULT_FRAUD_COST,
    friction_cost: float = DEFAULT_FRICTION_COST,
    grid_size: int = 501,
    target_recall: float | None = None,
) -> ThresholdResult:
    """Find the probability cut-off minimising expected business cost.

    ``Cost(threshold) = FN * fraud_cost + FP * friction_cost``

    Parameters
    ----------
    y_true, y_prob:
        Labels and predicted probabilities.
    fraud_cost:
        Loss per missed fraud (false negative).
    friction_cost:
        Friction loss per false positive (blocked genuine customer).
    grid_size:
        Resolution of the threshold grid in ``(0, 1)``.
    target_recall:
        When provided, restrict the search to thresholds whose recall is at
        least this value (a risk-appetite constraint), maximising precision
        subject to that recall floor.

    Returns
    -------
    ThresholdResult
    """
    if fraud_cost <= 0 or friction_cost <= 0:
        raise ValueError("costs must be positive")
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    positives = int(y_true_arr.sum())
    negatives = int(y_true_arr.size - positives)

    if positives == 0 or negatives == 0:
        LOGGER.warning("Threshold search skipped: single-class labels present")
        threshold = float(target_recall or 0.5)
        predictions = (y_prob_arr >= threshold).astype(int)
        cost = float(
            np.sum((y_true_arr == 1) & (predictions == 0)) * fraud_cost
            + np.sum((y_true_arr == 0) & (predictions == 1)) * friction_cost
        )
        return ThresholdResult(
            threshold=threshold,
            cost=cost,
            fraud_cost=fraud_cost,
            friction_cost=friction_cost,
            true_positives=int(np.sum((y_true_arr == 1) & (predictions == 1))),
            false_positives=int(np.sum((y_true_arr == 0) & (predictions == 1))),
            true_negatives=int(np.sum((y_true_arr == 0) & (predictions == 0))),
            false_negatives=int(np.sum((y_true_arr == 1) & (predictions == 0))),
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )

    order = np.argsort(-y_prob_arr, kind="mergesort")
    sorted_prob = y_prob_arr[order]
    sorted_true = y_true_arr[order]
    tp_cumulative = np.cumsum(sorted_true)
    fp_cumulative = np.cumsum(1 - sorted_true)

    thresholds = np.linspace(0.0, 1.0, grid_size)
    best: tuple[float, float] | None = None
    for threshold in thresholds:
        # Predict positive for every row with prob >= threshold.
        position = int(np.searchsorted(-sorted_prob, -threshold, side="right"))
        tp = int(tp_cumulative[position - 1]) if position else 0
        fp = int(fp_cumulative[position - 1]) if position else 0
        fn = positives - tp
        if target_recall is not None and tp / positives < target_recall:
            continue
        cost = fn * fraud_cost + fp * friction_cost
        if best is None or cost < best[1]:
            best = (float(threshold), float(cost))

    if best is None:  # recall floor unreachable on the grid
        LOGGER.warning(
            "Recall floor %.2f unreachable; retrying on observed probabilities",
            target_recall or 0.0,
        )
        return optimise_threshold(
            y_true_arr,
            y_prob_arr,
            fraud_cost=fraud_cost,
            friction_cost=friction_cost,
            grid_size=grid_size,
            target_recall=None,
        )

    threshold, cost = best
    predictions = (y_prob_arr >= threshold).astype(int)
    tp = int(np.sum((y_true_arr == 1) & (predictions == 1)))
    fp = int(np.sum((y_true_arr == 0) & (predictions == 1)))
    tn = int(np.sum((y_true_arr == 0) & (predictions == 0)))
    fn = int(np.sum((y_true_arr == 1) & (predictions == 0)))

    return ThresholdResult(
        threshold=threshold,
        cost=cost,
        fraud_cost=fraud_cost,
        friction_cost=friction_cost,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        recall=float(tp / (tp + fn)) if tp + fn else 0.0,
        f1=float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0,
    )


#: American-English alias for international readers / existing code bases.
optimize_threshold = optimise_threshold


def plot_confusion_matrix(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    threshold: float = 0.5,
    model_name: str = "model",
    save_path: Path | str | None = None,
    normalise: bool = False,
) -> Path:
    """Plot and save a confusion-matrix figure.

    Returns the path written (``save_path`` if given, else a temp file name).
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    matrix = confusion_matrix(
        y_true_arr, (y_prob_arr >= threshold).astype(int), labels=[0, 1]
    )
    display = matrix.astype(float)
    if normalise:
        denominators = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display, denominators, out=np.zeros_like(display), where=denominators > 0
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(display, cmap="Blues")
    for (row, column), value in np.ndenumerate(display):
        label = f"{value:.2f}" if normalise else f"{int(value):,}"
        ax.text(
            column,
            row,
            label,
            ha="center",
            va="center",
            color="white" if value > display.max() / 2 else "black",
            fontsize=11,
        )
    ax.set_xticks([0, 1], ["Predicted genuine", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual genuine", "Actual fraud"])
    ax.set_title(f"Confusion matrix - {model_name} @ p>={threshold:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    destination = (
        Path(save_path) if save_path else Path(f"confusion_matrix_{model_name}.png")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_precision_recall_curve(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    model_name: str = "model",
    save_path: Path | str | None = None,
    threshold_marker: float | None = None,
    ax: Any | None = None,
) -> Path | None:
    """Plot a precision-recall curve annotated with PR-AUC (average precision)."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    ap = average_precision_score(y_true_arr, y_prob_arr)
    baseline = float(y_true_arr.mean())

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color="#1f4e79", label=f"PR-AUC = {ap:.4f}")
    ax.axhline(
        baseline, linestyle="--", color="grey", label=f"Baseline = {baseline:.3f}"
    )
    if threshold_marker is not None:
        predictions = y_prob_arr >= threshold_marker
        point_precision = (
            float(y_true_arr[predictions].mean()) if predictions.any() else 1.0
        )
        point_recall = (
            float(predictions[y_true_arr == 1].mean()) if y_true_arr.sum() else 0.0
        )
        ax.scatter([point_recall], [point_precision], color="#c1121f", zorder=5)
        ax.annotate(
            f"operating point\np>={threshold_marker:.3f}",
            xy=(point_recall, point_precision),
            xytext=(-95, 22),
            textcoords="offset points",
            fontsize=8,
            color="#c1121f",
            arrowprops={
                "arrowstyle": "-",
                "color": "#c1121f",
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )
    ax.set_xlabel("Recall (fraud captured)")
    ax.set_ylabel("Precision (precision of alerts)")
    ax.set_title(f"Precision-Recall - {model_name}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    if not owns_figure:
        return None
    destination = (
        Path(save_path) if save_path else Path(f"precision_recall_{model_name}.png")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=150)
    plt.close(ax.figure)
    return destination


def plot_roc_curve(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    model_name: str = "model",
    save_path: Path | str | None = None,
) -> Path:
    """Plot and save a ROC curve."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    auc = roc_auc_score(y_true_arr, y_prob_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f4e79", label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destination = Path(save_path) if save_path else Path(f"roc_{model_name}.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_threshold_cost_curve(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    fraud_cost: float = DEFAULT_FRAUD_COST,
    friction_cost: float = DEFAULT_FRICTION_COST,
    model_name: str = "model",
    save_path: Path | str | None = None,
    target_recall: float | None = None,
) -> Path:
    """Plot expected cost versus decision threshold, marking the optimum."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for threshold in grid:
        predictions = (y_prob_arr >= threshold).astype(int)
        fn = np.sum((y_true_arr == 1) & (predictions == 0))
        fp = np.sum((y_true_arr == 0) & (predictions == 1))
        costs.append(fn * fraud_cost + fp * friction_cost)
    optimum = optimise_threshold(
        y_true_arr,
        y_prob_arr,
        fraud_cost=fraud_cost,
        friction_cost=friction_cost,
        target_recall=target_recall,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, costs, color="#1f4e79")
    ax.axvline(
        optimum.threshold,
        color="#c1121f",
        linestyle="--",
        label=f"optimal p={optimum.threshold:.3f} (cost {optimum.cost:,.0f})",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost")
    ax.set_title(f"Cost-sensitive threshold - {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destination = (
        Path(save_path) if save_path else Path(f"threshold_cost_{model_name}.png")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def evaluate_predictions(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    *,
    model_name: str = "model",
    fraud_cost: float = DEFAULT_FRAUD_COST,
    friction_cost: float = DEFAULT_FRICTION_COST,
    figures_dir: Path | str | None = None,
    threshold_grid_size: int = 501,
    target_recall: float | None = None,
    make_figures: bool = True,
) -> EvaluationReport:
    """Evaluate probabilities end-to-end and persist diagnostic figures.

    Returns
    -------
    EvaluationReport
        Metrics at the default (0.5) and cost-optimal thresholds.
    """
    started = time.perf_counter()
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = _sigmoid_to_probabilities(np.asarray(y_prob, dtype=np.float64))
    if y_true_arr.size != y_prob_arr.size:
        raise ValueError("y_true and y_prob must be the same length")

    threshold_default = 0.5
    metrics_default = compute_metrics(
        y_true_arr, y_prob_arr, threshold=threshold_default
    )
    optimum = optimise_threshold(
        y_true_arr,
        y_prob_arr,
        fraud_cost=fraud_cost,
        friction_cost=friction_cost,
        grid_size=threshold_grid_size,
        target_recall=target_recall,
    )
    metrics_optimal = compute_metrics(
        y_true_arr, y_prob_arr, threshold=optimum.threshold
    )

    if make_figures:
        directory = Path(figures_dir) if figures_dir else Path("outputs/figures")
        directory.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            y_true_arr,
            y_prob_arr,
            threshold=optimum.threshold,
            model_name=model_name,
            save_path=directory / f"confusion_matrix_{model_name}.png",
        )
        plot_precision_recall_curve(
            y_true_arr,
            y_prob_arr,
            model_name=model_name,
            save_path=directory / f"precision_recall_{model_name}.png",
            threshold_marker=optimum.threshold,
        )
        plot_roc_curve(
            y_true_arr,
            y_prob_arr,
            model_name=model_name,
            save_path=directory / f"roc_{model_name}.png",
        )
        plot_threshold_cost_curve(
            y_true_arr,
            y_prob_arr,
            fraud_cost=fraud_cost,
            friction_cost=friction_cost,
            model_name=model_name,
            save_path=directory / f"threshold_cost_{model_name}.png",
            target_recall=target_recall,
        )

    default_predictions = (y_prob_arr >= threshold_default).astype(int)
    default_fn = int(np.sum((y_true_arr == 1) & (default_predictions == 0)))
    default_fp = int(np.sum((y_true_arr == 0) & (default_predictions == 1)))
    report = EvaluationReport(
        model_name=model_name,
        threshold_default=threshold_default,
        threshold_optimal=optimum.threshold,
        metrics_at_default=metrics_default,
        metrics_at_optimal=metrics_optimal,
        cost_at_default=float(default_fn * fraud_cost + default_fp * friction_cost),
        cost_optimal=optimum.cost,
        fraud_cost=fraud_cost,
        friction_cost=friction_cost,
        n_samples=int(y_true_arr.size),
        fraud_rate=float(y_true_arr.mean()),
        extras={
            "expected_cost_per_transaction": optimum.expected_cost_per_transaction,
            "evaluation_seconds": round(time.perf_counter() - started, 2),
        },
    )
    LOGGER.info(report.summary_line())
    return report


def save_report(
    reports: Mapping[str, EvaluationReport] | Sequence[EvaluationReport],
    path: Path | str,
) -> Path:
    """Persist one or more :class:`EvaluationReport` objects as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = list(reports.values()) if isinstance(reports, Mapping) else list(reports)
    payload = {report.model_name: report.to_dict() for report in items}
    import json

    destination.write_text(json.dumps(payload, indent=2, default=float))
    return destination
