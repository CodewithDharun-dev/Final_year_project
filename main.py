#!/usr/bin/env python3
"""Command-line entry point for the financial fraud detection pipeline.

The pipeline runs the following stages, each of which can be enabled or
disabled from the CLI:

1. **Preprocess** - merge, clean, engineer, encode, chronological split,
   optional downcasting, optional SMOTE (training fold only).
2. **Feature selection** (``--run_shap_selection``) - baseline XGBoost +
   TreeSHAP ranking, top-30 shortlist exported to JSON.
3. **Tuning** (``--tune``) - Optuna search maximising validation PR-AUC.
4. **Training** - XGBoost/LightGBM/CatBoost (and the time-aware OOF stacking
   ensemble) on the selected features.
5. **Evaluation** - PR-AUC, ROC-AUC, F1, balanced accuracy, Brier, plus
   cost-sensitive threshold optimisation and saved figures.
6. **Explainability** (``--explain``) - SHAP, permutation importance, LIME and
   PDP artifacts.

Examples
--------
Full run with tuning and explanations::

    python main.py --data_path data/raw --tune --run_shap_selection --explain

Fast smoke test on a small subsample::

    python main.py --data_path data/raw --sample_rows 60000 --no-stacking

Reuse a previously tuned configuration::

    python main.py --data_path data/raw --config outputs/reports/run_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # ``resource`` is POSIX-only; Windows falls back to a 0.0 RSS estimate.
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

from src import __version__
from src.config import PipelineConfig, configure_logging
from src.evaluation import EvaluationReport, evaluate_predictions, save_report
from src.feature_selection import run_feature_selection
from src.models import (
    DEFAULT_MODEL_NAMES,
    StackingEnsemble,
    build_model,
    chronological_holdout,
    cross_validate_time_series,
    fit_estimator,
    save_model,
    tune_model,
)
from src.preprocessing import (
    ProcessedData,
    apply_resampling,
    build_processed_dataset,
    load_processed_dataset,
    select_features,
)
from src.xai import run_xai

LOGGER = logging.getLogger("ffdd.pipeline")

#: Directory names probed when ``--data_path`` does not directly contain the
#: transaction file (so the tool works from the repository root).
_DATA_SEARCH_HINTS: tuple[str, ...] = (
    ".",
    "data/raw",
    "data",
    "Dataset",
    "../Dataset",
    "../data",
    "..",
    "../..",
    "untitled folder/Dataset",
)


def _is_transaction_csv(path: Path) -> bool:
    """True when a path looks like a train transaction CSV (case-insensitive)."""
    name = path.name.lower()
    return "transaction" in name and "train" in name and path.suffix.lower() == ".csv"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def peak_rss_mb() -> float:
    """Return the process peak resident set size in megabytes.

    ``resource.getrusage`` reports bytes on macOS and kilobytes on Linux; on
    platforms without ``resource`` (Windows) a best-effort 0.0 is returned so
    the pipeline still runs.
    """
    if resource is None:  # pragma: no cover - Windows
        return 0.0
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


@dataclass
class StageTimer:
    """Context manager that records elapsed time and memory per stage."""

    name: str
    timings: dict[str, dict[str, float]] = field(default_factory=dict)

    def __enter__(self) -> StageTimer:
        self._started = time.perf_counter()
        self._rss_before = peak_rss_mb()
        LOGGER.info("=" * 72)
        LOGGER.info("STAGE START: %s (peak RSS %.0f MB)", self.name, self._rss_before)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        elapsed = time.perf_counter() - self._started
        rss_after = peak_rss_mb()
        self.timings[self.name] = {
            "seconds": round(elapsed, 2),
            "peak_rss_mb": round(rss_after, 1),
            "rss_delta_mb": round(rss_after - self._rss_before, 1),
        }
        if exc_type is None:
            LOGGER.info(
                "STAGE END: %s in %.1fs (peak RSS %.0f MB)",
                self.name,
                elapsed,
                rss_after,
            )
        else:
            LOGGER.error("STAGE FAILED: %s after %.1fs: %s", self.name, elapsed, exc)
        return False  # never suppress exceptions


def resolve_data_directory(requested: str, project_root: Path) -> Path:
    """Find the directory that actually contains ``train_transaction.csv``.

    Parameters
    ----------
    requested:
        The ``--data_path`` value (may be a directory or a CSV path).
    project_root:
        Base for relative resolution and fallback probing.

    Returns
    -------
    pathlib.Path
        Directory holding the transaction CSV.

    Raises
    ------
    FileNotFoundError
        If no candidate directory contains the file.
    """
    candidate = Path(requested).expanduser() if requested else None
    if candidate is None:
        candidate = Path(os.environ.get("FFDD_DATA_PATH", ""))
        if not str(candidate):
            candidate = None
    if candidate is None:
        candidate = project_root / "data" / "raw"
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    if candidate.suffix.lower() == ".csv" and candidate.exists():
        return candidate.parent
    if candidate.is_dir():
        probe = [candidate]
        for hint in _DATA_SEARCH_HINTS:
            probe.append((candidate / hint).resolve())
        for directory in probe:
            if directory.is_dir() and any(
                _is_transaction_csv(p) for p in directory.glob("*.csv")
            ):
                return directory
    raise FileNotFoundError(
        f"could not locate train_transaction.csv under {candidate}; "
        "pass --data_path pointing at the folder containing the CSVs, or set "
        "the FFDD_DATA_PATH environment variable"
    )


def detect_dataset_paths(data_dir: Path) -> tuple[Path, Path | None]:
    """Return the transaction and (optional) identity CSV paths."""
    transaction = identity = None
    for path in sorted(data_dir.glob("*.csv")):
        name = path.name.lower()
        if "transaction" in name and "train" in name:
            transaction = path
        elif "identity" in name and "train" in name:
            identity = path
    if transaction is None:
        raise FileNotFoundError(
            f"no train transaction CSV in {data_dir} "
            f"(found: {sorted(p.name for p in data_dir.glob('*.csv'))})"
        )
    return transaction, identity


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Production-grade IEEE-CIS fraud detection pipeline: leakage-safe "
            "preprocessing, SHAP feature selection, gradient-boosting tuning, "
            "stacking, cost-sensitive evaluation and explainability."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Every flag below defaults to ``None`` so that a value loaded from
    # ``--config`` is preserved unless the user explicitly overrides it on the
    # command line. The real (built-in) defaults live in ``PipelineConfig``.
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Directory containing the raw CSVs (or a path to the transaction "
        "CSV). Defaults to FFDD_DATA_PATH, then data/raw.",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Project root that owns data/, outputs/ and notebooks/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON configuration file (explicit CLI flags take precedence).",
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run Optuna hyper-parameter search before training.",
    )
    parser.add_argument(
        "--run_shap_selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run baseline XGBoost + TreeSHAP feature selection.",
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate SHAP/LIME/PDP explainability artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(DEFAULT_MODEL_NAMES),
        help="Base learners to train.",
    )
    parser.add_argument(
        "--top_n", type=int, default=None, help="Features retained by SHAP selection."
    )
    parser.add_argument(
        "--tuning_trials", type=int, default=None, help="Optuna trials per model."
    )
    parser.add_argument(
        "--tuning_timeout",
        type=int,
        default=None,
        help="Per-model Optuna timeout in seconds (0 disables the timeout).",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=None,
        help="Chronologically last fraction reserved for validation.",
    )
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=None,
        help="Drop columns whose missing fraction exceeds this value.",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default=None,
        choices=["time", "timeseries_cv"],
        help="'time' = chronological hold-out only; 'timeseries_cv' also runs "
        "expanding-window CV over the training period.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["frequency", "ordinal", "target"],
        help="Categorical encoding strategy.",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default=None,
        choices=["none", "smote", "borderline"],
        help="Over-sampling strategy applied to training folds only.",
    )
    parser.add_argument(
        "--resample_ratio",
        type=float,
        default=None,
        help="Target fraud:legitimate ratio after resampling.",
    )
    parser.add_argument(
        "--stacking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train the time-aware out-of-fold stacking ensemble.",
    )
    parser.add_argument(
        "--stacking_splits", type=int, default=None, help="OOF folds for stacking."
    )
    parser.add_argument(
        "--fraud_cost",
        type=float,
        default=None,
        help="Expected loss per missed fraud (false negative).",
    )
    parser.add_argument(
        "--friction_cost",
        type=float,
        default=None,
        help="Friction loss per false positive (blocked genuine customer).",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=None,
        help="Optional recall floor for threshold tuning.",
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Randomly subsample this many raw rows (smoke testing only).",
    )
    parser.add_argument(
        "--reuse_processed",
        action="store_true",
        help="Load data/processed artifacts instead of rebuilding them.",
    )
    parser.add_argument(
        "--random_state", type=int, default=None, help="Global random seed."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Parallel jobs for model fitting (-1 uses all cores).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-stage banner logging.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Merge a JSON config file, then CLI overrides, into a validated config.

    CLI flags use ``default=None`` and :meth:`PipelineConfig.with_overrides`
    ignores ``None`` values, so a value loaded from ``--config`` is preserved
    unless the user *explicitly* sets the corresponding flag.
    """
    base = PipelineConfig()
    if args.config:
        LOGGER.info("Loading configuration from %s", args.config)
        base = PipelineConfig.load(args.config)

    # ``--tuning_timeout 0`` means "disable the timeout", which maps to None.
    tuning_timeout = (
        args.tuning_timeout or None if args.tuning_timeout is not None else None
    )
    overrides = {
        "data_path": args.data_path,
        "project_root": args.project_root,
        "tune": args.tune,
        "run_shap_selection": args.run_shap_selection,
        "explain": args.explain,
        "train_models": tuple(args.models) if args.models else None,
        "top_n_features": args.top_n,
        "tuning_trials": args.tuning_trials,
        "tuning_timeout_seconds": tuning_timeout,
        "validation_fraction": args.validation_fraction,
        "split_method": args.split_method,
        "missing_drop_threshold": args.missing_threshold,
        "encoding_strategy": args.encoding,
        "resample_strategy": args.resample,
        "resample_ratio": args.resample_ratio,
        "stacking": args.stacking,
        "stacking_n_splits": args.stacking_splits,
        "fraud_cost": args.fraud_cost,
        "friction_cost": args.friction_cost,
        "target_recall": args.target_recall,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "log_level": args.log_level,
    }
    return base.with_overrides(**overrides).resolve_paths()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    """Everything the pipeline produced, returned for testing and notebooks."""

    config: PipelineConfig
    processed: ProcessedData
    reports: dict[str, EvaluationReport] = field(default_factory=dict)
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    fold_scores: dict[str, Any] = field(default_factory=dict)


def _subset_by_time(
    frame: pd.DataFrame, n_rows: int, time_column: str, random_state: int
) -> pd.DataFrame:
    """Randomly subsample rows but restore chronological order."""
    if n_rows >= len(frame):
        return frame
    sample = frame.sample(n=n_rows, random_state=random_state)
    return sample.sort_values(time_column, kind="mergesort").reset_index(drop=True)


def _load_and_prepare(
    config: PipelineConfig, args: argparse.Namespace
) -> ProcessedData:
    """Preprocess raw data (or reload cached artifacts)."""
    if args.reuse_processed and (config.processed_dir / "metadata.json").exists():
        LOGGER.info("Loading cached processed dataset from %s", config.processed_dir)
        return load_processed_dataset(config.processed_dir)

    from src.preprocessing import load_raw_data

    data_dir = resolve_data_directory(config.data_path, config.root)
    transaction_path, identity_path = detect_dataset_paths(data_dir)
    LOGGER.info("Transaction CSV: %s", transaction_path)
    LOGGER.info(
        "Identity CSV: %s", identity_path if identity_path else "not found (skipping)"
    )
    raw = load_raw_data(
        config,
        transaction_path=transaction_path,
        identity_path=identity_path,
        load_identity=identity_path is not None,
    )
    if args.sample_rows:
        LOGGER.warning(
            "SMOKE TEST: subsampling %s of %s rows",
            f"{args.sample_rows:,}",
            f"{len(raw):,}",
        )
        raw = _subset_by_time(
            raw,
            args.sample_rows,
            config.time_columns[0],
            config.random_state,
        )
    return build_processed_dataset(config, raw_df=raw)


def run_pipeline(config: PipelineConfig, args: argparse.Namespace) -> PipelineResult:
    """Execute every enabled stage and return the collected artifacts."""
    timings: dict[str, dict[str, float]] = {}
    artifacts: list[Path] = []
    config.ensure_directories()
    started = time.perf_counter()
    LOGGER.info("Python %s on %s", platform.python_version(), platform.platform())
    LOGGER.info("Configuration: %s", json.dumps(config.to_dict(), default=str))
    config_path = config.save(config.reports_dir / "run_config.json")
    artifacts.append(config_path)

    # ---- 1. Preprocess --------------------------------------------------
    with StageTimer("1_preprocess", timings):
        processed = _load_and_prepare(config, args)
    LOGGER.info("Dataset ready: %s", processed.summary())

    feature_names = processed.feature_names
    X_train, X_valid = processed.X_train, processed.X_valid
    y_train, y_valid = processed.y_train, processed.y_valid

    # ---- 2. Feature selection ------------------------------------------
    if config.run_shap_selection:
        with StageTimer("2_feature_selection", timings):
            selection = run_feature_selection(
                X_train,
                y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                top_n=config.top_n_features,
                n_estimators=config.baseline_n_estimators,
                shap_sample_size=config.shap_sample_size,
                output_dir=config.outputs_dir,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            )
            feature_names = selection.selected_features
            artifacts.extend(
                [
                    selection.selection_path,
                    selection.ranking_path,
                ]
            )
            LOGGER.info(
                "Reduced feature space from %d to %d columns",
                processed.X_train.shape[1],
                len(feature_names),
            )
    else:
        LOGGER.info("SHAP feature selection disabled; using all features")

    X_train_sel = select_features(X_train, feature_names)
    X_valid_sel = select_features(X_valid, feature_names)
    processed.feature_names = list(feature_names)

    # Keep a chronologically-ordered copy of the training period. Time-based CV
    # and stacking rely on index order == time order, so SMOTE must be applied
    # *inside* their folds (and only to the gradient-fit portion), never to this
    # whole frame up front.
    X_train_time = X_train_sel
    y_train_time = y_train

    resample_seconds = 0.0

    # ---- 4. Tuning + training ------------------------------------------
    tuned_params: dict[str, dict[str, Any]] = {}
    fold_scores: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    reports: dict[str, EvaluationReport] = {}

    if config.tune:
        with StageTimer("4a_optuna_tuning", timings):
            for model_name in config.train_models:
                tuning = tune_model(
                    X_train_time,
                    y_train_time,
                    X_valid_sel,
                    y_valid,
                    model_name=model_name,
                    n_trials=config.tuning_trials,
                    timeout=config.tuning_timeout_seconds,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                    resample_strategy=config.resample_strategy,
                    resample_ratio=config.resample_ratio,
                    study_name=f"{model_name}_ieee_cis",
                    storage=str(config.reports_dir / "optuna_studies.db"),
                )
                tuned_params[model_name] = tuning["best_params"]
                LOGGER.info(
                    "Tuned %s: PR-AUC %.5f over %d trials",
                    model_name,
                    tuning["best_value"],
                    tuning["n_trials"],
                )
            tuning_path = config.reports_dir / "best_params.json"
            tuning_path.write_text(json.dumps(tuned_params, indent=2))
            artifacts.append(tuning_path)

    # Expanding-window CV is useful on its own (stability check), so it runs
    # whenever tuning produced parameters OR the user asked for CV explicitly.
    if config.tune or config.split_method == "timeseries_cv":
        with StageTimer("4b_time_series_cv", timings):
            for model_name in config.train_models:
                fold_scores[model_name] = cross_validate_time_series(
                    X_train_time,
                    y_train_time,
                    model_name=model_name,
                    n_splits=config.n_time_folds,
                    validation_fraction=config.validation_fraction,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                    resample_strategy=config.resample_strategy,
                    resample_ratio=config.resample_ratio,
                    resample_k_neighbors=config.resample_k_neighbors,
                    params_overrides=tuned_params.get(model_name),
                )
            cv_path = config.reports_dir / "cv_scores.json"
            cv_path.write_text(json.dumps(fold_scores, indent=2))
            artifacts.append(cv_path)

    with StageTimer("4c_model_training", timings):
        # Carve the early-stopping slice from the end of the *chronological*
        # training period first, then resample only the gradient-fit portion so
        # synthetic samples never leak into the stopping slice.
        es_fraction = config.early_stopping_fraction if config.early_stopping else 0.0
        X_fit, y_fit, X_stop, y_stop = chronological_holdout(
            X_train_time, y_train_time, fraction=es_fraction
        )
        if config.resample_strategy != "none":
            with StageTimer("3_resampling", timings):
                before = len(X_fit)
                X_fit, y_fit = apply_resampling(
                    X_fit,
                    y_fit,
                    strategy=config.resample_strategy,
                    ratio=config.resample_ratio,
                    k_neighbors=config.resample_k_neighbors,
                    random_state=config.resample_random_state,
                )
                LOGGER.info(
                    "Training rows %s -> %s (early-stopping slice untouched: %s)",
                    f"{before:,}",
                    f"{len(X_fit):,}",
                    f"{len(X_stop):,}" if X_stop is not None else "none",
                )
            resample_seconds = timings["3_resampling"]["seconds"]
        LOGGER.info(
            "Training rows: %s (early-stopping slice: %s)",
            f"{len(X_fit):,}",
            f"{len(X_stop):,}" if X_stop is not None else "none",
        )
        for model_name in config.train_models:
            LOGGER.info("Training %s on %s rows", model_name, f"{len(X_fit):,}")
            model = build_model(
                model_name,
                y_train=y_fit,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                overrides=tuned_params.get(model_name),
            )
            fit_started = time.perf_counter()
            fit_estimator(
                model,
                X_fit,
                y_fit,
                X_stop=X_stop,
                y_stop=y_stop,
                early_stopping_rounds=(
                    config.early_stopping_rounds if config.early_stopping else None
                ),
            )
            fit_seconds = time.perf_counter() - fit_started
            probabilities = np.asarray(model.predict_proba(X_valid_sel))[:, 1]
            report = evaluate_predictions(
                y_valid,
                probabilities,
                model_name=model_name,
                fraud_cost=config.fraud_cost,
                friction_cost=config.friction_cost,
                figures_dir=config.figures_dir,
                threshold_grid_size=config.threshold_grid_size,
                target_recall=config.target_recall,
            )
            report.extras["fit_seconds"] = round(fit_seconds, 2)
            reports[model_name] = report
            fitted[model_name] = model
            artifact = save_model(model, config.models_dir / f"{model_name}.joblib")
            artifacts.append(artifact)
            LOGGER.info("%s | %s", model_name, report.summary_line())

    # ---- 5. Stacking ensemble ------------------------------------------
    if config.stacking and len(config.train_models) >= 2:
        with StageTimer("5_stacking", timings):
            ensemble = StackingEnsemble(
                config.train_models,
                meta_c=config.stacking_meta_c,
                n_splits=config.stacking_n_splits,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                resample_strategy=config.resample_strategy,
                resample_ratio=config.resample_ratio,
                params_overrides=tuned_params,
            )
            # Feed the chronologically-ordered frame; SMOTE (if enabled) is
            # applied inside each OOF fold's gradient-fit portion only.
            ensemble.fit(X_train_time, y_train_time)
            probabilities = ensemble.predict_proba(X_valid_sel)[:, 1]
            report = evaluate_predictions(
                y_valid,
                probabilities,
                model_name="stacking",
                fraud_cost=config.fraud_cost,
                friction_cost=config.friction_cost,
                figures_dir=config.figures_dir,
                threshold_grid_size=config.threshold_grid_size,
                target_recall=config.target_recall,
            )
            report.extras["oof_pr_auc"] = dict(ensemble.oof_scores_)
            reports["stacking"] = report
            fitted["stacking"] = ensemble
            artifacts.append(
                save_model(ensemble, config.models_dir / "stacking.joblib")
            )
            LOGGER.info("stacking | %s", report.summary_line())
    elif config.stacking:
        LOGGER.warning("Stacking requires at least two base models; skipped")

    # ---- 6. Reports -----------------------------------------------------
    with StageTimer("6_reporting", timings):
        report_path = save_report(
            reports, config.reports_dir / "evaluation_report.json"
        )
        artifacts.append(report_path)
        leaderboard = _build_leaderboard(reports)
        leaderboard_path = config.reports_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        artifacts.append(leaderboard_path)
        LOGGER.info("Leaderboard:\n%s", leaderboard.to_string(index=False))

    # ---- 7. Explainability ---------------------------------------------
    best_name = max(
        reports,
        key=lambda key: reports[key].metrics_at_optimal.get("pr_auc", float("nan")),
    )
    if config.explain:
        with StageTimer("7_explainability", timings):
            best_model = fitted[best_name]
            probabilities = np.asarray(best_model.predict_proba(X_valid_sel))[:, 1]
            xai_report = run_xai(
                best_model,
                X_train_sel,
                X_valid_sel,
                y_valid,
                probabilities,
                output_dir=config.figures_dir,
                threshold=reports[best_name].threshold_optimal,
                xai_sample_size=config.xai_sample_size,
                lime_sample_size=config.lime_sample_size,
                lime_num_samples=config.lime_num_samples,
                pdp_permutation_samples=config.pdp_permutation_samples,
                random_state=config.random_state,
            )
            artifacts.extend(xai_report.figures)
            artifacts.extend(xai_report.tables)

    # ---- 8. Run manifest ------------------------------------------------
    total_seconds = time.perf_counter() - started
    manifest = {
        "version": __version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total_seconds, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "stages": timings,
        "best_model": best_name,
        "best_pr_auc": reports[best_name].metrics_at_optimal["pr_auc"],
        "n_features_used": len(feature_names),
        "n_train_rows": len(X_fit),
        "n_valid_rows": len(X_valid_sel),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_valid": float(y_valid.mean()),
        "fold_scores": fold_scores,
        "artifacts": [str(p) for p in artifacts],
        "platform": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "resample_seconds": resample_seconds,
    }
    manifest_path = config.reports_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    artifacts.append(manifest_path)

    LOGGER.info("=" * 72)
    LOGGER.info(
        "PIPELINE COMPLETE in %.1fs (peak RSS %.0f MB). Best model: %s (PR-AUC %.4f)",
        total_seconds,
        peak_rss_mb(),
        best_name,
        reports[best_name].metrics_at_optimal["pr_auc"],
    )
    LOGGER.info(
        "Artifacts written: %d files under %s", len(artifacts), config.outputs_dir
    )

    return PipelineResult(
        config=config,
        processed=processed,
        reports=reports,
        timings=timings,
        artifacts=artifacts,
        fold_scores=fold_scores,
    )


def _build_leaderboard(reports: dict[str, EvaluationReport]) -> pd.DataFrame:
    """Tabulate headline metrics for every trained model."""
    rows = []
    for name, report in reports.items():
        optimal = report.metrics_at_optimal
        rows.append(
            {
                "model": name,
                "roc_auc": round(optimal["roc_auc"], 5),
                "pr_auc": round(optimal["pr_auc"], 5),
                "f1": round(optimal["f1"], 4),
                "balanced_accuracy": round(optimal["balanced_accuracy"], 4),
                "brier": round(optimal["brier"], 5),
                "precision": round(optimal["precision"], 4),
                "recall": round(optimal["recall"], 4),
                "threshold": round(report.threshold_optimal, 4),
                "expected_cost": round(report.cost_optimal, 1),
            }
        )
    frame = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, configure logging and run the pipeline.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a handled failure, ``130`` on interrupt.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except (TypeError, ValueError, KeyError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    config.ensure_directories()
    log_file = config.reports_dir / (
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    if args.quiet:
        config = config.with_overrides(log_level="WARNING")
    configure_logging(config.log_level, log_file)

    LOGGER.info("financial_fraud_detection v%s", __version__)
    LOGGER.info("Log file: %s", log_file)
    try:
        result = run_pipeline(config, args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.warning("Interrupted by user")
        return 130
    except FileNotFoundError as exc:
        LOGGER.error("Data error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard  # pragma: no cover
        LOGGER.error("Pipeline failed: %s", exc)
        LOGGER.debug("Traceback:\n%s", traceback.format_exc())
        return 1

    LOGGER.info("Done. %d artifacts.", len(result.artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
