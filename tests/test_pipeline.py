"""Unit and integration tests for the financial fraud detection pipeline.

Run with::

    pytest -q

The suite is intentionally fast: every test either uses a small synthetic
dataset or a tiny slice of the real schema, so it completes in under a minute
without touching the raw competition files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, humanise_feature, humanise_features
from src.evaluation import (
    DEFAULT_FRAUD_COST,
    DEFAULT_FRICTION_COST,
    compute_metrics,
    evaluate_predictions,
    optimise_threshold,
    save_report,
)
from src.feature_selection import (
    load_selected_features,
    rank_features_by_shap,
    save_selected_features,
)
from src.models import (
    StackingEnsemble,
    build_model,
    chronological_holdout,
    cross_validate_time_series,
    fit_estimator,
    load_model,
    normalise_storage,
    save_model,
)
from src.preprocessing import (
    ProcessedData,
    _clean_cognitive_columns,
    apply_resampling,
    build_processed_dataset,
    encode_categoricals,
    load_processed_dataset,
    reduce_mem_usage,
    split_by_time,
    time_series_folds,
    transform_with_encoding,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    """Small IEEE-CIS-shaped frame with a learnable fraud signal."""
    rng = np.random.default_rng(7)
    n_rows = 800
    fraud = (rng.random(n_rows) < 0.12).astype(int)
    amount = rng.gamma(2.0, 40.0, n_rows) + fraud * 120.0
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n_rows + 1),
            "isFraud": fraud,
            "TransactionDT": np.sort(rng.integers(86_400, 15_811_131, n_rows)),
            "TransactionAmt": amount,
            "ProductCD": rng.choice(["W", "C", "H", "S", "R"], n_rows),
            "card1": rng.integers(1_000, 9_000, n_rows),
            "card4": rng.choice(["visa", "mastercard", "discover"], n_rows),
            "P_emaildomain": rng.choice(["gmail.com", "aol.com", None], n_rows),
            "C13": rng.integers(0, 500, n_rows),
            "V12": rng.normal(size=n_rows),
            "id_31": rng.choice(["chrome 70", "safari 10", None], n_rows),
        }
    )
    frame.loc[rng.random(n_rows) < 0.30, "C13"] = np.nan
    frame["constant_col"] = 1.0
    frame["mostly_missing"] = np.nan
    frame.loc[:9, "mostly_missing"] = 1.0
    return frame


@pytest.fixture()
def small_config(tmp_path: Path) -> PipelineConfig:
    """Configuration pointing at a temporary project root."""
    return PipelineConfig(
        project_root=str(tmp_path),
        data_path=str(tmp_path / "data" / "raw"),
        top_n_features=5,
        shap_sample_size=200,
        baseline_n_estimators=30,
        xai_sample_size=200,
        lime_sample_size=50,
        lime_num_samples=100,
        pdp_permutation_samples=100,
        missing_drop_threshold=0.80,
    )


# --------------------------------------------------------------------------- #
# Memory optimisation
# --------------------------------------------------------------------------- #
def test_reduce_mem_usage_downsizes_dtypes() -> None:
    """float64 -> float32 and large ints -> narrow nullable ints."""
    frame = pd.DataFrame(
        {
            "small_int": np.array([1, 2, 3, 4 * 10_000], dtype=np.int64),
            "float": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
            "integral_float": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64),
            "text": ["a", "b", "a", "b"],
        }
    )
    before = frame.memory_usage(deep=True).sum()
    result = reduce_mem_usage(frame, verbose=False)
    after = result.memory_usage(deep=True).sum()

    assert result["float"].dtype == np.float32
    assert pd.api.types.is_integer_dtype(result["small_int"]) or str(
        result["small_int"].dtype
    ).startswith("int")
    assert after < before
    # Values are preserved through the downcast.
    np.testing.assert_allclose(
        result["float"].to_numpy(), frame["float"].to_numpy(), rtol=1e-6
    )


def test_reduce_mem_usage_does_not_mutate_input() -> None:
    """The helper must return a copy."""
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    _ = reduce_mem_usage(frame, verbose=False)
    assert frame["x"].dtype == np.float64


# --------------------------------------------------------------------------- #
# Time-aware splitting
# --------------------------------------------------------------------------- #
def test_split_by_time_is_chronological(synthetic_frame: pd.DataFrame) -> None:
    """Training timestamps must all precede validation timestamps."""
    train, valid = split_by_time(
        synthetic_frame, time_column="TransactionDT", validation_fraction=0.2
    )
    assert len(train) + len(valid) == len(synthetic_frame)
    assert train["TransactionDT"].max() <= valid["TransactionDT"].min()
    # Indexes are re-aligned from zero so X/y stay positionally consistent.
    assert list(valid.index[:3]) == [0, 1, 2]
    assert list(train.index[:3]) == [0, 1, 2]


def test_split_by_time_rejects_invalid_fraction(
    synthetic_frame: pd.DataFrame,
) -> None:
    """A fraction outside (0, 1) must raise."""
    with pytest.raises(ValueError):
        split_by_time(
            synthetic_frame, time_column="TransactionDT", validation_fraction=1.5
        )


def test_time_series_folds_have_no_lookahead() -> None:
    """Every training index must precede every validation index."""
    n_samples = 500
    values = np.arange(n_samples, dtype=np.float64)
    folds = time_series_folds(
        n_samples, n_splits=4, time_values=values, validation_fraction=0.2
    )
    assert len(folds) >= 2
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()
        assert len(np.intersect1d(train_idx, valid_idx)) == 0


def test_time_series_folds_without_time_values() -> None:
    """Index-based folds also respect ordering."""
    folds = time_series_folds(400, n_splits=4)
    assert folds
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()


def test_time_series_folds_small_sample_does_not_overcommit() -> None:
    """Fold sizing must not exceed the available rows (regression test)."""
    folds = time_series_folds(4_800, n_splits=5, validation_fraction=0.2)
    assert folds
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_encode_categoricals_marks_missing_and_handles_unseen() -> None:
    """Missing categories become a label and unseen ones a safe default."""
    train = pd.DataFrame(
        {
            "cat": ["a", "b", None, "a"],
            "num": [1.0, np.nan, 3.0, 4.0],
            "TransactionDT": [10, 20, 30, 40],
        }
    )
    valid = pd.DataFrame(
        {
            "cat": ["a", "zzz", None, "b"],
            "num": [2.0, 5.0, np.nan, 6.0],
            "TransactionDT": [50, 60, 70, 80],
        }
    )
    encoded_train, encoded_valid, state = encode_categoricals(
        train,
        valid,
        target_column="isFraud",
        target=pd.Series([0, 1, 0, 0]),
        time_column="TransactionDT",
        strategy="frequency",
    )
    assert not encoded_train.isna().any().any()
    assert not encoded_valid.isna().any().any()
    # "Missing" is the most frequent label in a 4-row training set (1 of 4).
    assert state["frequency_maps"]["cat"]["Missing"] == pytest.approx(0.25)
    # Unseen category maps to frequency 0 rather than NaN.
    assert encoded_valid["cat"].iloc[1] == pytest.approx(0.0)


def test_category_dtype_is_encoded_not_zeroed() -> None:
    """Regression: a ``category``-dtype column must be encoded, not coerced to 0.

    ``reduce_mem_usage`` turns low-cardinality strings into ``category``; if the
    detector skipped that dtype the column would be treated as numeric, coerced
    to ``NaN`` and filled with the median 0.0 - destroying the signal silently.
    """
    train = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="category"),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    valid = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="category"),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    encoded_train, encoded_valid, state = encode_categoricals(
        train, valid, target_column="isFraud", strategy="frequency"
    )
    assert "cat" in state["categorical_columns"]
    # Both categories occur half the time, so the encoded value must be 0.5 -
    # not the 0.0 a numeric mis-detection would have produced.
    assert encoded_train["cat"].to_numpy() == pytest.approx(0.5)
    assert encoded_valid["cat"].to_numpy() == pytest.approx(0.5)


def test_encode_categoricals_refuses_target_in_frame() -> None:
    """Passing the label inside the feature frame must fail loudly."""
    frame = pd.DataFrame({"cat": ["a", "b"], "isFraud": [0, 1]})
    with pytest.raises(ValueError):
        encode_categoricals(frame, frame, target_column="isFraud", strategy="frequency")


def test_target_encoding_requires_labels() -> None:
    """Target encoding without labels must raise."""
    frame = pd.DataFrame({"cat": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="requires the training labels"):
        encode_categoricals(frame, frame, target_column="isFraud", strategy="target")


def test_transform_with_encoding_round_trip() -> None:
    """A persisted state must reproduce the validation encoding exactly."""
    train = pd.DataFrame({"cat": ["a", "b", None, "a"], "num": [1.0, np.nan, 3.0, 4.0]})
    valid = pd.DataFrame({"cat": ["a", "b", None, "zzz"], "num": [9.0, 8.0, 7.0, 6.0]})
    _, encoded_valid, state = encode_categoricals(
        train, valid, target_column="isFraud", strategy="frequency"
    )
    restored = transform_with_encoding(valid, state)
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True), encoded_valid.reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def test_resampling_increases_minority_only() -> None:
    """SMOTE must change the training matrix, never the label distribution of a
    validation set (which it is not given)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 5)).astype("float32"), columns=list("abcde"))
    y = pd.Series((rng.random(400) < 0.05).astype(int))
    X_resampled, y_resampled = apply_resampling(
        X, y, strategy="smote", ratio=0.5, k_neighbors=3, random_state=0
    )
    assert len(X_resampled) > len(X)
    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.mean() > y.mean()
    minority_ratio = y_resampled.mean() / (1 - y_resampled.mean())
    assert minority_ratio == pytest.approx(0.5, abs=0.05)


def test_resampling_none_is_identity() -> None:
    """strategy='none' must return the inputs unchanged."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1])
    X_out, y_out = apply_resampling(X, y, strategy="none")
    pd.testing.assert_frame_equal(X_out, X)
    pd.testing.assert_series_equal(y_out, y)


def test_borderline_smote_is_available() -> None:
    """BorderlineSMOTE must be wired and produce a resampled frame."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 4)).astype("float32"), columns=list("wxyz"))
    y = pd.Series((rng.random(200) < 0.1).astype(int))
    X_resampled, y_resampled = apply_resampling(
        X, y, strategy="borderline", ratio=0.3, k_neighbors=3
    )
    assert len(X_resampled) > len(X)
    assert y_resampled.sum() > y.sum()


def test_resampling_rejects_bad_ratio() -> None:
    """Ratios outside (0, 1] are rejected."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError):
        apply_resampling(X, y, strategy="smote", ratio=1.5)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_compute_metrics_known_values() -> None:
    """Perfect predictions produce perfect metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(np.mean((y_prob - y_true) ** 2), abs=1e-9)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0


def test_compute_metrics_rejects_shape_mismatch() -> None:
    """Mismatched shapes are an error, not silently broadcast."""
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.1])


def test_threshold_optimiser_matches_cost_formula() -> None:
    """Optimal cost must equal FN*fraud_cost + FP*friction_cost at the choice."""
    rng = np.random.default_rng(3)
    y_true = (rng.random(2_000) < 0.1).astype(int)
    y_prob = np.clip(y_true * 0.5 + rng.random(2_000) * 0.5, 0, 1)
    result = optimise_threshold(
        y_true,
        y_prob,
        fraud_cost=DEFAULT_FRAUD_COST,
        friction_cost=DEFAULT_FRICTION_COST,
    )
    expected = result.false_negatives * DEFAULT_FRAUD_COST + (
        result.false_positives * DEFAULT_FRICTION_COST
    )
    assert result.cost == pytest.approx(expected)
    assert 0.0 <= result.threshold <= 1.0


def test_threshold_optimiser_prefers_low_threshold_when_fraud_is_costly() -> None:
    """A 100:1 cost ratio must not choose a conservative threshold."""
    rng = np.random.default_rng(4)
    y_true = (rng.random(1_000) < 0.2).astype(int)
    y_prob = np.clip(y_true * 0.4 + rng.random(1_000) * 0.6, 0, 1)
    cheap_friction = optimise_threshold(
        y_true, y_prob, fraud_cost=10_000.0, friction_cost=1.0
    )
    expensive_friction = optimise_threshold(
        y_true, y_prob, fraud_cost=10.0, friction_cost=1_000.0
    )
    assert cheap_friction.threshold <= expensive_friction.threshold


def test_threshold_optimiser_respects_recall_floor() -> None:
    """A recall constraint must be honoured when reachable."""
    rng = np.random.default_rng(5)
    y_true = (rng.random(1_000) < 0.2).astype(int)
    y_prob = np.clip(y_true * 0.4 + rng.random(1_000) * 0.6, 0, 1)
    result = optimise_threshold(y_true, y_prob, target_recall=0.9)
    assert result.recall >= 0.9 - 1e-9


def test_evaluate_predictions_writes_figures(tmp_path: Path) -> None:
    """The full evaluation path must persist figures and a JSON report."""
    rng = np.random.default_rng(6)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_prob = np.clip(y_true * 0.5 + rng.random(500) * 0.5, 0, 1)
    report = evaluate_predictions(
        y_true, y_prob, model_name="unit", figures_dir=tmp_path
    )
    written = {p.name for p in tmp_path.glob("*.png")}
    assert {"confusion_matrix_unit.png", "precision_recall_unit.png"} <= written
    assert report.metrics_at_optimal["pr_auc"] > 0.0
    path = save_report({"unit": report}, tmp_path / "report.json")
    payload = json.loads(path.read_text())
    assert payload["unit"]["model_name"] == "unit"


# --------------------------------------------------------------------------- #
# Feature selection helpers
# --------------------------------------------------------------------------- #
def test_shap_ranking_orders_by_magnitude() -> None:
    """Features with larger |SHAP| must rank first."""
    values = np.array([[1.0, -5.0, 0.5], [2.0, -4.0, 0.1]])
    ranking = rank_features_by_shap(values, ["a", "b", "c"])
    assert ranking.index[0] == "b"
    assert ranking.iloc[0] == pytest.approx(4.5)


def test_selected_features_json_round_trip(tmp_path: Path) -> None:
    """Persisted feature lists must reload identically."""
    path = save_selected_features(["f1", "f2"], tmp_path / "sel.json", top_n=2)
    assert load_selected_features(path) == ["f1", "f2"]
    assert json.loads(path.read_text())["n_features"] == 2


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_chronological_holdout_splits_from_the_end() -> None:
    """Early-stopping slice must be the most recent rows."""
    X = pd.DataFrame({"t": np.arange(100), "v": np.arange(100)})
    y = pd.Series(np.arange(100) % 2)
    X_fit, _, X_stop, _ = chronological_holdout(X, y, fraction=0.2)
    assert len(X_stop) == 20
    assert X_fit["t"].max() < X_stop["t"].min()


def test_fit_estimator_trains_and_predicts(synthetic_frame: pd.DataFrame) -> None:
    """The shared fitting helper must work for XGBoost with early stopping."""
    features = ["TransactionAmt", "C13", "V12", "card1"]
    X = synthetic_frame[features].fillna(0).astype("float32")
    y = synthetic_frame["isFraud"]
    model = build_model("xgboost", y_train=y, n_jobs=1)
    fit_estimator(
        model, X.iloc[:600], y.iloc[:600], X_stop=X.iloc[600:], y_stop=y.iloc[600:]
    )
    probabilities = model.predict_proba(X.iloc[600:])[:, 1]
    assert probabilities.shape == (len(X) - 600,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_cross_validation_reports_fold_scores(synthetic_frame: pd.DataFrame) -> None:
    """Time-series CV must return one PR-AUC per fold."""
    features = ["TransactionAmt", "C13", "V12", "card1"]
    X = synthetic_frame[features].fillna(0).astype("float32")
    y = synthetic_frame["isFraud"]
    result = cross_validate_time_series(
        X, y, model_name="lightgbm", n_splits=2, n_jobs=1
    )
    assert result["fold_scores"]
    assert 0.0 <= result["mean_pr_auc"] <= 1.0


def test_stacking_ensemble_uses_oof_meta_features(
    synthetic_frame: pd.DataFrame,
) -> None:
    """The time-aware stacker must fit, predict and expose OOF scores."""
    features = ["TransactionAmt", "C13", "V12", "card1"]
    X = synthetic_frame[features].fillna(0).astype("float32")
    y = synthetic_frame["isFraud"]
    ensemble = StackingEnsemble(
        ["xgboost", "lightgbm"], n_splits=2, n_jobs=1, meta_c=1.0
    )
    ensemble.fit(X, y)
    probabilities = ensemble.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    assert set(ensemble.oof_scores_) == {"xgboost", "lightgbm"}


def test_stacking_rejects_single_model() -> None:
    """Stacking needs at least two base learners."""
    with pytest.raises(ValueError):
        StackingEnsemble(["xgboost"])


def test_normalise_storage_builds_sqlite_url(tmp_path: Path) -> None:
    """A bare path must become a SQLAlchemy URL Optuna can parse."""
    url = normalise_storage(tmp_path / "study.db")
    assert url is not None and url.startswith("sqlite:///")
    assert "study.db" in url


def test_normalise_storage_passes_through_urls() -> None:
    """An existing URL must be returned unchanged."""
    assert normalise_storage("sqlite:///tmp/x.db") == "sqlite:///tmp/x.db"


def test_normalise_storage_none_means_in_memory() -> None:
    """``None`` disables persistence."""
    assert normalise_storage(None) is None


def test_model_persistence_round_trip(
    tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    """joblib artifacts must reload and reproduce predictions."""
    features = ["TransactionAmt", "C13", "V12", "card1"]
    X = synthetic_frame[features].fillna(0).astype("float32")
    y = synthetic_frame["isFraud"]
    model = build_model("xgboost", y_train=y, n_jobs=1, overrides={"n_estimators": 20})
    model.fit(X, y)
    path = save_model(model, tmp_path / "model.joblib")
    reloaded = load_model(path)
    np.testing.assert_allclose(
        model.predict_proba(X)[:, 1],
        reloaded.predict_proba(X)[:, 1],
        rtol=1e-6,
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_humanise_feature_adds_readable_labels() -> None:
    """Report labels must describe anonymised columns while keeping raw names."""
    assert humanise_feature("V91") == "V91 (Vesta feature 91)"
    assert humanise_feature("C13") == "C13 (count feature 13)"
    assert humanise_feature("D15") == "D15 (timedelta feature 15)"
    assert humanise_feature("M6") == "M6 (match flag M6)"
    assert humanise_feature("id_17") == "id_17 (identity feature 17)"
    assert humanise_feature("card6") == "card6 (card type (debit/credit))"
    assert (
        humanise_feature("transaction_hour_sin") == "transaction_hour_sin (hour (sin))"
    )
    # Unknown columns are returned untouched (never silently renamed).
    assert humanise_feature("some_new_feature") == "some_new_feature"
    assert humanise_features(["V1", "unknown"]) == ["V1 (Vesta feature 1)", "unknown"]


def test_config_rejects_invalid_threshold() -> None:
    """Missing-threshold bounds are validated at construction."""
    with pytest.raises(ValueError):
        PipelineConfig(missing_drop_threshold=1.5)


def test_config_round_trip(tmp_path: Path, small_config: PipelineConfig) -> None:
    """A config written to disk must reload with identical values."""
    path = small_config.save(tmp_path / "config.json")
    restored = PipelineConfig.load(path)
    assert restored.top_n_features == small_config.top_n_features
    assert restored.categorical_columns == small_config.categorical_columns


def test_config_unknown_override_is_rejected(small_config: PipelineConfig) -> None:
    """Typos in overrides must not be silently ignored."""
    with pytest.raises(TypeError):
        small_config.with_overrides(not_a_field=1)


# --------------------------------------------------------------------------- #
# End-to-end integration
# --------------------------------------------------------------------------- #
def test_build_processed_dataset_end_to_end(
    synthetic_frame: pd.DataFrame, small_config: PipelineConfig
) -> None:
    """Preprocessing must drop leakage columns and produce aligned matrices."""
    processed = build_processed_dataset(
        small_config, raw_df=synthetic_frame, save=False
    )
    assert isinstance(processed, ProcessedData)
    assert processed.X_train.shape[0] == len(processed.y_train)
    assert processed.X_valid.shape[0] == len(processed.y_valid)
    assert list(processed.X_train.columns) == list(processed.X_valid.columns)
    assert "isFraud" not in processed.X_train.columns
    assert "TransactionID" not in processed.X_train.columns
    # Constant and >80% missing columns are removed.
    assert "constant_col" not in processed.X_train.columns
    assert "mostly_missing" not in processed.X_train.columns
    # No missing values survive and everything is float32.
    assert not processed.X_train.isna().any().any()
    assert all(dtype == np.float32 for dtype in processed.X_train.dtypes)


# --------------------------------------------------------------------------- #
# Regression tests for fixed bugs
# --------------------------------------------------------------------------- #
def test_reduce_mem_usage_preserves_nan_not_sentinel() -> None:
    """NaN must survive as a genuine nullable-int NA, never a -127 sentinel."""
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = reduce_mem_usage(frame, verbose=False)
    # Integral float with a NaN -> nullable Int with real missingness.
    assert pd.api.types.is_extension_array_dtype(result["x"].dtype)
    assert result["x"].isna().tolist() == [False, False, True, False]
    # No injected sentinel values should appear in the data.
    assert not (result["x"].fillna(0) <= -127).any()


def test_config_with_overrides_ignores_none() -> None:
    """CLI-style ``None`` values must not clobber a loaded config (merge fix)."""
    config = PipelineConfig(tune=False, top_n_features=30)
    kept = config.with_overrides(tune=None, top_n_features=None)
    assert kept.tune is False
    assert kept.top_n_features == 30
    overridden = config.with_overrides(tune=True, top_n_features=50)
    assert overridden.tune is True
    assert overridden.top_n_features == 50


def test_clean_cognitive_columns_is_case_insensitive() -> None:
    """'NotFound' variants must all collapse to missing regardless of case."""
    frame = pd.DataFrame(
        {"dev": ["NotFound", "NOTFOUND", "notfound", "chrome 70", None]}
    )
    _clean_cognitive_columns(frame)
    assert frame["dev"].isna().sum() == 4  # three placeholders + original None
    assert frame["dev"].dropna().tolist() == ["chrome 70"]


def test_processed_dataset_round_trip_preserves_time(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Reloading a saved dataset must restore the time columns."""
    config = PipelineConfig(
        project_root=str(tmp_path),
        data_path=str(tmp_path / "data" / "raw"),
        missing_drop_threshold=0.80,
    )
    processed = build_processed_dataset(config, raw_df=synthetic_frame, save=True)
    reloaded = load_processed_dataset(config.processed_dir)
    assert len(reloaded.time_train) == len(processed.time_train)
    assert len(reloaded.time_valid) == len(processed.time_valid)
    pd.testing.assert_series_equal(
        reloaded.time_train,
        processed.time_train.reset_index(drop=True),
        check_names=False,
    )
