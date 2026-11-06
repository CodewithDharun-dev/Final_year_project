"""Data cleaning, memory downcasting, encoding, time-aware splits and SMOTE.

The module is deliberately side-effect free: every transformation returns new
objects (or writes only when an explicit ``path`` argument is supplied) and all
learned statistics are captured in plain dictionaries so that they can be
persisted and re-applied at inference time (see :func:`transform_with_encoding`).

Leakage-safety contract
-----------------------
* Every statistic (missing thresholds, medians, category vocabularies, target
  encoding, SMOTE neighbours) is fitted **only** on the training period.
* Target encoding is computed out-of-fold over time blocks, so the encoding of
  row *t* never uses the label of row *t*.
* SMOTE only ever sees ``X_train``; :func:`apply_resampling` refuses to touch
  validation or test matrices.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import PipelineConfig

__all__ = [
    "RESERVED_CATEGORY_TOKEN",
    "ProcessedData",
    "apply_downcasting",
    "apply_resampling",
    "build_processed_dataset",
    "detect_categorical_columns",
    "drop_high_missing_columns",
    "encode_categoricals",
    "engineer_features",
    "find_csv",
    "load_processed_dataset",
    "load_raw_data",
    "reduce_mem_usage",
    "select_features",
    "split_by_time",
    "time_series_folds",
    "transform_with_encoding",
]

LOGGER = logging.getLogger(__name__)

#: Sentinel used for missing categories before encoding.
MISSING_CATEGORY: str = "Missing"

#: Numeric sentinel emitted for categories unseen at fit time.  It is far
#: outside the observed feature range so tree models can isolate it.
RESERVED_CATEGORY_TOKEN: float = -999.0

#: Columns that are identifiers even if the config omits them.
_PROTECTED_COLUMNS: frozenset[str] = frozenset({"TransactionID", "isFraud"})

#: Placeholder detection for the IEEE-CIS ``"NotFound"`` literal.
_PLACEHOLDER_TOKENS: frozenset[str] = frozenset({"notfound", "missing", "", "nan"})

#: Canonical column prefixes for the identity-file join.
_IDENTITY_PREFIX_RE = re.compile(r"^id[-_]")


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedData:
    """Container for the leakage-safe train/validation matrices.

    Attributes
    ----------
    X_train, X_valid:
        Numeric feature matrices, identical schema, ``float32`` where possible.
    y_train, y_valid:
        Binary labels (``int8``) aligned to the matrices.
    time_train, time_valid:
        ``TransactionDT`` values aligned to the rows (for auditability).
    feature_names:
        Ordered list of model-ready columns.
    metadata:
        JSON-serialisable description of the transformations that were applied.
    """

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    time_train: pd.Series
    time_valid: pd.Series
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable, log-friendly summary of the split."""
        return (
            f"train={self.X_train.shape}, valid={self.X_valid.shape}, "
            f"fraud rate train={float(self.y_train.mean()):.3%}, "
            f"valid={float(self.y_valid.mean()):.3%}"
        )


# --------------------------------------------------------------------------- #
# 1. Memory optimisation
# --------------------------------------------------------------------------- #
def _nullable_int_dtype(min_value: float, max_value: float) -> str:
    """Pick the cheapest pandas nullable integer dtype able to hold the range.

    Note: pandas nullable ``Int8``/``Int16``/``Int32`` reserve their minimum
    value as the internal ``NA`` marker, so the usable bounds are one narrower
    than the equivalent ``numpy`` int (e.g. ``Int8`` holds ``[-127, 127]``).
    """
    if min_value >= -127 and max_value <= 127:
        return "Int8"
    if min_value >= -32_767 and max_value <= 32_767:
        return "Int16"
    if min_value >= -2_147_483_647 and max_value <= 2_147_483_647:
        return "Int32"
    return "Int64"


def reduce_mem_usage(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
    exclude_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Automatically downcast every column to the smallest safe dtype.

    * ``float64`` -> ``float32`` when the round-trip error is acceptable.
    * Integer-like floats -> nullable ``Int8/16/32`` (preserves ``NaN``).
    * Integer columns -> ``int8/int16/int32`` (or nullable equivalents).
    * Categorical/binomial string columns -> ``category`` with ``"Missing"``.

    The function modifies a copy of the frame and never converts object
    columns that are genuinely high-cardinality strings.

    Parameters
    ----------
    df:
        Input frame. Not modified in place.
    verbose:
        Emit memory statistics at ``INFO`` level.
    exclude_columns:
        Columns that must stay untouched (e.g. the target).

    Returns
    -------
    pandas.DataFrame
        A downcast copy of ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> reduce_mem_usage(pd.DataFrame({"a": [1.5, 2.5]})).dtypes["a"]
    dtype('float32')
    """
    protected = set(_PROTECTED_COLUMNS)
    if exclude_columns:
        protected |= set(exclude_columns)

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    result = df.copy()

    for column in result.columns:
        series = result[column]
        if column in protected:
            continue
        kind = series.dtype

        try:
            if pd.api.types.is_float_dtype(kind):
                non_null = series.dropna()
                if non_null.empty:
                    result[column] = series.astype("float32")
                    continue
                is_integral = bool((non_null == non_null.round()).all())
                min_value = float(non_null.min())
                max_value = float(non_null.max())
                if (
                    is_integral
                    and max_value < np.iinfo(np.int32).max
                    and min_value > np.iinfo(np.int32).min
                ):
                    # Preserve genuine NaN as pandas NA - do NOT inject a
                    # sentinel value, which could collide with real negatives.
                    result[column] = series.astype(
                        _nullable_int_dtype(min_value, max_value)
                    )
                else:
                    as_32 = series.astype("float32")
                    with np.errstate(invalid="ignore"):
                        round_trip_ok = bool(
                            np.allclose(
                                as_32.to_numpy(dtype=np.float64, na_value=np.nan),
                                series.to_numpy(dtype=np.float64, na_value=np.nan),
                                rtol=1e-5,
                                atol=1e-8,
                                equal_nan=True,
                            )
                        )
                    result[column] = as_32 if round_trip_ok else series

            elif pd.api.types.is_integer_dtype(kind):
                min_value = int(series.min()) if series.notna().any() else 0
                max_value = int(series.max()) if series.notna().any() else 0
                if pd.api.types.is_extension_array_dtype(kind):
                    result[column] = series.astype(
                        _nullable_int_dtype(min_value, max_value)
                    )
                elif (
                    min_value >= np.iinfo(np.uint8).min
                    and max_value <= np.iinfo(np.uint8).max
                ):
                    result[column] = series.astype(np.uint8)
                elif (
                    min_value >= np.iinfo(np.int8).min
                    and max_value <= np.iinfo(np.int8).max
                ):
                    result[column] = series.astype(np.int8)
                elif (
                    min_value >= np.iinfo(np.int16).min
                    and max_value <= np.iinfo(np.int16).max
                ):
                    result[column] = series.astype(np.int16)
                elif (
                    min_value >= np.iinfo(np.int32).min
                    and max_value <= np.iinfo(np.int32).max
                ):
                    result[column] = series.astype(np.int32)
                else:
                    result[column] = series.astype(np.int64)

            elif isinstance(kind, pd.CategoricalDtype):
                continue

            elif (
                pd.api.types.is_string_dtype(kind) or pd.api.types.is_object_dtype(kind)
            ) and not pd.api.types.is_numeric_dtype(kind):
                if series.isna().all():
                    continue
                cardinality = series.nunique(dropna=True)
                if (
                    cardinality <= max(2, int(len(series) * 0.5))
                    and cardinality <= 100_000
                ):
                    result[column] = series.astype("category")
        except (TypeError, ValueError, OverflowError) as exc:  # pragma: no cover
            LOGGER.debug("reduce_mem_usage skipped %s: %s", column, exc)

    gc.collect()
    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        LOGGER.info(
            "reduce_mem_usage: %.1f MB -> %.1f MB (%.1f%% reduction)",
            start_mem,
            end_mem,
            (1 - end_mem / start_mem) * 100 if start_mem else 0.0,
        )
    return result


def apply_downcasting(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    exclude_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Conditionally apply :func:`reduce_mem_usage` (config-aware wrapper)."""
    if not enabled:
        return df
    return reduce_mem_usage(df, verbose=True, exclude_columns=exclude_columns)


# --------------------------------------------------------------------------- #
# 2. Loading / merging
# --------------------------------------------------------------------------- #
def find_csv(directory: Path, filename: str) -> Path:
    """Locate ``filename`` inside ``directory`` case-insensitively.

    Raises
    ------
    FileNotFoundError
        If no case-insensitive match exists. The message lists what was found
        so that misnamed exports (e.g. ``Train_Transaction.csv``) are obvious.
    """
    directory = Path(directory)
    target = filename.lower()
    if not directory.is_dir():
        raise FileNotFoundError(f"data directory not found: {directory}")
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name.lower() == target:
            return path
    available = sorted(p.name for p in directory.glob("*.csv"))
    raise FileNotFoundError(
        f"{filename!r} not found in {directory}. CSV files present: {available}"
    )


def load_raw_data(
    config: PipelineConfig,
    *,
    transaction_path: Path | str | None = None,
    identity_path: Path | str | None = None,
    downcast: bool | None = None,
    load_identity: bool = True,
) -> pd.DataFrame:
    """Load and merge the transaction (and identity) tables.

    The merge is validated as ``one_to_one`` after normalising identity column
    names (the public test file uses ``id-01`` while train uses ``id_01``).

    Parameters
    ----------
    config:
        Pipeline configuration supplying file names and dtypes.
    transaction_path, identity_path:
        Explicit overrides; otherwise resolved case-insensitively under
        ``config.data_path``.
    downcast:
        Override ``config.downcast`` for this load.
    load_identity:
        Set ``False`` to load only the transaction table.

    Returns
    -------
    pandas.DataFrame
        ``train_transaction`` left-joined with ``train_identity``.
    """
    started = time.perf_counter()
    data_dir = Path(config.data_path)
    txn_path = (
        Path(transaction_path)
        if transaction_path
        else find_csv(data_dir, config.transaction_filename)
    )
    if not txn_path.exists():
        raise FileNotFoundError(f"transaction file not found: {txn_path}")

    LOGGER.info("Loading transactions from %s", txn_path)
    transaction = pd.read_csv(txn_path, low_memory=False)
    LOGGER.info(
        "  transactions: %s", f"{transaction.shape[0]:,} x {transaction.shape[1]}"
    )

    if not load_identity:
        merged = transaction
    else:
        id_path = (
            Path(identity_path)
            if identity_path
            else find_csv(data_dir, config.identity_filename)
        )
        LOGGER.info("Loading identity from %s", id_path)
        identity = pd.read_csv(id_path, low_memory=False)
        identity = identity.rename(columns=lambda c: _IDENTITY_PREFIX_RE.sub("id_", c))
        LOGGER.info("  identity: %s", f"{identity.shape[0]:,} x {identity.shape[1]}")

        if not identity["TransactionID"].is_unique:
            duplicate_count = int(identity["TransactionID"].duplicated().sum())
            LOGGER.warning(
                "identity has %d duplicate TransactionIDs; keeping the first",
                duplicate_count,
            )
            identity = identity.drop_duplicates("TransactionID", keep="first")

        merged = transaction.merge(
            identity,
            on="TransactionID",
            how="left",
            validate="one_to_one",
        )
        joined = merged[list(identity.columns[1:])].notna().any(axis=1).mean()
        LOGGER.info(
            "  merge: %s rows x %d cols; %.1f%% of transactions have identity data",
            f"{len(merged):,}",
            merged.shape[1],
            joined * 100,
        )

    should_downcast = config.downcast if downcast is None else downcast
    merged = apply_downcasting(
        merged,
        enabled=should_downcast,
        exclude_columns=[config.target_column],
    )
    LOGGER.info("Loaded raw data in %.1fs", time.perf_counter() - started)
    return merged


# --------------------------------------------------------------------------- #
# 3. Cleaning
# --------------------------------------------------------------------------- #
def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return every non-numeric, non-boolean column that needs encoding.

    ``pandas`` 3 deprecates ``select_dtypes(include="object")`` for string
    columns, so dtypes are inspected directly, which is also faster.

    ``category`` dtype is included deliberately: :func:`reduce_mem_usage`
    converts low-cardinality strings to ``category``, and treating those as
    numeric would coerce every value to ``NaN`` and then to ``0.0`` - silently
    destroying the signal.
    """
    return [
        column
        for column, dtype in df.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_datetime64_any_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    ]


def _clean_cognitive_columns(df: pd.DataFrame) -> list[str]:
    """Normalise the IEEE-CIS ``"NotFound"`` placeholder into ``"Missing"``.

    Detection is case-insensitive and whitespace-tolerant, so ``NotFound``,
    ``NOTFOUND`` and ``notfound`` are all collapsed to ``NA`` (which the
    categorical encoder later maps to ``"Missing"``).
    """
    touched: list[str] = []
    for column in detect_categorical_columns(df):
        series = df[column]
        try:
            lowered = series.astype("string").str.strip().str.lower()
        except (TypeError, ValueError):  # pragma: no cover
            continue
        if lowered.isin(_PLACEHOLDER_TOKENS).any():
            normalised = series.astype("string").str.strip()
            normalised = normalised.where(~lowered.isin(_PLACEHOLDER_TOKENS))
            # Collapse whitespace to keep device strings consistent.
            df[column] = normalised.str.replace(r"\s+", " ", regex=True)
            touched.append(column)
    return touched


def _split_value_into_features(
    df: pd.DataFrame, column: str, pattern: str, prefix: str
) -> list[str]:
    """Expand a compound string column (OS/browser, device) into sub-features."""
    extracted = df[column].astype("string").str.extract(pattern, expand=True)
    created: list[str] = []
    for index, part in enumerate(extracted.columns, start=1):
        name = f"{prefix}_{index}"
        df[name] = extracted[part].astype("string").str.lower()
        created.append(name)
    return created


def engineer_features(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    """Add interpretable transaction-time and structural features in place.

    Features
    --------
    * ``TransactionDT_log`` - compressed continuous time indicator.
    * ``transaction_day`` / ``transaction_dow`` / ``transaction_hour`` - the
      IEEE-CIS time base is seconds; days and hours of the synthetic week are
      stable monotone transforms of time (no target information).
    * ``email_match`` / ``email_domain_prefix`` - payer-payee mismatch signal.
    * ``amount_cent`` - trailing cents, a known fraud tells for round amounts.
    * ``id_31_*`` - browser/OS tokens extracted from the device string.
    * ``id_33_*`` - screen resolution height and width.

    Returns
    -------
    list[str]
        Names of the newly created columns (excludes pre-existing ones).
    """
    if config.time_unit_seconds <= 0:
        raise ValueError("time_unit_seconds must be positive")
    created: list[str] = []
    time_column = config.time_columns[0]

    if time_column in df.columns:
        seconds = df[time_column].astype("float32")
        df["TransactionDT_log"] = np.log1p(seconds.clip(lower=0)).astype("float32")
        unit = float(config.time_unit_seconds)
        day = (seconds // unit).astype("float32")
        df["transaction_day"] = day
        df["transaction_dow"] = (day % 7).astype("float32")
        df["transaction_hour"] = ((seconds % unit) / 3600.0).astype("float32")
        hourly = df["transaction_hour"]
        df["transaction_hour_sin"] = np.sin(2 * np.pi * hourly / 24).astype("float32")
        df["transaction_hour_cos"] = np.cos(2 * np.pi * hourly / 24).astype("float32")
        created += [
            "TransactionDT_log",
            "transaction_day",
            "transaction_dow",
            "transaction_hour",
            "transaction_hour_sin",
            "transaction_hour_cos",
        ]

    if "TransactionAmt" in df.columns:
        amount = df["TransactionAmt"].astype("float32")
        df["TransactionAmt_log"] = np.log1p(amount.clip(lower=0)).astype("float32")
        df["amount_cent"] = (np.mod(amount * 100.0, 100.0).round()).astype("float32")
        created += ["TransactionAmt_log", "amount_cent"]

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        payer = df["P_emaildomain"].astype("string").str.lower()
        payee = df["R_emaildomain"].astype("string").str.lower()
        df["email_match"] = (payer == payee).astype("float32")
        df["email_match"] = df["email_match"].where(payer.notna() & payee.notna())
        df["email_domain_prefix"] = payer.str.split(".").str[0].astype("string")
        created += ["email_match", "email_domain_prefix"]

    if "id_31" in df.columns:
        created += _split_value_into_features(
            df, "id_31", r"([a-z\-]+)", "id_31_browser"
        )
    if "id_33" in df.columns:
        created += _split_value_into_features(df, "id_33", r"(\d+)x(\d+)", "id_33_res")

    LOGGER.info("Engineered %d features: %s", len(created), created)
    return created


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.80,
    *,
    protected: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Drop columns whose missing fraction exceeds ``threshold``.

    Returns
    -------
    tuple[pandas.DataFrame, list[dict]]
        The reduced frame and an audit log of dropped columns with their
        observed missing fractions.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    protected_set = set(protected)
    fraction = df.isna().mean()
    to_drop = [
        column
        for column in df.columns
        if column not in protected_set and fraction[column] > threshold
    ]
    audit = [
        {"column": c, "missing_fraction": float(fraction[c]), "reason": "high_missing"}
        for c in to_drop
    ]
    if to_drop:
        LOGGER.info(
            "Dropping %d columns with > %.0f%% missing", len(to_drop), threshold * 100
        )
    return df.drop(columns=to_drop), audit


def _drop_constant_columns(
    df: pd.DataFrame, protected: Iterable[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Drop columns with no variation (after missing handling)."""
    protected_set = set(protected)
    constants = [
        column
        for column in df.columns
        if column not in protected_set and df[column].nunique(dropna=True) <= 1
    ]
    audit = [{"column": c, "reason": "constant"} for c in constants]
    if constants:
        LOGGER.info("Dropping %d constant columns", len(constants))
    return df.drop(columns=constants), audit


def _drop_duplicate_columns(
    df: pd.DataFrame, protected: Iterable[str], *, sample_rows: int = 20_000
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Remove feature columns that are exact copies of an earlier column.

    Uses a two-stage check: a hash on a random sample prunes candidates, then a
    full-frame equality test confirms, so the quadratic worst case is avoided.
    """
    protected_set = set(protected)
    candidates = [c for c in df.columns if c not in protected_set]
    if len(candidates) < 2:
        return df, []
    sample = df[candidates].head(sample_rows)
    buckets: dict[tuple[int, ...], list[str]] = {}
    for column in candidates:
        key_hash = int(pd.util.hash_pandas_object(sample[column], index=False).sum())
        dtype_key = str(sample[column].dtype)
        buckets.setdefault((key_hash, hash(dtype_key)), []).append(column)

    duplicates: list[dict[str, Any]] = []
    for columns in buckets.values():
        if len(columns) < 2:
            continue
        keep = columns[0]
        for other in columns[1:]:
            try:
                identical = df[keep].equals(df[other])
            except (TypeError, ValueError):  # pragma: no cover
                identical = False
            if identical:
                duplicates.append({"column": other, "duplicate_of": keep})
    if duplicates:
        LOGGER.info("Dropping %d duplicate feature columns", len(duplicates))
    return df.drop(columns=[d["column"] for d in duplicates]), duplicates


# --------------------------------------------------------------------------- #
# 4. Splitting
# --------------------------------------------------------------------------- #
def split_by_time(
    df: pd.DataFrame,
    *,
    time_column: str,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split ``df`` into train (older) and validation (newer).

    Rows are sorted by ``time_column``; the last ``validation_fraction`` of
    rows form the validation set. Ties on the boundary stay intact because the
    split is index based, not timestamp based.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if time_column not in df.columns:
        raise KeyError(f"time column {time_column!r} missing")
    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - validation_fraction))
    cut = max(1, min(cut, len(ordered) - 1))
    train = ordered.iloc[:cut].reset_index(drop=True)
    valid = ordered.iloc[cut:].reset_index(drop=True)
    LOGGER.info(
        "Time split at row %d: train <= %s < validation <= %s",
        cut,
        ordered[time_column].iloc[cut - 1],
        ordered[time_column].iloc[-1],
    )
    # Both frames are re-indexed from 0 so that positional and label alignment
    # stay identical across X, y and any downstream sampled subsets.
    return train, valid


def time_series_folds(
    n_samples: int,
    *,
    n_splits: int = 4,
    time_values: Sequence[float] | np.ndarray | None = None,
    validation_fraction: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build leakage-free expanding-window folds.

    Data must already be sorted chronologically. When ``time_values`` are
    given, folds are constructed so that every training index precedes every
    validation index, even if timestamps repeat.
    """
    if n_samples < 4:
        raise ValueError("need at least 4 samples to build time folds")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if time_values is None:
        # Each fold consumes `test_size` rows, and at least one row must remain
        # for training, so n_splits * test_size must stay below n_samples.
        test_size = max(1, int(n_samples * validation_fraction))
        effective_splits = max(1, min(n_splits, n_samples // 2))
        while (
            effective_splits > 1
            and effective_splits * test_size >= n_samples - test_size
        ):
            effective_splits -= 1
        splits = TimeSeriesSplit(n_splits=effective_splits, test_size=test_size)
        for train_idx, valid_idx in splits.split(np.arange(n_samples)):
            folds.append((train_idx, valid_idx))
        return folds

    values = np.asarray(time_values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    unique_times = np.unique(sorted_values)
    if len(unique_times) < n_splits + 1:
        raise ValueError(
            f"cannot build {n_splits} time folds from {len(unique_times)} time points"
        )
    # Reserve the first half of unique time points for growing training data.
    usable = unique_times[
        np.linspace(
            int(len(unique_times) * 0.45), len(unique_times) - 1, n_splits + 1
        ).astype(int)
    ]
    for start, end in itertools.pairwise(usable):
        train_mask = sorted_values < start
        valid_mask = (sorted_values >= start) & (sorted_values < end)
        train_idx = np.sort(order[train_mask])
        valid_idx = np.sort(order[valid_mask])
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


# --------------------------------------------------------------------------- #
# 5. Encoding
# --------------------------------------------------------------------------- #
@dataclass
class EncodingState:
    """Everything required to encode unseen data exactly like training data."""

    strategy: str
    categorical_columns: list[str]
    numeric_columns: list[str]
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    target_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    target_prior: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialise the state."""
        return {
            "strategy": self.strategy,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "numeric_fill_values": self.numeric_fill_values,
            "frequency_maps": self.frequency_maps,
            "target_maps": self.target_maps,
            "category_levels": self.category_levels,
            "target_prior": self.target_prior,
        }


def _numeric_fill_value(series: pd.Series, strategy: str) -> float:
    """Return the imputation constant for a numeric column.

    Parameters
    ----------
    series:
        Training-period column (may contain ``NaN``).
    strategy:
        ``"median"``, ``"mean"`` or ``"constant"`` (0.0).

    Returns
    -------
    float
        The fill value; ``0.0`` when the column is entirely missing.
    """
    if not series.notna().any():
        return 0.0
    if strategy == "mean":
        return float(series.mean())
    if strategy == "constant":
        return 0.0
    if strategy != "median":  # pragma: no cover - guarded by PipelineConfig
        raise ValueError(f"unknown numeric impute strategy {strategy!r}")
    return float(series.median())


def _normalise_categories(series: pd.Series) -> pd.Series:
    """Coerce a column to clean string categories with a ``"Missing"`` label."""
    as_string = series.astype("string").str.strip()
    as_string = as_string.str.replace(r"\s+", " ", regex=True)
    return as_string.fillna(MISSING_CATEGORY)


def _frequency_encoding(
    train_series: pd.Series,
    valid_series: pd.Series,
    column: str,
) -> tuple[pd.Series, pd.Series, dict[str, float], list[str]]:
    """Map each category to its share of training rows."""
    train_cat = _normalise_categories(train_series)
    valid_cat = _normalise_categories(valid_series)
    counts = train_cat.value_counts(normalize=True)
    mapping = {str(k): float(v) for k, v in counts.items()}
    train_encoded = train_cat.map(mapping).astype("float32")
    valid_encoded = valid_cat.map(mapping)
    valid_encoded = (
        valid_encoded.astype("float32").fillna(np.float32(0.0)).astype("float32")
    )
    return train_encoded, valid_encoded, mapping, sorted(mapping)


def _ordinal_encoding(
    train_series: pd.Series,
    valid_series: pd.Series,
    column: str,
) -> tuple[pd.Series, pd.Series, dict[str, float], list[str]]:
    """Map categories to dense integer codes by training frequency."""
    train_cat = _normalise_categories(train_series)
    valid_cat = _normalise_categories(valid_series)
    levels = train_cat.value_counts().index.tolist()
    mapping = {str(level): float(code) for code, level in enumerate(levels)}
    train_encoded = train_cat.map(mapping).astype("float32")
    valid_encoded = valid_cat.map(mapping).astype("float32")
    valid_encoded = valid_encoded.fillna(np.float32(RESERVED_CATEGORY_TOKEN))
    return train_encoded, valid_encoded, mapping, [str(x) for x in levels]


def _smoothed_target_map(
    categories: pd.Series, target: pd.Series, prior: float, smoothing: float
) -> dict[str, float]:
    """Bayesian-smoothed target statistic per category."""
    grouped = pd.DataFrame({"cat": categories, "y": target})
    stats = grouped.groupby("cat", observed=True)["y"].agg(["sum", "count"])
    encoded = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
    return {str(k): float(v) for k, v in encoded.items()}


def _target_encoding(
    train_series: pd.Series,
    valid_series: pd.Series,
    target: pd.Series,
    time_values: pd.Series,
    column: str,
    *,
    smoothing: float,
    n_folds: int,
) -> tuple[pd.Series, pd.Series, dict[str, float], list[str]]:
    """Out-of-fold, time-ordered target encoding.

    Training rows are encoded with an expanding window: the encoding for time
    block *k* is fitted on blocks ``< k`` only. The global map (all training
    rows) is then applied to validation data, which never influences fitting.
    """
    train_cat = _normalise_categories(train_series)
    valid_cat = _normalise_categories(valid_series)
    prior = float(target.mean())
    encoded = pd.Series(prior, index=train_cat.index, dtype="float64")

    folds = time_series_folds(
        len(train_cat),
        n_splits=max(2, n_folds),
        time_values=time_values.to_numpy(),
    )
    for train_idx, valid_idx in folds:
        mapping = _smoothed_target_map(
            train_cat.iloc[train_idx],
            target.iloc[train_idx],
            prior,
            smoothing,
        )
        encoded.iloc[valid_idx] = (
            train_cat.iloc[valid_idx].map(mapping).fillna(prior).to_numpy()
        )

    full_map = _smoothed_target_map(train_cat, target, prior, smoothing)
    valid_encoded = valid_cat.map(full_map).fillna(prior)
    return (
        encoded.astype("float32"),
        valid_encoded.astype("float32"),
        full_map,
        sorted(full_map),
    )


def encode_categoricals(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    target_column: str,
    target: pd.Series | np.ndarray | None = None,
    time_column: str | None = None,
    strategy: str = "frequency",
    smoothing: float = 20.0,
    target_folds: int = 5,
    explicit_categoricals: Sequence[str] | None = None,
    impute_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode categorical columns using statistics fitted on ``train`` only.

    Numeric columns are median-imputed (``'Missing'`` categories are mapped
    first, so categorical columns never see ``NaN``).

    Parameters
    ----------
    train, valid:
        Feature frames. Neither may contain ``target_column``; the labels are
        supplied separately via ``target`` so that no label can travel with the
        feature matrix.
    target:
        Aligned labels for ``train``. Required for ``strategy="target"``.
    time_column:
        Time column used to order the out-of-fold target encoding.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, dict]
        Encoded train/validation frames and a JSON-serialisable encoding state.
    """
    if strategy not in {"frequency", "ordinal", "target"}:
        raise ValueError(f"unknown encoding strategy {strategy!r}")
    for frame, name in ((train, "train"), (valid, "valid")):
        if target_column in frame.columns:
            raise ValueError(
                f"{name} frame must not contain the target column "
                f"{target_column!r}; it would leak into encodings"
            )

    train = train.reset_index(drop=True).copy()
    valid = valid.reset_index(drop=True).copy()
    target_series: pd.Series | None = None
    if target is not None:
        target_series = pd.Series(np.asarray(target).astype(int)).reset_index(drop=True)
        if len(target_series) != len(train):
            raise ValueError("target must be aligned with the training frame")
    elif strategy == "target":
        raise ValueError("strategy='target' requires the training labels")

    detected = detect_categorical_columns(train)
    requested = list(explicit_categoricals or [])
    categorical_columns = [
        c for c in dict.fromkeys(requested + detected) if c in train.columns
    ]
    numeric_columns = [c for c in train.columns if c not in categorical_columns]

    LOGGER.info(
        "Encoding %d categorical and %d numeric columns with %r strategy",
        len(categorical_columns),
        len(numeric_columns),
        strategy,
    )

    state = EncodingState(
        strategy=strategy,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )

    time_values = None
    if time_column is not None and time_column in train.columns:
        time_values = train[time_column]

    for column in categorical_columns:
        if strategy == "frequency":
            out = _frequency_encoding(train[column], valid[column], column)
            state.frequency_maps[column] = out[2]
        elif strategy == "ordinal":
            out = _ordinal_encoding(train[column], valid[column], column)
            state.category_levels[column] = out[3]
        else:
            if time_values is None:
                raise KeyError(
                    "target encoding requires the time column for out-of-fold splits"
                )
            out = _target_encoding(
                train[column],
                valid[column],
                target_series,
                time_values,
                column,
                smoothing=smoothing,
                n_folds=target_folds,
            )
            state.target_maps[column] = out[2]
            state.category_levels[column] = out[3]
        train[column] = out[0]
        valid[column] = out[1]

    for column in numeric_columns:
        # Force float64 before imputation: a downcast nullable Int* column would
        # otherwise reject a fractional median during fillna.
        numeric_train = pd.to_numeric(train[column], errors="coerce").astype("float64")
        numeric_valid = pd.to_numeric(valid[column], errors="coerce").astype("float64")
        fill = _numeric_fill_value(numeric_train, impute_strategy)
        state.numeric_fill_values[column] = fill
        train[column] = numeric_train.fillna(fill).astype("float32")
        valid[column] = numeric_valid.fillna(fill).astype("float32")

    if strategy == "target" and target_series is not None:
        state.target_prior = float(target_series.mean())

    return (
        train[numeric_columns + categorical_columns],
        valid[numeric_columns + categorical_columns],
        state.to_json(),
    )


def transform_with_encoding(df: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """Apply a persisted encoding state to new data (inference path).

    Missing categories map to the reserved sentinel for ordinal/target
    encodings and to frequency 0 for frequency encoding. Columns absent from
    ``state`` raise :class:`KeyError` to fail loudly rather than silently
    shifting the feature schema.
    """
    strategy = state["strategy"]
    columns: list[str] = list(state["numeric_columns"]) + list(
        state["categorical_columns"]
    )
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"input is missing trained feature columns: {missing}")

    output = pd.DataFrame(index=df.index)
    for column in state["numeric_columns"]:
        fill = float(state["numeric_fill_values"].get(column, 0.0))
        output[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .astype("float64")
            .fillna(fill)
            .astype("float32")
        )
    for column in state["categorical_columns"]:
        categories = _normalise_categories(df[column])
        if strategy == "frequency":
            mapping = state["frequency_maps"][column]
            output[column] = categories.map(mapping).fillna(0.0).astype("float32")
        elif strategy == "ordinal":
            mapping = state["category_levels"].get(column, [])
            codes = {str(level): float(i) for i, level in enumerate(mapping)}
            output[column] = (
                categories.map(codes).fillna(RESERVED_CATEGORY_TOKEN).astype("float32")
            )
        else:
            mapping = state["target_maps"][column]
            prior = float(
                state.get("target_prior")
                if state.get("target_prior") is not None
                else (np.mean(list(mapping.values())) if mapping else 0.0)
            )
            output[column] = categories.map(mapping).fillna(prior).astype("float32")
    return output[columns]


# --------------------------------------------------------------------------- #
# 6. Resampling (training folds only)
# --------------------------------------------------------------------------- #
def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    strategy: str = "none",
    ratio: float = 0.20,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Over-sample the fraud class with SMOTE / BorderlineSMOTE.

    Parameters
    ----------
    X, y:
        **Training fold only.** Passing validation/test data here violates the
        pipeline's leakage contract.
    strategy:
        ``"none"``, ``"smote"`` or ``"borderline"``.
    ratio:
        Desired ``fraud / legitimate`` ratio after sampling.
    k_neighbors:
        SMOTE neighbour count (auto-reduced for very small minority sets).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Resampled training matrix and labels (``random_state`` deterministic).

    Raises
    ------
    ValueError
        For an unknown strategy or a ratio outside ``(0, 1]``.
    """
    if strategy in {None, "none"}:
        return X, y
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must lie in (0, 1]")
    if strategy not in {"smote", "borderline"}:
        raise ValueError(f"unknown resampling strategy {strategy!r}")

    from imblearn.over_sampling import SMOTE, BorderlineSMOTE

    counts = y.value_counts()
    if len(counts) < 2:
        LOGGER.warning("Resampling skipped: only one class present")
        return X, y
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    if minority_count < 2:
        LOGGER.warning("Resampling skipped: minority class has < 2 samples")
        return X, y

    desired_minority = round(ratio * majority_count)
    if desired_minority <= minority_count:
        LOGGER.info("Resampling skipped: class ratio already >= %.2f", ratio)
        return X, y

    sampling_strategy = desired_minority / majority_count
    effective_k = int(min(k_neighbors, minority_count - 1))
    sampler_class = SMOTE if strategy == "smote" else BorderlineSMOTE
    sampler = sampler_class(
        sampling_strategy=sampling_strategy,
        k_neighbors=effective_k,
        random_state=random_state,
    )

    LOGGER.info(
        "Resampling with %s: minority %d -> %d (ratio %.2f, k=%d)",
        sampler_class.__name__,
        minority_count,
        desired_minority,
        ratio,
        effective_k,
    )
    started = time.perf_counter()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns).astype("float32")
    y_resampled = pd.Series(
        np.asarray(y_resampled).astype("int8"), name=y.name or "isFraud"
    )
    LOGGER.info(
        "Resampled %s -> %s in %.1fs",
        f"{len(X):,}",
        f"{len(X_resampled):,}",
        time.perf_counter() - started,
    )
    return X_resampled, y_resampled


# --------------------------------------------------------------------------- #
# 7. End-to-end dataset construction
# --------------------------------------------------------------------------- #
def select_features(
    X: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Subset ``X`` to ``feature_names`` preserving the given order."""
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise KeyError(f"features absent from frame: {missing[:10]} (… {len(missing)})")
    return X.loc[:, list(feature_names)]


def build_processed_dataset(
    config: PipelineConfig,
    *,
    raw_df: pd.DataFrame | None = None,
    save: bool | None = None,
) -> ProcessedData:
    """Run cleaning, feature engineering, encoding and the time split.

    Pipeline order (each step fits on training data only where relevant):

    1. Load/merge raw transaction + identity tables.
    2. Normalise placeholders and engineer features.
    3. Drop > ``missing_drop_threshold`` missing, constant and duplicate
       columns **using the training period only**.
    4. Chronological split into train (older) and validation (newer).
    5. Frequency/ordinal/target-encode categoricals (fitted on train).
    6. Median-impute numerics and downcast to ``float32``.
    7. Optionally persist ``X_train``/``X_valid``/``y_*`` and metadata.
    """
    started = time.perf_counter()
    config.ensure_directories()
    target = config.target_column
    time_column = config.time_columns[0]

    raw = raw_df if raw_df is not None else load_raw_data(config)
    if target not in raw.columns:
        raise KeyError(f"target column {target!r} missing from raw data")
    if time_column not in raw.columns:
        raise KeyError(f"time column {time_column!r} missing from raw data")

    frame = raw.replace([np.inf, -np.inf], np.nan).copy()
    cleaned = _clean_cognitive_columns(frame)
    LOGGER.info(
        "Normalised placeholder tokens in %d columns: %s", len(cleaned), cleaned
    )

    if config.add_engineered_features:
        engineered_features = engineer_features(frame, config)
    else:
        engineered_features = []

    # The split must exist before any statistic is fitted.
    train_period, valid_period = split_by_time(
        frame, time_column=time_column, validation_fraction=config.validation_fraction
    )
    protected_for_drop = {target, *config.id_columns, *config.time_columns}

    # Each helper drops from the *training* frame and reports what it removed;
    # the validation frame must then drop the identical column set so that both
    # periods keep an identical schema.
    train_period, audit_missing = drop_high_missing_columns(
        train_period, config.missing_drop_threshold, protected=protected_for_drop
    )
    dropped = [row["column"] for row in audit_missing]
    valid_period = valid_period.drop(
        columns=[c for c in dropped if c in valid_period.columns]
    )

    audit: dict[str, Any] = {
        "dropped_high_missing": audit_missing,
        "dropped_constant": [],
        "dropped_duplicates": [],
        "engineered_features": engineered_features,
    }
    if config.constant_drop:
        train_period, constants = _drop_constant_columns(
            train_period, protected_for_drop
        )
        columns = [row["column"] for row in constants]
        valid_period = valid_period.drop(
            columns=[c for c in columns if c in valid_period.columns]
        )
        audit["dropped_constant"] = constants

    if config.drop_duplicate_columns:
        train_period, duplicates = _drop_duplicate_columns(
            train_period, protected_for_drop
        )
        columns = [row["column"] for row in duplicates]
        valid_period = valid_period.drop(
            columns=[c for c in columns if c in valid_period.columns]
        )
        audit["dropped_duplicates"] = duplicates

    feature_columns = [
        c for c in train_period.columns if c not in set(config.id_columns) | {target}
    ]
    y_train = train_period[target].astype("int8")
    y_valid = valid_period[target].astype("int8")

    explicit_categoricals = [
        c for c in config.categorical_columns if c in feature_columns
    ]
    X_train, X_valid, encoding_state = encode_categoricals(
        train_period[feature_columns],
        valid_period[feature_columns],
        target_column=target,
        target=y_train,
        time_column=time_column,
        strategy=config.encoding_strategy,
        smoothing=config.target_encoding_smoothing,
        target_folds=config.target_encoding_folds,
        explicit_categoricals=explicit_categoricals,
        impute_strategy=config.numeric_impute_strategy,
    )

    if config.add_missing_indicators:
        for column in feature_columns:
            indicator = f"{column}__missing"
            X_train[indicator] = (
                train_period[column].isna().astype("float32").to_numpy()
            )
            X_valid[indicator] = (
                valid_period[column].isna().astype("float32").to_numpy()
            )

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")
    if X_train.isna().any().any() or X_valid.isna().any().any():
        raise RuntimeError("encoded matrices still contain missing values")

    metadata = {
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "target_column": target,
        "time_column": time_column,
        "time_split": {
            "train_max": float(train_period[time_column].max()),
            "valid_min": float(valid_period[time_column].min()),
            "valid_max": float(valid_period[time_column].max()),
        },
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_valid": float(y_valid.mean()),
        "encoding": encoding_state,
        "audit": audit,
        "config": config.to_dict(),
    }

    processed = ProcessedData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        time_train=train_period[time_column].reset_index(drop=True),
        time_valid=valid_period[time_column].reset_index(drop=True),
        feature_names=list(X_train.columns),
        metadata=metadata,
    )

    should_save = config.save_intermediate if save is None else save
    if should_save:
        save_processed_dataset(processed, config.processed_dir)

    LOGGER.info(
        "Preprocessing finished in %.1fs: %s",
        time.perf_counter() - started,
        processed.summary(),
    )
    return processed


def save_processed_dataset(processed: ProcessedData, directory: Path) -> None:
    """Persist matrices (parquet), labels and metadata JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    processed.X_train.to_parquet(directory / "X_train.parquet", index=False)
    processed.X_valid.to_parquet(directory / "X_valid.parquet", index=False)
    processed.y_train.to_frame().to_parquet(directory / "y_train.parquet", index=False)
    processed.y_valid.to_frame().to_parquet(directory / "y_valid.parquet", index=False)
    processed.time_train.rename("time").to_frame().to_parquet(
        directory / "time_train.parquet", index=False
    )
    processed.time_valid.rename("time").to_frame().to_parquet(
        directory / "time_valid.parquet", index=False
    )
    (directory / "metadata.json").write_text(
        json.dumps(processed.metadata, indent=2, default=str)
    )
    LOGGER.info("Saved processed dataset to %s", directory)


def load_processed_dataset(directory: Path) -> ProcessedData:
    """Reload a dataset written by :func:`save_processed_dataset`."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    X_train = pd.read_parquet(directory / "X_train.parquet")
    X_valid = pd.read_parquet(directory / "X_valid.parquet")
    feature_names = list(metadata["feature_names"])
    X_train = X_train.loc[:, feature_names]
    X_valid = X_valid.loc[:, feature_names]
    y_train = pd.read_parquet(directory / "y_train.parquet").iloc[:, 0].astype("int8")
    y_valid = pd.read_parquet(directory / "y_valid.parquet").iloc[:, 0].astype("int8")

    def _load_time(name: str) -> pd.Series:
        path = directory / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path).iloc[:, 0].reset_index(drop=True)
        LOGGER.warning(
            "Missing %s (dataset was written before time columns were persisted); "
            "returning an empty series",
            path.name,
        )
        return pd.Series(dtype="float64")

    return ProcessedData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        time_train=_load_time("time_train"),
        time_valid=_load_time("time_valid"),
        feature_names=feature_names,
        metadata=metadata,
    )
