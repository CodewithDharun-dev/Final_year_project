"""Central configuration for the financial fraud detection pipeline.

All tunable knobs live in :class:`PipelineConfig`, a frozen dataclass that is
constructed from CLI arguments (see :mod:`main`) or loaded from
``config.json``.  Keeping a single, typed configuration object prevents the
implicit global state that makes ML pipelines hard to reproduce and audit.

Design notes
------------
* The dataclass is frozen so that configuration cannot be mutated halfway
  through a run; use :func:`dataclasses.replace` to derive variants.
* Paths are resolved eagerly by :meth:`PipelineConfig.resolve_paths` so that
  every module receives absolute paths regardless of the working directory.
* Leakage-safe defaults are chosen for every option that could leak: the
  validation split is chronological and SMOTE is restricted to training folds.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DEFAULT_CATEGORICAL_COLUMNS",
    "DEFAULT_ID_COLUMNS",
    "DEFAULT_TIME_COLUMNS",
    "FEATURE_LABELS",
    "PipelineConfig",
    "configure_logging",
    "humanise_feature",
    "humanise_features",
]

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Column groups specific to the IEEE-CIS Fraud Detection schema.
# --------------------------------------------------------------------------- #
#: Identifier / target columns that must never reach a model.
DEFAULT_ID_COLUMNS: tuple[str, ...] = ("TransactionID",)

#: Target column.
DEFAULT_TARGET_COLUMN: str = "isFraud"

#: Raw time columns.  ``TransactionDT`` is *seconds since a reference epoch*
#: (not a calendar datetime) and is treated as a continuous temporal indicator.
DEFAULT_TIME_COLUMNS: tuple[str, ...] = ("TransactionDT",)

#: Genuinely categorical (string) columns.  The ``M*`` flags and other
#: low-cardinality string columns that ``reduce_mem_usage`` converts to
#: ``category`` dtype are auto-detected at encode time and frequency-encoded,
#: so they do not need to be listed here explicitly.
DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
)

#: Canonical (interchangeable) transaction / identity file names.  The loader
#: performs case-insensitive matching so ``Train_Transaction.csv`` also works.
DEFAULT_TRANSACTION_FILENAME = "train_transaction.csv"
DEFAULT_IDENTITY_FILENAME = "train_identity.csv"

# --------------------------------------------------------------------------- #
# Human-readable display labels.
#
# The IEEE-CIS dataset ships *anonymised* column names (Vesta deliberately never
# published what C1-C14, D1-D15 or V1-V339 mean), so plots that use the raw
# names are correct but unreadable. ``humanise_feature`` adds a short
# descriptive suffix for reports while the pipeline keeps the raw names as the
# canonical schema (feature lists, parquet columns, model inputs).
# --------------------------------------------------------------------------- #
FEATURE_LABELS: dict[str, str] = {
    # --- transaction level -------------------------------------------------
    "TransactionAmt": "transaction amount",
    "TransactionAmt_log": "log amount",
    "TransactionDT": "transaction time (s)",
    "TransactionDT_log": "log time",
    "transaction_day": "day index",
    "transaction_dow": "day of week",
    "transaction_hour": "hour of day",
    "transaction_hour_sin": "hour (sin)",
    "transaction_hour_cos": "hour (cos)",
    "amount_cent": "amount cents",
    "ProductCD": "product code",
    "email_match": "email match flag",
    "email_domain_prefix": "payer email domain",
    # --- card / address ----------------------------------------------------
    "card1": "card id",
    "card2": "card attribute 2",
    "card3": "card attribute 3",
    "card4": "card network",
    "card5": "card attribute 5",
    "card6": "card type (debit/credit)",
    "addr1": "billing region",
    "addr2": "billing country",
    "dist1": "distance 1",
    "dist2": "distance 2",
    "P_emaildomain": "payer email domain",
    "R_emaildomain": "recipient email domain",
    # --- identity / device -------------------------------------------------
    "DeviceType": "device type",
    "DeviceInfo": "device info",
}

#: Regex rules for the anonymised blocks (``V``/``C``/``D``/``M``/``id_``).
_FEATURE_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    (r"^V(\d+)$", "Vesta feature {n}"),
    (r"^C(\d+)$", "count feature {n}"),
    (r"^D(\d+)$", "timedelta feature {n}"),
    (r"^M(\d+)$", "match flag M{n}"),
    (r"^id[_-](\d+)(?:_|$)", "identity feature {n}"),
)


def humanise_feature(name: str) -> str:
    """Return a report-ready label for a raw feature name.

    Known columns get a hand-written description; the anonymised IEEE-CIS
    blocks (``V*``, ``C*``, ``D*``, ``M*``, ``id_*``) get a neutral category
    label because their true meaning was never published by the competition
    organisers. The raw name is always preserved as a prefix so the label can
    still be traced back to the data.

    Parameters
    ----------
    name:
        Raw column / feature name.

    Returns
    -------
    str
        ``"<raw name> (<description>)"``, or the raw name when unknown.
    """
    if name in FEATURE_LABELS:
        return f"{name} ({FEATURE_LABELS[name]})"
    for pattern, template in _FEATURE_PREFIX_RULES:
        match = re.match(pattern, name)
        if match:
            return f"{name} ({template.format(n=match.group(1))})"
    return name


def humanise_features(names: Iterable[str]) -> list[str]:
    """Vectorised :func:`humanise_feature` for a sequence of feature names."""
    return [humanise_feature(name) for name in names]


EncodingStrategy = Literal["frequency", "ordinal", "target"]
ImputeStrategy = Literal["median", "mean", "constant"]
SmoteStrategy = Literal["none", "smote", "borderline"]
Level = int


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable, fully-typed configuration for one pipeline run.

    Parameters
    ----------
    data_path:
        Directory containing the raw CSVs, or an explicit path to the
        transaction CSV (its parent directory is then used for identity data).
    project_root:
        Directory that owns ``data/``, ``outputs/``, ``notebooks/``.
    missing_drop_threshold:
        Columns whose training-period missing fraction exceeds this value are
        dropped (default ``0.80``, i.e. drop when > 80 % missing).
    split_method:
        ``"time"`` for a chronological hold-out, ``"timeseries_cv"`` to also
        produce expanding-window folds for tuning.
    validation_fraction:
        Fraction of rows (by time) reserved for validation in ``"time"`` mode.
    encoding_strategy:
        Categorical encoding used for string columns.
    target_encoding_smoothing:
        Smoothing constant for target encoding (higher = stronger shrinkage).
    target_encoding_folds:
        Inner time-ordered folds used for out-of-fold target encoding.
    numeric_impute_strategy:
        Imputation for numeric columns.
    resample_strategy:
        ``"none"``, ``"smote"`` or ``"borderline"`` (BorderlineSMOTE).
    resample_ratio:
        Target ``fraud : legitimate`` ratio after over-sampling, e.g. ``0.2``
        means the minority class grows to 20 % of the majority count.  Use
        ``1.0`` for fully balanced samples.
    resample_k_neighbors:
        ``k_neighbors`` passed to SMOTE.
    """

    # ---- paths -----------------------------------------------------------
    data_path: str = "data/raw"
    project_root: str = "."
    transaction_filename: str = DEFAULT_TRANSACTION_FILENAME
    identity_filename: str = DEFAULT_IDENTITY_FILENAME

    # ---- schema ----------------------------------------------------------
    target_column: str = DEFAULT_TARGET_COLUMN
    id_columns: tuple[str, ...] = DEFAULT_ID_COLUMNS
    time_columns: tuple[str, ...] = DEFAULT_TIME_COLUMNS
    categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS

    # ---- preprocessing ---------------------------------------------------
    missing_drop_threshold: float = 0.80
    constant_drop: bool = True
    drop_duplicate_columns: bool = True
    encoding_strategy: EncodingStrategy = "frequency"
    target_encoding_smoothing: float = 20.0
    target_encoding_folds: int = 5
    numeric_impute_strategy: ImputeStrategy = "median"
    add_missing_indicators: bool = False
    add_engineered_features: bool = True
    downcast: bool = True
    time_unit_seconds: int = 86_400

    # ---- split -----------------------------------------------------------
    split_method: Literal["time", "timeseries_cv"] = "time"
    validation_fraction: float = 0.20
    n_time_folds: int = 4

    # ---- resampling ------------------------------------------------------
    resample_strategy: SmoteStrategy = "none"
    resample_ratio: float = 0.20
    resample_k_neighbors: int = 5
    resample_random_state: int = 42

    # ---- feature selection ----------------------------------------------
    baseline_n_estimators: int = 250
    shap_sample_size: int = 20_000
    top_n_features: int = 30
    run_shap_selection: bool = True

    # ---- tuning ----------------------------------------------------------
    tune: bool = False
    tuning_trials: int = 40
    tuning_timeout_seconds: int | None = 5_400

    # ---- models ----------------------------------------------------------
    train_models: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")
    stacking: bool = True
    stacking_meta_c: float = 1.0
    stacking_n_splits: int = 5

    # ---- early stopping --------------------------------------------------
    early_stopping: bool = True
    early_stopping_rounds: int = 100
    early_stopping_fraction: float = 0.10

    # ---- evaluation ------------------------------------------------------
    fraud_cost: float = 500.0
    friction_cost: float = 5.0
    threshold_grid_size: int = 501
    target_recall: float | None = None

    # ---- explainability --------------------------------------------------
    explain: bool = True
    xai_sample_size: int = 5_000
    lime_sample_size: int = 500
    lime_num_samples: int = 3_000
    pdp_permutation_samples: int = 2_000

    # ---- reproducibility / runtime --------------------------------------
    random_state: int = 42
    n_jobs: int = -1
    log_level: str = "INFO"
    save_intermediate: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived directory helpers
    # ------------------------------------------------------------------ #
    @property
    def root(self) -> Path:
        """Absolute project root."""
        return Path(self.project_root).expanduser().resolve()

    @property
    def data_dir(self) -> Path:
        """Absolute path of the raw data directory."""
        return self._resolve_data_input()

    @property
    def raw_dir(self) -> Path:
        """``data/raw`` under the project root."""
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        """``data/processed`` under the project root."""
        return self.root / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        """``outputs`` under the project root."""
        return self.root / "outputs"

    @property
    def models_dir(self) -> Path:
        """``outputs/models`` under the project root."""
        return self.outputs_dir / "models"

    @property
    def figures_dir(self) -> Path:
        """``outputs/figures`` under the project root."""
        return self.outputs_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        """``outputs/reports`` under the project root."""
        return self.outputs_dir / "reports"

    @property
    def notebooks_dir(self) -> Path:
        """``notebooks`` under the project root."""
        return self.root / "notebooks"

    # ------------------------------------------------------------------ #
    # Validation and factory helpers
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        """Validate invariants that would otherwise fail deep inside a run."""
        if not 0.0 < self.missing_drop_threshold < 1.0:
            raise ValueError("missing_drop_threshold must lie in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.validation_fraction > 0.5:
            LOGGER.warning(
                "validation_fraction=%.2f leaves < 50%% of rows for training",
                self.validation_fraction,
            )
        if self.resample_ratio <= 0:
            raise ValueError("resample_ratio must be positive")
        if self.top_n_features <= 0:
            raise ValueError("top_n_features must be positive")
        if self.fraud_cost <= 0 or self.friction_cost <= 0:
            raise ValueError("fraud_cost and friction_cost must be positive")
        if self.split_method not in {"time", "timeseries_cv"}:
            raise ValueError(f"unknown split_method {self.split_method!r}")
        if self.encoding_strategy not in {"frequency", "ordinal", "target"}:
            raise ValueError(f"unknown encoding {self.encoding_strategy!r}")
        if self.resample_strategy not in {"none", "smote", "borderline"}:
            raise ValueError(f"unknown resample_strategy {self.resample_strategy!r}")
        if self.resample_k_neighbors < 1:
            raise ValueError("resample_k_neighbors must be >= 1")
        if not 0.0 <= self.early_stopping_fraction < 0.5:
            raise ValueError("early_stopping_fraction must lie in [0, 0.5)")

    def _resolve_data_input(self) -> Path:
        """Resolve ``data_path`` to a directory, accepting a CSV path too."""
        candidate = Path(self.data_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.root / candidate).resolve()
        if candidate.suffix.lower() == ".csv":
            return candidate.parent
        return candidate

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.figures_dir,
            self.reports_dir,
            self.notebooks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_paths(self) -> PipelineConfig:
        """Return a copy whose path fields are absolute."""
        return replace(
            self,
            data_path=str(self._resolve_data_input()),
            project_root=str(self.root),
        )

    def with_overrides(self, **overrides: Any) -> PipelineConfig:
        """Return a validated copy with the given fields replaced.

        ``None`` values are ignored, which lets CLI parsers pass optional
        arguments straight through.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in self.__dataclass_fields__.values()}
        if unknown:
            raise TypeError(f"unknown configuration fields: {sorted(unknown)}")
        return replace(self, **clean)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["extra"] = dict(self.extra)
        return payload

    def save(self, path: Path | str) -> Path:
        """Write the configuration to ``path`` as pretty JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination

    @classmethod
    def load(cls, path: Path | str) -> PipelineConfig:
        """Load a configuration previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        payload.pop("extra", None)
        for key in (
            "id_columns",
            "time_columns",
            "categorical_columns",
            "train_models",
        ):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)


def configure_logging(
    level: str | Level = "INFO",
    log_file: Path | str | None = None,
) -> None:
    """Configure root logging with a rich, timestamped format.

    Parameters
    ----------
    level:
        Logging level name (``"INFO"``) or numeric level.
    log_file:
        Optional path that additionally receives every log record.
    """
    numeric_level = (
        level if isinstance(level, int) else logging.getLevelName(str(level).upper())
    )
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)

    for handler in list(root.handlers):
        if getattr(handler, "_ffdd_handler", False):
            root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    console._ffdd_handler = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        file_handler._ffdd_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    for noisy in ("matplotlib", "PIL", "numba", "optuna", "lightgbm", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
