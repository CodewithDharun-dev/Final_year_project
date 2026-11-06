# Financial Fraud Detection Pipeline (IEEE-CIS)

A production-oriented, leakage-safe machine learning pipeline for card-not-present
transaction fraud detection, built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

The project is deliberately modular: every stage is a small, typed, independently
testable module, and `main.py` only orchestrates them. The emphasis is on the things
that actually decide whether a fraud model survives contact with production -
**honest validation, calibrated cost-based thresholds, and explanations an analyst can
defend** - rather than on squeezing the last 0.001 of AUC.

---

## Table of contents

1. [Problem framing](#problem-framing)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data layout](#data-layout)
5. [Usage](#usage)
6. [Module reference](#module-reference)
7. [Leakage-safety guarantees](#leakage-safety-guarantees)
8. [Benchmarks](#benchmarks)
9. [Testing](#testing)
10. [Design decisions worth knowing](#design-decisions-worth-knowing)
11. [Troubleshooting](#troubleshooting)

---

## Problem framing

| Property | Value | Consequence |
|---|---|---|
| Transactions (train) | 590,540 x 434 (after identity merge) | Needs downcasting to fit comfortably in RAM |
| Fraud rate | **3.50 %** | PR-AUC (average precision) is the target metric, not accuracy |
| Validation | newest 20 % of `TransactionDT` | Chronological, never random |
| Class costs | 500 : 5 (fraud : friction) | Cost-optimal threshold ≈ 0.03-0.07, *far* below 0.5 |

Two decisions follow directly and are enforced in code:

* **Accuracy is never a primary metric.** A model that never flags fraud is 96.5 %
  accurate. The pipeline reports ROC-AUC, PR-AUC, F1, balanced accuracy, Brier score
  and Matthews correlation, and always tunes the threshold against expected cost.
* **Time is a barrier, not a column.** Every split, fold, encoding and resampling
  operation is fitted on the past and evaluated on the future.

---

## Architecture

```
financial_fraud_detection/
│
├── data/
│   ├── raw/                     # train_transaction.csv, train_identity.csv  (gitignored)
│   └── processed/               # X_train/X_valid parquet, labels, metadata.json
│
├── notebooks/
│   ├── 01_eda_and_visualization.ipynb
│   └── 02_model_prototyping.ipynb
│
├── src/
│   ├── config.py                # frozen dataclass config + logging setup
│   ├── preprocessing.py         # merge, clean, downcast, encode, split, SMOTE
│   ├── feature_selection.py     # baseline XGBoost + TreeSHAP -> top-30 JSON
│   ├── models.py                # Optuna, XGB/LGBM/CatBoost, OOF stacking
│   ├── evaluation.py            # metrics, cost-sensitive threshold, figures
│   └── xai.py                   # SHAP, permutation importance, LIME, PDP
│
├── outputs/
│   ├── models/                  # .joblib artifacts (gitignored)
│   ├── figures/                 # ROC/PR/confusion/SHAP/LIME/PDP plots
│   ├── reports/                 # run manifest, metrics JSON, CV scores, logs
│   └── selected_features.json   # the SHAP top-30 list
│
├── tests/test_pipeline.py       # 37 unit + integration tests
├── main.py                      # CLI orchestration
├── requirements.txt             # fully pinned, verified together
├── pyproject.toml               # ruff + pytest configuration
└── .gitignore
```

### Data flow

```
raw CSVs ──► load_raw_data ──► merge(identity) ──► reduce_mem_usage
                                                     │
                              engineer_features ─────┤
                                                     ▼
                                        split_by_time  (train / validation)
                                                     │
                    ┌────────────────────────────────┼─────────────────────────────┐
                    ▼                                ▼                             ▼
        statistics fitted on TRAIN only      encode_categoricals          validation untouched
        (missing threshold, medians,         (frequency/target/ordinal)   until scoring
         category vocabularies)                     │
                                                    ▼
                                 run_feature_selection  (baseline XGB + TreeSHAP)
                                                    │
                                                    ▼
                                      top-30 feature matrix (parquet)
                                                    │
                             ┌──────────────────────┼──────────────────────┐
                             ▼                      ▼                      ▼
                      XGBoost/LightGBM          Optuna              StackingEnsemble
                       /CatBoost               (PR-AUC)          (time-ordered OOF)
                             └──────────────────────┼──────────────────────┘
                                                    ▼
                                    evaluate_predictions + cost threshold
                                                    │
                                                    ▼
                                        run_xai (SHAP / LIME / PDP)
```

---

## Installation

Python **3.12** is required (the pins were resolved and verified together on
3.12.7). A clean virtual environment is strongly recommended:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Development / notebook / testing tooling (pytest, ruff, nbconvert):
python -m pip install -r requirements-dev.txt

# `lime` is published as an sdist only, so install it without resolving deps:
python -m pip install --no-deps lime==0.2.0.1
```

> **Version note.** `scikit-learn` is held at **1.8.0** because
> `imbalanced-learn 0.14.2` imports the private symbol
> `sklearn.utils.validation._is_pandas_df`, which scikit-learn 1.9 removed.
> Bumping either package requires re-testing `from imblearn.over_sampling import SMOTE`.

Verify the environment:

```bash
python -c "import src.preprocessing, src.feature_selection, src.models, src.evaluation, src.xai; print('ok')"
pytest -q          # 37 tests, ~9 seconds
```

---

## Data layout

Place the Kaggle files in `data/raw/` (or point `--data_path` / the
`FFDD_DATA_PATH` environment variable at wherever they live):

```
data/raw/
├── train_transaction.csv     # 590,540 rows
└── train_identity.csv        # 144,233 rows, left-joined on TransactionID
```

File names are matched **case-insensitively**, so `Train_Transaction.csv` also works.
If `--data_path` does not directly contain the transaction file, the CLI probes
`data/raw`, `data`, `Dataset`, `../Dataset`, `..`, `../..` and
`untitled folder/Dataset` (and honours `FFDD_DATA_PATH`) before failing with a
message listing what it found. The notebooks use the same `FFDD_DATA_PATH`
variable.

The identity table is optional: without it the pipeline logs a warning and proceeds
with transaction features only.

---

## Usage

### Full pipeline

```bash
python main.py \
    --data_path data/raw \
    --tune \
    --run_shap_selection \
    --explain
```

### Common variants

```bash
# Fast smoke test (60k rows, no tuning, no stacking) - ~40 s
python main.py --data_path data/raw --sample_rows 60000 --no-stacking

# Baseline benchmark on the full dataset (no Optuna) - a few minutes
python main.py --data_path data/raw --explain

# Reuse a previous tuned configuration (only flags you pass override it)
python main.py --config outputs/reports/run_config.json
```

### CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | `data/raw` | Folder containing the CSVs (or a CSV path) |
| `--project_root` | `.` | Root owning `data/`, `outputs/`, `notebooks/` |
| `--config` | – | JSON config to load; only *explicitly passed* CLI flags override it |
| `--tune` / `--no-tune` | `--no-tune` | Run the Optuna PR-AUC search |
| `--run_shap_selection` / `--no-run_shap_selection` | `--run_shap_selection` | TreeSHAP top-k selection |
| `--explain` / `--no-explain` | `--no-explain` | SHAP / LIME / PDP bundle |
| `--models` | all three | Subset of `xgboost lightgbm catboost` |
| `--top_n` | `30` | Features retained by SHAP |
| `--tuning_trials` / `--tuning_timeout` | `40` / `5400` | Optuna budget per model |
| `--validation_fraction` | `0.20` | Chronologically last share held out |
| `--split_method` | `time` | `time` = hold-out only; `timeseries_cv` also runs expanding-window CV |
| `--missing_threshold` | `0.80` | Drop columns above this missing fraction |
| `--encoding` | `frequency` | `frequency`, `ordinal` or `target` |
| `--resample` / `--resample_ratio` | `none` / `0.20` | `smote` / `borderline`, training folds only |
| `--stacking` / `--no-stacking` | `--stacking` | Time-ordered OOF ensemble |
| `--fraud_cost` / `--friction_cost` | `500` / `5` | Cost model for threshold selection |
| `--target_recall` | – | Optional recall floor for threshold search |
| `--sample_rows` | – | Subsample raw rows (smoke testing only) |
| `--reuse_processed` | off | Load `data/processed` instead of rebuilding |
| `--random_state` / `--n_jobs` / `--log_level` | `42` / `-1` / `INFO` | Runtime knobs |

### Programmatic use

```python
from src.config import PipelineConfig
from src.preprocessing import build_processed_dataset
from src.feature_selection import run_feature_selection
from src.models import build_model, fit_estimator, chronological_holdout
from src.evaluation import evaluate_predictions

config = PipelineConfig(data_path="data/raw", project_root=".")
processed = build_processed_dataset(config)

selection = run_feature_selection(
    processed.X_train,
    processed.y_train,
    X_valid=processed.X_valid,
    y_valid=processed.y_valid,
    top_n=30,
    output_dir="outputs",
)

X_train = processed.X_train[selection.selected_features]
y_train = processed.y_train
X_fit, y_fit, X_stop, y_stop = chronological_holdout(X_train, y_train, fraction=0.10)

model = build_model("xgboost", y_train=y_fit)
fit_estimator(model, X_fit, y_fit, X_stop=X_stop, y_stop=y_stop)

report = evaluate_predictions(
    processed.y_valid,
    model.predict_proba(processed.X_valid[selection.selected_features])[:, 1],
    model_name="xgboost",
    figures_dir="outputs/figures",
)
print(report.summary_line())
```

### Artifacts produced

| Path | Contents |
|---|---|
| `outputs/models/*.joblib` | Fitted XGBoost, LightGBM, CatBoost and stacking artifacts |
| `outputs/selected_features.json` | SHAP top-30 feature list |
| `outputs/shap_feature_ranking.csv` | Full feature ranking with mean \|SHAP\| |
| `outputs/figures/roc_*.png` | ROC curves per model |
| `outputs/figures/precision_recall_*.png` | PR curves with the operating point marked |
| `outputs/figures/confusion_matrix_*.png` | Confusion matrices at the cost-optimal threshold |
| `outputs/figures/threshold_cost_*.png` | Expected-cost curve versus threshold |
| `outputs/figures/shap_summary_beeswarm.png` | Global SHAP beeswarm |
| `outputs/figures/permutation_importance.png` | Permutation importance ranking |
| `outputs/figures/lime_*.html` / `.png` | Local explanations for a FP/FN/TP case |
| `outputs/figures/pdp_*.png` | Partial dependence for amount/C/V/card features |
| `outputs/reports/evaluation_report.json` | All metrics, both thresholds |
| `outputs/reports/leaderboard.csv` | Model comparison table |
| `outputs/reports/run_manifest.json` | Stage timings, peak RSS, artifacts, platform |
| `outputs/reports/best_params.json` | Optuna best hyper-parameters |
| `outputs/reports/cv_scores.json` | Per-fold PR-AUC / ROC-AUC |
| `outputs/reports/pipeline_*.log` | Full run log |

---

## Module reference

### `src/config.py`

`PipelineConfig` is a **frozen dataclass**: one typed object holds every knob, so a
run is reproducible from a single JSON file (`--config`). Invalid values fail at
construction rather than hours into a run. `configure_logging` installs a timestamped
console + file handler and silences noisy third-party loggers.

### `src/preprocessing.py`

* `reduce_mem_usage(df)` - downcasts `float64 → float32`, integer-like floats →
  nullable `Int8/16/32`, integers → `int8/16/32`, and low-cardinality strings →
  `category`. On the full dataset this cut memory **68.9 % (1,984 MB → 617 MB)**.
* `load_raw_data(config)` - case-insensitive file discovery, `id-01 → id_01`
  normalisation, validated `one_to_one` merge.
* `engineer_features(df)` - `TransactionDT_log`, day/dow/hour (+ sin/cos), amount
  `log1p`, trailing cents, payer/payee email match, browser and screen-resolution
  tokens.
* `drop_high_missing_columns` / `_drop_constant_columns` / `_drop_duplicate_columns` -
  each returns an audit trail that is persisted in `metadata.json`.
* `split_by_time(df, ...)` - chronological hold-out, both frames re-indexed from 0.
* `time_series_folds(n, ...)` - expanding-window folds that guarantee
  `train_idx.max() < valid_idx.min()`.
* `encode_categoricals(...)` - frequency / ordinal / **out-of-fold target** encoding,
  fitted on training data only. Missing categories become the explicit `"Missing"`
  level; unseen validation categories map to a sentinel.
* `transform_with_encoding(df, state)` - applies a persisted encoding at inference.
* `apply_resampling(X, y, ...)` - SMOTE / BorderlineSMOTE with a controllable ratio,
  **training data only**.

### `src/feature_selection.py`

Fits a deliberately light XGBoost (250 trees) on the training period, computes exact
**TreeSHAP** contributions for a 20k-row background sample, ranks by mean \|SHAP\| and
exports the top-k list plus a full ranking CSV and two figures. The validation period
is only *read* to report the baseline score, never to pick features.

### `src/models.py`

* `build_xgboost` / `build_lightgbm` / `build_catboost` - imbalance-aware builders
  (`scale_pos_weight`, translated to `class_weights` for CatBoost).
* `fit_estimator(...)` - normalises early stopping across the three libraries
  (XGBoost `eval_set`, LightGBM `eval_X/eval_y` + callbacks, CatBoost `eval_set`).
* `chronological_holdout(...)` - carves the early-stopping slice from the **end of the
  training period**, so the evaluation period is never used for stopping.
* `make_optuna_objective` / `tune_model(...)` - TPE search maximising validation
  **PR-AUC**, with a SQLite study so runs can resume.
* `cross_validate_time_series(...)` - expanding-window CV with resampling confined to
  each training fold.
* `StackingEnsemble` - XGBoost + LightGBM + CatBoost with a regularised
  `LogisticRegression` meta-learner trained on **time-ordered out-of-fold**
  probabilities, then base learners refit on the full training period.

### `src/evaluation.py`

`compute_metrics` returns ROC-AUC, PR-AUC, F1, balanced accuracy, Brier, precision,
recall, MCC, FPR@80 % recall and the confusion counts. `optimise_threshold` performs a
vectorised grid search minimising

```
Cost(threshold) = False_Negatives × Fraud_Cost + False_Positives × Friction_Cost
```

with an optional recall floor. Plot helpers write ROC, PR, confusion-matrix and
cost-versus-threshold figures.

### `src/xai.py`

`shap_global_explanations` (beeswarm + importance bar), `permutation_importance_plot`
(PR-AUC drop when a feature is shuffled), `lime_explanations` (HTML + PNG + weights CSV
for a chosen false negative / true positive / false positive), and
`partial_dependence_plots` (model-agnostic 1-D PDP for `TransactionAmt`, `C13`, `V12`,
`card1`, …). `run_xai` orchestrates all of it and is failure-tolerant per step.

---

## Leakage-safety guarantees

These are the invariants the pipeline is built around. Each is covered by a test.

| Risk | Mitigation | Test |
|---|---|---|
| Random split leaks future rows | `split_by_time` orders by `TransactionDT` | `test_split_by_time_is_chronological` |
| CV folds trained on the future | `time_series_folds` asserts `train.max < valid.min` | `test_time_series_folds_have_no_lookahead` |
| Encoding sees validation labels | Encoders fitted on train rows only; target encoding is out-of-fold over time blocks | `test_encode_categoricals_marks_missing_and_handles_unseen` |
| SMOTE synthesises from validation/test | `apply_resampling` is only ever called on training folds | `test_resampling_increases_minority_only` |
| Early stopping peeks at the evaluation set | Slice taken from the end of the *training* period (`chronological_holdout`) | `test_chronological_holdout_splits_from_the_end` |
| Stacking meta-learner sees the future | Meta-features built from expanding-window OOF predictions | `test_stacking_ensemble_uses_oof_meta_features` |
| Feature selection uses validation labels | SHAP ranking fitted and computed on training rows only | (covered by `run_feature_selection` design) |
| Target/ID columns reach the model | `build_processed_dataset` raises if they are present | `test_build_processed_dataset_end_to_end` |

> **Why not `sklearn.ensemble.StackingClassifier` with `TimeSeriesSplit`?**
> Its internal `cross_val_predict` requires a *partition* of the data and raises
> `ValueError: cross_val_predict only works for partitions` for `TimeSeriesSplit`
> (the earliest samples are never in a test fold). Falling back to random K-fold would
> have leaked the future into the meta-learner, so `StackingEnsemble` implements the
> OOF loop explicitly. A `backend="sklearn"` option exists for benchmarking only and
> logs a leakage warning.

---

## Benchmarks

Measured on the **full training set** with the CLI defaults (no Optuna), using
the SHAP top-30 features. Validation = newest 20 % of `TransactionDT`
(118,108 rows, 3.44 % fraud), training = 472,432 rows.

| Model | ROC-AUC | PR-AUC | F1 @ opt. | Balanced acc. | Brier | Threshold | Recall | Precision | Expected cost |
|---|---|---|---|---|---|---|---|---|---|
| **Stacking (XGB+LGBM+CatBoost)** | **0.9031** | **0.4949** | **0.1634** | **0.7907** | **0.0234** | 0.014 | 0.9107 | **0.0897** | 369,205 |
| LightGBM | 0.8994 | 0.4932 | 0.1413 | 0.7653 | 0.0315 | 0.040 | **0.9318** | 0.0764 | **367,310** |
| CatBoost | 0.8938 | 0.4828 | 0.1377 | 0.7565 | 0.0517 | 0.098 | 0.9213 | 0.0744 | 392,830 |
| XGBoost | 0.8922 | 0.4829 | 0.1385 | 0.7574 | 0.0376 | 0.072 | 0.9200 | 0.0749 | 393,540 |

**How to read this**

* **Stacking wins on discrimination and calibration** (best ROC-AUC, PR-AUC, F1,
  balanced accuracy and Brier), and its OOF base scores (LightGBM 0.507,
  CatBoost 0.493, XGBoost 0.484) show the meta-learner is combining genuinely
  different signals.
* **LightGBM has the lowest expected cost** (367,310 vs 369,205) - it trades a
  little PR-AUC for higher recall at a lower threshold. For a cost-driven
  deployment, LightGBM is the pragmatic single-model choice.
* No-skill PR-AUC (predicting the base rate) is **0.034**; every model is ~14x
  better.
* At a 100:1 cost ratio the optimal thresholds are **low** (0.014-0.098), giving
  ~91-93 % recall at 7-9 % precision. That is the correct operating point when a
  missed fraud costs 100x a false decline.
* The single baseline XGBoost used *inside* feature selection, trained on all
  **416** features, reaches ROC-AUC 0.9052 / PR-AUC **0.5312** - i.e. the 30-feature
  models give up ~0.04 PR-AUC for a 14x smaller feature set. If accuracy matters more
  than scoring latency, use `--top_n 100` or `--no-run_shap_selection`.

**Runtime & memory** (8-core Apple arm64, 8 GB RAM, Python 3.12.7):

| Stage | Seconds |
|---|---|
| 1. Preprocess (merge, clean, encode, split) | 24.0 |
| 2. SHAP feature selection (416 -> 30) | 23.2 |
| 3. Train XGBoost + LightGBM + CatBoost | 114.3 |
| 4. Stacking (5 time-ordered OOF folds) | 276.8 |
| 5. Explainability (SHAP, permutation, LIME, PDP) | 13.4 |
| **Total** | **451.7** |

`reduce_mem_usage` cut the raw frame from **1,984 MB to 617 MB (-68.9 %)**; peak
process RSS was **3.3 GB** including the stacking ensemble.

### Bugs the benchmarks caught

Three defects were only visible once the full pipeline ran. They are documented
because they are the kind of silent failure that makes a fraud model look fine in
a notebook and disappoint in production.

1. **LightGBM was silently a 2-tree model.** With `scale_pos_weight` ~27 the
   weighted `binary_logloss` on the hold-out rises after two iterations, so early
   stopping fired immediately (PR-AUC 0.283). Switching the early-stopping metric
   to `average_precision` - aligned with the tuning objective - recovered
   **0.483**, a +0.20 PR-AUC difference.
2. **Nine `M*` fraud flags were being zeroed out.** `reduce_mem_usage` converts
   low-cardinality strings to `category` dtype, and the categorical detector
   skipped `category`. Those columns were then treated as numeric, coerced to
   `NaN` and median-filled with 0.0 - destroying the `M1`-`M9` mismatch/identity
   signals. Fixing the detector raised the all-feature baseline from PR-AUC 0.513
   to **0.531** and is guarded by
   `test_category_dtype_is_encoded_not_zeroed`.
3. **The stacking meta-learner was badly calibrated** because
   `class_weight="balanced"` multiplied its probabilities by the 28:1 imbalance
   ratio (Brier 0.115). Dropping the class weight brings Brier to **0.023**, and
   the cost-optimal threshold then lands at 0.014 - close to the theoretical
   5/(500+5) = 0.0099 implied by the cost ratio, which is good evidence the
   probabilities are now nearly calibrated.

### Resampling note

SMOTE / BorderlineSMOTE are implemented and tested, but on this dataset they **did
not help**: a 40k-row comparison gave PR-AUC 0.30 with BorderlineSMOTE (ratio 0.3)
versus 0.47 without it, because the boosters already use class weights and SMOTE
additionally shifts the score distribution. Use `--resample smote` only if you
re-tune the threshold (the pipeline does this automatically) and validate on your
own hold-out.

---

## Testing

```bash
pytest -q                    # 37 tests, ~9 s
pytest --cov=src -q          # with coverage (install pytest-cov)
ruff check . && ruff format --check .
```

Coverage includes memory downcasting, chronological splitting, fold construction,
encoding (including unseen categories and target-encoding guards), SMOTE behaviour,
metric correctness, threshold optimality under different cost ratios, model
persistence, stacking shape/finiteness, config validation, and an end-to-end
`build_processed_dataset` run on a synthetic IEEE-CIS-shaped frame.

Notebooks are validated with `nbmake`:

```bash
pytest --nbmake notebooks/ -q
```

---

## Design decisions worth knowing

**Why PR-AUC for tuning?** With 3.5 % positives, ROC-AUC stays high even for models
that are useless in the alert band. Average precision directly rewards ranking fraud
above genuine traffic, which is what an alert queue needs.

**Why a 500:5 cost ratio?** A missed fraud typically costs the transaction plus
chargeback fees and reputational damage (order 500); a false positive costs support
time, a declined customer and some churn (order 5). Under calibration the implied
threshold is `5/(500+5) ≈ 0.01`; the empirically optimised threshold lands in the
0.03-0.07 range because scores are not perfectly calibrated. Both the default (0.5)
and the cost-optimal threshold are reported so the trade-off is explicit.

**Why frequency encoding by default?** It is leak-free by construction and stable under
distribution shift. Target encoding is available (`--encoding target`) and is computed
out-of-fold over expanding time blocks, but it is more sensitive to the train/test
period shift in this dataset.

**Why keep `TransactionDT`?** It is a continuous time index, and time genuinely carries
signal (fraud rate drifts). It is monotonically transformed (`TransactionDT_log`,
day/dow/hour) rather than treated as a categorical ID.

**Why `float32` everywhere?** Trees are insensitive to the precision loss, and halving
the matrix size keeps the full 590k x 418 matrix well inside 8 GB of RAM while
allowing the stacking ensemble to fit.

**Why is the validation set never resampled?** Resampling the evaluation set changes
the base rate and therefore the meaning of PR-AUC, precision and the cost-optimal
threshold. Only training folds are resampled.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: ... train_transaction.csv` | Point `--data_path` at the folder containing the CSVs (the error lists what it found) |
| `ImportError: cannot import name '_is_pandas_df'` | scikit-learn was upgraded past 1.8.0; reinstall `scikit-learn==1.8.0` |
| `ModuleNotFoundError: lime` | `pip install --no-deps lime==0.2.0.1` |
| Memory pressure during stacking | Use `--no-stacking`, or `--models xgboost lightgbm` |
| Optuna runs too long | Lower `--tuning_trials` or set `--tuning_timeout` |
| Very low precision at the threshold | The 100:1 cost ratio prioritises recall by design; raise `--friction_cost` or set `--target_recall` |
| `cross_val_predict only works for partitions` | You are using sklearn's stacker with `TimeSeriesSplit`; use `StackingEnsemble` instead |

---

## License & data

The IEEE-CIS dataset is provided under the Kaggle competition rules and is **not**
included in this repository (see `.gitignore`). Code in this repository is released for
educational and research use.
