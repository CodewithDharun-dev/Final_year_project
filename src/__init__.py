"""Financial Fraud Detection - production-grade IEEE-CIS fraud pipeline.

This package implements a modular, leakage-safe, memory-efficient machine
learning pipeline for card-not-present transaction fraud detection:

* :mod:`src.preprocessing`   - cleaning, downcasting, encoding, SMOTE.
* :mod:`src.feature_selection` - baseline XGBoost + TreeSHAP top-k ranking.
* :mod:`src.models`          - Optuna tuning, XGBoost/LightGBM/CatBoost,
                               out-of-fold stacking ensemble.
* :mod:`src.evaluation`      - PR-AUC/ROC-AUC metrics, cost-sensitive
                               threshold optimisation, diagnostic figures.
* :mod:`src.xai`             - SHAP, LIME and PDP explainability artifacts.

The command-line entry point is :mod:`main`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
