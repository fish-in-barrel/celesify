"""
Model training and hyperparameter search.

This module handles:
- Training baseline Random Forest models
- Hyperparameter search via RandomizedSearchCV
- Managing class weighting strategies
- Grid/search space definition
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from celesify.core.constants import RANDOM_STATE
from celesify.core.logging import log

SERVICE = "training"


def train_baseline_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_jobs: int,
) -> SklearnRandomForestClassifier:
    """
    Train baseline Random Forest with default hyperparameters.

    Used as a control to measure improvement from hyperparameter tuning.
    Defaults: n_estimators=100, no max_depth constraint.

    Args:
        x_train: Training features.
        y_train: Training target (class labels).
        n_jobs: Number of parallel jobs (-1 for all cores).

    Returns:
        Trained RandomForestClassifier instance.
    """
    log(SERVICE, "Training baseline RandomForestClassifier with default hyperparameters.")

    model = SklearnRandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)

    log(SERVICE, f"Baseline model trained on {len(x_train)} samples with {x_train.shape[1]} features.")
    return model


def create_search_space(class_weight_options: list[Any]) -> dict[str, list[Any]]:
    """
    Define hyperparameter search space for RandomizedSearchCV.

    Based on CLAUDE.md Phase 3 recommendations.

    Args:
        class_weight_options: List of class_weight values to try
                             (typically [None, "balanced"] or [None]).

    Returns:
        Dictionary of parameter distributions.
    """
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3],
        "class_weight": class_weight_options,
    }


def run_randomized_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv_splits: int,
    n_jobs: int,
    class_weight_options: list[Any],
) -> tuple[SklearnRandomForestClassifier, dict[str, Any], list[dict[str, Any]], float]:
    """
    Execute RandomizedSearchCV for hyperparameter tuning.

    Performs stratified cross-validation to find best hyperparameters.

    Args:
        x_train: Training features.
        y_train: Training target (class labels).
        n_iter: Number of parameter combinations to sample.
        cv_splits: Number of cross-validation folds.
        n_jobs: Number of parallel jobs (-1 for all cores).
        class_weight_options: Class weight strategies to try.

    Returns:
        Tuple of (best_model, best_params, top_5_results, best_cv_score).
    """
    log(
        SERVICE,
        f"Starting RandomizedSearchCV: n_iter={n_iter}, cv_splits={cv_splits}, "
        f"n_jobs={n_jobs}, class_weight_options={class_weight_options}",
    )

    param_space = create_search_space(class_weight_options)

    search = RandomizedSearchCV(
        estimator=SklearnRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs),
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )

    search.fit(x_train, y_train)

    # Extract top 5 results with timing information
    top_5_results = _extract_top_trials(search.cv_results_, n_top=5)

    best_params = cast(dict[str, Any], search.best_params_)
    best_score = float(search.best_score_)

    log(SERVICE, f"Best CV score (f1_macro): {best_score:.4f}")
    log(SERVICE, f"Best parameters: {best_params}")

    # Retrain on full training set with best params
    final_model = SklearnRandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        **best_params,
    )
    final_model.fit(x_train, y_train)

    return final_model, best_params, top_5_results, best_score


def _extract_top_trials(
    cv_results: dict[str, Any],
    n_top: int = 5,
) -> list[dict[str, Any]]:
    """
    Extract top N trials by CV score from RandomizedSearchCV results.

    Args:
        cv_results: cv_results_ from RandomizedSearchCV.
        n_top: Number of top trials to extract.

    Returns:
        List of trial dictionaries with params, scores, and timing.
    """
    ranks = np.asarray(cv_results.get("rank_test_score", []), dtype=float)
    mean_scores = np.asarray(cv_results.get("mean_test_score", []), dtype=float)
    std_scores = np.asarray(cv_results.get("std_test_score", []), dtype=float)
    mean_fit_times = np.asarray(cv_results.get("mean_fit_time", []), dtype=float)
    std_fit_times = np.asarray(cv_results.get("std_fit_time", []), dtype=float)
    mean_score_times = np.asarray(cv_results.get("mean_score_time", []), dtype=float)
    std_score_times = np.asarray(cv_results.get("std_score_time", []), dtype=float)
    params_list = cast(list[dict[str, Any]], cv_results.get("params", []))

    top_results: list[dict[str, Any]] = []

    if len(params_list) > 0 and len(mean_scores) == len(params_list):
        order = np.argsort(-mean_scores)  # Descending order (best first)
        for rank_position, idx in enumerate(order[:n_top], start=1):
            idx_int = int(idx)
            params = params_list[idx_int] if idx_int < len(params_list) else {}

            top_results.append(
                {
                    "rank": int(ranks[idx_int]) if len(ranks) > idx_int else rank_position,
                    "mean_test_score": float(mean_scores[idx_int]),
                    "std_test_score": float(std_scores[idx_int]) if len(std_scores) > idx_int else 0.0,
                    "mean_fit_time": float(mean_fit_times[idx_int]) if len(mean_fit_times) > idx_int else 0.0,
                    "std_fit_time": float(std_fit_times[idx_int]) if len(std_fit_times) > idx_int else 0.0,
                    "mean_score_time": float(mean_score_times[idx_int]) if len(mean_score_times) > idx_int else 0.0,
                    "std_score_time": float(std_score_times[idx_int]) if len(std_score_times) > idx_int else 0.0,
                    "params": _make_jsonable(params),
                }
            )

    return top_results


def _make_jsonable(obj: Any) -> Any:
    """Convert sklearn parameter dict to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_jsonable(item) for item in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
