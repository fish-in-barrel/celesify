"""Model training: baseline RF and RandomizedSearchCV hyperparameter tuning."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from celesify.core.constants import RANDOM_STATE
from celesify.core.json_utils import as_jsonable
from celesify.core.logging import log

SERVICE = "training"


def train_baseline_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_jobs: int,
) -> SklearnRandomForestClassifier:
    """
    Train a baseline RandomForestClassifier with sklearn defaults.

    Used to establish a performance ceiling on minimal feature engineering.

    Args:
        x_train: Training features
        y_train: Training target
        n_jobs: Number of jobs for parallel processing (-1 for all cores)

    Returns:
        Fitted RandomForestClassifier with defaults (n_estimators=100, no tuning)
    """
    log(SERVICE, "Training baseline RandomForestClassifier on cleaned features.")
    baseline_model = SklearnRandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    baseline_model.fit(x_train, y_train)
    return baseline_model


def run_randomized_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter_used: int,
    cv_splits: int,
    n_jobs: int,
    class_weight_space: list[Any],
) -> tuple[SklearnRandomForestClassifier, dict[str, Any], list[dict[str, Any]], float]:
    """
    Run RandomizedSearchCV over hyperparameter space for RF model.

    Search space:
    - n_estimators: [100, 200, 300, 500]
    - max_depth: [None, 10, 20, 30]
    - min_samples_split: [2, 5, 10]
    - max_features: ['sqrt', 'log2', 0.3]
    - class_weight: [None, 'balanced'] if imbalance detected, else [None]

    Scoring: f1_macro (accounts for class imbalance)
    CV: Stratified k-fold with shuffle

    Args:
        x_train: Training features
        y_train: Training target
        n_iter_used: Number of random parameter combinations to try
        cv_splits: Number of cross-validation folds
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
        class_weight_space: List of class_weight options to search (typically [None] or [None, 'balanced'])

    Returns:
        Tuple of (best_model, best_params, top_5_results, best_cv_score)
        - best_model: Refitted model on full training data with best params
        - best_params: Dictionary of best hyperparameters found
        - top_5_results: List of top 5 trial results with scores and times
        - best_cv_score: Best cross-validation f1_macro score
    """
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3],
        "class_weight": class_weight_space,
    }

    search = RandomizedSearchCV(
        estimator=SklearnRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs),
        param_distributions=param_distributions,
        n_iter=n_iter_used,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)

    cv_results = cast(dict[str, Any], search.cv_results_)
    ranks = np.asarray(cv_results.get("rank_test_score", []), dtype=float)
    mean_scores = np.asarray(cv_results.get("mean_test_score", []), dtype=float)
    std_scores = np.asarray(cv_results.get("std_test_score", []), dtype=float)
    mean_fit_times = np.asarray(cv_results.get("mean_fit_time", []), dtype=float)
    std_fit_times = np.asarray(cv_results.get("std_fit_time", []), dtype=float)
    mean_score_times = np.asarray(cv_results.get("mean_score_time", []), dtype=float)
    std_score_times = np.asarray(cv_results.get("std_score_time", []), dtype=float)
    params_list = cast(list[dict[str, Any]], cv_results.get("params", []))

    # Extract top 5 trials by mean CV score
    top_5_results: list[dict[str, Any]] = []
    if len(params_list) > 0 and len(mean_scores) == len(params_list):
        order = np.argsort(-mean_scores)
        for rank_position, idx in enumerate(order[:5], start=1):
            params = params_list[int(idx)] if int(idx) < len(params_list) else {}
            top_5_results.append(
                {
                    "rank": int(ranks[int(idx)]) if len(ranks) > int(idx) else rank_position,
                    "mean_test_score": float(mean_scores[int(idx)]),
                    "std_test_score": float(std_scores[int(idx)]) if len(std_scores) > int(idx) else 0.0,
                    "mean_fit_time": float(mean_fit_times[int(idx)]) if len(mean_fit_times) > int(idx) else 0.0,
                    "std_fit_time": float(std_fit_times[int(idx)]) if len(std_fit_times) > int(idx) else 0.0,
                    "mean_score_time": float(mean_score_times[int(idx)]) if len(mean_score_times) > int(idx) else 0.0,
                    "std_score_time": float(std_score_times[int(idx)]) if len(std_score_times) > int(idx) else 0.0,
                    "params": as_jsonable(params),
                }
            )

    best_params = cast(dict[str, Any], search.best_params_)
    best_score = float(search.best_score_)

    # Refit best model on full training data
    tuned_model = SklearnRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **best_params)
    tuned_model.fit(x_train, y_train)

    return tuned_model, best_params, top_5_results, best_score
