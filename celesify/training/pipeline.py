"""
Training pipeline orchestration.

Coordinates all phases of model training:
1. Data loading and preparation
2. Baseline model training and evaluation
3. Hyperparameter search and tuned model training
4. Artifact export (metrics, model files, feature importance)

This module uses specialized submodules for each concern:
- data_handling: Loading and preparing datasets
- model_training: Training and hyperparameter search
- evaluation: Computing metrics and importance scores
- export: Writing artifacts to disk
- config: Managing training configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from celesify.core.constants import CLASS_LABEL_ORDER, RANDOM_STATE
from celesify.core.logging import log
from celesify.core.paths import resolve_training_paths
from celesify.training import data_handling, evaluation, export, model_training
from celesify.training.config import TrainingConfig

SERVICE = "training"


def _check_processed_data_exists(processed_dir: Path) -> bool:
    """Check if any valid processed data split exists."""
    candidates = [
        ("train_clean.parquet", "test_clean.parquet"),
        ("train.parquet", "test.parquet"),
    ]
    return any(
        (processed_dir / train).exists() and (processed_dir / test).exists()
        for train, test in candidates
    )


def _get_preprocessing_summary(preprocess_report: dict) -> dict:
    """Extract preprocessing summary for inclusion in metrics artifacts."""
    return evaluation._extract_preprocessing_summary(preprocess_report)


def _train_and_evaluate_baseline(
    x_clean_train: pd.DataFrame,
    y_clean_train: pd.Series,
    x_clean_test: pd.DataFrame,
    y_clean_test: pd.Series,
    clean_feature_columns: list[str],
    dataset_shapes: dict[str, list[int]],
    dataset_metadata: dict[str, str],
    preprocess_report: dict,
    config: TrainingConfig,
    class_mapping: dict[str, int],
) -> None:
    """
    Phase 1: Train baseline model and save metrics.

    The baseline uses default Random Forest hyperparameters and serves as
    a control to measure improvement from tuning.
    """
    log(SERVICE, "=== PHASE 1: Baseline Model ===")

    baseline_model = model_training.train_baseline_model(
        x_clean_train, y_clean_train, config.n_jobs
    )

    baseline_eval = evaluation.evaluate_model(baseline_model, x_clean_test, y_clean_test)

    baseline_metrics = evaluation.format_baseline_metrics(
        evaluation=baseline_eval,
        class_label_order=CLASS_LABEL_ORDER,
        class_mapping=class_mapping,
        feature_columns=clean_feature_columns,
        clean_train_shape=x_clean_train.shape,
        clean_test_shape=x_clean_test.shape,
        clean_variant=dataset_metadata["variant_clean"],
        clean_train_name=dataset_metadata["clean_train_file"],
        clean_test_name=dataset_metadata["clean_test_file"],
        preprocess_report=preprocess_report,
        random_state=config.random_state,
    )

    export.save_metrics_json(
        output_dir=Path(resolve_training_paths()[1]),
        filename="baseline_metrics.json",
        metrics=baseline_metrics,
    )


def _train_and_search_tuned(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    engineered_feature_columns: list[str],
    dataset_metadata: dict[str, str],
    preprocess_report: dict,
    config: TrainingConfig,
    class_mapping: dict[str, int],
    class_weight_space: list,
) -> None:
    """
    Phase 2: Hyperparameter search and tuned model training.

    Runs RandomizedSearchCV to find best hyperparameters, then trains
    final model on full training set with best params.
    """
    log(SERVICE, "=== PHASE 2: Hyperparameter Search (Tuned Model) ===")

    tuned_model, best_params, top_5_results, best_cv_score = model_training.run_randomized_search(
        x_train=x_train,
        y_train=y_train,
        n_iter=config.n_iter,
        cv_splits=config.cv_splits,
        n_jobs=config.n_jobs,
        class_weight_options=class_weight_space,
    )

    tuned_eval = evaluation.evaluate_model(tuned_model, x_test, y_test)

    # Format and save all metrics/params
    tuned_metrics = evaluation.format_tuned_metrics(
        evaluation=tuned_eval,
        best_params=best_params,
        best_cv_score=best_cv_score,
        top_5_results=top_5_results,
        class_label_order=CLASS_LABEL_ORDER,
        class_mapping=class_mapping,
        feature_columns=engineered_feature_columns,
        dataset_shapes={
            "train": list(x_train.shape),
            "test": list(x_test.shape),
        },
        dataset_variant=dataset_metadata["variant_engineered"],
        split_files={
            "train": dataset_metadata["engineered_train_file"],
            "test": dataset_metadata["engineered_test_file"],
        },
        preprocess_report=preprocess_report,
        n_iter_used=config.n_iter,
        cv_splits=config.cv_splits,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )

    best_params_export = evaluation.format_best_params(
        best_params=best_params,
        best_cv_score=best_cv_score,
        top_5_results=top_5_results,
        n_iter_used=config.n_iter,
    )

    top_trials_export = evaluation.format_top_trials(
        top_results=top_5_results,
        n_iter_used=config.n_iter,
    )

    models_dir = Path(resolve_training_paths()[1])

    export.save_metrics_json(models_dir, "tuned_metrics.json", tuned_metrics)
    export.save_metrics_json(models_dir, "best_params.json", best_params_export)
    export.save_metrics_json(models_dir, "top_trials.json", top_trials_export)

    # Phase 3: Feature importance
    log(SERVICE, "=== PHASE 3: Feature Importance ===")

    feature_importance_items = evaluation.extract_feature_importance(
        tuned_model, engineered_feature_columns
    )

    importance_export = evaluation.format_feature_importance(feature_importance_items)

    export.save_metrics_json(
        models_dir, "feature_importance.json", importance_export
    )

    # Phase 4: Export models
    log(SERVICE, "=== PHASE 4: Model Export ===")

    export.save_model_joblib(models_dir, "model.joblib", tuned_model)

    # Try ONNX export (non-fatal if it fails)
    onnx_success = export.export_onnx_model(
        models_dir, tuned_model, x_train.iloc[:1]
    )

    if not onnx_success:
        log(SERVICE, "Warning: ONNX export failed; joblib model is available.")


def run() -> None:
    """
    Execute the complete training pipeline.

    Pipeline stages:
    1. Setup: resolve paths, check data availability
    2. Load: read data, metadata, config
    3. Baseline: train default RF, evaluate, export metrics
    4. Search: hyperparameter tuning via RandomizedSearchCV
    5. Tuned: train final model, export metrics and models
    6. Importance: compute feature importance
    7. Export: save ONNX and joblib artifacts
    """
    processed_dir, models_dir = resolve_training_paths()
    models_dir.mkdir(parents=True, exist_ok=True)

    log(SERVICE, f"Processed dir: {processed_dir}")
    log(SERVICE, f"Models dir: {models_dir}")

    # === Setup ===
    if not _check_processed_data_exists(processed_dir):
        log(SERVICE, "Processed parquet files not found; writing placeholder artifacts.")
        required_files = [
            str(processed_dir / "train_clean.parquet"),
            str(processed_dir / "test_clean.parquet"),
            str(processed_dir / "train.parquet"),
            str(processed_dir / "test.parquet"),
        ]
        export.write_skip_placeholder(models_dir, required_files)
        return

    # === Load ===
    log(SERVICE, f"Using random_state={RANDOM_STATE}, class_labels={CLASS_LABEL_ORDER}")

    config = TrainingConfig()

    preprocess_report = data_handling.load_preprocessing_report(
        processed_dir / "preprocessing_report.json"
    )

    clean_train, clean_test, engineered_train, engineered_test, metadata = (
        data_handling.load_datasets(processed_dir)
    )
    data_handling.log_dataset_info(clean_train, clean_test, engineered_train, engineered_test)

    # Extract features and targets
    (x_clean_train, y_clean_train, x_clean_test, y_clean_test, clean_feature_columns) = (
        data_handling.extract_features_and_target(clean_train, clean_test)
    )
    (x_engineered_train, y_engineered_train, x_engineered_test, y_engineered_test, engineered_feature_columns) = (
        data_handling.extract_features_and_target(engineered_train, engineered_test)
    )

    # Apply subsampling if configured (for quick validation runs)
    if config.max_train_rows > 0:
        engineered_train, engineered_test = data_handling.subsample_train_data(
            engineered_train, engineered_test, config.max_train_rows
        )
        clean_train, clean_test = data_handling.subsample_train_data(
            clean_train, clean_test, config.max_train_rows
        )

        # Re-extract after subsampling
        (x_clean_train, y_clean_train, x_clean_test, y_clean_test, _) = (
            data_handling.extract_features_and_target(clean_train, clean_test)
        )
        (x_engineered_train, y_engineered_train, x_engineered_test, y_engineered_test, _) = (
            data_handling.extract_features_and_target(engineered_train, engineered_test)
        )

    # Determine class weighting strategy
    class_mapping = data_handling.get_class_mapping(preprocess_report)
    imbalance_flagged = data_handling.get_imbalance_recommendation(preprocess_report)
    class_weight_space = [None, "balanced"] if imbalance_flagged else [None]

    log(SERVICE, f"Class mapping: {class_mapping}")
    log(SERVICE, f"Imbalance flagged: {imbalance_flagged}, class_weight_space: {class_weight_space}")

    # === Baseline Training ===
    _train_and_evaluate_baseline(
        x_clean_train=x_clean_train,
        y_clean_train=y_clean_train,
        x_clean_test=x_clean_test,
        y_clean_test=y_clean_test,
        clean_feature_columns=clean_feature_columns,
        dataset_shapes={
            "train": list(x_clean_train.shape),
            "test": list(x_clean_test.shape),
        },
        dataset_metadata=metadata,
        preprocess_report=preprocess_report,
        config=config,
        class_mapping=class_mapping,
    )

    # === Hyperparameter Search & Tuned Training ===
    _train_and_search_tuned(
        x_train=x_engineered_train,
        y_train=y_engineered_train,
        x_test=x_engineered_test,
        y_test=y_engineered_test,
        engineered_feature_columns=engineered_feature_columns,
        dataset_metadata=metadata,
        preprocess_report=preprocess_report,
        config=config,
        class_mapping=class_mapping,
        class_weight_space=class_weight_space,
    )

    log(SERVICE, "=== Training Pipeline Complete ===")
