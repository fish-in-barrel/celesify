from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from celesify.core.constants import CLASS_LABEL_ORDER, RANDOM_STATE
from celesify.core.logging import log
from celesify.core.paths import resolve_training_paths
from celesify.training.evaluation import evaluate_model
from celesify.training.export import export_joblib_model, export_onnx_optimized, extract_feature_importance
from celesify.training.reporting import (
    DatasetInfo,
    SearchMetadata,
    SearchResults,
    build_baseline_metrics_report,
    build_best_params_report,
    build_preprocessing_summary,
    build_top_trials_report,
    build_tuned_metrics_report,
    write_baseline_artifacts,
    write_placeholder_artifacts,
    write_tuned_artifacts,
)
from celesify.training.training import run_randomized_search, train_baseline_model
from celesify.training.utils import (
    apply_max_train_rows,
    get_class_mapping,
    get_imbalance_recommendation,
    get_int_env,
    load_split_variant,
)

SERVICE = "training"


def run() -> None:
    processed_dir, models_dir = resolve_training_paths()
    models_dir.mkdir(parents=True, exist_ok=True)

    clean_candidates = [
        ("train_clean.parquet", "test_clean.parquet", "cleaned"),
        ("train.parquet", "test.parquet", "engineered"),
    ]
    engineered_candidates = [
        ("train.parquet", "test.parquet", "engineered"),
        ("train_clean.parquet", "test_clean.parquet", "cleaned"),
    ]
    report_file = processed_dir / "preprocessing_report.json"

    log(SERVICE, f"Processed dir: {processed_dir}")
    log(SERVICE, f"Models dir: {models_dir}")

    if not any((processed_dir / train).exists() and (processed_dir / test).exists() for train, test, _ in clean_candidates):
        log(SERVICE, "Processed parquet files not found; writing scaffold placeholder artifacts.")
        write_placeholder_artifacts(models_dir)
        return

    log(SERVICE, f"Using random_state={RANDOM_STATE} and class labels={CLASS_LABEL_ORDER}")

    preprocess_report = {}
    if report_file.exists():
        preprocess_report = json.loads(report_file.read_text(encoding="utf-8"))
        log(SERVICE, "Loaded preprocessing_report.json for metadata and imbalance guidance.")
    else:
        log(SERVICE, "preprocessing_report.json not found; continuing with safe defaults.")

    clean_train_df, clean_test_df, clean_variant, clean_train_name, clean_test_name = load_split_variant(processed_dir, clean_candidates)
    engineered_train_df, engineered_test_df, engineered_variant, engineered_train_name, engineered_test_name = load_split_variant(
        processed_dir,
        engineered_candidates,
    )

    if "class" not in clean_train_df.columns or "class" not in clean_test_df.columns:
        raise ValueError("Expected target column 'class' in both train and test parquet files.")
    if "class" not in engineered_train_df.columns or "class" not in engineered_test_df.columns:
        raise ValueError("Expected target column 'class' in both train and test parquet files.")

    clean_feature_columns: list[str] = [str(col) for col in clean_train_df.columns if col != "class"]
    engineered_feature_columns: list[str] = [str(col) for col in engineered_train_df.columns if col != "class"]
    if not clean_feature_columns:
        raise ValueError("No feature columns found after removing target column 'class'.")
    if not engineered_feature_columns:
        raise ValueError("No feature columns found in engineered training split after removing target column 'class'.")

    x_clean_train: pd.DataFrame = cast(pd.DataFrame, clean_train_df[clean_feature_columns])
    y_clean_train = clean_train_df["class"].astype(int)
    x_clean_test: pd.DataFrame = cast(pd.DataFrame, clean_test_df[clean_feature_columns])
    y_clean_test = clean_test_df["class"].astype(int)

    x_train: pd.DataFrame = cast(pd.DataFrame, engineered_train_df[engineered_feature_columns])
    y_train = engineered_train_df["class"].astype(int)
    x_test: pd.DataFrame = cast(pd.DataFrame, engineered_test_df[engineered_feature_columns])
    y_test = engineered_test_df["class"].astype(int)

    log(
        SERVICE,
        "Loaded datasets: "
        f"clean_train={x_clean_train.shape}, clean_test={x_clean_test.shape}, "
        f"engineered_train={x_train.shape}, engineered_test={x_test.shape}",
    )

    class_mapping = get_class_mapping(preprocess_report)
    imbalance_flagged = get_imbalance_recommendation(preprocess_report)
    class_weight_space = [None, "balanced"] if imbalance_flagged else [None]

    n_iter_used = max(1, get_int_env("TRAINING_N_ITER", 20))
    cv_splits = max(2, get_int_env("TRAINING_CV_SPLITS", 5))
    n_jobs = get_int_env("TRAINING_N_JOBS", -1)
    max_train_rows = get_int_env("TRAINING_MAX_TRAIN_ROWS", 0)

    engineered_train_df, x_train, clean_train_df, x_clean_train = apply_max_train_rows(
        engineered_train_df,
        clean_train_df,
        engineered_feature_columns,
        clean_feature_columns,
        max_train_rows,
    )
    y_train = engineered_train_df["class"].astype(int)
    y_clean_train = clean_train_df["class"].astype(int)

    baseline_eval_model = train_baseline_model(x_clean_train, y_clean_train, n_jobs)
    baseline_eval = evaluate_model(baseline_eval_model, x_clean_test, y_clean_test)

    # Build and write baseline metrics report
    preprocessing_summary = build_preprocessing_summary(preprocess_report)
    baseline_dataset_info = DatasetInfo(
        train_file=clean_train_name,
        test_file=clean_test_name,
        variant=clean_variant,
        train_shape=x_clean_train.shape,
        test_shape=x_clean_test.shape,
        feature_columns=clean_feature_columns,
    )
    baseline_metrics_report = build_baseline_metrics_report(
        baseline_eval,
        baseline_dataset_info,
        preprocessing_summary,
        class_mapping,
    )
    write_baseline_artifacts(models_dir, baseline_metrics_report)
    log(
        SERVICE,
        "Starting RandomizedSearchCV for clean features "
        f"backend=sklearn_cpu, n_iter={n_iter_used}, cv={cv_splits}, n_jobs={n_jobs}, "
        f"class_weight_space={class_weight_space}",
    )
    clean_tuned_model, clean_best_params, clean_top_5_results, clean_best_score = run_randomized_search(
        x_clean_train,
        y_clean_train,
        n_iter_used,
        cv_splits,
        n_jobs,
        class_weight_space,
    )
    clean_tuned_eval = evaluate_model(clean_tuned_model, x_clean_test, y_clean_test)

    # Build and write clean-tuned metrics reports
    clean_tuned_dataset_info = DatasetInfo(
        train_file=clean_train_name,
        test_file=clean_test_name,
        variant=clean_variant,
        train_shape=x_clean_train.shape,
        test_shape=x_clean_test.shape,
        feature_columns=clean_feature_columns,
    )
    clean_search_metadata = SearchMetadata(
        n_iter=n_iter_used,
        cv_splits=cv_splits,
        n_jobs=n_jobs,
    )
    clean_search_results = SearchResults(
        best_params=clean_best_params,
        best_cv_score=clean_best_score,
        top_5_results=clean_top_5_results,
    )
    clean_tuned_metrics_report = build_tuned_metrics_report(
        clean_tuned_eval,
        clean_tuned_dataset_info,
        preprocessing_summary,
        clean_search_metadata,
        clean_search_results,
        class_mapping,
    )
    clean_best_params_report = build_best_params_report(
        clean_search_metadata,
        clean_search_results,
    )
    clean_top_trials_report = build_top_trials_report(
        clean_search_metadata,
        clean_search_results,
    )
    write_tuned_artifacts(
        models_dir,
        "clean_tuned",
        clean_tuned_metrics_report,
        clean_best_params_report,
        clean_top_trials_report,
    )

    log(
        SERVICE,
        "Starting RandomizedSearchCV for engineered features "
        f"backend=sklearn_cpu, n_iter={n_iter_used}, cv={cv_splits}, n_jobs={n_jobs}, "
        f"class_weight_space={class_weight_space}",
    )
    tuned_model, best_params, top_5_results, best_score = run_randomized_search(
        x_train,
        y_train,
        n_iter_used,
        cv_splits,
        n_jobs,
        class_weight_space,
    )
    tuned_eval = evaluate_model(tuned_model, x_test, y_test)

    # Build and write engineered-tuned metrics reports
    engineered_dataset_info = DatasetInfo(
        train_file=engineered_train_name,
        test_file=engineered_test_name,
        variant=engineered_variant,
        train_shape=x_train.shape,
        test_shape=x_test.shape,
        feature_columns=engineered_feature_columns,
    )
    engineered_search_metadata = SearchMetadata(
        n_iter=n_iter_used,
        cv_splits=cv_splits,
        n_jobs=n_jobs,
    )
    engineered_search_results = SearchResults(
        best_params=best_params,
        best_cv_score=best_score,
        top_5_results=top_5_results,
    )
    tuned_metrics_report = build_tuned_metrics_report(
        tuned_eval,
        engineered_dataset_info,
        preprocessing_summary,
        engineered_search_metadata,
        engineered_search_results,
        class_mapping,
        feature_engineering_payload=preprocess_report.get("feature_engineering", {}),
    )
    best_params_report = build_best_params_report(
        engineered_search_metadata,
        engineered_search_results,
    )
    top_trials_report = build_top_trials_report(
        engineered_search_metadata,
        engineered_search_results,
    )
    write_tuned_artifacts(
        models_dir,
        "",
        tuned_metrics_report,
        best_params_report,
        top_trials_report,
    )

    # Extract and export feature importance from engineered tuned model
    extract_feature_importance(tuned_model, engineered_feature_columns, models_dir / "feature_importance.json")

    # Export primary model to joblib
    export_joblib_model(tuned_model, models_dir / "model.joblib")

    # Export to ONNX with memory optimizations
    model_onnx_path = models_dir / "model.onnx"
    onnx_status_path = models_dir / "onnx_export_status.json"
    onnx_error_log = models_dir / "onnx_export_error.log"

    export_onnx_optimized(
        tuned_model,
        len(engineered_feature_columns),
        model_onnx_path,
        onnx_status_path,
        onnx_error_log,
    )

    log(SERVICE, "Phase 3 training pipeline completed successfully.")
