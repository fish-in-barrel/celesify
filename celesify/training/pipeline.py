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
from typing import Any, cast

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


def run() -> None:
    """
    Execute the complete training pipeline.

    Phases:
    1. Load data and configuration
    2. Train and evaluate baseline model
    3. Run hyperparameter search and train tuned model
    4. Extract feature importance
    5. Export all artifacts (metrics JSON, models, ONNX)
    """
    processed_dir, models_dir = resolve_training_paths()
    models_dir.mkdir(parents=True, exist_ok=True)

    log(SERVICE, f"Processed dir: {processed_dir}")
    log(SERVICE, f"Models dir: {models_dir}")

    # Check for input data
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

    if max_train_rows > 0 and len(engineered_train_df) > max_train_rows:
        class_counts = cast(pd.Series, engineered_train_df["class"].value_counts().sort_index())
        target_counts = cast(pd.Series, ((class_counts / len(engineered_train_df)) * max_train_rows).round().astype(int))
        target_counts[target_counts < 1] = 1

        while int(target_counts.sum()) > max_train_rows:
            reducible = target_counts[target_counts > 1]
            if reducible.empty:
                break
            reducible_index = reducible.index[0]
            target_counts.loc[reducible_index] = int(target_counts.loc[reducible_index]) - 1

        sampled_index_parts: list[int] = []
        for cls, count in target_counts.items():
            class_slice = engineered_train_df[engineered_train_df["class"] == cls]
            take_n = min(int(count), len(class_slice))
            sampled_index_parts.extend(class_slice.sample(n=take_n, random_state=RANDOM_STATE, replace=False).index.tolist())

        sampled_indices = pd.Index(sampled_index_parts)
        sampled_engineered = engineered_train_df.loc[sampled_indices].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        sampled_clean = clean_train_df.loc[sampled_indices].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        engineered_train_df = sampled_engineered
        clean_train_df = sampled_clean
        x_train = cast(pd.DataFrame, engineered_train_df[engineered_feature_columns])
        y_train = engineered_train_df["class"].astype(int)
        x_clean_train = cast(pd.DataFrame, clean_train_df[clean_feature_columns])
        y_clean_train = clean_train_df["class"].astype(int)
        log(SERVICE, f"Applied TRAINING_MAX_TRAIN_ROWS={max_train_rows}; sampled train rows={len(engineered_train_df)}")

    baseline_eval_model = SklearnRandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    log(SERVICE, "Training baseline RandomForestClassifier on cleaned features.")
    baseline_eval_model.fit(x_clean_train, y_clean_train)

    baseline_eval = evaluate_model(baseline_eval_model, x_clean_test, y_clean_test)
    baseline_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_variant": clean_variant,
        "split_files": {
            "train": clean_train_name,
            "test": clean_test_name,
        },
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "feature_columns": clean_feature_columns,
        "dataset_shapes": {
            "train": list(x_clean_train.shape),
            "test": list(x_clean_test.shape),
        },
        "n_features": len(clean_feature_columns),
        "preprocessing_summary": {
            "rows_removed_missing": preprocess_report.get("rows_removed_missing"),
            "rows_removed_malformed": preprocess_report.get("rows_removed_malformed"),
            "rows_removed_total": preprocess_report.get("rows_removed_total"),
            "feature_count_before": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_before"),
            "feature_count_after": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_after"),
        },
        **baseline_eval,
    }
    write_json(models_dir / "baseline_metrics.json", as_jsonable(baseline_payload))
    log(SERVICE, "Wrote baseline_metrics.json")
    def _write_tuned_artifacts(
        *,
        model_tag: str,
        output_prefix: str,
        model: SklearnRandomForestClassifier,
        x_eval: pd.DataFrame,
        y_eval: pd.Series,
        feature_columns: list[str],
        dataset_variant: str,
        train_file_name: str,
        test_file_name: str,
        feature_engineering_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tuned_eval = evaluate_model(model, x_eval, y_eval)
        tuned_payload = {
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "dataset_variant": dataset_variant,
            "split_files": {
                "train": train_file_name,
                "test": test_file_name,
            },
            "class_label_order": CLASS_LABEL_ORDER,
            "class_mapping": class_mapping,
            "feature_columns": feature_columns,
            "dataset_shapes": {
                "train": list(x_train.shape if feature_columns == engineered_feature_columns else x_clean_train.shape),
                "test": list(x_test.shape if feature_columns == engineered_feature_columns else x_clean_test.shape),
            },
            "feature_engineering": feature_engineering_payload or {},
            "preprocessing_summary": {
                "rows_removed_missing": preprocess_report.get("rows_removed_missing"),
                "rows_removed_malformed": preprocess_report.get("rows_removed_malformed"),
                "rows_removed_total": preprocess_report.get("rows_removed_total"),
                "feature_count_before": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_before"),
                "feature_count_after": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_after"),
            },
            "n_iter_used": n_iter_used,
            "cv_splits": cv_splits,
            "n_jobs": n_jobs,
            "search_backend": "sklearn_cpu",
            **tuned_eval,
        }
        best_params_payload = {
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_iter_used": n_iter_used,
            "scoring": "f1_macro",
            "best_cv_score": float(tuned_eval.get("best_cv_score", 0.0)),
            "search_backend": "sklearn_cpu",
            "best_params": tuned_eval.get("best_params", {}),
            "top_5_results": tuned_eval.get("top_5_results", []),
        }
        write_json(models_dir / f"{output_prefix}_metrics.json", as_jsonable(tuned_payload))
        write_json(models_dir / f"best_params_{output_prefix}.json", as_jsonable(best_params_payload))
        write_json(
            models_dir / f"top_trials_{output_prefix}.json",
            {
                "status": "completed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scoring": "f1_macro",
                "n_iter_used": n_iter_used,
                "top_5_results": tuned_eval.get("top_5_results", []),
            },
        )
        log(SERVICE, f"Wrote {output_prefix}_metrics.json")
        log(SERVICE, f"Wrote best_params_{output_prefix}.json")
        log(SERVICE, f"Wrote top_trials_{output_prefix}.json")

        joblib.dump(model, models_dir / f"{output_prefix}_model.joblib")
        log(SERVICE, f"Wrote {output_prefix}_model.joblib")
        return tuned_payload

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
    clean_tuned_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_variant": clean_variant,
        "split_files": {
            "train": clean_train_name,
            "test": clean_test_name,
        },
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "feature_columns": clean_feature_columns,
        "dataset_shapes": {
            "train": list(x_clean_train.shape),
            "test": list(x_clean_test.shape),
        },
        "preprocessing_summary": {
            "rows_removed_missing": preprocess_report.get("rows_removed_missing"),
            "rows_removed_malformed": preprocess_report.get("rows_removed_malformed"),
            "rows_removed_total": preprocess_report.get("rows_removed_total"),
            "feature_count_before": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_before"),
            "feature_count_after": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_after"),
        },
        "n_iter_used": n_iter_used,
        "cv_splits": cv_splits,
        "n_jobs": n_jobs,
        "search_backend": "sklearn_cpu",
        "best_params": clean_best_params,
        "best_cv_score": clean_best_score,
        "top_5_results": clean_top_5_results,
        **clean_tuned_eval,
    }
    clean_best_params_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_iter_used": n_iter_used,
        "scoring": "f1_macro",
        "best_cv_score": clean_best_score,
        "search_backend": "sklearn_cpu",
        "best_params": clean_best_params,
        "top_5_results": clean_top_5_results,
    }
    write_json(models_dir / "clean_tuned_metrics.json", as_jsonable(clean_tuned_payload))
    write_json(models_dir / "best_params_clean_tuned.json", as_jsonable(clean_best_params_payload))
    write_json(
        models_dir / "top_trials_clean_tuned.json",
        {
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scoring": "f1_macro",
            "n_iter_used": n_iter_used,
            "top_5_results": clean_top_5_results,
        },
    )
    log(SERVICE, "Wrote clean_tuned_metrics.json")
    log(SERVICE, "Wrote best_params_clean_tuned.json")
    log(SERVICE, "Wrote top_trials_clean_tuned.json")

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
    tuned_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_variant": engineered_variant,
        "split_files": {
            "train": engineered_train_name,
            "test": engineered_test_name,
        },
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "feature_columns": engineered_feature_columns,
        "dataset_shapes": {
            "train": list(x_train.shape),
            "test": list(x_test.shape),
        },
        "feature_engineering": preprocess_report.get("feature_engineering", {}),
        "preprocessing_summary": {
            "rows_removed_missing": preprocess_report.get("rows_removed_missing"),
            "rows_removed_malformed": preprocess_report.get("rows_removed_malformed"),
            "rows_removed_total": preprocess_report.get("rows_removed_total"),
            "feature_count_before": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_before"),
            "feature_count_after": (preprocess_report.get("dataset_comparison") or {}).get("feature_count_after"),
        },
        "n_iter_used": n_iter_used,
        "cv_splits": cv_splits,
        "n_jobs": n_jobs,
        "search_backend": "sklearn_cpu",
        "best_params": best_params,
        "best_cv_score": best_score,
        "top_5_results": top_5_results,
        **tuned_eval,
    }
    write_json(models_dir / "tuned_metrics.json", as_jsonable(tuned_payload))
    write_json(models_dir / "best_params.json", as_jsonable({
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_iter_used": n_iter_used,
        "scoring": "f1_macro",
        "best_cv_score": best_score,
        "search_backend": "sklearn_cpu",
        "best_params": best_params,
        "top_5_results": top_5_results,
    }))
    write_json(
        models_dir / "top_trials.json",
        {
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scoring": "f1_macro",
            "n_iter_used": n_iter_used,
            "top_5_results": top_5_results,
        },
    )
    log(SERVICE, "Wrote tuned_metrics.json")
    log(SERVICE, "Wrote best_params.json")
    log(SERVICE, "Wrote top_trials.json")

    tuned_importances = np.asarray(getattr(tuned_model, "feature_importances_", np.zeros(len(engineered_feature_columns))))
    feature_importance_items = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(engineered_feature_columns, tuned_importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    feature_importance_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "mdi",
        "feature_importance": feature_importance_items,
    }
    write_json(models_dir / "feature_importance.json", feature_importance_payload)
    log(SERVICE, "Wrote feature_importance.json")

    model_joblib_path = models_dir / "model.joblib"
    joblib.dump(tuned_model, model_joblib_path)
    log(SERVICE, "Wrote model.joblib")

    x_sample = x_train.iloc[:1].astype(np.float32)
    model_onnx_path = models_dir / "model.onnx"
    onnx_status_path = models_dir / "onnx_export_status.json"
    onnx_error_log = models_dir / "onnx_export_error.log"

    try:
        onnx_result = convert_sklearn(
            tuned_model,
            initial_types=[("input", FloatTensorType([None, x_sample.shape[1]]))],
        )
        onnx_model = onnx_result[0] if isinstance(onnx_result, tuple) else onnx_result
        model_onnx_path.write_bytes(onnx_model.SerializeToString())
        if onnx_error_log.exists():
            onnx_error_log.unlink()
        write_json(
            onnx_status_path,
            {
                "status": "completed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_onnx_path),
                "source_model_type": f"{tuned_model.__class__.__module__}.{tuned_model.__class__.__name__}",
                "onnx_export_model_type": f"{tuned_model.__class__.__module__}.{tuned_model.__class__.__name__}",
            },
        )
        log(SERVICE, "Wrote model.onnx")
    except Exception as exc:
        error_summary = str(exc).splitlines()[0][:500]
        onnx_error_log.write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            onnx_status_path,
            {
                "status": "failed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": error_summary,
                "error_log": str(onnx_error_log),
            },
        )
        log(SERVICE, f"ONNX export failed ({type(exc).__name__}); details written to {onnx_error_log}")
        raise RuntimeError(f"ONNX export failed; see {onnx_error_log}") from None

    log(SERVICE, "Phase 3 training pipeline completed successfully.")
