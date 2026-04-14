from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from celesify.core.constants import CLASS_ENCODING, CLASS_LABEL_ORDER, RANDOM_STATE
from celesify.core.json_utils import as_jsonable, write_json
from celesify.core.logging import log
from celesify.core.paths import resolve_training_paths

SERVICE = "training"


DEFAULT_CLASS_MAP = CLASS_ENCODING


def load_split_variant(
    processed_dir: Path,
    candidates: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    for train_name, test_name, variant_name in candidates:
        train_file = processed_dir / train_name
        test_file = processed_dir / test_name
        if train_file.exists() and test_file.exists():
            return (
                pd.read_parquet(train_file),
                pd.read_parquet(test_file),
                variant_name,
                train_file.name,
                test_file.name,
            )
    candidate_list = ", ".join(f"{train}/{test}" for train, test, _ in candidates)
    raise FileNotFoundError(f"No matching parquet split found. Checked: {candidate_list}")


def get_imbalance_recommendation(preprocess_report: dict) -> bool:
    recommendation = preprocess_report.get("imbalance_recommendation")
    if isinstance(recommendation, str) and "balanced" in recommendation.lower():
        return True

    nested_assessment = preprocess_report.get("imbalance_assessment")
    if isinstance(nested_assessment, dict):
        nested_recommendation = nested_assessment.get("recommendation")
        if isinstance(nested_recommendation, str) and "balanced" in nested_recommendation.lower():
            return True

    ratio = preprocess_report.get("majority_minority_ratio")
    if ratio is None:
        ratio = preprocess_report.get("imbalance_ratio")
    if ratio is None and isinstance(preprocess_report.get("class_balance"), dict):
        ratio = preprocess_report["class_balance"].get("majority_minority_ratio")
    if ratio is None and isinstance(nested_assessment, dict):
        ratio = nested_assessment.get("majority_to_minority_ratio")

    if ratio is not None:
        try:
            return float(ratio) > 2.0
        except (TypeError, ValueError):
            return False
    return False


def get_class_mapping(preprocess_report: dict) -> dict[str, int]:
    class_mapping = preprocess_report.get("class_mapping")
    if class_mapping is None:
        class_mapping = preprocess_report.get("target_encoding")
    if isinstance(class_mapping, dict):
        parsed = {}
        for name, encoded in class_mapping.items():
            try:
                parsed[str(name)] = int(encoded)
            except (TypeError, ValueError):
                continue
        if parsed:
            return parsed
    return DEFAULT_CLASS_MAP


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=CLASS_LABEL_ORDER)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = {}
    for label in CLASS_LABEL_ORDER:
        label_report = report.get(str(label), {}) if isinstance(report, dict) else {}
        if not isinstance(label_report, dict):
            continue
        per_class_metrics[str(label)] = {
            "precision": float(label_report.get("precision", 0.0)),
            "recall": float(label_report.get("recall", 0.0)),
            "f1_score": float(label_report.get("f1-score", 0.0)),
            "support": int(label_report.get("support", 0)),
        }

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log(SERVICE, f"Invalid {name}={raw!r}; using default {default}.")
        return default


def run_randomized_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter_used: int,
    cv_splits: int,
    n_jobs: int,
    class_weight_space: list[Any],
) -> tuple[SklearnRandomForestClassifier, dict[str, Any], list[dict[str, Any]], float]:
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

    tuned_model = SklearnRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **best_params)
    tuned_model.fit(x_train, y_train)

    return tuned_model, best_params, top_5_results, best_score


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
        placeholder = {
            "status": "skipped_no_processed_data",
            "required_files": [str(processed_dir / "train_clean.parquet"), str(processed_dir / "test_clean.parquet"), str(processed_dir / "train.parquet"), str(processed_dir / "test.parquet")],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(models_dir / "baseline_metrics.json", placeholder)
        write_json(models_dir / "tuned_metrics.json", placeholder)
        write_json(models_dir / "best_params.json", {"status": "not_run"})
        write_json(models_dir / "feature_importance.json", {"status": "not_run"})
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
