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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

RANDOM_STATE = 42
CLASS_LABEL_ORDER = [0, 1, 2]
DEFAULT_CLASS_MAP = {"STAR": 0, "GALAXY": 1, "QSO": 2}

try:
    from cuml.ensemble import RandomForestClassifier as CumlRandomForestClassifier  # type: ignore[reportMissingImports]
except Exception:
    CumlRandomForestClassifier = None


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [training] {message}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


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


def try_import_cuml() -> bool:
    return CumlRandomForestClassifier is not None


def get_random_forest_classifier() -> type[Any]:
    if CumlRandomForestClassifier is not None:
        return CumlRandomForestClassifier
    return SklearnRandomForestClassifier


def is_cuml_estimator(model: Any) -> bool:
    return model.__class__.__module__.startswith("cuml.")


def resolve_data_paths() -> tuple[Path, Path]:
    candidate_roots = [Path("/workspace"), Path.cwd().resolve()]
    candidate_roots.extend(Path(__file__).resolve().parents)

    seen_roots: set[Path] = set()
    for root in candidate_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)

        outputs_root = root if root.name == "outputs" else root / "outputs"
        try:
            (outputs_root / "models").mkdir(parents=True, exist_ok=True)
            return outputs_root / "processed", outputs_root / "models"
        except (PermissionError, OSError, IndexError):
            continue

    raise RuntimeError("Unable to resolve writable outputs directory for training artifacts.")


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log(f"Invalid {name}={raw!r}; using default {default}.")
        return default


def supports_class_weight(estimator_cls: type[Any]) -> bool:
    return estimator_cls is SklearnRandomForestClassifier


def main() -> None:
    processed_dir, models_dir = resolve_data_paths()
    models_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_dir / "train.parquet"
    test_file = processed_dir / "test.parquet"
    report_file = processed_dir / "preprocessing_report.json"

    log(f"Processed dir: {processed_dir}")
    log(f"Models dir: {models_dir}")

    if not train_file.exists() or not test_file.exists():
        log("Processed parquet files not found; writing scaffold placeholder artifacts.")
        placeholder = {
            "status": "skipped_no_processed_data",
            "required_files": [str(train_file), str(test_file)],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(models_dir / "baseline_metrics.json", placeholder)
        write_json(models_dir / "tuned_metrics.json", placeholder)
        write_json(models_dir / "best_params.json", {"status": "not_run"})
        write_json(models_dir / "feature_importance.json", {"status": "not_run"})
        return

    log(f"Using random_state={RANDOM_STATE} and class labels={CLASS_LABEL_ORDER}")
    log(f"cuML available: {try_import_cuml()}")

    preprocess_report = {}
    if report_file.exists():
        preprocess_report = json.loads(report_file.read_text(encoding="utf-8"))
        log("Loaded preprocessing_report.json for metadata and imbalance guidance.")
    else:
        log("preprocessing_report.json not found; continuing with safe defaults.")

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    if "class" not in train_df.columns or "class" not in test_df.columns:
        raise ValueError("Expected target column 'class' in both train and test parquet files.")

    feature_columns: list[str] = [str(col) for col in train_df.columns if col != "class"]
    if not feature_columns:
        raise ValueError("No feature columns found after removing target column 'class'.")

    x_train: pd.DataFrame = cast(pd.DataFrame, train_df[feature_columns])
    y_train = train_df["class"].astype(int)
    x_test: pd.DataFrame = cast(pd.DataFrame, test_df[feature_columns])
    y_test = test_df["class"].astype(int)

    log(
        "Loaded datasets: "
        f"train={x_train.shape}, test={x_test.shape}, features={len(feature_columns)}"
    )

    class_mapping = get_class_mapping(preprocess_report)
    imbalance_flagged = get_imbalance_recommendation(preprocess_report)
    class_weight_space = [None, "balanced"] if imbalance_flagged else [None]

    # Defaults preserve Phase 3 requirements; env overrides support faster local smoke tests.
    n_iter_used = max(1, get_int_env("TRAINING_N_ITER", 20))
    cv_splits = max(2, get_int_env("TRAINING_CV_SPLITS", 5))
    n_jobs = get_int_env("TRAINING_N_JOBS", -1)
    max_train_rows = get_int_env("TRAINING_MAX_TRAIN_ROWS", 0)

    if max_train_rows > 0 and len(train_df) > max_train_rows:
        class_counts = cast(pd.Series, train_df["class"].value_counts().sort_index())
        target_counts = cast(pd.Series, ((class_counts / len(train_df)) * max_train_rows).round().astype(int))
        target_counts[target_counts < 1] = 1

        while int(target_counts.sum()) > max_train_rows:
            reducible = target_counts[target_counts > 1]
            if reducible.empty:
                break
            target_counts.loc[reducible.idxmax()] -= 1

        sampled_parts = []
        for cls, count in target_counts.items():
            class_slice = train_df[train_df["class"] == cls]
            take_n = min(int(count), len(class_slice))
            sampled_parts.append(class_slice.sample(n=take_n, random_state=RANDOM_STATE, replace=False))

        sampled = pd.concat(sampled_parts, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        train_df = sampled
        x_train = cast(pd.DataFrame, train_df[feature_columns])
        y_train = train_df["class"].astype(int)
        log(f"Applied TRAINING_MAX_TRAIN_ROWS={max_train_rows}; sampled train rows={len(train_df)}")

    # Keep baseline on sklearn for direct comparability of reported baseline metrics.
    baseline_model = SklearnRandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    log("Training baseline RandomForestClassifier.")
    baseline_model.fit(x_train, y_train)

    baseline_eval = evaluate_model(baseline_model, x_test, y_test)
    baseline_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "feature_columns": feature_columns,
        "dataset_shapes": {
            "train": list(x_train.shape),
            "test": list(x_test.shape),
        },
        **baseline_eval,
    }
    write_json(models_dir / "baseline_metrics.json", as_jsonable(baseline_payload))
    log("Wrote baseline_metrics.json")

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.3],
    }
    top_k = max(1, get_int_env("TRAINING_TOP_K", 3))

    broad_search_backend = "cuml" if try_import_cuml() else "sklearn"
    broad_estimator_cls: type[Any] = (
        cast(type[Any], CumlRandomForestClassifier)
        if broad_search_backend == "cuml"
        else SklearnRandomForestClassifier
    )

    broad_param_distributions = dict(param_distributions)
    if supports_class_weight(broad_estimator_cls):
        broad_param_distributions["class_weight"] = class_weight_space
    else:
        log("cuML broad search detected: omitting unsupported 'class_weight' search parameter.")

    log(
        "Starting broad RandomizedSearchCV "
        f"backend={broad_search_backend}, n_iter={n_iter_used}, cv={cv_splits}, n_jobs={n_jobs}, "
        f"class_weight_space={class_weight_space}"
    )
    search = RandomizedSearchCV(
        estimator=broad_estimator_cls(random_state=RANDOM_STATE),
        param_distributions=broad_param_distributions,
        n_iter=n_iter_used,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    try:
        search.fit(x_train, y_train)
    except Exception as exc:
        if broad_search_backend == "cuml":
            log(f"cuML broad search failed ({type(exc).__name__}); retrying broad search with sklearn.")
            broad_search_backend = "sklearn_fallback"
            broad_param_distributions = dict(param_distributions)
            broad_param_distributions["class_weight"] = class_weight_space
            search = RandomizedSearchCV(
                estimator=SklearnRandomForestClassifier(random_state=RANDOM_STATE),
                param_distributions=broad_param_distributions,
                n_iter=n_iter_used,
                scoring="f1_macro",
                cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
                refit=True,
                verbose=1,
            )
            search.fit(x_train, y_train)
        else:
            raise

    ranked_candidates: list[dict[str, Any]] = []
    cv_results = getattr(search, "cv_results_", None)
    if isinstance(cv_results, dict) and "rank_test_score" in cv_results and "params" in cv_results:
        candidates = list(zip(cv_results["rank_test_score"], cv_results["params"], cv_results["mean_test_score"]))
        candidates.sort(key=lambda item: item[0])
        ranked_candidates = [
            {
                "rank": int(rank),
                "params": cast(dict[str, Any], params),
                "mean_test_score": float(mean_score),
            }
            for rank, params, mean_score in candidates[:top_k]
        ]
    else:
        ranked_candidates = [
            {
                "rank": 1,
                "params": cast(dict[str, Any], search.best_params_),
                "mean_test_score": float(search.best_score_),
            }
        ]

    confirmation_grid: list[dict[str, Any]] = []
    for candidate in ranked_candidates:
        base_params = dict(candidate["params"])
        if "class_weight" in base_params:
            confirmation_grid.append(base_params)
            continue

        if imbalance_flagged:
            for cw in class_weight_space:
                p = dict(base_params)
                p["class_weight"] = cw
                confirmation_grid.append(p)
        else:
            p = dict(base_params)
            p["class_weight"] = None
            confirmation_grid.append(p)

    seen_param_keys: set[str] = set()
    unique_confirmation_grid: list[dict[str, Any]] = []
    for params in confirmation_grid:
        key = json.dumps(as_jsonable(params), sort_keys=True)
        if key in seen_param_keys:
            continue
        seen_param_keys.add(key)
        unique_confirmation_grid.append(params)

    log(
        "Starting sklearn CPU confirmation on shortlisted candidates "
        f"(top_k={top_k}, candidates={len(unique_confirmation_grid)})."
    )
    best_confirm_score = -1.0
    best_confirm_params: dict[str, Any] = {}
    confirmation_results: list[dict[str, Any]] = []
    for params in unique_confirmation_grid:
        sklearn_model = SklearnRandomForestClassifier(random_state=RANDOM_STATE, **params)
        scores = cross_val_score(
            sklearn_model,
            x_train,
            y_train,
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=n_jobs,
        )
        mean_score = float(np.mean(scores))
        confirmation_results.append({"params": params, "mean_cv_f1_macro": mean_score})
        if mean_score > best_confirm_score:
            best_confirm_score = mean_score
            best_confirm_params = params

    if not best_confirm_params:
        best_confirm_params = cast(dict[str, Any], search.best_params_)
        if "class_weight" not in best_confirm_params:
            best_confirm_params["class_weight"] = class_weight_space[-1] if imbalance_flagged else None

    best_params_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_iter_used": n_iter_used,
        "top_k_used": top_k,
        "scoring": "f1_macro",
        "best_cv_score": float(search.best_score_),
        "broad_search_backend": broad_search_backend,
        "best_params_broad_search": search.best_params_,
        "best_broad_cv_score": float(search.best_score_),
        "best_params": best_confirm_params,
        "best_params_confirmed_cpu": best_confirm_params,
        "best_confirmed_cpu_cv_score": best_confirm_score,
        "shortlist": ranked_candidates,
        "cpu_confirmation_results": confirmation_results,
    }
    write_json(models_dir / "best_params.json", as_jsonable(best_params_payload))
    log("Wrote best_params.json")

    tuned_model = SklearnRandomForestClassifier(random_state=RANDOM_STATE, **best_confirm_params)
    tuned_model.fit(x_train, y_train)
    tuned_eval = evaluate_model(tuned_model, x_test, y_test)
    tuned_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "feature_columns": feature_columns,
        "dataset_shapes": {
            "train": list(x_train.shape),
            "test": list(x_test.shape),
        },
        "n_iter_used": n_iter_used,
        "top_k_used": top_k,
        "cv_splits": cv_splits,
        "n_jobs": n_jobs,
        "search_backend": broad_search_backend,
        "best_params": best_confirm_params,
        "best_params_broad_search": search.best_params_,
        "best_broad_cv_score": float(search.best_score_),
        "best_confirmed_cpu_cv_score": best_confirm_score,
        **tuned_eval,
    }
    write_json(models_dir / "tuned_metrics.json", as_jsonable(tuned_payload))
    log("Wrote tuned_metrics.json")

    tuned_importances = np.asarray(getattr(tuned_model, "feature_importances_", np.zeros(len(feature_columns))))
    feature_importance_items = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(feature_columns, tuned_importances)
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
    log("Wrote feature_importance.json")

    model_joblib_path = models_dir / "model.joblib"
    joblib.dump(tuned_model, model_joblib_path)
    log("Wrote model.joblib")

    x_sample = x_train.iloc[:1].astype(np.float32)
    model_onnx_path = models_dir / "model.onnx"
    onnx_status_path = models_dir / "onnx_export_status.json"
    onnx_error_log = models_dir / "onnx_export_error.log"

    try:
        onnx_export_estimator = tuned_model
        if is_cuml_estimator(tuned_model):
            log("cuML estimator detected; fitting sklearn surrogate for ONNX export compatibility.")
            surrogate_params = {
                "random_state": RANDOM_STATE,
                "n_estimators": search.best_params_.get("n_estimators", 100),
                "max_depth": search.best_params_.get("max_depth", None),
                "min_samples_split": search.best_params_.get("min_samples_split", 2),
                "max_features": search.best_params_.get("max_features", "sqrt"),
            }
            if "class_weight" in search.best_params_:
                surrogate_params["class_weight"] = search.best_params_["class_weight"]

            onnx_export_estimator = SklearnRandomForestClassifier(**surrogate_params)
            onnx_export_estimator.fit(x_train, y_train)

        onnx_result = convert_sklearn(
            onnx_export_estimator,
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
                "onnx_export_model_type": f"{onnx_export_estimator.__class__.__module__}.{onnx_export_estimator.__class__.__name__}",
            },
        )
        log("Wrote model.onnx")
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
        log(f"ONNX export failed ({type(exc).__name__}); details written to {onnx_error_log}")
        raise RuntimeError(f"ONNX export failed; see {onnx_error_log}") from None

    log("Phase 3 training pipeline completed successfully.")


if __name__ == "__main__":
    main()
