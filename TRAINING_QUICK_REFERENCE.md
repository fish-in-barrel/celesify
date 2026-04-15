# Training Module: Quick Reference Guide

## Module Overview (TL;DR)

| Module | Lines | Purpose | Key Functions |
|--------|-------|---------|----------------|
| `config.py` | 87 | Configuration management | `TrainingConfig` class |
| `data_handling.py` | 360 | Data loading & prep | Load datasets, extract features, subsample |
| `model_training.py` | 180 | Model training & search | Train baselines, RandomizedSearchCV |
| `evaluation.py` | 360 | Metrics & formatting | Evaluate, extract importance, format results |
| `export.py` | 170 | Artifact persistence | Save JSON, joblib, ONNX |
| `pipeline.py` | 303 | Orchestration | `run()` entry point, phase coordination |

## Running the Pipeline

### Standard run (all defaults):
```bash
cd /workspaces/celesify
python -m celesify.training.main
```

### Custom configuration via environment variables:
```bash
# Use more search iterations
TRAINING_N_ITER=50 python -m celesify.training.main

# Quick validation run
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_MAX_TRAIN_ROWS=3000 python -m celesify.training.main

# For debugging (single job, smaller CV)
TRAINING_N_JOBS=1 TRAINING_CV_SPLITS=2 python -m celesify.training.main
```

## Configuration Options

```python
from celesify.training.config import TrainingConfig

config = TrainingConfig()
print(config.n_iter)        # 20 (default) or TRAINING_N_ITER env var
print(config.cv_splits)     # 5 (default) or TRAINING_CV_SPLITS env var
print(config.n_jobs)        # -1 (default) or TRAINING_N_JOBS env var
print(config.max_train_rows) # 0 (no limit) or TRAINING_MAX_TRAIN_ROWS env var
```

## Key Imports

```python
# Main entry point
from celesify.training.pipeline import run

# Individual modules (for testing/debugging)
from celesify.training.data_handling import (
    load_datasets,
    extract_features_and_target,
    subsample_train_data,
    get_imbalance_recommendation,
    get_class_mapping,
)

from celesify.training.model_training import (
    train_baseline_model,
    run_randomized_search,
    create_search_space,
)

from celesify.training.evaluation import (
    evaluate_model,
    extract_feature_importance,
    format_baseline_metrics,
    format_tuned_metrics,
)

from celesify.training.export import (
    save_metrics_json,
    save_model_joblib,
    export_onnx_model,
)

from celesify.training.config import TrainingConfig
```

## Code Examples

### Example 1: Train a baseline model
```python
from celesify.training.data_handling import load_datasets, extract_features_and_target
from celesify.training.model_training import train_baseline_model
from celesify.training.evaluation import evaluate_model
from pathlib import Path

processed_dir = Path("outputs/processed")
clean_train, clean_test, _, _, _ = load_datasets(processed_dir)

x_train, y_train, x_test, y_test, features = extract_features_and_target(clean_train, clean_test)

model = train_baseline_model(x_train, y_train, n_jobs=-1)
metrics = evaluate_model(model, x_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
```

### Example 2: Run hyperparameter search
```python
from celesify.training.model_training import run_randomized_search

tuned_model, best_params, top_5, best_score = run_randomized_search(
    x_train, y_train,
    n_iter=20,
    cv_splits=5,
    n_jobs=-1,
    class_weight_options=[None, "balanced"]
)

print(f"Best CV score: {best_score:.4f}")
print(f"Best params:\n{best_params}")
```

### Example 3: Extract feature importance
```python
from celesify.training.evaluation import extract_feature_importance

importance_list = extract_feature_importance(tuned_model, features)

# Top 5 most important features
for item in importance_list[:5]:
    print(f"{item['feature']}: {item['importance']:.4f}")
```

### Example 4: Save artifacts
```python
from celesify.training.export import save_metrics_json, save_model_joblib, export_onnx_model
from pathlib import Path

models_dir = Path("outputs/models")
models_dir.mkdir(parents=True, exist_ok=True)

# Save metrics
save_metrics_json(models_dir, "my_metrics.json", metrics)

# Save model
save_model_joblib(models_dir, "my_model.joblib", tuned_model)

# Try ONNX (graceful if fails)
x_sample = x_train.iloc[:1]
success = export_onnx_model(models_dir, tuned_model, x_sample)
if not success:
    print("ONNX export failed; joblib model available as fallback")
```

## Output Artifacts

After a full run, these files are created in `outputs/models/`:

```
outputs/models/
├── baseline_metrics.json           # Baseline RF results
├── tuned_metrics.json              # Tuned RF results  
├── best_params.json                # Best hyperparameters
├── top_trials.json                 # Top 5 CV trials
├── feature_importance.json         # Feature rankings (MDI)
├── model.joblib                    # Serialized RF model
├── model.onnx                      # ONNX version (if successful)
└── onnx_export_status.json         # ONNX export metadata
```

## Metrics JSON Structure

### baseline_metrics.json / tuned_metrics.json
```json
{
  "status": "completed",
  "timestamp_utc": "2026-04-15T...",
  "random_state": 42,
  "dataset_variant": "cleaned",
  "split_files": {"train": "train_clean.parquet", "test": "test_clean.parquet"},
  "class_label_order": [0, 1, 2],
  "class_mapping": {"STAR": 0, "GALAXY": 1, "QSO": 2},
  "feature_columns": ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"],
  "dataset_shapes": {"train": [80000, 8], "test": [20000, 8]},
  "n_features": 8,
  "accuracy": 0.9742,
  "f1_macro": 0.9623,
  "confusion_matrix": [[...], [...], [...]],
  "per_class_metrics": {
    "0": {"precision": 0.98, "recall": 0.99, "f1_score": 0.985, "support": 5000},
    "1": {...},
    "2": {...}
  }
}
```

### feature_importance.json
```json
{
  "status": "completed",
  "timestamp_utc": "2026-04-15T...",
  "method": "mdi",
  "feature_importance": [
    {"feature": "redshift", "importance": 0.312},
    {"feature": "u", "importance": 0.245},
    {"feature": "z", "importance": 0.198},
    ...
  ]
}
```

## Debugging Tips

### Check configuration is loaded correctly:
```python
from celesify.training.config import TrainingConfig
import os

os.environ["TRAINING_N_ITER"] = "5"
config = TrainingConfig()
assert config.n_iter == 5  # ✓
```

### Verify data loads without errors:
```python
from celesify.training.data_handling import load_datasets
from pathlib import Path

try:
    datasets = load_datasets(Path("outputs/processed"))
    print(f"✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Data load failed: {e}")
```

### Test a single model training:
```python
from celesify.training.model_training import train_baseline_model

model = train_baseline_model(x_train, y_train, n_jobs=1)
print(f"✓ Model trained with {model.n_estimators} trees")
```

### Check metrics output format:
```python
from celesify.training.evaluation import evaluate_model
import json

metrics = evaluate_model(model, x_test, y_test)
# Verify structure
assert "accuracy" in metrics
assert "f1_macro" in metrics
assert "confusion_matrix" in metrics
assert "per_class_metrics" in metrics
print(f"✓ Metrics structure valid")
```

## Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| `FileNotFoundError: No matching parquet split found` | Preprocessing didn't run or wrong path | Run preprocessing service first |
| `ValueError: Expected target column 'class'` | Processed data corrupted or wrong format | Regenerate via preprocessing |
| `ONNX export failed` | sklearn/skl2onnx version mismatch | Check requirements.txt versions |
| `TRAINING_N_ITER` not recognized | Typo in env var name | Use exact name: `TRAINING_N_ITER` |
| Out of memory during search | Too many combinations or large data | Use `TRAINING_MAX_TRAIN_ROWS` to subsample |

## Performance Tuning

For faster iteration during development:

```bash
# Quick smoke test (~2 min)
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_MAX_TRAIN_ROWS=5000 python -m celesify.training.main

# Medium validation (~20 min)  
TRAINING_N_ITER=10 TRAINING_CV_SPLITS=3 TRAINING_MAX_TRAIN_ROWS=20000 python -m celesify.training.main

# Full run (~60 min)
TRAINING_N_ITER=20 TRAINING_CV_SPLITS=5 python -m celesify.training.main
```

## Container Execution

```bash
# Run from docker-compose
docker compose up training

# Or run directly
docker run \
  -v $(pwd)/outputs:/workspace/outputs \
  celesify-training \
  python -m celesify.training.main
```
