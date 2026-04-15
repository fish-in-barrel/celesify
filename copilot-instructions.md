# copilot-instructions.md — Copilot Reference

**See [.instructions.md](.instructions.md) for full project context and guidelines.**

This is a quick reference for GitHub Copilot Chat.

## TL;DR

- **Project**: Classify stellar objects (STAR/GALAXY/QSO) via scikit-learn Random Forest on SDSS17 dataset
- **Architecture**: Docker Compose (preprocessing → training → Streamlit)
- **State**: Fully functional; training module refactored into 6 modular files (April 2026)
- **Docs**: 
  - [TRAINING_MODULE_ARCHITECTURE.md](TRAINING_MODULE_ARCHITECTURE.md) — Technical guide
  - [TRAINING_QUICK_REFERENCE.md](TRAINING_QUICK_REFERENCE.md) — Quick start
  - [CLAUDE.md](CLAUDE.md) — Full project plan

## Key Conventions

| Item | Value |
|------|-------|
| Python | 3.11+ |
| Random State | Always `RANDOM_STATE=42` from `celesify.core.constants` |
| Class Encoding | `STAR=0, GALAXY=1, QSO=2` |
| Type Hints | Full (all public functions) |
| Logging | Use `celesify.core.logging.log()`, not `print()` |
| Scoring | `f1_macro` (macro-averaged F1) |
| CV | `StratifiedKFold` (preserves class distribution) |
| Model Export | joblib (always), ONNX (non-fatal if fails) |

## Training Module (6 Files)

```
celesify/training/
├── config.py          → TrainingConfig, env var handling
├── data_handling.py   → Load datasets, extract features, validate
├── model_training.py  → Baseline RF, RandomizedSearchCV
├── evaluation.py      → Metrics computation, formatting
├── export.py          → Save JSON, joblib, ONNX
├── pipeline.py        → run() orchestration
└── main.py            → Entrypoint
```

**Environment Variables**:
- `TRAINING_N_ITER` (default: 20)
- `TRAINING_CV_SPLITS` (default: 5)
- `TRAINING_N_JOBS` (default: -1)
- `TRAINING_MAX_TRAIN_ROWS` (default: 0 = no limit)

## Hyperparameter Search Space

```python
n_estimators:    [100, 200, 300, 500]
max_depth:       [None, 10, 20, 30]
min_samples_split: [2, 5, 10]
max_features:    ["sqrt", "log2", 0.3]
class_weight:    [None, "balanced"]  # if imbalance > 2.0
```

## Data Flow

```
data/raw/star_classification.csv
    ↓ (preprocessing)
outputs/processed/{train,test}.parquet + preprocessing_report.json
    ↓ (training)
outputs/models/{baseline_metrics, tuned_metrics, best_params, feature_importance, model.joblib, model.onnx}
    ↓ (streamlit)
http://localhost:8501 (dashboard)
```

## Quick Commands

```bash
# Full pipeline
docker compose up

# Training only (CPU, ~60 min)
TRAINING_N_ITER=20 python -m celesify.training.main

# Quick validation (~2 min)
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_MAX_TRAIN_ROWS=3000 python -m celesify.training.main

# Test imports
python -c "from celesify.training.pipeline import run; print('✓')"
```

## Code Patterns

### Logging
```python
from celesify.core.logging import log
log("training", f"Training with n_iter={config.n_iter}")
```

### Type Hints
```python
def evaluate_model(model: RandomForestClassifier, x_test: pd.DataFrame, 
                   y_test: pd.Series) -> dict[str, Any]:
    """Compute evaluation metrics."""
```

### Constants
```python
from celesify.core.constants import RANDOM_STATE, CLASS_ENCODING, CLASS_LABEL_ORDER
# Use RANDOM_STATE in: train/test split, CV, model init, etc.
```

### JSON Export
```python
from celesify.core.json_utils import write_json, as_jsonable
write_json(Path("metrics.json"), as_jsonable(metrics_dict))
```

## Avoid

❌ `print()` — use `log()`  
❌ Non-42 random seeds — use `RANDOM_STATE`  
❌ Direct logging.getLogger() — use `celesify.core.logging.log()`  
❌ ONNX export failures blocking run — make graceful with try/except  
❌ Committing `data/raw/`, `outputs/`, `*.pyc` — use .gitignore  

## See Also

- [CLAUDE.md](CLAUDE.md) — Full project plan
- [.instructions.md](.instructions.md) — Complete context & guidelines
- [TRAINING_MODULE_ARCHITECTURE.md](TRAINING_MODULE_ARCHITECTURE.md) — Module design & testing
- [TRAINING_QUICK_REFERENCE.md](TRAINING_QUICK_REFERENCE.md) — Code examples & debugging
