# copilot-instructions.md — Implementation Guide for celesify

This document provides implementation patterns, conventions, and common workflows for the celesify codebase.
It is a **companion to CLAUDE.md** — refer to CLAUDE.md for full architectural overview and phase status.

---

## Project Status

**Phases Complete:**
- ✅ Phase 1: Environment & repo setup
- ✅ Phase 2: Data ingestion & preprocessing (with advanced feature engineering)
- ✅ Phase 3: Modelling & tuning (three model variants: baseline, clean-tuned, engineered-tuned)
- ✅ Phase 4: Streamlit dashboard (three pages: Data Explorer, Model Evaluation, Upload & Infer)

**Phase Active:**
- ⏳ Phase 5: Analysis, writing & submission (placeholder; not yet started)

**Current Focus Areas:**
- Bug fixes and performance optimization in production paths (Docker + local Rye)
- Feature extensions to the Streamlit dashboard
- Model artifact validation and ONNX export reliability

---

## Code Organization

### Package Structure

The main Python package lives in `/celesify/` and is organized by functional domain:

```
celesify/
├── __init__.py
├── core/
│   ├── constants.py         # Fixed project constants (class encoding, feature names, etc.)
│   ├── logging.py           # Timestamped logging to stdout
│   ├── paths.py             # Cross-platform path resolution (data/, outputs/)
│   └── json_utils.py        # Consistent JSON read/write (pretty-print, error handling)
├── preprocessing/
│   ├── main.py              # Entry point: `python -m celesify.preprocessing.main`
│   └── pipeline.py          # Core logic: loading, cleaning, feature engineering, export
├── training/
│   ├── main.py              # Entry point: `python -m celesify.training.main`
│   └── pipeline.py          # Core logic: three model training, hyperparameter search, export
└── streamlit_app/
    ├── main.py              # Entry point: `streamlit run celesify/streamlit_app/main.py`
    ├── app.py               # Streamlit multi-page orchestrator
    ├── common.py            # Shared UI utilities (layout, colors, formatted tables, etc.)
    ├── page_data_explorer.py     # Tab 1: Data exploration visuals
    ├── page_performance_metrics.py # Tab 2: Model comparison and evaluation
    ├── page_upload_infer.py       # Tab 3: CSV upload or manual feature input + prediction
    └── assets/              # Static files (CSS, images, etc.)
```

### Docker Service Wrappers

Legacy entry points in `/services/` are thin wrappers that import and invoke the package:

```
services/
├── preprocessing/
│   ├── Dockerfile           # Pulls python:3.11-slim, installs from requirements.txt
│   ├── requirements.txt      # Generated from pyproject.toml; do NOT edit directly
│   └── preprocess.py        # Calls: from celesify.preprocessing import main; main()
├── training/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py             # Calls: from celesify.training import main; main()
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py               # Calls: from celesify.streamlit_app import app; app.run()
```

**Key insight:** Both Docker and local Rye execution invoke the same Python package code.
This ensures consistency and makes debugging simpler — no duplicated logic across entry points.

### Entry Points

All three workflows are invoked identically regardless of execution context:

| Workflow | Local (Rye) | Docker |
|----------|------------|--------|
| Preprocessing | `rye run preprocess` | `docker compose up preprocessing` |
| Training | `rye run train` | `docker compose up training` |
| Streamlit | `rye run streamlit-app` | `docker compose up streamlit` |

---

## Key Conventions

### Random State & Reproducibility

**Rule:** Use `random_state=42` throughout for reproducible splits, CV folds, and model initialization.

**Locations:**
- `celesify/core/constants.py`: `RANDOM_STATE = 42`
- `celesify/preprocessing/pipeline.py`: Stratified split, feature engineering sampling
- `celesify/training/pipeline.py`: RF initialization, RandomizedSearchCV, cross-validation

### Class Encoding

**Fixed mapping** in `celesify/core/constants.py`:
```python
CLASS_ENCODING = {"STAR": 0, "GALAXY": 1, "QSO": 2}
CLASS_LABEL_ORDER = [0, 1, 2]
```

**Always use:** Integer labels (0, 1, 2) internally; store and report with these IDs.
**For presentation:** Map back to string labels in Streamlit dashboard and reports.

### Feature Sets

Two distinct feature sets are maintained in parallel:

| Name | Features | Files | Purpose |
|------|----------|-------|---------|
| **Clean** | 8 original | `train_clean.parquet`, `test_clean.parquet` | Baseline + sanity check |
| **Engineered** | 28 derived | `train.parquet`, `test.parquet` | Primary model + deployment |

**Clean features** (8): `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`

**Engineered features** (28): Clean 8 + 
- Color pairs (9): `u-g`, `g-r`, `r-i`, `i-z`, `u-r`, `g-i`, `g-z`, `r-z`, `u-z`
- Band statistics (5): mean, std, min, max, range (computed across photometric bands)
- Redshift interactions (5): redshift × {`color_u_g`, `color_g_r`, `color_r_i`, `color_i_z`, `color_g_z`}

**Implementation:** See `celesify/preprocessing/pipeline.py` for feature engineering logic.

### Model Artifacts & Naming

**Three RF models are always trained in order:**

1. **Baseline RF** (clean features)
   - Sklearn defaults: `n_estimators=100`, no hyperparameter tuning
   - Artifacts: `model_baseline.joblib`, `baseline_metrics.json`

2. **Tuned RF (clean features)**
   - RandomizedSearchCV on clean features (5-fold stratified CV, `n_iter=20` default)
   - Artifacts: `model_clean_tuned.joblib`, `clean_tuned_metrics.json`, `best_params_clean_tuned.json`

3. **Tuned RF (engineered features) — PRIMARY**
   - RandomizedSearchCV on engineered features
   - Artifacts: `model.joblib`, `tuned_metrics.json`, `best_params.json`, `feature_importance.json`
   - ONNX export: `model.onnx`, `onnx_export_status.json`

**Key rule:** Baseline-first order is strict. Never run searches until baseline is fully saved.

### Logging & Output

**Style:** Timestamp + service name + message to stdout

```python
from celesify.core.logging import log

log("Starting preprocessing...")
log(f"Shape: {df.shape}, Classes: {class_counts}")
```

**Docker:** Automatically captures stdout; visible in `docker compose logs`
**Local:** Visible in terminal when running `rye run preprocess` etc.

### JSON Utilities

**Consistent read/write for metrics, params, and feature importance:**

```python
from celesify.core.json_utils import read_json, write_json

# Save
write_json({"accuracy": 0.95, "f1": 0.92}, "outputs/models/metrics.json")

# Load
metrics = read_json("outputs/models/metrics.json")
```

**Benefits:** Pretty-printing, error handling, consistent encoding.

### Paths & Cross-Platform

**Always use:** `celesify/core/paths.py` to resolve data and output directories

```python
from celesify.core.paths import resolve_preprocessing_paths, resolve_training_paths

train_path, test_path, report_path = resolve_preprocessing_paths()
# Returns correct paths whether running locally or in Docker
```

**Why:** Docker mounts shared volumes at `/workspace/outputs`; local runs use `./outputs`.
The `paths.py` module handles the conditional logic.

---

## Common Tasks

### Task 1: Modify Preprocessing Pipeline

**File:** `celesify/preprocessing/pipeline.py`

**Steps:**
1. Open `celesify/preprocessing/pipeline.py`
2. Identify the function you want to modify (e.g., `engineer_features()`)
3. Update the logic
4. **Test locally first:** `rye run preprocess` (use `TRAINING_MAX_TRAIN_ROWS=3000` env var to test on a subset)
5. Verify outputs in `outputs/processed/` (Parquet files + `preprocessing_report.json`)
6. Commit and test in Docker: `docker compose up preprocessing`

**Common modifications:**
- Add a new color pair: add tuple to `COLOR_FEATURES` list (e.g., `("u", "i", "color_u_i")`)
- Add a new statistic: add computation logic in `engineer_features()` (e.g., median, quartiles)
- Change missing-value strategy: modify logic in `handle_missing_values()`

**Validation checklist:**
- ✅ Both clean and engineered Parquets are produced
- ✅ `preprocessing_report.json` includes class counts and feature stats
- ✅ No errors in logs; shape and class distribution printed

### Task 2: Change Model Hyperparameters or Search Space

**File:** `celesify/training/pipeline.py`

**Steps:**
1. Open `celesify/training/pipeline.py`
2. Locate the search space dictionary in the tuned RF training function
3. Modify ranges (e.g., `max_depth`, `n_estimators`, `class_weight`)
4. **Test with quick override:** 
   ```bash
   TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_MAX_TRAIN_ROWS=3000 rye run train
   ```
   (This runs one iteration per parameter, 2-fold CV, on 3000 training rows — finishes in ~30s)
5. Verify `best_params.json` and metrics are produced
6. Run full training once validated: `rye run train`

**Common modifications:**
- Expand search space: add new values to lists (e.g., `n_estimators: [100, 200, 300, 500, 700]`)
- Restrict space: remove values (e.g., remove 'balanced' from `class_weight` for speed)
- Change number of iterations: `n_iter=50` (default 20) for more thorough search

**Validation checklist:**
- ✅ Baseline model trains first; `baseline_metrics.json` exists and contains accuracy/F1
- ✅ Search completes and best params are saved to JSON
- ✅ Final model artifacts (`model.joblib`, `model.onnx`) are produced
- ✅ Feature importance extracted; `feature_importance.json` is non-empty

### Task 3: Add a Dashboard Page or Widget

**Files:** `celesify/streamlit_app/app.py`, `celesify/streamlit_app/page_*.py`, `celesify/streamlit_app/common.py`

**Steps:**
1. Create a new page file: `celesify/streamlit_app/page_my_feature.py`
2. Define a `render()` function that calls Streamlit functions (`st.metric()`, `st.plotly_chart()`, etc.)
3. Import and register the page in `celesify/streamlit_app/app.py` (in the `pages` dict)
4. **Test locally:** `rye run streamlit-app` (opens http://localhost:8501)
5. Navigate to your new page in the sidebar and validate

**Example page structure:**
```python
# celesify/streamlit_app/page_my_feature.py
import streamlit as st
from celesify.core.json_utils import read_json
from celesify.streamlit_app.common import render_title, colored_metric

def render():
    render_title("My Feature", "Description of what this shows")
    
    # Load data
    metrics = read_json("outputs/models/tuned_metrics.json")
    
    # Display
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.dataframe(metrics)
```

**Validation checklist:**
- ✅ Page renders without errors
- ✅ All data loaded from JSON artifacts (not recalculated)
- ✅ Layout is clean and responsive

### Task 4: Add a New Feature or Column

**File:** `celesify/preprocessing/pipeline.py`

**Steps:**
1. Identify the type of feature (color pair, statistic, interaction, etc.)
2. Add the feature to the appropriate list in `pipeline.py` (e.g., `COLOR_FEATURES`, `FEATURE_INTERACTIONS`)
3. Ensure the feature engineering function processes it (e.g., `engineer_features()`)
4. Run preprocessing: `rye run preprocess`
5. Verify the new feature appears in `train.parquet` with a sensible distribution
6. Run training to ensure models don't break: `rye run train`

**Validation:**
- ✅ New feature in Parquet schema
- ✅ No NaN or inf values
- ✅ Training completes and produces metrics

### Task 5: Validate Model Exports (ONNX)

**File:** `celesify/training/pipeline.py` (export section)

**Steps:**
1. Run training: `rye run train`
2. Check `outputs/models/onnx_export_status.json` for status
3. If failed, check `outputs/models/onnx_export_error.log` for details
4. To debug manually:
   ```python
   import onnx
   from onnxruntime import InferenceSession
   
   model = onnx.load("outputs/models/model.onnx")
   onnx.checker.check_model(model)  # Validates structure
   
   sess = InferenceSession("outputs/models/model.onnx")  # Test runtime load
   print(sess.get_inputs())
   print(sess.get_outputs())
   ```

**Common issues:**
- **Issue:** `skl2onnx` version mismatch → **Solution:** Use pinned versions in `pyproject.toml` (onnx==1.21.0, skl2onnx==1.20.0, onnxruntime==1.21.0)
- **Issue:** ONNX export silently fails → **Solution:** Check `onnx_export_error.log` for traceback

---

## File Dependencies & Data Flow

```
┌─────────────────────────────────────────┐
│  data/raw/star_classification.csv       │ (User downloads from Kaggle or uses fallback API)
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  celesify/preprocessing/main.py          │ (Entry point)
│  celesify/preprocessing/pipeline.py      │ (Core logic)
└────────────┬────────────────────────────┘
             │
             ├─► outputs/processed/train.parquet (28 engineered features, 80% of data)
             ├─► outputs/processed/test.parquet (28 engineered features, 20% of data)
             ├─► outputs/processed/train_clean.parquet (8 original features, 80% of data)
             ├─► outputs/processed/test_clean.parquet (8 original features, 20% of data)
             └─► outputs/processed/preprocessing_report.json (metadata)
             │
             ▼
┌─────────────────────────────────────────┐
│  celesify/training/main.py               │ (Entry point)
│  celesify/training/pipeline.py           │ (Core logic)
└────────────┬────────────────────────────┘
             │
             ├─► outputs/models/model_baseline.joblib
             ├─► outputs/models/baseline_metrics.json
             ├─► outputs/models/model_clean_tuned.joblib
             ├─► outputs/models/clean_tuned_metrics.json
             ├─► outputs/models/best_params_clean_tuned.json
             ├─► outputs/models/model.joblib (PRIMARY)
             ├─► outputs/models/tuned_metrics.json
             ├─► outputs/models/best_params.json
             ├─► outputs/models/feature_importance.json
             ├─► outputs/models/model.onnx
             └─► outputs/models/onnx_export_status.json
             │
             ▼
┌─────────────────────────────────────────┐
│  celesify/streamlit_app/main.py          │ (Entry point)
│  celesify/streamlit_app/app.py           │ (Orchestrator)
│  celesify/streamlit_app/page_*.py        │ (Render functions)
└─────────────────────────────────────────┘
       │
       └─► Loads from outputs/models/ + outputs/processed/ at startup
           Displays dashboards, predictions, visualizations
```

---

## Testing & Validation

### Smoke Test: Full Pipeline (5–10 minutes)

```bash
# 1. Preprocess (on subset for speed)
TRAINING_MAX_TRAIN_ROWS=10000 rye run preprocess

# 2. Train (quick validation only)
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_MAX_TRAIN_ROWS=3000 rye run train

# 3. Check outputs exist
ls -la outputs/processed/
ls -la outputs/models/

# 4. Verify JSON artifacts are valid
python -c "import json; print(json.load(open('outputs/models/baseline_metrics.json')))"
```

### Full Production Run (30–60 minutes, depending on hardware)

```bash
# 1. Full preprocessing
rye run preprocess

# 2. Full training (all three model variants, 20 iterations each)
rye run train

# 3. Start dashboard
rye run streamlit-app
```

### Docker Validation

```bash
# Build and run all three services
docker compose up

# Watch logs in another terminal
docker compose logs -f

# When dashboard is ready, navigate to http://localhost:8501
```

### Environment Overrides (for quick validation)

Set before running `rye run train`:

```bash
TRAINING_N_ITER=1              # Number of RandomizedSearchCV iterations (default 20)
TRAINING_CV_SPLITS=2           # K-fold CV splits (default 5)
TRAINING_N_JOBS=1              # Parallel jobs (default -1 for all cores)
TRAINING_MAX_TRAIN_ROWS=3000   # Limit training set size for testing (default None)
```

**Example:**
```bash
TRAINING_N_ITER=1 TRAINING_CV_SPLITS=2 TRAINING_N_JOBS=1 TRAINING_MAX_TRAIN_ROWS=3000 rye run train
```

---

## Known Decisions & Rationales

### Why CPU-only, not GPU/CUDA?

**Decision:** Standardized on CPU-only scikit-learn for training and inference.

**Rationale:**
- Dataset size (~100k rows, tabular RF) shows diminishing returns with GPU acceleration
- CUDA setup overhead (toolkit version management, image pulls, runtime compatibility) increased iteration time
- CPU training with `n_jobs=-1` parallelism is fast enough for hyperparameter search
- Portability: CPU-only runs reliably across Windows/Mac/Linux without GPU-specific configuration

### Why Three Model Variants?

**Decision:** Train baseline (defaults) → tuned (same features) → tuned (engineered features).

**Rationale:**
- Baseline establishes a reproducible lower bound on performance
- Tuned on clean features isolates the impact of hyperparameter search alone
- Tuned on engineered features captures both engineering + tuning improvements
- Allows paper to report: "tuning alone improves F1 by X%; engineering + tuning improves by Y%"

### Why Both Joblib and ONNX Exports?

**Decision:** Save models in both joblib and ONNX formats.

**Rationale:**
- **Joblib:** Native scikit-learn format, full feature support, used for inference in Streamlit
- **ONNX:** Open interoperability standard, enables deployment on non-Python platforms (JavaScript, mobile, etc.)
- Maintains reproducibility of the exact fitted model while enabling broader deployment

### Why Drop Rows for Missing Values?

**Decision:** Drop any row with missing values; do not impute.

**Rationale:**
- Missing values affect < 1% of data
- No systematic bias detected in missing patterns
- Simpler to validate and reproduce than imputation strategies
- Avoids introducing model-dependent assumptions into preprocessing

---

## Troubleshooting

### Issue: Kaggle API key not found

**Error message:** `Kaggle API key not found at ~/.kaggle/kaggle.json`

**Solution:**
1. Download your Kaggle API key from https://www.kaggle.com/settings/account
2. Create `~/.kaggle/` directory and save the JSON file
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
4. Retry preprocessing: `rye run preprocess`

**Alternative:** Manually download `star_classification.csv` from Kaggle and place in `data/raw/`.

---

### Issue: ONNX export fails

**Error message:** `skl2onnx converter error...` or ONNX checker fails

**Solution:**
1. Check `outputs/models/onnx_export_error.log` for detailed traceback
2. Verify pinned versions in `pyproject.toml`: `onnx==1.21.0`, `skl2onnx==1.20.0`, `onnxruntime==1.21.0`
3. Reinstall training dependencies:
   ```bash
   rye sync --features training-core --features training-onnx
   ```
4. Run training again: `rye run train`

**If issue persists:**
- Reduce search complexity: set `TRAINING_N_ITER=1` and run a quick test
- Check `onnx_export_status.json` for validation details
- The training pipeline does not fail if ONNX export fails; metrics are always saved

---

### Issue: Training runs out of memory or is very slow

**Error:** `MemoryError` or training takes > 2 hours

**Solutions:**
1. **Reduce data:** Set `TRAINING_MAX_TRAIN_ROWS=10000` to test on subset
2. **Reduce CV complexity:** `TRAINING_CV_SPLITS=2` (default 5)
3. **Reduce search iterations:** `TRAINING_N_ITER=5` (default 20)
4. **Reduce parallelism:** `TRAINING_N_JOBS=2` (default -1 for all cores)

**Example quick run:**
```bash
TRAINING_N_ITER=5 TRAINING_CV_SPLITS=2 TRAINING_N_JOBS=2 TRAINING_MAX_TRAIN_ROWS=10000 rye run train
```

---

### Issue: Streamlit app won't load or shows "Model not found"

**Error:** 404 or file not found error in browser console

**Solution:**
1. Ensure training has completed: `ls outputs/models/model.joblib`
2. Check that Parquet files exist: `ls outputs/processed/train.parquet`
3. Restart Streamlit: Stop with Ctrl+C and re-run `rye run streamlit-app`
4. Clear Streamlit cache: `rm -rf ~/.streamlit && rye run streamlit-app`

---

### Issue: Docker build fails with "pip install requirements.txt" error

**Error:** `ERROR: Could not find a version that satisfies the requirement...`

**Solution:**
1. Regenerate `requirements.txt` from `pyproject.toml`:
   ```bash
   rye lock
   ```
2. Commit the updated lockfile to git
3. Rebuild Docker images:
   ```bash
   docker compose build --no-cache
   ```

---

### Issue: Windows/WSL — Training is very slow or crashes with memory errors

**Error:** `MemoryError` or training hangs after starting

**Root cause:** WSL default resource limits are too low (often 50% RAM, 1 CPU core).

**Solution:** Increase WSL memory and CPU limits

1. **On Windows host**, create or edit `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   memory=8GB
   processors=4
   swap=4GB
   ```

2. **Save and restart WSL:**
   ```powershell
   wsl --shutdown
   wsl --list --verbose
   ```

3. **Verify new limits inside WSL:**
   ```bash
   free -h          # Check available memory
   nproc             # Check CPU cores
   ```

4. **Retry training:**
   ```bash
   rye run train
   ```

**Recommended settings for ONNX export:**
- `memory=8GB` (minimum)
- `processors=4` (minimum; 8+ is ideal)
- `swap=4GB` (prevents swapping to disk)

**Dev container in Windows/WSL:**
If using VS Code dev container, also ensure Docker Desktop is configured with adequate memory:
1. Open Docker Desktop → Settings → Resources
2. Set Memory to at least 6GB
3. Set CPUs to at least 4
4. Restart Docker

---

### Issue: Preprocessing skips feature engineering (only produces `*_clean.parquet`)

**Error:** No `train.parquet` or `test.parquet` file; only clean files

**Solution:**
1. Check logs for skipped engineering steps
2. Verify feature engineering function is enabled in `pipeline.py`
3. Run preprocessing again with verbose logging:
   ```bash
   rye run preprocess
   ```
4. Look for lines like: `Generating engineered features...`

If issue persists, the engineering logic may have been commented out or removed. Verify the `engineer_features()` function exists and is called in `preprocess()`.

---

### Issue: Test set predictions differ between runs (not reproducible)

**Error:** Running inference twice gives different predictions

**Solution:**
1. Verify `random_state=42` is set in model loading/prediction code
2. Ensure ONNX runtime is deterministic:
   ```python
   import onnxruntime as ort
   sess = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
   ```
3. Check that no external randomness is introduced in preprocessing or feature scaling

---

## Development Container Setup (VS Code)

The project includes a `.devcontainer/` configuration for consistent development environments across Windows, Mac, and Linux.

### Prerequisites

- Docker Desktop installed and running
- VS Code Remote - Containers extension installed

### Opening the Dev Container

1. Open the project in VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
3. Wait for the container to build (~2–5 minutes on first run)

### Dev Container Specifications

**Image:** Python 3.11 slim + Rye + Git + Docker CLI + Starship shell

**Memory & CPU for Windows/WSL:**
- **Minimum:** 6 GB RAM, 4 CPU cores
- **Recommended:** 8 GB RAM, 6 CPU cores
- **For ONNX export:** 8 GB RAM + 4 cores

**To adjust in Windows/WSL:**

1. Create/edit `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   memory=8GB
   processors=6
   swap=4GB
   localhostForwarding=true
   ```

2. Restart WSL:
   ```powershell
   wsl --shutdown
   # Then restart container in VS Code
   ```

3. Inside the container, verify:
   ```bash
   free -h       # Should show 8GB available
   nproc         # Should show 6 cores
   ```

### Troubleshooting Dev Container on Windows/WSL

**Issue:** Container startup is very slow or hangs

**Solution:**
1. Increase memory/CPU in `.wslconfig` (see above)
2. Rebuild container from scratch:
   - In VS Code: `Dev Containers: Rebuild Container`
3. Clear Docker disk space:
   ```bash
   docker system prune -a --volumes
   ```

**Issue:** "docker: command not found" in container

**Solution:**
- Docker-in-Docker is pre-configured. If missing, ensure Docker daemon is running on host.
- In Windows: Start Docker Desktop
- In Mac: Start Docker Desktop
- In Linux: Ensure Docker daemon is running

**Issue:** Slow file I/O in the container

**Solution:**
- On Windows/WSL, file I/O between host and container is slower than native
- To improve: Keep all work inside the container (do not edit files on host and sync to container)
- Alternatively, use Docker volume mounts for better performance:
  ```bash
  docker volume create celesify_data
  docker run -v celesify_data:/workspace ...
  ```

---

## References & Links

- **CLAUDE.md:** Full architectural overview and phase status
- **README.md:** User-facing documentation and quick-start guide
- **pyproject.toml:** Dependency definitions and Rye scripts
- **docker-compose.yml:** Docker service orchestration
- **Rye:** https://rye.astral.sh/
- **Task runner:** https://taskfile.dev/
- **scikit-learn docs:** https://scikit-learn.org/
- **ONNX spec:** https://onnx.ai/
- **Streamlit docs:** https://docs.streamlit.io/
