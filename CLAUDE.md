# CLAUDE.md — celesify

This file describes the project structure, implementation plan, and working conventions for
this codebase. Read it before making changes.

---

## Project overview

Classify stellar objects (stars, galaxies, quasars) from the SDSS17 dataset using a Random
Forest model. The project answers two questions:

1. How accurately can a tuned RF model classify objects in SDSS17, and how much does
   tuning improve on the default configuration?
2. Which features drive separation between classes, and what do those rankings reveal
   about the underlying data?

Dataset: 100,000 objects, 17 features, 3 target classes. Source: fedesoriano (2022) via Kaggle.

---

## Repository structure

The project uses a **hybrid package + service architecture**: modular Python code in `/celesify/`
is orchestrated via `/services/` Dockerfiles and managed locally via Rye and Taskfile.

```
.
├── CLAUDE.md                         # This file — project overview & implementation status
├── copilot-instructions.md           # Companion guide — patterns, conventions, common tasks
├── README.md                         # User-facing documentation
├── pyproject.toml                    # Rye project config with optional dependency groups
├── Taskfile.yml                      # Task runner for local development
├── requirements.lock                 # Rye lockfile (reproducible deps)
├── docker-compose.yml                # Defines all three services + shared volume
├── .devcontainer/                    # VS Code dev container config
│
├── celesify/                         # Main Python package (importable)
│   ├── __init__.py
│   ├── core/                         # Shared utilities
│   │   ├── constants.py              # Project constants (class encoding, feature names, etc.)
│   │   ├── logging.py                # Logging helper
│   │   ├── paths.py                  # Path resolution for data/outputs
│   │   └── json_utils.py             # JSON read/write utilities
│   ├── preprocessing/
│   │   ├── main.py                   # Entry point (run via 'rye run preprocess' or Docker)
│   │   ├── pipeline.py               # Orchestrator: coordinates all phases
│   │   ├── loading.py                # Phase 1: CSV file discovery & Kaggle download
│   │   ├── cleaning.py               # Phase 2: Schema validation, missing values, target encoding
│   │   ├── features.py               # Phase 3: Stratified split & feature engineering
│   │   └── exports.py                # Phase 4: Parquet export & report generation
│   ├── training/
│   │   ├── main.py                   # Entry point (run via 'rye run train' or Docker)
│   │   └── pipeline.py               # Baseline RF, hyperparameter search, ONNX export
│   └── streamlit_app/
│       ├── main.py                   # Entry point (run via 'rye run streamlit-app' or Docker)
│       ├── app.py                    # Streamlit app orchestrator
│       ├── common.py                 # Shared UI utilities
│       ├── page_data_explorer.py     # Data exploration page
│       ├── page_performance_metrics.py # Model evaluation page
│       ├── page_upload_infer.py      # Upload & inference page
│       └── assets/                   # Static files (images, CSS, etc.)
│
├── services/                         # Legacy Docker entry points (delegate to package)
│   ├── preprocessing/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── preprocess.py             # Thin wrapper calling celesify.preprocessing
│   ├── training/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── train.py                  # Thin wrapper calling celesify.training
│   └── streamlit/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app.py                    # Thin wrapper calling celesify.streamlit_app
│
├── data/
│   └── raw/                          # Place the downloaded SDSS17 CSV here
├── outputs/                          # Shared Docker volume mount point
│   ├── processed/                    # Parquet files from preprocessing service
│   │   ├── train.parquet             # 28 engineered features (80% of data)
│   │   ├── test.parquet              # 28 engineered features (20% of data)
│   │   ├── train_clean.parquet       # 8 original features (80% of data)
│   │   ├── test_clean.parquet        # 8 original features (20% of data)
│   │   └── preprocessing_report.json # Metadata: class counts, feature stats, decisions
│   └── models/                       # Model artifacts from training service
│       ├── model.joblib              # Tuned RF (engineered features)
│       ├── model_baseline.joblib     # Baseline RF (clean features)
│       ├── model_clean_tuned.joblib  # Tuned RF (clean features)
│       ├── model.onnx                # ONNX export of tuned RF (engineered)
│       ├── baseline_metrics.json     # Baseline performance (clean features)
│       ├── clean_tuned_metrics.json  # Tuned performance (clean features)
│       ├── tuned_metrics.json        # Tuned performance (engineered features)
│       ├── best_params.json          # Best hyperparams (engineered search)
│       ├── best_params_clean_tuned.json # Best hyperparams (clean search)
│       ├── feature_importance.json   # MDI scores from tuned engineered model
│       ├── top_trials.json           # Full search history (engineered)
│       ├── top_trials_clean_tuned.json # Full search history (clean)
│       ├── onnx_export_status.json   # ONNX export validation status
│       └── onnx_export_error.log     # ONNX export error details (if any)
│
└── figures/                          # Exported figures at 300 DPI for papers/reports
```

**Key insight:** All Python code lives in `celesify/` as a unified package. The `services/` 
Dockerfiles only provide container orchestration and delegate execution to the package. This 
allows identical code to run locally (via Rye) or in Docker (via compose), and makes refactoring 
easier without duplicating logic across service entry points.

---

## Architecture

Three Docker services communicate only via a shared named volume (`outputs/`). No
networking or API layer between preprocessing and training.

| Service          | Base image              | Reads from        | Writes to              |
|------------------|-------------------------|-------------------|------------------------|
| `preprocessing`  | python:3.11-slim        | `data/raw/*.csv`  | `outputs/processed/`   |
| `training`       | python:3.11-slim (CPU sklearn) | `outputs/processed/` | `outputs/models/`  |
| `streamlit`      | python:3.11-slim        | `outputs/models/` | —                      |

To run the full stack:

```bash
docker compose up
```

No other setup required.

### Runtime direction update (April 2026)

The project is now intentionally CPU-first for training (`scikit-learn` Random Forest with
`n_jobs=-1`) and no longer depends on CUDA/cuML for the default workflow.

Rationale for moving away from CUDA/GPU in this project:
- Dataset size and model family (`~100k` rows, tabular RF) did not produce enough wall-clock
  gains to offset engineering overhead.
- CUDA image pulls, environment bootstrapping, and container rebuild times increased the
  iteration loop for development and debugging.
- GPU-specific configuration (toolkit/runtime/version compatibility) reduced portability across
  machines and contributor setups.
- CPU-only sklearn runs are reproducible, simpler to maintain, and fast enough for the current
  tuning/search scope.

This tradeoff prioritizes total developer productivity and reliability over peak hardware
throughput.

### Developer environment direction update (April 2026)

The development container is standardized on a lean Python image rather than a CUDA/NVIDIA base
image. The goal is faster rebuilds, lower image pull size, and fewer environment-specific
failures during day-to-day development.

Dev-container SSH setup direction:
- Use SSH agent forwarding into the container instead of copying private keys into the image.
- Keep key material on the host and expose only the agent socket in the dev-container runtime.
- Validate access with a non-destructive check (for example `ssh -T git@github.com`) after
  container startup.

Shell experience direction:
- Install and enable Starship in the dev-container for a consistent prompt across local and
  container sessions.
- Keep Starship configuration lightweight and commit-safe (no machine-specific secrets).

Python project management direction:
- Adopt Rye as the project manager for dependency resolution, lockfile control, virtual
  environment management, and task execution.
- Prefer checked-in lockfiles and deterministic sync/install steps for reproducibility.
- Use Rye-managed scripts/tasks so local runs and CI runs execute the same commands.

Rationale for adopting Rye:
- Improves reproducibility with consistent, locked dependency graphs.
- Reduces bootstrap drift between contributors and CI/deployment environments.
- Supports deployable workflows by making build/install behavior deterministic and repeatable.
- Simplifies Python toolchain management (interpreter + environment + dependencies) in one place.

---

## Implementation phases

### Phase 1 — Environment & repo setup (Days 1–2)
- [x] Initialize Git repo with the structure above
- [x] Write `docker-compose.yml` with three service definitions and shared volume
- [x] Create skeleton `Dockerfile` + `requirements.txt` for each service
- [x] Verify `docker compose up` completes without errors before writing model code
- [x] Pin all dependency versions
- [x] Switch dev-container from NVIDIA/CUDA image to a lean Python development image
- [x] Configure SSH agent forwarding in dev-container workflow (no private key copy into container)
- [x] Add Starship prompt setup in dev-container bootstrap
- [x] Standardize Python dependency/tooling workflow on Rye for reproducible local/CI execution

### Phase 2 — Data ingestion & preprocessing (Days 2–3) ✅ COMPLETE
Work is organized into focused submodules within `celesify/preprocessing/`:
- [loading.py](celesify/preprocessing/loading.py) — Phase 1: CSV discovery & Kaggle download
- [cleaning.py](celesify/preprocessing/cleaning.py) — Phase 2: Schema validation, missing values, target encoding
- [features.py](celesify/preprocessing/features.py) — Phase 3: Stratified split & feature engineering
- [exports.py](celesify/preprocessing/exports.py) — Phase 4: Parquet export & report generation
- [pipeline.py](celesify/preprocessing/pipeline.py) — Orchestrator: imports & coordinates all phases

Entry point: `celesify/preprocessing/main.py` (run via `rye run preprocess` or Docker)

**Core preprocessing steps**
- [x] Load SDSS17 CSV with pandas; print shape and class distribution
- [x] Drop non-informative columns: object IDs, plate numbers, fiber IDs, MJD
- [x] Inspect and handle missing values (strategy: drop rows with any missing value)
- [x] Encode target labels (`STAR`, `GALAXY`, `QSO` → `0`, `1`, `2`)
- [x] Inspect photometric bands for skew; log summary but do not apply transforms
- [x] Stratified 80/20 train/test split — preserve class proportions
- [x] Add Kaggle API fallback download when no local CSV exists (`fedesoriano/stellar-classification-dataset-sdss17`)
- [x] Save preprocessing metadata report to `outputs/processed/preprocessing_report.json`

**Feature engineering (exceeds original plan)**

After identifying class separability issues in raw 8-feature space, advanced feature engineering 
was implemented to improve model expressiveness:

| Feature set | File (train/test) | Features | Purpose |
|---|---|---|---|
| **Clean** (8) | `train_clean.parquet` / `test_clean.parquet` | `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift` | Baseline RF training; sanity-check for model quality before engineering |
| **Engineered** (28) | `train.parquet` / `test.parquet` | Clean 8 + 9 color features + 5 band statistics + 5 redshift-color interactions | Primary feature set; supports tuned RF on richer feature space |

**Engineered features** include:
- **Color features** (9): `u-g`, `g-r`, `r-i`, `i-z`, `u-r`, `g-i`, `g-z`, `r-z`, `u-z` — contrast relative magnitudes to improve separation
- **Band statistics** (5): mean, std, min, max, range across photometric bands — capture magnitude variability
- **Redshift-color interactions** (5): products of redshift with key color pairs — encode redshift-dependent class properties

The engineered feature set is derived from:
- [Carrasco et al. (2015)](https://arxiv.org/abs/1405.5298): RF + color features for photometric quasar classification
- [Wu et al. (2012)](https://arxiv.org/abs/1204.6197): color-color and redshift-color criteria for quasar separation
- [Hickox et al. (2017)](https://arxiv.org/abs/1709.04468): optical-IR colors for obscured quasar identification

**Phase 2 documented decisions:**
- Missing-value strategy: drop rows (< 1% affected; no systematic bias detected)
- Imbalance handling direction: `class_weight='balanced'` recommended when majority/minority ratio > 2.0 (applies to this dataset)
- Transform policy: compute and log skew for numeric columns; no log transform applied (distributions not severely skewed)
- Feature engineering: build explicit color features + statistical summaries + redshift interactions from raw magnitudes
- Module design: separate concerns into focused submodules for clarity, testability, and maintainability

### Phase 3 — Modelling & tuning (Days 3–6) ✅ COMPLETE
Work happens in `celesify/training/pipeline.py` (entry point: `celesify/training/main.py`).

**Training workflow**

The pipeline executes **three sequential RF training experiments** on different feature sets:

1. **Baseline RF (clean features only)**
   - Trains `RandomForestClassifier` with sklearn defaults (`n_estimators=100`, no max_depth, no class weighting)
   - Purpose: establish a performance ceiling on minimal feature engineering
   - Artifacts: `model_baseline.joblib`, `baseline_metrics.json`

2. **Tuned RF (clean features)**
   - Runs `RandomizedSearchCV` over the same feature set (clean 8 features)
   - Search space: `n_estimators` [100, 200, 300, 500], `max_depth` [None, 10, 20, 30], `min_samples_split` [2, 5, 10], `max_features` ['sqrt', 'log2', 0.3], `class_weight` [None, 'balanced']
   - CV: 5-fold stratified k-fold, `n_iter=20` (configurable via `TRAINING_N_ITER`)
   - Purpose: quantify tuning gains on baseline feature set
   - Artifacts: `model_clean_tuned.joblib`, `clean_tuned_metrics.json`, `best_params_clean_tuned.json`

3. **Tuned RF (engineered features) — PRIMARY MODEL**
   - Runs `RandomizedSearchCV` over the engineered 28-feature set
   - Same search space and CV as Step 2
   - Purpose: primary model for deployment; captures engineering improvements
   - Artifacts: `model.joblib`, `tuned_metrics.json`, `best_params.json`, `feature_importance.json`

**Feature importance & export**

- [x] Extract Mean Decrease Impurity (MDI) scores from the final tuned engineered model
- [x] Rank features and save to `outputs/models/feature_importance.json` (used by Streamlit dashboard)
- [x] Export ONNX artifact from tuned engineered model: `outputs/models/model.onnx`
- [x] Validate ONNX with checker and test-load in runtime; save status to `onnx_export_status.json`

**Implementation notes**

- All training uses sklearn CPU execution (`n_jobs=-1`) with `random_state=42` for reproducibility
- Baseline-first ordering is strict: baseline metrics saved before any hyperparameter search begins
- Imbalance handling: `class_weight='balanced'` is included in the search space and used when selected by CV
- ONNX export uses verified stack: `onnx==1.21.0`, `skl2onnx==1.20.0`, `onnxruntime==1.21.0`
- Environment overrides supported (for quick validation): `TRAINING_N_ITER`, `TRAINING_CV_SPLITS`, `TRAINING_N_JOBS`, `TRAINING_MAX_TRAIN_ROWS`
- Quick smoke test: set `TRAINING_N_ITER=1`, `TRAINING_CV_SPLITS=2`, `TRAINING_N_JOBS=1`, `TRAINING_MAX_TRAIN_ROWS=3000`

**Phase 3 validation outcome:**
- ✅ All three models train end-to-end without errors
- ✅ Metrics and best params consistently saved before final export
- ✅ ONNX artifact is valid, checker-passes, and loads in `onnxruntime`
- ✅ Feature importance extracted and ranked from engineered model
- ✅ If ONNX export fails, details written to `onnx_export_error.log` with non-blocking status

### Phase 4 — Streamlit dashboard (Days 6–8) ✅ COMPLETE
Work happens in `celesify/streamlit_app/` (entry point: `celesify/streamlit_app/main.py`).

All three dashboard pages are implemented and load models/artifacts from the shared volume at startup:

1. **Data Explorer** (`page_data_explorer.py`)
   - Class distribution bar chart
   - Per-feature histograms (raw data)
   - Correlation heatmap of photometric bands
   - Dataset summary statistics

2. **Model Evaluation** (`page_performance_metrics.py`)
   - Confusion matrix heatmap (normalized and raw counts)
   - Per-class precision/recall/F1 tables (baseline vs. clean-tuned vs. engineered-tuned)
   - Feature importance bar chart (MDI scores from tuned engineered model)
   - Side-by-side comparison of baseline vs. tuned metrics

3. **Upload & Infer** (`page_upload_infer.py`)
   - Accept feature input: either manual slider input for all 28 features or CSV upload
   - Return predicted class and per-class probability scores
   - Uses the primary tuned engineered model

**Phase 4 validation outcome:**
- ✅ All three pages load without errors
- ✅ Models and metrics correctly loaded from shared volume
- ✅ All figures regenerated from saved JSON artifacts (not recomputed from training data)
- ✅ Inference returns valid predictions and probabilities

### Phase 5 — Analysis, writing & submission (Days 8–12)
- [ ] Export all figures at 300 DPI to `figures/` for paper inclusion
- [ ] Fill in paper sections IV–VI using actual result values
- [ ] Compare baseline vs tuned: report delta in percentage points, not just absolute values
- [ ] Address the computational constraint explicitly: document what search space was feasible and what was left out
- [ ] Final README: must cover `docker compose up` as the only required step
- [ ] Freeze `requirements.txt` files from the running containers

---

## Local Development Setup

The project uses **Rye** for reproducible Python dependency and environment management, plus **Taskfile** for convenience task runners.

**Initial setup (one-time):**

1. Install [Rye](https://rye.astral.sh/) if not already installed
2. Sync the dev environment (includes all optional dependency groups):
   ```bash
   rye sync --all-features
   ```
   Or sync specific feature groups:
   ```bash
   rye sync --features preprocessing
   rye sync --features training-core --features training-onnx
   rye sync --features streamlit
   ```

**Running the pipeline locally (instead of Docker):**

```bash
# Preprocessing
rye run preprocess

# Training (baseline + tuned models)
rye run train

# Streamlit dashboard (runs on http://localhost:8501)
rye run streamlit-app
```

**Using Taskfile for convenience:**

```bash
# View available tasks
task -l

# Sync environment for specific phase
task sync-preprocess
task sync-train
task sync-streamlit

# Or run full sync
task
```

**Key insight:** Running locally via Rye executes identical Python code as the Docker services,
so validation and debugging can happen without container overhead. The `pyproject.toml`
defines all dependencies and entry points; Rye manages the virtual environment and lockfile.

**Docker workflow (production):**

```bash
docker compose up
```

This orchestrates all three services via shared named volume. No additional setup needed
beyond having Docker installed and credentials for Kaggle (if the CSV is not locally present).

---

## Key libraries

| Library             | Feature group      | Purpose                                      |
|---------------------|-------------------|----------------------------------------------|
| numpy, pandas       | core              | Numerical and data frame operations          |
| pyarrow             | core              | Parquet file I/O                             |
| kaggle              | preprocessing     | Kaggle API fallback for dataset download     |
| scikit-learn        | training-core     | RF model, GridSearchCV, metrics              |
| imbalanced-learn    | training-core     | Class weight balancing utilities             |
| joblib              | training-core     | Model serialization                          |
| onnx, skl2onnx      | training-onnx     | ONNX export and conversion                   |
| onnxruntime         | training-onnx     | ONNX model validation and inference          |
| streamlit           | streamlit         | Dashboard and interactive UI                 |
| plotly, matplotlib  | streamlit         | Interactive and static visualization         |
| seaborn             | streamlit         | Statistical visualization (heatmaps, etc.)   |

All versions are pinned in `pyproject.toml` for reproducibility.

---

## Evaluation metrics

- **Primary:** overall accuracy + macro-averaged F1 (both required; accuracy alone is
  misleading under class imbalance)
- **Secondary:** per-class confusion matrix; per-class precision, recall, F1
- **Feature analysis:** MDI scores ranked; SHAP values if computed

Always report baseline and tuned metrics together in the same table.

---

## Conventions

- All random states use `random_state=42` for reproducibility
- Log every major step to stdout with a timestamp; Docker captures this automatically
- Save intermediate artifacts (metrics JSON, feature importance JSON) before saving the
  model — if training crashes during export, metrics are preserved
- Do not commit the raw CSV or any model artifacts to Git; add `data/` and `outputs/` to
  `.gitignore`
- The paper's computational constraint narrative depends on honest documentation of what
  `n_iter` was actually used in the search and why — record this in code comments

---

## References

- fedesoriano (2022). Stellar Classification Dataset – SDSS17. Kaggle.
- Baron, D. (2019). Machine Learning in Astronomy: A Practical Overview. NASA/IPAC.
- Kavlakoglu, E. (n.d.). What is Random Forest? IBM.
- NASA Goddard Space Flight Center (2025). Nancy Grace Roman Space Telescope.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
