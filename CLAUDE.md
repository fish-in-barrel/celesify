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

```
.
├── CLAUDE.md                    # This file
├── docker-compose.yml           # Defines all three services + shared volume
├── data/
│   └── raw/                     # Place the downloaded SDSS17 CSV here
├── services/
│   ├── preprocessing/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── preprocess.py        # Ingests CSV, cleans, splits, writes Parquet
│   ├── training/
│   │   ├── Dockerfile           # Built on NVIDIA CUDA base; falls back to CPU
│   │   ├── requirements.txt
│   │   └── train.py             # Baseline RF, hyperparameter search, export
│   └── streamlit/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app.py               # Data explorer, results dashboard, upload + infer
├── outputs/                     # Shared Docker volume mount point (see below)
│   ├── processed/               # Parquet files written by preprocessing service
│   └── models/                  # joblib and ONNX model artifacts
└── figures/                     # Exported matplotlib/seaborn figures for the paper
```

---

## Architecture

Three Docker services communicate only via a shared named volume (`outputs/`). No
networking or API layer between preprocessing and training.

| Service          | Base image              | Reads from        | Writes to              |
|------------------|-------------------------|-------------------|------------------------|
| `preprocessing`  | python:3.11-slim        | `data/raw/*.csv`  | `outputs/processed/`   |
| `training`       | nvidia/cuda (+ CPU fallback) | `outputs/processed/` | `outputs/models/`  |
| `streamlit`      | python:3.11-slim        | `outputs/models/` | —                      |

To run the full stack:

```bash
docker compose up
```

No other setup required. GPU acceleration is opt-in via the NVIDIA Container Toolkit;
hosts without it run on CPU automatically.

---

## Implementation phases

### Phase 1 — Environment & repo setup (Days 1–2)
- [x] Initialize Git repo with the structure above
- [x] Write `docker-compose.yml` with three service definitions and shared volume
- [x] Create skeleton `Dockerfile` + `requirements.txt` for each service
- [x] Verify `docker compose up` completes without errors before writing model code
- [x] Pin all dependency versions

### Phase 2 — Data ingestion & preprocessing (Days 2–3)
Work happens in `services/preprocessing/preprocess.py`.

- [x] Load SDSS17 CSV with pandas; print shape and class distribution
- [x] Drop non-informative columns: object IDs, plate numbers, fiber IDs, MJD
- [x] Inspect and handle missing values (log strategy: imputation or row removal)
- [x] Encode target labels (`STAR`, `GALAXY`, `QSO` → integers)
- [x] Apply any needed transformations (check photometric bands for skew; log-scale if needed)
- [x] Stratified 80/20 train/test split — preserve class proportions
- [x] Write train and test sets to `outputs/processed/` as Parquet files
- [x] Log class counts in both splits to confirm stratification worked
- [x] Add Kaggle API fallback download when no local CSV exists (`fedesoriano/stellar-classification-dataset-sdss17`)
- [x] Save preprocessing metadata report to `outputs/processed/preprocessing_report.json`

**Decision to make early:** if quasars are significantly underrepresented, decide whether
to use `class_weight='balanced'` in the RF or SMOTE via imbalanced-learn. This affects
how F1 is reported in the paper. Make the call here and document it.

**Phase 2 documented decisions:**
- Missing-value strategy: drop rows with missing values.
- Imbalance handling direction: recommend `class_weight='balanced'` when majority/minority ratio > 2.0.
- Transform policy: compute and log skew for numeric columns; no log transform currently applied.

### Phase 3 — Modelling & tuning (Days 3–6)
Work happens in `services/training/train.py`.

**Phase 3 status update:**
- Phase 3 is implemented and validated end-to-end on the generated parquet inputs.
- Training inputs are produced at `outputs/processed/train.parquet` and `outputs/processed/test.parquet` inside the shared Docker volume mount (`/workspace/outputs/processed/`).
- `outputs/` is a named Docker volume in compose; artifacts are not written to host `./outputs` by default unless the script is run locally outside Docker.
- Preprocessing metadata is available at `outputs/processed/preprocessing_report.json` and includes class counts, proportions, skew metrics, and imbalance recommendation.
- Target encoding remains fixed as `STAR=0`, `GALAXY=1`, `QSO=2`.
- The imbalance signal remains greater than 2.0, so `class_weight='balanced'` stays enabled in the tuning search space.
- Feature set after preprocessing drop step is `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`, plus encoded `class`.
- Random seed continuity remains `random_state=42` in all Phase 3 training/splitting/CV logic.
- cuML is now installed in the rebuilt training image and imports successfully from `cuml.ensemble`.

**Phase 3 implementation notes learned during validation:**
- Quick smoke tests are supported through environment overrides in `train.py`: `TRAINING_N_ITER`, `TRAINING_CV_SPLITS`, `TRAINING_N_JOBS`, and `TRAINING_MAX_TRAIN_ROWS`.
- Keep the defaults for full runs, but use the quick-test overrides for local validation and ONNX debugging.
- The training container now prefers cuML when available and falls back to sklearn only if the import fails.
- The tuned-model workflow now uses a two-stage pattern: broad search on cuML (GPU) when available, then CPU sklearn confirmation/refit on shortlisted candidates before final artifact export.
- If GPU/cuML broad search fails at runtime, training automatically falls back to sklearn broad search to keep the pipeline running.
- Candidate shortlist size for CPU confirmation is configurable with `TRAINING_TOP_K` (default 3).
- Docker GPU access is configured through `gpus: all` plus `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility` in `docker-compose.yml`.
- ONNX export initially failed with the older converter stack because `skl2onnx` could not serialize the current RandomForest layout correctly.
- The validated training stack for ONNX export is `onnx==1.21.0`, `onnxruntime==1.21.0`, and `skl2onnx==1.20.0`.
- The training image now uses `nvidia/cuda:12.4.1-devel-ubuntu22.04` by default, and `services/training/requirements.txt` pins `cuml-cu12==26.2.0` for GPU RF support.
- `onnxruntime` is now included in `services/training/requirements.txt` so the exported model can be smoke-tested in the same environment.
- The exported ONNX file was validated with both the ONNX checker and a runtime session load.

**Phase 3 implementation guardrails for next agent:**
- Keep baseline-first order strict: complete baseline training + artifact writes before running any search.
- Persist baseline/tuned metrics and best params JSON before final model export.
- Ensure confusion matrix and per-class metrics use the encoded class IDs consistently with the preprocessing mapping.
- Record the actual `n_iter` used for `RandomizedSearchCV` in code/comments and saved outputs for paper reproducibility.
- If you need a fast validation run, set `TRAINING_N_ITER=1`, `TRAINING_CV_SPLITS=2`, `TRAINING_N_JOBS=1`, and `TRAINING_MAX_TRAIN_ROWS=3000`.
- For parity-focused runs, keep final reported metrics/artifacts tied to the CPU sklearn refit stage, even when broad search ran on GPU.

**Step 1 — Baseline model**
- [x] Load Parquet from shared volume
- [x] Train `RandomForestClassifier` with sklearn defaults (`n_estimators=100`, no max_depth)
- [x] Evaluate on test set: overall accuracy, macro-averaged F1
- [x] Generate and save confusion matrix
- [x] Save baseline metrics to `outputs/models/baseline_metrics.json`

Do not touch hyperparameters until the baseline is fully evaluated and saved.

**Step 2 — Hyperparameter search**
- [x] Use `RandomizedSearchCV` with `cv=5` (stratified k-fold) and `n_jobs=-1`
- [x] Search over:
  - `n_estimators`: [100, 200, 300, 500]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `max_features`: ['sqrt', 'log2', 0.3]
  - `class_weight`: [None, 'balanced'] — if imbalance was flagged in Phase 2
- [x] Set `n_iter` based on available time/hardware (default 20; quick-test override supported)
- [x] Save best params to `outputs/models/best_params.json`
- [x] Evaluate tuned model on test set; save metrics to `outputs/models/tuned_metrics.json`

**Step 3 — Feature importance**
- [x] Extract MDI scores from the trained RF (`estimator.feature_importances_`)
- [x] Rank and save to `outputs/models/feature_importance.json`
- [ ] If SHAP is installed and time permits, compute SHAP values and save separately

**Step 4 — Export**
- [x] Save final model as `outputs/models/model.joblib`
- [x] Export ONNX artifact as `outputs/models/model.onnx`

**GPU note:** `train.py` should attempt `from cuml.ensemble import RandomForestClassifier`
at runtime and fall back to `from sklearn.ensemble import RandomForestClassifier` silently
if cuML is unavailable. The API is nearly identical; the fallback requires no other code changes.

**Validation outcome:**
- ONNX artifact `outputs/models/model.onnx` is valid, checker-passes, and loads in `onnxruntime`.
- If ONNX export fails again, the script now writes details to `outputs/models/onnx_export_error.log` and a compact status file, rather than flooding the terminal.

### Phase 4 — Streamlit dashboard (Days 6–8)
Work happens in `services/streamlit/app.py`. Load the model artifact from the shared
volume at startup; all inference runs in-process.

Three views:

1. **Data explorer** — class distribution bar chart, per-feature histograms, correlation
   heatmap of photometric bands
2. **Results dashboard** — confusion matrix heatmap, per-class precision/recall/F1 table,
   feature importance bar chart (baseline vs tuned side-by-side)
3. **Upload & infer** — accept a CSV or manual input of feature values; return predicted
   class and per-class probability

- [ ] Confirm the app loads the model and returns predictions before building UI
- [ ] All figures should be reproducible from the saved metric/importance JSON files
  (not regenerated from training data at runtime)

### Phase 5 — Analysis, writing & submission (Days 8–12)
- [ ] Export all figures at 300 DPI to `figures/` for paper inclusion
- [ ] Fill in paper sections IV–VI using actual result values
- [ ] Compare baseline vs tuned: report delta in percentage points, not just absolute values
- [ ] Address the computational constraint explicitly: document what search space was
  feasible and what was left out
- [ ] Final README: must cover `docker compose up` as the only required step
- [ ] Freeze `requirements.txt` files from the running containers

---

## Key libraries

| Library             | Service        | Purpose                                      |
|---------------------|----------------|----------------------------------------------|
| pandas              | preprocessing  | CSV ingestion, cleaning, splitting           |
| numpy               | preprocessing  | Numerical operations                         |
| scikit-learn        | training       | RF model, GridSearchCV, metrics              |
| cuML (optional)     | training       | Drop-in GPU-accelerated RF                   |
| imbalanced-learn    | training       | SMOTE if class imbalance warrants it         |
| SHAP (optional)     | training       | Model-agnostic feature importance            |
| joblib              | training       | Model serialization                          |
| seaborn / matplotlib| streamlit      | All visualization                            |
| streamlit           | streamlit      | Dashboard and upload interface               |

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
- NVIDIA Corporation (2024). cuML v24.10.
