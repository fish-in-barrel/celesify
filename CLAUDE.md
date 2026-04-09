# CLAUDE.md — SDSS17 Stellar Classification

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
- [ ] Initialize Git repo with the structure above
- [ ] Write `docker-compose.yml` with three service definitions and shared volume
- [ ] Create skeleton `Dockerfile` + `requirements.txt` for each service
- [ ] Verify `docker compose up` completes without errors before writing model code
- [ ] Pin all dependency versions

### Phase 2 — Data ingestion & preprocessing (Days 2–3)
Work happens in `services/preprocessing/preprocess.py`.

- [ ] Load SDSS17 CSV with pandas; print shape and class distribution
- [ ] Drop non-informative columns: object IDs, plate numbers, fiber IDs, MJD
- [ ] Inspect and handle missing values (log strategy: imputation or row removal)
- [ ] Encode target labels (`STAR`, `GALAXY`, `QSO` → integers)
- [ ] Apply any needed transformations (check photometric bands for skew; log-scale if needed)
- [ ] Stratified 80/20 train/test split — preserve class proportions
- [ ] Write train and test sets to `outputs/processed/` as Parquet files
- [ ] Log class counts in both splits to confirm stratification worked

**Decision to make early:** if quasars are significantly underrepresented, decide whether
to use `class_weight='balanced'` in the RF or SMOTE via imbalanced-learn. This affects
how F1 is reported in the paper. Make the call here and document it.

### Phase 3 — Modelling & tuning (Days 3–6)
Work happens in `services/training/train.py`.

**Step 1 — Baseline model**
- [ ] Load Parquet from shared volume
- [ ] Train `RandomForestClassifier` with sklearn defaults (`n_estimators=100`, no max_depth)
- [ ] Evaluate on test set: overall accuracy, macro-averaged F1
- [ ] Generate and save confusion matrix
- [ ] Save baseline metrics to `outputs/models/baseline_metrics.json`

Do not touch hyperparameters until the baseline is fully evaluated and saved.

**Step 2 — Hyperparameter search**
- [ ] Use `RandomizedSearchCV` with `cv=5` (stratified k-fold) and `n_jobs=-1`
- [ ] Search over:
  - `n_estimators`: [100, 200, 300, 500]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `max_features`: ['sqrt', 'log2', 0.3]
  - `class_weight`: [None, 'balanced'] — if imbalance was flagged in Phase 2
- [ ] Set `n_iter` based on available time/hardware (start at 20; increase if feasible)
- [ ] Save best params to `outputs/models/best_params.json`
- [ ] Evaluate tuned model on test set; save metrics to `outputs/models/tuned_metrics.json`

**Step 3 — Feature importance**
- [ ] Extract MDI scores from the trained RF (`estimator.feature_importances_`)
- [ ] Rank and save to `outputs/models/feature_importance.json`
- [ ] If SHAP is installed and time permits, compute SHAP values and save separately

**Step 4 — Export**
- [ ] Save final model as `outputs/models/model.joblib`
- [ ] Export ONNX artifact as `outputs/models/model.onnx`

**GPU note:** `train.py` should attempt `from cuml.ensemble import RandomForestClassifier`
at runtime and fall back to `from sklearn.ensemble import RandomForestClassifier` silently
if cuML is unavailable. The API is nearly identical; the fallback requires no other code changes.

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
