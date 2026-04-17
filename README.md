# celesify

Stellar object classification pipeline for the SDSS17 dataset using a CPU-first Random Forest workflow.

The project runs as three Docker services:
- `preprocessing`: loads/cleans data and writes train/test parquet files
- `training`: trains baseline + tuned models and exports artifacts
- `streamlit`: serves a dashboard for results and inference

## Quickstart

1. Clone this repository and change into the project directory.
2. Open the folder in VS Code and choose **Reopen in Container** (recommended).
3. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` (recommended primary data path).
4. (Alternative) Put `star_classification.csv` in `data/raw/` if you prefer not to use Kaggle credentials.
5. Start the full stack:

```bash
docker compose up --build
```

6. Open Streamlit at `http://localhost:8501`.

That single `docker compose up --build` command is the primary run path for the full pipeline.

## VS Code Dev Container

This repository includes a Dev Container configuration in `.devcontainer/`.
If you open the project in VS Code and choose **Reopen in Container**, the development environment is provisioned for you (Python tooling, SSH client, and project setup scripts).

Notes:
- SSH auth inside the Dev Container works via agent forwarding from the host; keys are not copied into the container.
- You can validate GitHub SSH access with `ssh -T git@github.com` (GitHub may return exit code 1 even when authentication succeeds).

## Environment Files (.envrc)

Use `.envrc` for local environment variables such as Kaggle credentials and the
numeric UID/GID used by Docker Compose. The file is git-ignored.

If you are not using the VS Code Dev Container, install `direnv` first (or source `.envrc` manually).

Quick setup:

```bash
cp .envrc.template .envrc
# edit .envrc and set KAGGLE_USERNAME / KAGGLE_KEY if needed
direnv allow    # or source .envrc manually
```

The template also exports `LOCAL_UID` and `LOCAL_GID` from the current shell, so
`docker compose` runs the service containers with the same numeric user as your
developer session.

Preprocessing will use these values to download the dataset only when no CSV exists in `data/raw/`.

## Prerequisites

- Docker Engine (with Compose v2 support, i.e. `docker compose`)
- Network access to pull Docker base images
- Recommended workflow: VS Code with Dev Containers support
- Kaggle account with `KAGGLE_USERNAME` and `KAGGLE_KEY` in your shell environment (recommended primary data path)
- `direnv` installed when using `.envrc` outside the Dev Container

## System Requirements

- OS: Linux, macOS, or Windows with Docker Desktop/Engine
- CPU: Multi-core recommended (training uses parallel CPU jobs)
- RAM: 8 GB minimum, 16 GB recommended
- Disk: At least 5 GB free for images, containers, dataset, and artifacts
- No GPU required (CPU-first workflow)

## Container Resource Sizing (Dev + Training)

Use this section when choosing Docker/WSL resource limits for the Dev Container and the
`training` service.

### Minimum (works, but slower)

- CPU: 2 vCPU
- Memory: 6 GB RAM
- Swap: 2 GB
- Disk: 8 GB free

Expected behavior:
- Dev Container starts and basic editing/linting works.
- Preprocessing should complete reliably.
- Training completes, but hyperparameter search can be noticeably slower.

### Recommended (balanced day-to-day)

- CPU: 4-6 vCPU
- Memory: 10-12 GB RAM
- Swap: 4 GB
- Disk: 12-20 GB free

Expected behavior:
- Smooth Dev Container experience in VS Code.
- Good parallel performance for `RandomizedSearchCV` with sklearn (`n_jobs=-1`).
- Reasonable runtime for baseline + tuned training pipeline.

### Optimal (if your machine can spare it)

- CPU: 8+ vCPU
- Memory: 14-16 GB RAM
- Swap: 4-8 GB
- Disk: 20+ GB free

Expected behavior:
- Fastest local iteration for repeated training/tuning runs.
- Better responsiveness while running Streamlit and training in parallel.

### Recommendation For Your Current WSL Settings

Your current limits (`10 GB` RAM, `4 GB` swap, `8` CPU) are already strong for this
project and close to the practical sweet spot.

- Keep `10 GB / 4 GB / 8 CPU` for regular development and training.
- If you see host pressure (fan noise, system lag), reduce CPUs first (for example to 6)
  before reducing RAM.
- If training is the priority and your host remains responsive, increasing RAM to
  `12-14 GB` can improve stability during heavier parallel search runs.

### If You Need To Run On Lower Resources

Use training environment overrides to reduce load:

```bash
TRAINING_N_JOBS=1
TRAINING_N_ITER=5
TRAINING_CV_SPLITS=3
TRAINING_MAX_TRAIN_ROWS=30000
docker compose up --build
```

These reduce CPU and memory pressure at the cost of search quality and runtime fidelity.

## Project Layout

```text
.
├── celesify/                # Python package (preprocessing, training, streamlit app)
├── services/                # Dockerfiles per service
├── data/raw/                # Input dataset location
├── outputs/processed/       # Preprocessed parquet + reports
├── outputs/models/          # Metrics + model exports (joblib, ONNX)
├── docker-compose.yml
├── pyproject.toml
└── Taskfile.yml
```

## Outputs

The pipeline writes artifacts under `outputs/`:
- `outputs/processed/preprocessing_report.json`
- `outputs/models/baseline_metrics.json`
- `outputs/models/tuned_metrics.json`
- `outputs/models/best_params.json`
- `outputs/models/feature_importance.json`
- `outputs/models/model.joblib`
- `outputs/models/model.onnx`

## Local (Non-Docker) Development (Optional)

This repository also supports local runs with Rye and Task:

```bash
task sync-all
task run-preprocess
task run-train
task run-streamlit
```

Docker remains the default and recommended execution path.