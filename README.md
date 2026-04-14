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

Use `.envrc` for local secrets such as Kaggle credentials. The file is git-ignored.

If you are not using the VS Code Dev Container, install `direnv` first (or source `.envrc` manually).

Quick setup:

```bash
cp .envrc.template .envrc
# edit .envrc and set KAGGLE_USERNAME / KAGGLE_KEY
direnv allow    # or source .envrc manually
```

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
`