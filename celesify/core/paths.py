from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_preprocessing_paths() -> tuple[Path, Path]:
    repo_root = _repo_root()
    local_raw = repo_root / "data" / "raw"
    local_processed = repo_root / "outputs" / "processed"
    workspace_raw = Path("/workspace/data/raw")
    workspace_processed = Path("/workspace/outputs/processed")

    if any(local_raw.glob("*.csv")):
        return local_raw, local_processed

    cwd = Path.cwd().resolve()
    if cwd == Path("/workspace") and (workspace_raw.exists() or workspace_processed.parent.exists()):
        return workspace_raw, workspace_processed

    return local_raw, local_processed


def resolve_training_paths() -> tuple[Path, Path]:
    candidate_roots = [Path.cwd().resolve(), Path("/workspace"), _repo_root()]
    candidate_roots.extend(Path(__file__).resolve().parents)

    seen_roots: set[Path] = set()
    fallback: tuple[Path, Path] | None = None
    for root in candidate_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)

        outputs_root = root if root.name == "outputs" else root / "outputs"
        try:
            (outputs_root / "models").mkdir(parents=True, exist_ok=True)
            processed_dir = outputs_root / "processed"
            models_dir = outputs_root / "models"
            has_inputs = (processed_dir / "train.parquet").exists() and (processed_dir / "test.parquet").exists()
            if has_inputs:
                return processed_dir, models_dir
            if fallback is None:
                fallback = (processed_dir, models_dir)
        except (PermissionError, OSError, IndexError):
            continue

    if fallback is not None:
        return fallback

    raise RuntimeError("Unable to resolve writable outputs directory for training artifacts.")
