from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [training] {message}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    processed_dir = Path("/workspace/outputs/processed")
    models_dir = Path("/workspace/outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_dir / "train.parquet"
    test_file = processed_dir / "test.parquet"

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

    log("Found processed data files. Full model training implementation is pending.")


if __name__ == "__main__":
    main()
