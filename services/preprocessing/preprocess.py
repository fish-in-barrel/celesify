from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [preprocessing] {message}")


def main() -> None:
    raw_dir = Path("/workspace/data/raw")
    processed_dir = Path("/workspace/outputs/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not csv_files:
        log("No CSV files found in /workspace/data/raw; scaffold run completed.")
        return

    log(f"Found {len(csv_files)} CSV file(s). Full preprocessing is not implemented yet.")


if __name__ == "__main__":
    main()
