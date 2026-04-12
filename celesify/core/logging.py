from __future__ import annotations

from datetime import datetime, timezone


def log(service: str, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [{service}] {message}")
