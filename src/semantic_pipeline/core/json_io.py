"""Small, dependency-free JSON persistence helpers for visual metadata."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def write_json_atomically(path: str | Path, data: Any, overwrite: bool = False) -> None:
    output_path = Path(path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --resume or --force"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        last_error: PermissionError | None = None
        for attempt in range(5):
            try:
                os.replace(temporary_path, output_path)
                last_error = None
                break
            except PermissionError as error:
                last_error = error
                if attempt < 4:
                    time.sleep(0.25 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
