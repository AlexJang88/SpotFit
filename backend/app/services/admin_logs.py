from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterator

from app.core.logging_config import LOG_FILE_NAME, log_directory, log_file_name


LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _log_files() -> list[Path]:
    directory = log_directory()
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.glob(f"{log_file_name()}*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _reverse_lines(path: Path, block_size: int = 64 * 1024) -> Iterator[str]:
    """Read a UTF-8 JSONL file newest-first without loading the whole file."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        remainder = b""
        while position > 0:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size) + remainder
            parts = chunk.split(b"\n")
            remainder = parts[0]
            for raw in reversed(parts[1:]):
                if raw:
                    yield raw.decode("utf-8", errors="replace")
        if remainder:
            yield remainder.decode("utf-8", errors="replace")


def read_logs(limit: int = 200, minimum_level: str = "INFO", log_date: date | None = None) -> dict:
    threshold = LEVELS[minimum_level]
    records: list[dict] = []
    for path in _log_files():
        for line in _reverse_lines(path):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if LEVELS.get(str(record.get("level")), 0) < threshold:
                continue
            timestamp = str(record.get("timestamp", ""))
            if log_date is not None and not timestamp.startswith(log_date.isoformat()):
                continue
            records.append(record)
            if len(records) >= limit:
                return {"count": len(records), "logs": records}
    return {"count": len(records), "logs": records}
