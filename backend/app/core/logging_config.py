from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
LOG_FILE_NAME = "spotfit-api.log"
_CONFIGURED = False


def log_directory() -> Path:
    configured = os.getenv("APP_LOG_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.cwd() / "logs" / "backend"


def log_file_name() -> str:
    configured = os.getenv("APP_LOG_FILE", "").strip()
    return Path(configured).name if configured else LOG_FILE_NAME


def log_file_path() -> Path:
    return log_directory() / log_file_name()


def set_request_id(value: str) -> Token[str]:
    return REQUEST_ID.set(value)


def reset_request_id(token: Token[str]) -> None:
    REQUEST_ID.reset(token)


def current_request_id() -> str:
    return REQUEST_ID.get()


def _timezone():
    name = os.getenv("APP_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name == "Asia/Seoul":
            return timezone(timedelta(hours=9), name="Asia/Seoul")
        return datetime.now().astimezone().tzinfo


def timestamp(created: float | None = None) -> str:
    value = datetime.now().timestamp() if created is None else created
    return datetime.fromtimestamp(value, tz=_timezone()).isoformat(timespec="milliseconds")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = current_request_id()
        return True


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line so logs remain readable and API-filterable."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for field in ("method", "path", "status_code", "duration_ms", "client_ip"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class LocalTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return timestamp(record.created)


def configure_logging() -> Path:
    global _CONFIGURED
    path = log_file_path()
    if _CONFIGURED:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    retention = max(1, int(os.getenv("APP_LOG_RETENTION_DAYS", "30")))
    context_filter = RequestContextFilter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        LocalTimeFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"
        )
    )
    console.addFilter(context_filter)

    file_handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        interval=1,
        backupCount=retention,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonLogFormatter())
    file_handler.addFilter(context_filter)

    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(handler, "_spotfit_handler", False) for handler in root.handlers):
        console._spotfit_handler = True  # type: ignore[attr-defined]
        file_handler._spotfit_handler = True  # type: ignore[attr-defined]
        root.addHandler(console)
        root.addHandler(file_handler)
    _CONFIGURED = True
    return path
