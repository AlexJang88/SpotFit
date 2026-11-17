from __future__ import annotations

import logging
import re
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging_config import reset_request_id, set_request_id, timestamp


logger = logging.getLogger("spotfit.api")
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
GENERIC_SERVER_ERROR = "요청을 처리하는 중 서버 오류가 발생했습니다. 관리자에게 문의해 주세요."
GENERIC_UNAVAILABLE_ERROR = "서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."


def _timestamp() -> str:
    return timestamp()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _payload(request: Request, status_code: int, detail, code: str) -> dict:
    return {
        "detail": jsonable_encoder(detail),
        "error": {
            "code": code,
            "status": status_code,
            "request_id": _request_id(request),
            "timestamp": _timestamp(),
        },
    }


def _exception_info(exception: BaseException):
    source = exception.__cause__ or exception
    return type(source), source, source.__traceback__


def install_error_handling(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "").strip()
        request_id = supplied if SAFE_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = set_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            # Nginx appends the direct peer; use the last value to avoid a spoofed leading entry.
            forwarded_for = request.headers.get("X-Forwarded-For", "").rsplit(",", 1)[-1].strip()
            log_level = logging.DEBUG if request.url.path == "/api/spot_fit" else logging.INFO
            logger.log(
                log_level,
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "client_ip": forwarded_for or (request.client.host if request.client else None),
                },
            )
            return response
        finally:
            reset_request_id(token)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exception: RequestValidationError):
        logger.warning(
            "request validation failed: %s",
            exception.errors(),
            extra={"method": request.method, "path": request.url.path, "status_code": 422},
        )
        return JSONResponse(
            status_code=422,
            content=_payload(request, 422, exception.errors(), "VALIDATION_ERROR"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exception: StarletteHTTPException):
        status_code = exception.status_code
        if status_code >= 500:
            logger.error(
                "service exception (%s): %s",
                status_code,
                exception.detail,
                exc_info=_exception_info(exception),
                extra={"method": request.method, "path": request.url.path, "status_code": status_code},
            )
            detail = GENERIC_UNAVAILABLE_ERROR if status_code == 503 else GENERIC_SERVER_ERROR
        else:
            logger.warning(
                "http exception (%s): %s",
                status_code,
                exception.detail,
                extra={"method": request.method, "path": request.url.path, "status_code": status_code},
            )
            detail = exception.detail
        return JSONResponse(
            status_code=status_code,
            content=_payload(request, status_code, detail, f"HTTP_{status_code}"),
            headers=exception.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exception: Exception):
        logger.error(
            "unhandled server exception",
            exc_info=_exception_info(exception),
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "request_id": _request_id(request),
            },
        )
        return JSONResponse(
            status_code=500,
            content=_payload(request, 500, GENERIC_SERVER_ERROR, "INTERNAL_SERVER_ERROR"),
            headers={"X-Request-ID": _request_id(request)},
        )
