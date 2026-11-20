from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query

from app.services.admin_logs import read_logs
from app.services.crime_model_training import verify_admin_key


router = APIRouter(prefix="/admin/logs", tags=["admin_logs"])


@router.get("")
def logs(
    limit: int = Query(default=200, ge=1, le=1000),
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_date: date | None = Query(default=None, alias="date"),
    admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key"),
) -> dict:
    """Return newest persistent API logs. Requires the model administrator key."""
    try:
        verify_admin_key(admin_key)
        return read_logs(limit=limit, minimum_level=level, log_date=log_date)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail="관리자 인증에 실패했습니다.") from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail="관리자 로그 조회 API가 비활성화되어 있습니다.") from error
