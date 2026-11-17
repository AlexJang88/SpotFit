from fastapi import APIRouter, Header, HTTPException, status

from app.services.crime_model_training import verify_admin_key
from app.services.service_model_bootstrap import bootstrap_status, start_model_bootstrap

router = APIRouter(prefix="/models", tags=["model_bootstrap"])


@router.get("/status")
def model_status() -> dict:
    return bootstrap_status()


@router.post("/bootstrap", status_code=status.HTTP_202_ACCEPTED)
def retry_bootstrap(admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key")) -> dict:
    try:
        verify_admin_key(admin_key)
        started = start_model_bootstrap(force=True)
        if not started:
            raise HTTPException(409, "모델 복구가 이미 실행 중이거나 비활성화되어 있습니다.")
        return bootstrap_status()
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/property/refresh")
def property_refresh_status() -> dict:
    from app.services.property_refresh_job import refresh_status
    return refresh_status()


@router.post("/property/refresh", status_code=status.HTTP_202_ACCEPTED)
def property_refresh(admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key")) -> dict:
    """Retrain the apartment sale / jeonse / wolse models from the newest raw data (called by the Airflow DAG)."""
    from app.services.property_refresh_job import start_refresh
    try:
        verify_admin_key(admin_key)
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except RuntimeError as error:                       # admin key not configured on the server
        raise HTTPException(503, str(error)) from error
    try:
        return start_refresh()
    except RuntimeError as error:                       # already running, or the script is missing
        raise HTTPException(409, str(error)) from error
