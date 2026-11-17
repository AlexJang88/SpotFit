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
