from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.schemas.crime_safety import CrimeSafetyForecastResponse
from app.services.child_crime_exposure import ChildCrimeExposureError, forecast_payload
from app.services.crime_model_training import start_training, training_status, verify_admin_key
from app.services.recommender_v1 import RecommenderV1Error
from app.services.safety_category_scores import get_safety_category_scores


router = APIRouter(prefix="/safety/crime", tags=["crime_safety"])


@router.get("/category-scores")
def category_scores(horizon: int = Query(default=1, ge=1, le=2)) -> dict:
    """교통사고·범죄·안전 인프라의 자치구별 백분위 점수를 반환한다."""
    try:
        return get_safety_category_scores(horizon)
    except (ChildCrimeExposureError, RecommenderV1Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@router.get("/district-forecasts", response_model=CrimeSafetyForecastResponse)
def district_forecasts(
    district: str | None = Query(default=None, description="선택한 서울 자치구만 반환"),
    horizon: int = Query(default=1, ge=1, le=2, description="1년 또는 2년 뒤 예측"),
) -> CrimeSafetyForecastResponse:
    """서울 25개 구의 아동 지역범죄 노출 예측점수와 상대순위를 반환한다."""
    try:
        payload = forecast_payload(horizon)
    except (ChildCrimeExposureError, OSError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error

    if district is not None:
        selected = [item for item in payload["forecasts"] if item["district"] == district.strip()]
        if not selected:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"서울 자치구를 찾을 수 없습니다: {district}",
            )
        payload["forecasts"] = selected
        payload["district_count"] = 1
    return CrimeSafetyForecastResponse.model_validate(payload)


@router.get("/model-status")
def crime_model_status() -> dict:
    return training_status()


@router.post("/models/train", status_code=status.HTTP_202_ACCEPTED)
def train_crime_models(
    force: bool = Query(default=False, description="모델이 있어도 강제로 다시 학습"),
    admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key"),
) -> dict:
    try:
        verify_admin_key(admin_key)
        return start_training(force=force)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409 if "실행 중" in str(error) else 503, detail=str(error)) from error

