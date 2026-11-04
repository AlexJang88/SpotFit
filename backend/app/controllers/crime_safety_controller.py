from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.crime_safety import CrimeSafetyForecastResponse
from app.services.crime_safety_forecast import CrimeSafetyForecastError, predict_all_districts
from app.services.recommender_v1 import RecommenderV1Error
from app.services.safety_category_scores import get_safety_category_scores


router = APIRouter(prefix="/safety/crime", tags=["crime_safety"])


@router.get("/category-scores")
def category_scores() -> dict:
    """교통사고·범죄·안전 인프라의 자치구별 백분위 점수를 반환한다."""
    try:
        return get_safety_category_scores()
    except (RecommenderV1Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@router.get("/district-forecasts", response_model=CrimeSafetyForecastResponse)
def district_forecasts(
    district: str | None = Query(default=None, description="선택한 서울 자치구만 반환"),
) -> CrimeSafetyForecastResponse:
    """서울 25개 구의 다음 연도 범죄안전 상대등급을 반환한다."""
    try:
        payload = predict_all_districts()
    except CrimeSafetyForecastError as error:
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

