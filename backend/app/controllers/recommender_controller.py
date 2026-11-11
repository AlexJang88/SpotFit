from fastapi import APIRouter, HTTPException, Query, status

from app.services.recommender_v1 import RecommenderV1Error, get_district_forecasts


router = APIRouter(prefix="/recommender", tags=["recommender_v1"])


@router.get("/district-forecasts")
def district_forecasts(district: str | None = Query(default=None)) -> dict:
    """Return results produced by every model delivered in recommender/v1."""
    try:
        payload = get_district_forecasts()
    except (RecommenderV1Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    if district:
        district_name = district.strip()
        selected = [row for row in payload["forecasts"] if row["district"] == district_name]
        if not selected:
            raise HTTPException(status_code=404, detail=f"자치구를 찾을 수 없습니다: {district}")
        neighborhoods = [
            row for row in payload["neighborhoods"] if row["district"] == district_name
        ]
        payload = {
            **payload,
            "district_count": 1,
            "forecasts": selected,
            "neighborhood_count": len(neighborhoods),
            "neighborhoods": neighborhoods,
        }
    return payload
