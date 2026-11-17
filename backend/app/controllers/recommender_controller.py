from fastapi import APIRouter, Header, HTTPException, Query, status

from app.services.recommender_v1 import RecommenderV1Error, get_district_forecasts
from app.services.crime_model_training import verify_admin_key
from app.services.traffic_accident_v51 import (
    TrafficAccidentV51Error,
    ensure_models as ensure_traffic_models,
    forecast_payload as traffic_forecast_payload,
    model_status as traffic_model_status,
)


router = APIRouter(prefix="/recommender", tags=["recommender_v1"])


@router.get("/model-status")
def recommender_model_status() -> dict:
    """Show the temporary traffic-v51 artifacts alongside recommender-v1."""
    return {"version": "v1", "traffic_accident": traffic_model_status()}


@router.post("/models/ensure", status_code=status.HTTP_202_ACCEPTED)
def ensure_recommender_models(
    force: bool = Query(default=False),
    admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key"),
) -> dict:
    """Create missing v51 traffic artifacts under models/recommender/v1."""
    try:
        verify_admin_key(admin_key)
        return {"version": "v1", "traffic_accident": ensure_traffic_models(force=force)}
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (RuntimeError, TrafficAccidentV51Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/traffic-accident/district-forecasts")
def traffic_accident_district_forecasts(
    horizon: int = Query(default=2, ge=2, le=3),
    district: str | None = Query(default=None),
) -> dict:
    """Return temporary model-51 traffic forecasts for 2027 or 2028."""
    try:
        payload = traffic_forecast_payload(horizon)
    except (TrafficAccidentV51Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if district:
        selected = [row for row in payload["forecasts"] if row["district"] == district.strip()]
        if not selected:
            raise HTTPException(status_code=404, detail=f"자치구를 찾을 수 없습니다: {district}")
        payload = {**payload, "district_count": 1, "forecasts": selected}
    return payload


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
