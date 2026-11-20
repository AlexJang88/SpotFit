from fastapi import APIRouter, Header, HTTPException, Query, status

from app.services.recommender_v1 import RecommenderV1Error, get_district_forecasts
from app.services.crime_model_training import verify_admin_key
from app.services.final_model_artifacts import traffic_forecasts
from app.services.neighborhood_recommendations import service_catalog
from app.services.service_model_bootstrap import bootstrap_status, start_model_bootstrap


router = APIRouter(prefix="/recommender", tags=["recommender_v1"])


@router.get("/model-status")
def recommender_model_status() -> dict:
    """Show readiness of the delivered final models."""
    return {"version": "v1", **bootstrap_status()}


@router.post("/models/ensure", status_code=status.HTTP_202_ACCEPTED)
def ensure_recommender_models(
    force: bool = Query(default=False),
    admin_key: str | None = Header(default=None, alias="X-Model-Admin-Key"),
) -> dict:
    """Restore missing final model artifacts in the background."""
    try:
        verify_admin_key(admin_key)
        return {"version": "v1", "started": start_model_bootstrap(force=force), **bootstrap_status()}
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/traffic-accident/district-forecasts")
def traffic_accident_district_forecasts(
    horizon: int = Query(default=2, ge=2, le=3),
    district: str | None = Query(default=None),
) -> dict:
    """Return final v94 Feature-weighted safety scores."""
    try:
        frame, metadata = traffic_forecasts(horizon)
        payload = {**metadata, "district_count": len(frame), "forecasts": frame.to_dict('records')}
    except (RecommenderV1Error, OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if district:
        selected = [row for row in payload["forecasts"] if row["district"] == district.strip()]
        if not selected:
            raise HTTPException(status_code=404, detail=f"자치구를 찾을 수 없습니다: {district}")
        payload = {**payload, "district_count": 1, "forecasts": selected}
    return payload


@router.get("/district-forecasts")
def district_forecasts(district: str | None = Query(default=None), horizon: int = Query(default=2, ge=2, le=3)) -> dict:
    """Return results produced by every model delivered in recommender/v1."""
    try:
        catalog = service_catalog(horizon)
        payload = {**catalog, 'forecasts': catalog['districts'], 'district_count': len(catalog['districts']),
                   'neighborhood_count': len(catalog['neighborhoods'])}
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
