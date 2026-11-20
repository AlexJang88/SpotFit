import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.admin_boundaries import AdminBoundaryStatus

router = APIRouter(tags=["admin_boundaries"])
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "admin_boundary_registry.json"


@router.get("/admin-boundaries/status", response_model=AdminBoundaryStatus)
def admin_boundary_status() -> AdminBoundaryStatus:
    """Return the boundary version actually deployed by SpotFit, not today's date."""
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return AdminBoundaryStatus.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="행정동 기준정보를 불러오지 못했습니다.") from exc
