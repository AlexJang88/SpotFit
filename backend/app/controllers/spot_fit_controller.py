from fastapi import APIRouter

from app.views.spot_fit_view import SpotFitResponse

router = APIRouter(tags=["spot_fit"])


@router.get("/spot_fit", response_model=SpotFitResponse)
def spot_fit() -> SpotFitResponse:
    """Process liveness only; database and storage are not checked."""
    return SpotFitResponse(status="ok")
