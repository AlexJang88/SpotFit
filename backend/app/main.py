from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.controllers.crime_safety_controller import router as crime_safety_router
from app.controllers.admin_boundaries_controller import router as admin_boundaries_router
from app.controllers.recommender_controller import router as recommender_router
from app.controllers.spot_fit_controller import router as spot_fit_router
from app.services.traffic_accident_v51 import ensure_models as ensure_traffic_models


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Temporary presentation setup: publish model 51 into recommender/v1 only
    # when its service artifacts are missing. A final traffic model can replace
    # this adapter without changing the rest of startup.
    try:
        ensure_traffic_models()
    except Exception:
        logger.exception("Failed to prepare temporary traffic model v51")
    yield

app = FastAPI(
    title="spot_fit API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.include_router(spot_fit_router, prefix="/api")
app.include_router(crime_safety_router, prefix="/api")
app.include_router(recommender_router, prefix="/api")
app.include_router(admin_boundaries_router, prefix="/api")
