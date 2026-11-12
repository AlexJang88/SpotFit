from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.controllers.admin_boundaries_controller import router as admin_boundaries_router
from app.controllers.admin_logs_controller import router as admin_logs_router
from app.controllers.crime_safety_controller import router as crime_safety_router
from app.controllers.model_bootstrap_controller import router as model_bootstrap_router
from app.controllers.neighborhood_controller import router as neighborhood_router
from app.controllers.recommender_controller import router as recommender_router
from app.controllers.spot_fit_controller import router as spot_fit_router
from app.core.error_handling import install_error_handling
from app.core.logging_config import configure_logging
from app.services.service_model_bootstrap import start_model_bootstrap


log_path = configure_logging()
logger = logging.getLogger("spotfit.lifecycle")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("SpotFit API starting; persistent_log=%s", log_path)
    try:
        # Restore missing property, traffic and child-crime service models in background.
        start_model_bootstrap()
        yield
    finally:
        logger.info("SpotFit API stopping")


app = FastAPI(
    lifespan=lifespan,
    title="spot_fit API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
install_error_handling(app)
app.include_router(spot_fit_router, prefix="/api")
app.include_router(crime_safety_router, prefix="/api")
app.include_router(recommender_router, prefix="/api")
app.include_router(admin_boundaries_router, prefix="/api")
app.include_router(neighborhood_router, prefix="/api")
app.include_router(model_bootstrap_router, prefix="/api")
app.include_router(admin_logs_router, prefix="/api")
