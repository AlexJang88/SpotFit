from fastapi import FastAPI

from app.controllers.crime_safety_controller import router as crime_safety_router
from app.controllers.admin_boundaries_controller import router as admin_boundaries_router
from app.controllers.recommender_controller import router as recommender_router
from app.controllers.spot_fit_controller import router as spot_fit_router

app = FastAPI(
    title="spot_fit API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
app.include_router(spot_fit_router, prefix="/api")
app.include_router(crime_safety_router, prefix="/api")
app.include_router(recommender_router, prefix="/api")
app.include_router(admin_boundaries_router, prefix="/api")
