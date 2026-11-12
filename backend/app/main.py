from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.property_model_bootstrap import start_background_bootstrap
from app.controllers.crime_safety_controller import router as crime_safety_router
from app.controllers.admin_boundaries_controller import router as admin_boundaries_router
from app.controllers.recommender_controller import router as recommender_router
from app.controllers.spot_fit_controller import router as spot_fit_router
from app.controllers.neighborhood_controller import router as neighborhood_router

@asynccontextmanager
async def lifespan(_: FastAPI):
    # 매매·전세·월세 최종 모델이 models/recommender/v1에 없으면 백그라운드로 만든다(서버 시작을 막지 않음).
    start_background_bootstrap()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="spot_fit API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
app.include_router(spot_fit_router, prefix="/api")
app.include_router(crime_safety_router, prefix="/api")
app.include_router(recommender_router, prefix="/api")
app.include_router(admin_boundaries_router, prefix="/api")
app.include_router(neighborhood_router, prefix="/api")
