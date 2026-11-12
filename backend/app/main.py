from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.service_model_bootstrap import start_model_bootstrap
from app.controllers.crime_safety_controller import router as crime_safety_router
from app.controllers.admin_boundaries_controller import router as admin_boundaries_router
from app.controllers.recommender_controller import router as recommender_router
from app.controllers.spot_fit_controller import router as spot_fit_router
from app.controllers.neighborhood_controller import router as neighborhood_router
from app.controllers.model_bootstrap_controller import router as model_bootstrap_router

@asynccontextmanager
async def lifespan(_: FastAPI):
    # 공개 서비스에 필요한 부동산·교통·아동범죄 모델/예측표를 누락 시 복구한다.
    start_model_bootstrap()
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
app.include_router(model_bootstrap_router, prefix="/api")
