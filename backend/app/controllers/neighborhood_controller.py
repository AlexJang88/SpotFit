import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.services import neighborhood_recommendations as service
from app.services.child_crime_exposure import ChildCrimeExposureError
from app.services.recommender_v1 import RecommenderV1Error

router = APIRouter(prefix="/neighborhoods", tags=["neighborhood_recommendations"])
logger = logging.getLogger(__name__)


def call_service(func, **kwargs):
    try:
        return func(**kwargs)
    except LookupError as error:
        # A missing dictionary key is an artifact error, not a user 404.
        if isinstance(error, KeyError):
            logger.exception("Invalid neighborhood model artifact")
            raise HTTPException(503, "모델 산출물 형식이 올바르지 않습니다.") from error
        raise HTTPException(404, str(error)) from error
    except (RecommenderV1Error, ChildCrimeExposureError, OSError, ValueError) as error:
        logger.exception("Neighborhood service data unavailable")
        raise HTTPException(503, "추천 데이터를 준비하지 못했습니다. 모델 산출물을 확인해 주세요.") from error


@router.get("/catalog")
def catalog(horizon: int = Query(default=2, ge=2, le=3)):
    return call_service(service.service_catalog, horizon=horizon)


@router.get("/recommendations")
def recommendations(
    budget_manwon: float = Query(gt=0, le=1_000_000_000, allow_inf_nan=False),
    transaction: Literal["sale", "rent"] = "sale",
    horizon: int = Query(default=2, ge=2, le=3),
    area_m2: float | None = Query(default=None, gt=0, le=1000, allow_inf_nan=False),
    limit: int = Query(default=5, ge=1, le=5),
    min_budget_manwon: float = Query(default=0, ge=0, le=1_000_000_000, allow_inf_nan=False),
):
    if min_budget_manwon > budget_manwon:
        raise HTTPException(422, "예산 하한은 상한보다 클 수 없습니다.")
    return call_service(service.recommend, budget_manwon=budget_manwon, transaction=transaction,
                        horizon=horizon, area_m2=area_m2, limit=limit, min_budget_manwon=min_budget_manwon)


@router.get("/area-presets")
def area_presets(transaction: Literal["sale", "rent"] = "sale"):
    return call_service(service.popular_areas, transaction=transaction)


@router.get("/detail")
def detail(
    district: str = Query(min_length=1, max_length=30),
    neighborhood: str = Query(min_length=1, max_length=50),
    horizon: int = Query(default=2, ge=2, le=3),
    transaction: Literal["sale", "rent"] = "sale",
    area_m2: float | None = Query(default=None, gt=0, le=1000, allow_inf_nan=False),
):
    return call_service(service.neighborhood_detail, district=district, neighborhood=neighborhood,
                        horizon=horizon, transaction=transaction, area_m2=area_m2)
