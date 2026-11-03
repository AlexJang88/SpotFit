from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SafetyGrade = Literal["A", "B", "C", "D", "E"]


class CrimeSafetyDistrictForecast(BaseModel):
    district: str = Field(description="서울특별시 자치구명")
    data_year: int = Field(description="예측 입력에 사용한 최신 기준연도")
    prediction_year: int = Field(description="1년 후 예측 대상연도")
    predicted_score: float = Field(description="범죄안전 대체점수 예측값")
    predicted_rank: int = Field(ge=1, le=25, description="서울 25개 자치구 내 예상 상대순위")
    predicted_grade: SafetyGrade = Field(description="상위 20%부터 A~E로 구분한 예상 상대등급")
    previous_rank: int = Field(ge=1, le=25)
    previous_grade: SafetyGrade
    grade_change_expected: bool = Field(description="전년도 대비 상대등급 전환 예상 여부")
    grade_change_direction: Literal["상승", "유지", "하락"]


class CrimeSafetyForecastResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    target: str
    data_year: int
    prediction_year: int
    district_count: int
    grade_definition: dict[SafetyGrade, str]
    warning: str
    forecasts: list[CrimeSafetyDistrictForecast]

