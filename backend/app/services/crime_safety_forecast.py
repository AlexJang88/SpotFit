from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = (
    ROOT / "artifacts" / "crime_safety_forecast" / "grade_transition_rolling_2021_2023"
    / "crime_safety_direct_best_ml_exception.joblib"
)
DEFAULT_DATA = (
    ROOT / "data" / "interim" / "안전" / "범죄안전지수"
    / "서울시_자치구_범죄안전_대체지수_2018_2023.csv"
)
RATE_COLUMNS = [
    "5대범죄_인구1만명당",
    "집객시설_인구1만명당",
    "주점업체_인구1만명당",
    "스트레스인지율",
    "경찰관서_인구1만명당",
]
GRADE_MAP = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
GRADE_DEFINITION = {
    "A": "서울 25개 구 중 상위 20%",
    "B": "상위 20~40%",
    "C": "상위 40~60%",
    "D": "상위 60~80%",
    "E": "하위 20%",
}
MODEL_FEATURES = ["자치구"] + RATE_COLUMNS + [f"{column}_전년차" for column in RATE_COLUMNS] + ["기준연도"]


class CrimeSafetyForecastError(RuntimeError):
    pass


def _configured_path(environment_name: str, default: Path) -> Path:
    value = os.getenv(environment_name)
    return Path(value).expanduser().resolve() if value else default


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    path = _configured_path("CRIME_SAFETY_MODEL_PATH", DEFAULT_MODEL)
    if path.is_file():
        bundle = joblib.load(path)
    else:
        model_uri = os.getenv(
            "CRIME_SAFETY_MODEL_URI", "models:/crime_safety_direct_score_1y/1"
        )
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise CrimeSafetyForecastError(
                f"로컬 모델이 없고 MLFLOW_TRACKING_URI도 설정되지 않았습니다: {path}"
            )
        try:
            mlflow.set_tracking_uri(tracking_uri)
            bundle = {
                "model": mlflow.sklearn.load_model(model_uri),
                "algorithm": "Ridge_1",
                "features": MODEL_FEATURES,
            }
        except Exception as error:
            raise CrimeSafetyForecastError(
                f"MLflow 범죄안전 모델을 불러오지 못했습니다: {model_uri}"
            ) from error
    required = {"model", "algorithm", "features"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise CrimeSafetyForecastError("범죄안전 모델 번들의 필수 항목이 없습니다.")
    return bundle


def _load_feature_frame(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    path = _configured_path("CRIME_SAFETY_DATA_PATH", DEFAULT_DATA)
    if not path.is_file():
        raise CrimeSafetyForecastError(f"범죄안전 입력 데이터가 없습니다: {path}")
    data = pd.read_csv(path, encoding="utf-8-sig")
    latest_year = int(data["연도"].max())
    previous_year = latest_year - 1
    current = data[data["연도"] == latest_year].copy()
    previous = data[data["연도"] == previous_year].set_index("자치구")
    if len(current) != 25 or current["자치구"].nunique() != 25:
        raise CrimeSafetyForecastError(f"최신 {latest_year}년 자료가 서울 25개 구 구조가 아닙니다.")
    if previous.index.nunique() != 25:
        raise CrimeSafetyForecastError(f"전년 {previous_year}년 자료가 서울 25개 구 구조가 아닙니다.")

    current = current.rename(columns={"연도": "기준연도"})
    for column in RATE_COLUMNS:
        current[f"{column}_전년차"] = current[column] - current["자치구"].map(previous[column])
    missing = [column for column in features if column not in current]
    if missing:
        raise CrimeSafetyForecastError(f"모델 입력 feature가 데이터에 없습니다: {missing}")
    if current[features].isna().any().any():
        raise CrimeSafetyForecastError("모델 입력 feature에 결측값이 있습니다.")
    return current, previous.reset_index(), latest_year


def predict_all_districts() -> dict:
    bundle = load_model_bundle()
    features = list(bundle["features"])
    current, _, latest_year = _load_feature_frame(features)
    current["예측점수"] = bundle["model"].predict(current[features])
    current["예측순위"] = current["예측점수"].rank(ascending=False, method="first").astype(int)
    current["예측등급번호"] = ((current["예측순위"] - 1) // 5 + 1).astype(int)
    current["예측등급"] = current["예측등급번호"].map(GRADE_MAP)

    if "서울25개구_순위" in current:
        current["전년순위"] = current["서울25개구_순위"].astype(int)
    else:
        current["전년순위"] = current["범죄안전_대체점수"].rank(
            ascending=False, method="first"
        ).astype(int)
    current["전년등급번호"] = ((current["전년순위"] - 1) // 5 + 1).astype(int)
    current["전년등급"] = current["전년등급번호"].map(GRADE_MAP)

    def direction(row: pd.Series) -> str:
        if row["예측등급번호"] < row["전년등급번호"]:
            return "상승"
        if row["예측등급번호"] > row["전년등급번호"]:
            return "하락"
        return "유지"

    current["등급변화방향"] = current.apply(direction, axis=1)
    current = current.sort_values("예측순위")
    forecasts = [
        {
            "district": row["자치구"],
            "data_year": latest_year,
            "prediction_year": latest_year + 1,
            "predicted_score": round(float(row["예측점수"]), 4),
            "predicted_rank": int(row["예측순위"]),
            "predicted_grade": row["예측등급"],
            "previous_rank": int(row["전년순위"]),
            "previous_grade": row["전년등급"],
            "grade_change_expected": row["등급변화방향"] != "유지",
            "grade_change_direction": row["등급변화방향"],
        }
        for _, row in current.iterrows()
    ]
    return {
        "model_name": "crime_safety_direct_score_1y",
        "model_version": os.getenv("CRIME_SAFETY_MODEL_VERSION", "1"),
        "algorithm": str(bundle["algorithm"]),
        "target": "다음 연도 범죄안전 대체점수",
        "data_year": latest_year,
        "prediction_year": latest_year + 1,
        "district_count": len(forecasts),
        "grade_definition": GRADE_DEFINITION,
        "warning": (
            "공식 지역안전등급이 아닌 서울 25개 구 상대 대체등급입니다. "
            "등급 전환 경보 모델의 검증 정밀도는 42.3%이므로 확정값으로 해석하지 마세요."
        ),
        "forecasts": forecasts,
    }

