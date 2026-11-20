from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.traffic_accident_persistence import TrafficSafetyPersistenceModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "교통사고" / "50차"
PROCESSED = ROOT / "data" / "processed" / "교통사고_50차_Persistence_기준모델데이터.xlsx"


def validation_rows(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_grade = f"{horizon}년후_위험등급"
    target_year = f"{horizon}년후_목표연도"
    frame = data[data[target_grade].notna() & data[target_year].isin(CV_YEARS[horizon])].copy()
    frame["실제안전등급"] = 6 - frame[target_grade].astype(int)
    frame["50차_예측안전등급"] = 6 - frame["현재연도_위험등급"].astype(int)
    frame["정확여부"] = (frame["실제안전등급"] == frame["50차_예측안전등급"]).astype(int)
    frame["한등급이내"] = (
        (frame["실제안전등급"] - frame["50차_예측안전등급"]).abs() <= 1
    ).astype(int)
    frame["절대등급오차"] = (
        frame["실제안전등급"] - frame["50차_예측안전등급"]
    ).abs()
    return frame[["연도", target_year, "자치구", "현재연도_종합위험점수",
                  "현재연도_위험등급", "실제안전등급", "50차_예측안전등급",
                  "정확여부", "한등급이내", "절대등급오차"]].rename(
        columns={target_year: "검증목표연도"}
    )


def summarize(detail: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict]:
    yearly = detail.groupby("검증목표연도", as_index=False).agg(
        표본수=("자치구", "size"),
        정확등급정확도=("정확여부", "mean"),
        한등급이내정확도=("한등급이내", "mean"),
        평균등급오차=("절대등급오차", "mean"),
    )
    metrics = {
        "예측간격": horizon,
        "검증연도": list(CV_YEARS[horizon]),
        "표본수": int(len(detail)),
        "정확등급정확도": float(yearly["정확등급정확도"].mean()),
        "한등급이내정확도": float(yearly["한등급이내정확도"].mean()),
        "평균등급오차": float(yearly["평균등급오차"].mean()),
        "최저연도정확도": float(yearly["정확등급정확도"].min()),
    }
    return yearly, metrics


def forecast(data: pd.DataFrame, horizon: int, model: TrafficSafetyPersistenceModel) -> pd.DataFrame:
    future = data[data["연도"].eq(2025)].copy().sort_values("자치구")
    result = future[["자치구", "연도", "현재연도_종합위험점수", "현재연도_위험등급"]].copy()
    result = result.rename(columns={"연도": "기준연도"})
    result["예측연도"] = 2025 + horizon
    result["50차_예측안전점수"] = model.predict_safety_score(future)
    result["50차_예측안전순위"] = model.predict_safety_rank(future)
    result["50차_예측안전등급"] = model.predict(future)
    return result.sort_values("50차_예측안전순위")


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    details, yearlies, summaries, forecasts = {}, {}, [], []
    for horizon in (2, 3):
        model = TrafficSafetyPersistenceModel(horizon)
        path = OUTPUT / f"교통사고_50차_Persistence_{2025+horizon}년_{horizon}년후_모델.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        assert (loaded.predict(data[data["연도"].eq(2025)]) ==
                6 - data.loc[data["연도"].eq(2025), "현재연도_위험등급"].to_numpy()).all()
        detail = validation_rows(data, horizon)
        yearly, summary = summarize(detail, horizon)
        details[horizon], yearlies[horizon] = detail, yearly
        summaries.append(summary)
        forecasts.append(forecast(data, horizon, loaded))

    summary_frame = pd.DataFrame(summaries)
    forecast_frame = pd.concat(forecasts, ignore_index=True)
    info = {
        "모델명": "교통사고 50차 Persistence 공식 기준모델",
        "역할": "51차 이후 장기예측 모델의 공식 비교 기준",
        "방식": "현재 안전등급·점수·순위를 미래에도 그대로 유지",
        "학습여부": "학습 없음",
        "등급정의": "안전 1등급=가장 안전, 안전 5등급=가장 위험",
        "기준연도": 2025,
        "예측연도": [2027, 2028],
        "성능": {f"{row['예측간격']}년후": row for row in summaries},
    }
    (OUTPUT / "model_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.ExcelWriter(OUTPUT / "50차_Persistence_검증결과.xlsx", engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="성능요약", index=False)
        yearlies[2].to_excel(writer, sheet_name="2년_연도별성능", index=False)
        yearlies[3].to_excel(writer, sheet_name="3년_연도별성능", index=False)
        details[2].to_excel(writer, sheet_name="2년_검증상세", index=False)
        details[3].to_excel(writer, sheet_name="3년_검증상세", index=False)
        forecast_frame.to_excel(writer, sheet_name="2027_2028예측", index=False)
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="50차_성능요약", index=False)
        details[2].to_excel(writer, sheet_name="50차_2년검증데이터", index=False)
        details[3].to_excel(writer, sheet_name="50차_3년검증데이터", index=False)
        forecast_frame.to_excel(writer, sheet_name="50차_2027_2028예측", index=False)
        pd.DataFrame([
            {"항목": "모델", "내용": "Persistence"},
            {"항목": "정의", "내용": "현재 안전등급을 미래에도 그대로 사용"},
            {"항목": "등급방향", "내용": "1등급 가장 안전, 5등급 가장 위험"},
        ]).to_excel(writer, sheet_name="50차_모델설명", index=False)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
