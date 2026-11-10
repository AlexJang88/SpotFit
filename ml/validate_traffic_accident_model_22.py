from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.train_traffic_accident_1y_best import DATA_PATH, TARGET
from ml.train_traffic_accident_incremental_models import CURRENT, add_external_features, metrics


ROOT = Path(__file__).resolve().parents[1]
MODEL_1_DIR = ROOT / "models" / "교통사고" / "1차"
MODEL_22_DIR = ROOT / "models" / "교통사고" / "22차~40차" / "22차"
EVALUATION_PATH = MODEL_22_DIR / "모델평가_및_2026예측.xlsx"
YEARS = [2021, 2022, 2023, 2024, 2025]


def metric_row(method: str, actual, predicted, **extra) -> dict:
    return {"검증방법": method, **extra, **metrics(actual, predicted)}


def load_first_model_predictions() -> pd.DataFrame:
    info = json.loads((MODEL_1_DIR / "model_info.json").read_text(encoding="utf-8"))
    selected = info["최종후보"]
    workbook = MODEL_1_DIR / "모델평가_및_2026예측.xlsx"
    cv = pd.read_excel(workbook, sheet_name="교차검증예측")
    cv = cv[(cv["후보명"] == selected) & cv["예측연도"].isin(YEARS[:-1])].copy()
    cv = cv[["기준연도", "예측연도", "자치구", TARGET, "예측값"]]
    holdout = pd.read_excel(workbook, sheet_name="2025홀드아웃검증")
    prediction_column = "예측_다음연도_교통사고건수"
    holdout = holdout[["기준연도", "예측연도", "자치구", TARGET, prediction_column]].rename(
        columns={prediction_column: "예측값"}
    )
    result = pd.concat([cv, holdout], ignore_index=True)
    result["검증방법"] = "1차 안전기준표 모델"
    return result


def rolling_validation(frame: pd.DataFrame, estimator, features: list[str]) -> pd.DataFrame:
    labeled = frame[frame[TARGET].notna()].copy()
    outputs = []
    for year in YEARS:
        train = labeled[labeled["예측연도"] < year]
        valid = labeled[labeled["예측연도"] == year]
        model = clone(estimator).fit(train[features], train[TARGET])
        prediction = np.maximum(model.predict(valid[features]), 0)
        part = valid[["기준연도", "예측연도", "자치구", TARGET, CURRENT]].copy()
        part["예측값"] = prediction
        part["검증방법"] = "22차 롤링 모델"
        part["학습행수"] = len(train)
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def leave_one_district_out_time_validation(
    frame: pd.DataFrame, estimator, features: list[str]
) -> pd.DataFrame:
    labeled = frame[frame[TARGET].notna()].copy()
    outputs = []
    for year in YEARS:
        year_rows = labeled[labeled["예측연도"] == year]
        for district in sorted(year_rows["자치구"].unique()):
            train = labeled[(labeled["예측연도"] < year) & (labeled["자치구"] != district)]
            valid = year_rows[year_rows["자치구"] == district]
            model = clone(estimator).fit(train[features], train[TARGET])
            prediction = np.maximum(model.predict(valid[features]), 0)
            part = valid[["기준연도", "예측연도", "자치구", TARGET, CURRENT]].copy()
            part["예측값"] = prediction
            part["검증방법"] = "자치구 제외+시간 분할"
            part["학습행수"] = len(train)
            outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def grouped_metrics(data: pd.DataFrame, group: str, method: str) -> pd.DataFrame:
    rows = []
    for value, part in data.groupby(group, sort=True):
        rows.append(metric_row(method, part[TARGET], part["예측값"], **{group: value, "표본수": len(part)}))
    return pd.DataFrame(rows)


def main() -> None:
    payload = joblib.load(MODEL_22_DIR / "교통사고_22차_예측모델.joblib")
    estimator = payload["model"]
    features = payload["features"]
    raw = pd.read_excel(DATA_PATH, sheet_name="모델데이터")
    frame = add_external_features(raw)

    rolling = rolling_validation(frame, estimator, features)
    persistence = rolling[["기준연도", "예측연도", "자치구", TARGET, CURRENT, "학습행수"]].copy()
    persistence["예측값"] = persistence[CURRENT]
    persistence["검증방법"] = "현재값 유지 기준모델"
    first_model = load_first_model_predictions()

    comparison_predictions = pd.concat([first_model, persistence, rolling], ignore_index=True, sort=False)
    summary_rows = []
    yearly_rows = []
    for method, part in comparison_predictions.groupby("검증방법", sort=False):
        summary_rows.append(metric_row(method, part[TARGET], part["예측값"], 표본수=len(part)))
        yearly_rows.append(grouped_metrics(part, "예측연도", method))
    summary = pd.DataFrame(summary_rows).sort_values("MAE").reset_index(drop=True)
    yearly = pd.concat(yearly_rows, ignore_index=True)

    loo = leave_one_district_out_time_validation(frame, estimator, features)
    loo_summary = pd.DataFrame([
        metric_row("자치구 제외+시간 분할", loo[TARGET], loo["예측값"], 표본수=len(loo))
    ])
    loo_yearly = grouped_metrics(loo, "예측연도", "자치구 제외+시간 분할")
    loo_district = grouped_metrics(loo, "자치구", "자치구 제외+시간 분할")

    with pd.ExcelWriter(
        EVALUATION_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        summary.to_excel(writer, sheet_name="추가검증_요약", index=False)
        yearly.to_excel(writer, sheet_name="롤링_연도별성능", index=False)
        comparison_predictions.to_excel(writer, sheet_name="롤링_예측", index=False)
        loo_summary.to_excel(writer, sheet_name="자치구제외_요약", index=False)
        loo_yearly.to_excel(writer, sheet_name="자치구제외_연도별", index=False)
        loo_district.to_excel(writer, sheet_name="자치구제외_자치구별", index=False)
        loo.to_excel(writer, sheet_name="자치구제외_예측", index=False)

    result = {
        "검증대상": "교통사고 22차 ExtraTrees",
        "검증연도": YEARS,
        "롤링검증": "각 예측연도 이전 연도만 학습",
        "자치구제외검증": "각 예측연도 이전 자료 중 대상 자치구 전체를 제외",
        "자치구제외검증횟수": len(loo),
        "전체비교": summary.to_dict(orient="records"),
        "자치구제외전체": loo_summary.iloc[0].to_dict(),
    }
    (MODEL_22_DIR / "추가검증_결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n자치구 제외+시간 분할")
    print(loo_summary.to_string(index=False))
    print("\n연도별")
    print(loo_yearly.to_string(index=False))


if __name__ == "__main__":
    main()
