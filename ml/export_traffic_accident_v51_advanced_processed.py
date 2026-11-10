from __future__ import annotations

import json

import pandas as pd

from ml.improve_traffic_accident_v51_advanced import DATA, MODEL_DIR, REPORT, add_features


def main() -> None:
    base = pd.read_excel(DATA, sheet_name="51차점수등급데이터")
    advanced, feature_sets = add_features(base)
    raw_features = list(pd.read_excel(DATA, sheet_name="변수가중치")["변수"])
    score_features = [f"점수_{name}" for name in raw_features]
    history_features = [name for name in feature_sets["all"] if name not in raw_features + score_features]
    district_features = [name for name in feature_sets["all_district"] if name.startswith("구_")]
    advanced["고도화_데이터구분"] = advanced["데이터구분"]

    model_columns = [
        "서울시", "연도", "예측연도", "자치구", "고도화_데이터구분",
        "다음연도_종합위험점수", "다음연도_위험등급",
        *raw_features, *score_features, *history_features, *district_features,
    ]
    final_data = advanced[model_columns].copy()
    configuration = pd.DataFrame([
        ["최종 목표", "다음 연도 서울 25개 자치구 상대 위험등급 1~5"],
        ["구성 A", "Gradient Boosting depth=1 · 다음 연도 위험점수 예측"],
        ["구성 A 보정", "예측점수 60% + 현재점수 40%"],
        ["구성 B", "Gradient Boosting depth=3 · 다음 연도 점수 변화량 예측"],
        ["최종 앙상블", "구성 A 75% + 구성 B 25%"],
        ["등급 산출", "예측 위험순위를 내림차순 정렬하여 5개 자치구씩 1~5등급"],
        ["정확등급 정확도", "60.0%"],
        ["±1등급 정확도", "90.4%"],
        ["평균 등급오차", "0.496등급"],
    ], columns=["항목", "내용"])
    feature_definition = pd.DataFrame([
        ["원천 변수", len(raw_features), ", ".join(raw_features)],
        ["5분위 점수", len(score_features), ", ".join(score_features)],
        ["시계열 파생변수", len(history_features), ", ".join(history_features)],
        ["자치구 고정변수", len(district_features), "서울 25개 자치구 원-핫 변수"],
    ], columns=["변수군", "컬럼수", "설명"])
    validation = pd.read_excel(REPORT, sheet_name="연도별검증")
    best_name = pd.read_excel(REPORT, sheet_name="모델비교").iloc[0]["모델"]
    validation = validation[validation["모델"] == best_name].copy()

    with pd.ExcelWriter(DATA, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        final_data.to_excel(writer, sheet_name="고도화51차_최종데이터", index=False)
        configuration.to_excel(writer, sheet_name="고도화51차_모델구성", index=False)
        feature_definition.to_excel(writer, sheet_name="고도화51차_변수정의", index=False)
        validation.to_excel(writer, sheet_name="고도화51차_연도별검증", index=False)

    metadata = {
        "행수": len(final_data), "전체컬럼수": len(final_data.columns),
        "원천변수수": len(raw_features), "점수변수수": len(score_features),
        "시계열파생변수수": len(history_features), "자치구변수수": len(district_features),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(DATA)


if __name__ == "__main__":
    main()
