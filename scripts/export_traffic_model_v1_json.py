from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "안전" / "교통사고" / "1차"
REPORT_DIR = ROOT / "artifacts" / "reports" / "traffic_model_v1"
OUTPUT = REPORT_DIR / "교통사고_1차_초기모델.json"


def records(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((MODEL_DIR / "모델정보.json").read_text(encoding="utf-8"))
    saved = joblib.load(MODEL_DIR / "교통사고_1차_초기모델.joblib")
    pipeline = saved["pipeline"]
    estimator = pipeline.named_steps["model"]

    payload = {
        "schema_version": "1.0",
        "artifact_type": "traffic_accident_next_year_forecast_model_summary",
        "model_id": "safety-traffic-accident-v1-initial",
        "model_name": "안전-교통사고-1차 초기모델",
        "purpose": "안전기준표 교통사고 항목으로 서울 자치구별 다음 연도 교통사고 건수 예측",
        "prediction_horizon": "1년",
        "algorithm": {
            "name": type(estimator).__name__,
            "library": "scikit-learn",
            "parameters": estimator.get_params(),
            "executable_artifact": "교통사고_1차_초기모델.joblib",
            "json_execution_note": "이 JSON은 공유·검토용 요약이다. 실제 예측 실행에는 동일 폴더의 joblib 파일과 scikit-learn 환경이 필요하다.",
        },
        "data_split": {
            "training": {
                "input_years": [2017, 2018, 2019, 2020, 2021, 2022, 2023],
                "target_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
                "rows": 175,
            },
            "final_validation": {"input_year": 2024, "target_year": 2025, "rows": 25},
            "split_rule": "시간 순서 분리; 2025년 검증 자료는 모델 선택과 학습에 사용하지 않음",
        },
        "features": metadata["입력변수"],
        "target": metadata["예측목표"],
        "excluded_features": metadata["제외"],
        "safety_score_weights": metadata["안전점수가중치"],
        "final_validation_metrics": metadata["최종검증지표"],
        "metric_note": metadata["주의"],
        "candidate_model_comparison": records(MODEL_DIR / "후보모델_비교.csv"),
        "feature_importance": records(MODEL_DIR / "변수중요도.csv"),
        "validation_predictions_2025": records(MODEL_DIR / "2025년_최종검증_예측결과.csv"),
        "source_files": {
            "traffic_accidents": "TAAS_서울_고속국도제외_2010-2014/2015-2019/2020-2024/2025.xlsx",
            "population": "자치구별_주민등록인구_2017_2025.csv",
            "medical_workplaces": "KOSIS_시군구별_적용인구현황_2017_2024_한글정상.csv",
            "seatbelt": "자치구별_운전자안전벨트착용_2008_2025.csv",
            "drunk_driving_experience": "자치구별_음주운전_2008_2025.csv",
        },
        "official_sources": [
            "한국도로교통공단 교통사고분석시스템(TAAS)",
            "서울특별시 서울 열린데이터광장 주민등록인구",
            "국민건강보험공단 KOSIS 시군구별 적용인구 현황(TX_35001_A008)",
            "질병관리청 지역사회건강조사",
            "행정안전부 지역안전지수 산출 기준",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
