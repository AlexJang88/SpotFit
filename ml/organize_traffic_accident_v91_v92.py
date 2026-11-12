from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "87차.피해자수Poisson기간별최종"
OUT91 = ROOT / "models" / "교통사고" / "91차.87차기반Persistence기준모델"
OUT92 = ROOT / "models" / "교통사고" / "92차.87차승계모델"

SRC2 = SOURCE / "교통사고_87차_2년후_피해자수Poisson모델.joblib"
SRC3 = SOURCE / "교통사고_87차_3년후_피해자수Poisson모델.joblib"


def persistence_map(features: list[str]) -> dict[str, dict[str, object]]:
    return {
        feature: {
            "spec": "Persistence",
            "components": {"Persistence": "Persistence"},
        }
        for feature in features
    }


def save_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT91.mkdir(parents=True, exist_ok=True)
    OUT92.mkdir(parents=True, exist_ok=True)

    source2 = joblib.load(SRC2)
    source3 = joblib.load(SRC3)

    # 91차: 변수·Poisson 가중치·5분위 등급 산식은 87차와 같고,
    # 미래 지표 예측만 마지막 관측값을 유지하는 Persistence로 고정한다.
    model91_2 = copy.deepcopy(source2)
    model91_2["version"] = 91
    model91_2["horizon"] = 2
    model91_2["baseline"] = "Persistence"
    model91_2["forecast_models"] = {2: persistence_map(model91_2["features"])}

    model91_3 = copy.deepcopy(source3)
    model91_3["version"] = 91
    model91_3["horizon"] = 3
    model91_3["baseline"] = "Persistence"
    model91_3["forecast_model"] = persistence_map(model91_3["features"])

    joblib.dump(model91_2, OUT91 / "교통사고_91차_2년후_Persistence기준모델.joblib")
    joblib.dump(model91_3, OUT91 / "교통사고_91차_3년후_Persistence기준모델.joblib")

    # 92차: 87차에서 검증된 기간별 모델 객체와 규칙을 그대로 승계한다.
    model92_2 = copy.deepcopy(source2)
    model92_2["version"] = 92
    model92_2["horizon"] = 2
    model92_2["source_version"] = 87
    model92_3 = copy.deepcopy(source3)
    model92_3["version"] = 92
    model92_3["source_version"] = 87

    joblib.dump(model92_2, OUT92 / "교통사고_92차_2년후_87차승계모델.joblib")
    joblib.dump(model92_3, OUT92 / "교통사고_92차_3년후_87차승계모델.joblib")

    common = {
        "Y": "다음 연도 어린이 교통사고 사망자 수+부상자 수",
        "인구처리": "어린이 인구 Poisson exposure",
        "등급산정근간": "미래지표→5분위→Poisson 계수가중치→종합점수→안전 1~5등급",
        "안전등급방향": "1=가장 안전, 5=가장 위험",
    }
    save_json(
        OUT91 / "model_info.json",
        {
            "모델명": "교통사고 91차 87차 기반 Persistence 기준모델",
            "역할": "92차 성능 비교용 기준모델",
            "87차대비차이": "미래 지표별 예측값 대신 마지막 관측값을 2년·3년 후에도 그대로 사용",
            "유지사항": "87차의 기간별 변수, Poisson 계수 가중치, 5분위와 안전등급 산정방식 유지",
            "2년모델파일": "교통사고_91차_2년후_Persistence기준모델.joblib",
            "3년모델파일": "교통사고_91차_3년후_Persistence기준모델.joblib",
            **common,
        },
    )
    save_json(
        OUT92 / "model_info.json",
        {
            "모델명": "교통사고 92차 87차 승계모델",
            "역할": "91차 Persistence 기준모델과 비교할 현재 모델",
            "원본": "87차 피해자수 Poisson 기간별 최종모델",
            "2년구성": "87차 2년 모델(84차 검증 조합) 그대로",
            "3년구성": "87차 3년 모델(86차 검증 조합) 그대로",
            "2년모델파일": "교통사고_92차_2년후_87차승계모델.joblib",
            "3년모델파일": "교통사고_92차_3년후_87차승계모델.joblib",
            **common,
        },
    )


if __name__ == "__main__":
    main()
