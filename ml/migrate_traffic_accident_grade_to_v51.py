from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib

from ml.train_best_child_traffic_accident_rate import ROOT


SOURCE_VERSION = 49
TARGET_VERSION = 51
SOURCE_DIR = ROOT / "models" / "교통사고" / "41차~49차" / f"{SOURCE_VERSION}차"
TARGET_DIR = ROOT / "models" / "교통사고" / f"{TARGET_VERSION}차"
PROCESSED_DIR = ROOT / "data" / "processed"


def migrate_model(source_name: str, target_name: str) -> None:
    bundle = joblib.load(SOURCE_DIR / source_name)
    bundle["version"] = TARGET_VERSION
    bundle["source_base_model_version"] = SOURCE_VERSION
    joblib.dump(bundle, TARGET_DIR / target_name)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    migrate_model(
        "교통사고_49차_가중치5분위_점수등급모델.joblib",
        "교통사고_51차_가중치5분위_점수등급모델.joblib",
    )
    migrate_model(
        "교통사고_49차_개선등급예측모델.joblib",
        "교통사고_51차_개선등급예측모델.joblib",
    )

    reports = {
        "49차_가중치5분위_점수등급평가.xlsx": "51차_가중치5분위_점수등급평가.xlsx",
        "49차_개선등급예측평가.xlsx": "51차_개선등급예측평가.xlsx",
    }
    for source, target in reports.items():
        shutil.copy2(SOURCE_DIR / source, TARGET_DIR / target)

    shutil.copy2(
        PROCESSED_DIR / "교통사고_49차_최종모델데이터.xlsx",
        PROCESSED_DIR / "교통사고_51차_최종모델데이터.xlsx",
    )

    metadata = {
        "모델명": "교통사고 51차 가중치 5분위 점수·등급 예측모델",
        "기반모델": "49차 어린이 교통사고 피해자율 예측모델",
        "등급체계": "서울 25개 자치구 상대순위 5분위, 1등급 고위험·5등급 저위험",
        "가중치": "49차 Ridge 표준화계수 절댓값을 합계 100%로 환산",
        "점수예측모델": "Ridge alpha=10",
        "개선등급예측모델": "선형 SVM C=10 + 순위보정",
        "2021_2025_평균_정확등급정확도_pct": 56.0,
        "2021_2025_평균_1칸이내정확도_pct": 92.8,
        "평균등급오차": 0.512,
        "원천변수수": 13,
        "상태": "공식 등급예측 모델",
    }
    (TARGET_DIR / "model_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(TARGET_DIR)
    print(PROCESSED_DIR / "교통사고_51차_최종모델데이터.xlsx")


if __name__ == "__main__":
    main()
