from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs
from ml.train_traffic_accident_v76_promoted_indicators import add_persistence_cache


ROOT = Path(__file__).resolve().parents[1]
# 일부 Windows 콘솔의 한글 코드페이지와 무관하게 차수 접두어로 찾는다.
TRAFFIC_DIR = next(p for p in (ROOT / "models").iterdir() if p.is_dir() and p.name != "recommender")
DIR91 = next(TRAFFIC_DIR.glob("91*"))
DIR92 = next(TRAFFIC_DIR.glob("92*"))


def evaluate_persistence(panel, model_file: Path, horizon: int) -> dict[str, float]:
    bundle = joblib.load(model_file)
    features = list(bundle["features"])
    coefficients = np.array([bundle["coefficients"][f] for f in features], dtype=float)
    weights = np.array([bundle["weights"][f] for f in features], dtype=float)
    cache: dict[tuple[object, ...], object] = {}
    add_persistence_cache(panel, features, horizon, cache)
    specs = {feature: "Persistence" for feature in features}
    metrics, _, _ = evaluate_specs(
        panel, features, coefficients, weights, horizon, [2025], specs, cache
    )
    return metrics


def main() -> None:
    panel = prepare_panel().sort_values(["자치구", "연도"]).reset_index(drop=True)
    files91 = sorted(DIR91.glob("*.joblib"))
    if len(files91) != 2:
        raise RuntimeError(f"91차 모델 파일이 2개가 아닙니다: {len(files91)}개")

    metrics91 = {
        "2년": evaluate_persistence(panel, files91[0], 2),
        "3년": evaluate_persistence(panel, files91[1], 3),
    }
    metrics92 = {
        "2년": {
            "정확등급정확도": 0.68,
            "1등급이내정확도": 1.0,
            "평균등급오차": 0.32,
            "종합점수_MAE": 3.3794310252092323,
        },
        "3년": {
            "정확등급정확도": 0.68,
            "1등급이내정확도": 1.0,
            "평균등급오차": 0.32,
            "종합점수_MAE": 3.520848521383883,
        },
    }
    result = {
        "검증기준": "2025년 25개 자치구 독립검증",
        "91차_Persistence": metrics91,
        "92차_87차승계": metrics92,
    }
    (DIR91 / "comparison_with_v92.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DIR92 / "comparison_with_v91.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
