from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.train_traffic_accident_1y_best import DATA_PATH, SCORE, TARGET, build_safety_features
from ml.train_traffic_accident_incremental_models import (
    CURRENT,
    MODEL_ROOT,
    MODEL_SPECS,
    TREND_FEATURES,
    TYPE_FEATURES,
    add_external_features,
    train_one,
)


COMBINATION_SPECS = {
    9: {"name": "현재 사고 + 자동차 등록대수", "features": [CURRENT, "자동차등록대수"]},
    10: {"name": "현재 사고 + 도로연장", "features": [CURRENT, "도로연장_km"]},
    11: {"name": "현재 사고 + 총 인구수", "features": [CURRENT, "총인구수"]},
    12: {"name": "현재 사고 + 최근 사고 추세", "features": [CURRENT, *TREND_FEATURES]},
    13: {"name": "현재 사고 + 사고 유형별 건수", "features": [CURRENT, *TYPE_FEATURES]},
    14: {"name": "현재 사고 + 어린이·노인 인구", "features": [CURRENT, "어린이_0_12세_인구수", "노인_65세이상_인구수"]},
}

ADVANCED_SPECS = {
    15: {"name": "12차 + 자동차 등록대수", "features": [CURRENT, *TREND_FEATURES, "자동차등록대수"]},
    16: {"name": "12차 + 도로연장", "features": [CURRENT, *TREND_FEATURES, "도로연장_km"]},
    17: {"name": "12차 + 총 인구수", "features": [CURRENT, *TREND_FEATURES, "총인구수"]},
    18: {"name": "12차 + 사고 유형별 건수", "features": [CURRENT, *TREND_FEATURES, *TYPE_FEATURES]},
    19: {"name": "12차 + 어린이·노인 인구", "features": [CURRENT, *TREND_FEATURES, "어린이_0_12세_인구수", "노인_65세이상_인구수"]},
    20: {"name": "12차 + 확보 변수 전체", "features": [
        CURRENT, *TREND_FEATURES,
        "자동차등록대수", "도로연장_km", "총인구수",
        *TYPE_FEATURES,
        "어린이_0_12세_인구수", "노인_65세이상_인구수",
    ]},
}


def read_result(version: int) -> dict:
    info = json.loads((MODEL_ROOT / f"{version}차" / "model_info.json").read_text(encoding="utf-8"))
    if version == 1:
        result = {
            "차수": "1차",
            "설계": "안전기준표만",
            "선정모델": info["최종후보"],
            "입력특성수": len(info["최종입력특성"]),
        }
    else:
        result = {
            "차수": f"{version}차",
            "설계": info["설계"],
            "선정모델": info["선정모델"],
            "입력특성수": len(info["입력특성"]),
        }
    metric = info["2025_홀드아웃성능"]
    result.update({
        "MAE": metric["MAE"],
        "RMSE": metric["RMSE"],
        "R2": metric["R2"],
        "WMAPE_pct": metric["WMAPE_pct"],
        "오차기반참고정확도_pct": metric.get("오차기반정확도_pct", 100 - metric["WMAPE_pct"]),
        "sMAPE_pct": metric["sMAPE_pct"],
    })
    return result


def main() -> None:
    MODEL_SPECS.update(COMBINATION_SPECS)
    MODEL_SPECS.update(ADVANCED_SPECS)
    raw = pd.read_excel(DATA_PATH, sheet_name="모델데이터")
    frame = add_external_features(build_safety_features(raw))
    for version in [20]:
        train_one(version, frame)

    summary = pd.DataFrame([read_result(v) for v in range(1, 21)])
    design = []
    all_combinations = {**COMBINATION_SPECS, **ADVANCED_SPECS}
    for version in range(1, 21):
        if version == 1:
            design.append({"차수": "1차", "공통": "안전기준표", "추가변수": "없음"})
        elif version in all_combinations:
            spec = all_combinations[version]
            common = "12차(안전기준표+현재 사고+최근 추세)" if version >= 15 else "2차(안전기준표+현재 사고)"
            base = {CURRENT, *TREND_FEATURES} if version >= 15 else {CURRENT}
            design.append({"차수": f"{version}차", "공통": common,
                           "추가변수": ", ".join(f for f in spec["features"] if f not in base)})
        else:
            info = json.loads((MODEL_ROOT / f"{version}차" / "model_info.json").read_text(encoding="utf-8"))
            design.append({"차수": f"{version}차", "공통": "안전기준표", "추가변수": ", ".join(info["추가특성"])})

    output = MODEL_ROOT / "모델_1차_20차_성능비교.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능비교", index=False)
        pd.DataFrame(design).to_excel(writer, sheet_name="차수별설계", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
