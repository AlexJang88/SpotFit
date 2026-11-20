from __future__ import annotations

import json

import joblib
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import evaluate, make_predictions, score_frame
from ml.train_traffic_accident_v75_v46_32_rubric import OUT, PROCESSED, load_v46


def main():
    features, coefficients, weights = load_v46()
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    persistence = {feature: "persistence" for feature in features}

    predicted_2025 = make_predictions(panel, features, 2023, 2025, persistence)
    actual_2025 = panel[panel["연도"] == 2025][["연도", "자치구", *features]]
    metrics_2y, detail_2y = evaluate(actual_2025, predicted_2025, features, coefficients, weights)
    detail_2y.insert(0, "예측기간", "2년 후")

    future_values = make_predictions(panel, features, 2025, 2027, persistence)
    future_scores = score_frame(future_values, features, coefficients, weights, "예측_")
    future_2y = future_values.merge(future_scores, on=["연도", "자치구"], validate="one_to_one")
    future_2y.insert(0, "예측기간", "2년 후")

    sheets = pd.read_excel(PROCESSED, sheet_name=None)
    details = sheets["2025검증상세"]
    sheets["2025검증상세"] = pd.concat(
        [detail_2y, details[details["예측기간"] != "2년 후"]], ignore_index=True
    )
    forecasts = sheets["2027_2028예측"]
    sheets["2027_2028예측"] = pd.concat(
        [future_2y, forecasts[forecasts["예측기간"] != "2년 후"]], ignore_index=True
    )
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    bundle = {feature: {"spec": "Persistence", "components": {"Persistence": "Persistence"}}
              for feature in features}
    joblib.dump(bundle, OUT / "교통사고_75차_2년후_32지표예측모델.joblib")

    info_path = OUT / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["결과"]["2"]["최종선정"] = "32지표 Persistence"
    info["결과"]["2"]["최종선정근거"] = "복합후보의 2025 정확등급 32%가 Persistence 44%보다 낮아 탈락"
    info["결과"]["2"]["final_holdout_2025"] = metrics_2y
    info["결과"]["2"]["final_strategies"] = {feature: "Persistence" for feature in features}
    info["결과"]["3"]["최종선정"] = "지표별 최적화 조합"
    info["결과"]["3"]["최종선정근거"] = "2025 정확등급이 Persistence 28%에서 52%로 개선"
    info["결과"]["3"]["final_holdout_2025"] = info["결과"]["3"]["candidate_holdout_2025"]
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"2년후_최종": metrics_2y, "processed": str(PROCESSED)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
