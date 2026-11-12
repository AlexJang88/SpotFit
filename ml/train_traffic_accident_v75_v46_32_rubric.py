from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import score_frame
from ml.train_traffic_accident_v72_indicator_forecast import (
    DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache,
)
from ml.train_traffic_accident_v73_indicator_ensemble import (
    evaluate_specs, fit_and_forecast, label, select_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = 75
OUT = ROOT / "models" / "교통사고" / "75차.46차32지표등급"
PROCESSED = ROOT / "data" / "processed" / "교통사고_75차_46차32지표등급_2027_2028.xlsx"
HORIZONS = (2, 3)


def load_v46():
    folder = ROOT / "models" / "교통사고" / "41차~49차" / "46차"
    info = json.loads((folder / "model_info.json").read_text(encoding="utf-8"))
    bundle = joblib.load(next(folder.glob("*.joblib")))
    features = info["최종변수"]
    coefficients = np.asarray(bundle["model"].named_steps["model"].coef_, float)
    weights = np.abs(coefficients) / np.abs(coefficients).sum()
    if len(features) != 32 or len(coefficients) != 32:
        raise ValueError("46차 32개 지표·계수 구성이 아닙니다.")
    return features, coefficients, weights


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features, coefficients, weights = load_v46()
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    missing = [feature for feature in features if feature not in panel.columns]
    if missing:
        raise ValueError(f"46차 지표 누락: {missing}")

    weight_table = pd.DataFrame({
        "지표": features,
        "46차표준화계수": coefficients,
        "위험방향": np.where(coefficients >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
        "46차고정가중치_pct": weights * 100,
    }).sort_values("46차고정가중치_pct", ascending=False)

    results, logs, metrics, details, forecasts, strategy_rows = {}, [], [], [], [], []
    for horizon in HORIZONS:
        data, methods, cache, numeric_all, _ = build_prediction_cache(panel, features, horizon)
        persistence = {feature: "Persistence" for feature in features}
        selected, dev_metrics, log = select_ensemble(
            panel, features, coefficients, weights, horizon, methods, cache, persistence
        )
        base_dev, base_dev_rows, _ = evaluate_specs(
            panel, features, coefficients, weights, horizon, DEV_YEARS, persistence, cache
        )
        candidate_hold, candidate_rows, candidate_detail = evaluate_specs(
            panel, features, coefficients, weights, horizon, [HOLDOUT_YEAR], selected, cache
        )
        base_hold, base_rows, _ = evaluate_specs(
            panel, features, coefficients, weights, horizon, [HOLDOUT_YEAR], persistence, cache
        )

        # 개발구간 정확등급이 개선된 조합만 사용하되 2025 결과를 별도로 공개한다.
        final_specs = selected if dev_metrics["정확등급정확도"] > base_dev["정확등급정확도"] else persistence
        future_values, bundle = fit_and_forecast(
            panel, data, features, horizon, final_specs, numeric_all
        )
        future_scores = score_frame(future_values, features, coefficients, weights, "예측_")
        future = future_values.merge(future_scores, on=["연도", "자치구"], validate="one_to_one")
        future.insert(0, "예측기간", f"{horizon}년 후")
        forecasts.append(future)
        joblib.dump(bundle, OUT / f"교통사고_75차_{horizon}년후_32지표예측모델.joblib")

        log.insert(0, "예측기간", f"{horizon}년 후"); logs.append(log)
        candidate_rows["구분"] = "75차후보_2025독립"
        base_rows["구분"] = "32지표Persistence_2025독립"
        base_dev_rows["구분"] = "32지표Persistence_개발"
        metrics.extend([candidate_rows, base_rows, base_dev_rows])
        candidate_detail.insert(0, "예측기간", f"{horizon}년 후"); details.append(candidate_detail)
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": feature,
                              "후보예측법": label(selected[feature]),
                              "최종예측법": label(final_specs[feature])} for feature in features)
        results[str(horizon)] = {
            "final_strategies": {feature: label(spec) for feature, spec in final_specs.items()},
            "candidate_dev": dev_metrics,
            "candidate_holdout_2025": candidate_hold,
            "persistence_dev": base_dev,
            "persistence_holdout_2025": base_hold,
        }

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="46차32지표고정가중치", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="32지표별예측법", index=False)
        pd.concat(metrics, ignore_index=True).to_excel(writer, sheet_name="5등급성능", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="모델탐색과정", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    info = {
        "모델명": "교통사고 75차 46차 32지표 미래예측·안전등급 모델",
        "구조": "미래 32개 지표 각각 예측→지표별 5분위→46차 Ridge 계수 기반 고정가중치→종합위험점수→안전 1~5등급",
        "지표수": 32,
        "가중치근거": "46차 Ridge alpha=0.1 표준화계수 절댓값을 100% 환산",
        "안전등급방향": "1=가장 안전, 5=가장 위험",
        "결과": results,
        "processed": str(PROCESSED.relative_to(ROOT)),
    }
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
