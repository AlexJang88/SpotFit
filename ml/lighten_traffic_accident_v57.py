from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "56차"
OUTPUT = ROOT / "models" / "교통사고" / "57차"
REPORT = ROOT / "artifacts" / "reports" / "traffic_v57_lightweight"


def importance_order(component: dict) -> list[str]:
    fitted = component["model"].named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(fitted.coef_, dtype=float)).mean(axis=0)
    pairs = sorted(zip(component["features"], values), key=lambda item: item[1], reverse=True)
    return [name for name, _ in pairs]


def size_candidates(total: int) -> list[int]:
    anchors = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, total]
    return sorted({min(total, value) for value in anchors if min(total, value) >= 8})


def evaluate(data: pd.DataFrame, bundle: dict, horizon: int, features_a: list[str], features_b: list[str]):
    target = f"{horizon}년후_위험등급"
    year_col = f"{horizon}년후_목표연도"
    labeled = data[data[target].notna()].copy()
    rows = []
    for year in CV_YEARS[horizon]:
        train = labeled[labeled[year_col] < year]
        valid = labeled[labeled[year_col] == year]
        actual = valid[target].astype(int).to_numpy()
        predictions = []
        for component, features in ((bundle["component_a"], features_a), (bundle["component_b"], features_b)):
            model = clone(component["model"]).fit(train[features], train[target].astype(int))
            predictions.append(risk(model, valid[features]))
        combined = bundle["weight_a"] * predictions[0] + (1 - bundle["weight_a"]) * predictions[1]
        predicted = grades_from_risk(combined)
        distance = np.abs(actual - predicted)
        rows.append(
            {
                "예측간격": horizon,
                "검증목표연도": year,
                "정확등급정확도": float(np.mean(actual == predicted)),
                "±1등급정확도": float(np.mean(distance <= 1)),
                "평균등급오차": float(np.mean(distance)),
            }
        )
    detail = pd.DataFrame(rows)
    metrics = {
        "exact": float(detail["정확등급정확도"].mean()),
        "within1": float(detail["±1등급정확도"].mean()),
        "error": float(detail["평균등급오차"].mean()),
        "min_exact": float(detail["정확등급정확도"].min()),
    }
    return metrics, detail


def refit_component(component: dict, data: pd.DataFrame, horizon: int, features: list[str]) -> dict:
    target = f"{horizon}년후_위험등급"
    labeled = data[data[target].notna()]
    result = {key: value for key, value in component.items() if key not in {"model", "features"}}
    result["features"] = features
    result["model"] = clone(component["model"]).fit(labeled[features], labeled[target].astype(int))
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    final_metrics = {}
    final_details = []
    selections = {}

    for horizon in (2, 3):
        source_file = SOURCE / f"교통사고_56차_{2025 + horizon}년_{horizon}년후_모델.joblib"
        bundle = joblib.load(source_file)
        order_a = importance_order(bundle["component_a"])
        order_b = importance_order(bundle["component_b"])
        trials = []
        best = None
        for size_a in size_candidates(len(order_a)):
            for size_b in size_candidates(len(order_b)):
                # Select by importance, but preserve the original training-column order.
                # Tree models with a fixed random seed can still change when columns are
                # merely permuted, so this keeps the comparison against 56차 fair.
                selected_a = set(order_a[:size_a])
                selected_b = set(order_b[:size_b])
                features_a = [f for f in bundle["component_a"]["features"] if f in selected_a]
                features_b = [f for f in bundle["component_b"]["features"] if f in selected_b]
                metrics, detail = evaluate(data, bundle, horizon, features_a, features_b)
                row = {
                    "예측간격": horizon,
                    "구성A변수수": size_a,
                    "구성B변수수": size_b,
                    "고유변수수": len(set(features_a) | set(features_b)),
                    **metrics,
                }
                trials.append(row)
                candidate = (metrics["exact"], metrics["within1"], -metrics["error"], metrics["min_exact"], -row["고유변수수"])
                if best is None or candidate > best[0]:
                    best = (candidate, row, features_a, features_b, detail)
        pd.DataFrame(trials).sort_values(
            ["exact", "within1", "error", "min_exact", "고유변수수"],
            ascending=[False, False, True, False, True],
        ).to_excel(REPORT / f"{horizon}년후_변수개수조합탐색.xlsx", index=False)

        _, best_row, features_a, features_b, detail = best
        fitted = {key: value for key, value in bundle.items() if key not in {"component_a", "component_b"}}
        fitted.update(
            version=57,
            selection="56차 중요도 기반 구성별 변수 개수 교차검증",
            component_a=refit_component(bundle["component_a"], data, horizon, features_a),
            component_b=refit_component(bundle["component_b"], data, horizon, features_b),
            v56_component_a_count=len(bundle["component_a"]["features"]),
            v56_component_b_count=len(bundle["component_b"]["features"]),
        )
        joblib.dump(fitted, OUTPUT / f"교통사고_57차_{2025 + horizon}년_{horizon}년후_모델.joblib")
        final_metrics[horizon] = {key: best_row[key] for key in ("exact", "within1", "error", "min_exact")}
        final_details.append(detail)
        selections[horizon] = {
            "component_a_features": features_a,
            "component_b_features": features_b,
            "component_a_count": len(features_a),
            "component_b_count": len(features_b),
            "unique_count": len(set(features_a) | set(features_b)),
        }
        print(json.dumps({"horizon": horizon, "best": best_row}, ensure_ascii=False), flush=True)

    summary = pd.DataFrame(
        [
            {
                "차수": 57,
                "예측간격": horizon,
                "구성A변수수": selections[horizon]["component_a_count"],
                "구성B변수수": selections[horizon]["component_b_count"],
                "고유변수수": selections[horizon]["unique_count"],
                **final_metrics[horizon],
            }
            for horizon in (2, 3)
        ]
    )
    feature_rows = []
    for horizon in (2, 3):
        for component_name in ("component_a", "component_b"):
            for order, feature in enumerate(selections[horizon][f"{component_name}_features"], 1):
                feature_rows.append({"예측간격": horizon, "구성요소": component_name, "순서": order, "변수": feature})
    with pd.ExcelWriter(OUTPUT / "57차_검증결과.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능요약", index=False)
        pd.concat(final_details, ignore_index=True).to_excel(writer, sheet_name="연도별검증", index=False)
        pd.DataFrame(feature_rows).to_excel(writer, sheet_name="모델별변수", index=False)

    info = {
        "모델명": "교통사고 57차 경량화 장기등급 모델",
        "기반": "56차",
        "선정방식": "앙상블 구성별 중요도 상위 변수 개수 조합 교차검증",
        "변수수": {f"{h}년후": {k: v for k, v in selections[h].items() if k.endswith("count")} for h in (2, 3)},
        "성능": {
            "2년후": final_metrics[2],
            "3년후": final_metrics[3],
            "평균정확등급": float(np.mean([final_metrics[2]["exact"], final_metrics[3]["exact"]])),
            "평균_1칸이내": float(np.mean([final_metrics[2]["within1"], final_metrics[3]["within1"]])),
            "평균등급오차": float(np.mean([final_metrics[2]["error"], final_metrics[3]["error"]])),
        },
    }
    (OUTPUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
