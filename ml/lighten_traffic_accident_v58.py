from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.lighten_traffic_accident_v57 import evaluate, importance_order, refit_component
from ml.search_traffic_accident_v52_forward_backward import prepare_data


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "57차"
OUTPUT = ROOT / "models" / "교통사고" / "58차"
REPORT = ROOT / "artifacts" / "reports" / "traffic_v58_fine_lightweight"


def candidate_sizes(total: int, horizon: int, component: str) -> list[int]:
    lower = {
        (2, "a"): max(15, total - 7),
        (2, "b"): max(45, total - 7),
        (3, "a"): max(35, total - 5),
        (3, "b"): max(45, total - 7),
    }[(horizon, component)]
    return list(range(lower, total + 1))


def is_not_worse(metrics: dict, baseline: dict) -> bool:
    eps = 1e-12
    return (
        metrics["exact"] + eps >= baseline["exact"]
        and metrics["within1"] + eps >= baseline["within1"]
        and metrics["error"] <= baseline["error"] + eps
        and metrics["min_exact"] + eps >= baseline["min_exact"]
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    selections = {}
    final_metrics = {}
    details = []

    for horizon in (2, 3):
        bundle = joblib.load(
            SOURCE / f"교통사고_57차_{2025 + horizon}년_{horizon}년후_모델.joblib"
        )
        original_a = list(bundle["component_a"]["features"])
        original_b = list(bundle["component_b"]["features"])
        baseline, _ = evaluate(data, bundle, horizon, original_a, original_b)
        rank_a = importance_order(bundle["component_a"])
        rank_b = importance_order(bundle["component_b"])
        trials = []
        accepted = []

        for size_a in candidate_sizes(len(original_a), horizon, "a"):
            selected_a = set(rank_a[:size_a])
            features_a = [feature for feature in original_a if feature in selected_a]
            for size_b in candidate_sizes(len(original_b), horizon, "b"):
                selected_b = set(rank_b[:size_b])
                features_b = [feature for feature in original_b if feature in selected_b]
                metrics, detail = evaluate(data, bundle, horizon, features_a, features_b)
                row = {
                    "예측간격": horizon,
                    "구성A변수수": size_a,
                    "구성B변수수": size_b,
                    "고유변수수": len(set(features_a) | set(features_b)),
                    "57차이상": is_not_worse(metrics, baseline),
                    **metrics,
                }
                trials.append(row)
                if row["57차이상"]:
                    # Performance first; when completely tied, prefer fewer features.
                    score = (
                        metrics["exact"],
                        metrics["within1"],
                        -metrics["error"],
                        metrics["min_exact"],
                        -row["고유변수수"],
                        -size_a - size_b,
                    )
                    accepted.append((score, row, features_a, features_b, detail))

        trial_frame = pd.DataFrame(trials).sort_values(
            ["57차이상", "exact", "within1", "error", "min_exact", "고유변수수"],
            ascending=[False, False, False, True, False, True],
        )
        trial_frame.to_excel(REPORT / f"{horizon}년후_1개단위_경량화탐색.xlsx", index=False)
        best = max(accepted, key=lambda item: item[0])
        _, best_row, features_a, features_b, detail = best

        fitted = {key: value for key, value in bundle.items() if key not in {"component_a", "component_b"}}
        fitted.update(
            version=58,
            selection="57차 구성별 중요도 경계구간 1개 단위 경량화",
            component_a=refit_component(bundle["component_a"], data, horizon, features_a),
            component_b=refit_component(bundle["component_b"], data, horizon, features_b),
            v57_component_a_count=len(original_a),
            v57_component_b_count=len(original_b),
        )
        joblib.dump(
            fitted,
            OUTPUT / f"교통사고_58차_{2025 + horizon}년_{horizon}년후_모델.joblib",
        )
        final_metrics[horizon] = {
            key: best_row[key] for key in ("exact", "within1", "error", "min_exact")
        }
        selections[horizon] = {
            "component_a_features": features_a,
            "component_b_features": features_b,
            "component_a_count": len(features_a),
            "component_b_count": len(features_b),
            "unique_count": len(set(features_a) | set(features_b)),
        }
        details.append(detail)
        print(json.dumps({"horizon": horizon, "baseline": baseline, "best": best_row}, ensure_ascii=False), flush=True)

    summary = pd.DataFrame(
        [
            {
                "차수": 58,
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
        for component in ("component_a", "component_b"):
            for order, feature in enumerate(selections[horizon][f"{component}_features"], 1):
                feature_rows.append(
                    {"예측간격": horizon, "구성요소": component, "순서": order, "변수": feature}
                )
    with pd.ExcelWriter(OUTPUT / "58차_검증결과.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능요약", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="연도별검증", index=False)
        pd.DataFrame(feature_rows).to_excel(writer, sheet_name="모델별변수", index=False)

    info = {
        "모델명": "교통사고 58차 추가 경량화 장기등급 모델",
        "기반": "57차",
        "선정방식": "구성별 중요도 경계구간 1개 단위 탐색, 57차 전 성능지표 이상만 허용",
        "변수수": {
            f"{horizon}년후": {
                key: value for key, value in selections[horizon].items() if key.endswith("count")
            }
            for horizon in (2, 3)
        },
        "성능": {
            "2년후": final_metrics[2],
            "3년후": final_metrics[3],
            "평균정확등급": float(np.mean([final_metrics[2]["exact"], final_metrics[3]["exact"]])),
            "평균_1칸이내": float(np.mean([final_metrics[2]["within1"], final_metrics[3]["within1"]])),
            "평균등급오차": float(np.mean([final_metrics[2]["error"], final_metrics[3]["error"]])),
        },
    }
    (OUTPUT / "model_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
