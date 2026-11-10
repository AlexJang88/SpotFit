from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v42 import CV_YEARS, metric_values, pipeline
from ml.train_traffic_accident_v44 import FEATURES as V44_FEATURES, prepare_panel


VERSION = 45
OUT = ROOT / "models" / "교통사고" / f"{VERSION}차"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)


def evaluate(panel, features: list[str], alpha: float) -> dict:
    yearly = []
    estimator = pipeline(Ridge(alpha=alpha), True)
    for year in CV_YEARS:
        train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
        valid = panel[panel["예측연도"] == year]
        model = clone(estimator).fit(train[features], train[TARGET])
        prediction = np.maximum(model.predict(valid[features]), 0)
        yearly.append(metric_values(valid[TARGET], prediction))
    return {
        "평균_MAE": np.mean([row["MAE"] for row in yearly]),
        "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
        "평균_R2": np.mean([row["R2"] for row in yearly]),
        "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    _, all_features, _ = build_panel()
    candidates = [feature for feature in all_features if feature not in V44_FEATURES]
    tested: dict[tuple[str, ...], dict] = {}

    def test(features, design):
        key = tuple(dict.fromkeys(features))
        if key in tested:
            return tested[key]
        best = None
        for alpha in ALPHAS:
            result = evaluate(panel, list(key), alpha)
            row = {
                "설계": design,
                "변수수": len(key),
                "alpha": alpha,
                **result,
                "변수": " | ".join(key),
            }
            if best is None or (row["평균_MAE"], row["평균_RMSE"]) < (best["평균_MAE"], best["평균_RMSE"]):
                best = row
        tested[key] = best
        return best

    base = test(V44_FEATURES, "44차 기준")
    addition_rows = [test([*V44_FEATURES, feature], f"1개 추가: {feature}") for feature in candidates]
    removal_rows = [
        test([feature for feature in V44_FEATURES if feature != removed], f"1개 제거: {removed}")
        for removed in V44_FEATURES
    ]
    top_additions = [
        row["변수"].split(" | ")[-1]
        for row in sorted(addition_rows, key=lambda row: row["평균_MAE"])[:15]
    ]
    pair_rows = [
        test([*V44_FEATURES, first, second], f"2개 추가: {first} + {second}")
        for first, second in itertools.combinations(top_additions, 2)
    ]
    top_swap_candidates = top_additions[:10]
    swap_rows = [
        test(
            [feature for feature in V44_FEATURES if feature != removed] + [added],
            f"교체: {removed} → {added}",
        )
        for removed in V44_FEATURES
        for added in top_swap_candidates
    ]
    results = pd.DataFrame([base, *addition_rows, *removal_rows, *pair_rows, *swap_rows])
    results = results.sort_values(["평균_MAE", "평균_RMSE", "변수수"]).reset_index(drop=True)
    results.insert(0, "순위", range(1, len(results) + 1))
    selected = results.iloc[0]
    selected_features = selected["변수"].split(" | ")
    selected_alpha = float(selected["alpha"])

    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    estimator = pipeline(Ridge(alpha=selected_alpha), True)
    holdout_model = clone(estimator).fit(train[selected_features], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[selected_features]), 0)
    holdout_metrics = metric_values(holdout[TARGET], prediction)

    info44 = json.loads((ROOT / "models" / "교통사고" / "44차" / "model_info.json").read_text(encoding="utf-8"))
    metrics44 = info44["2025_독립검증"]
    rate_2023 = panel[panel["연도"] == 2023].set_index("자치구")[CURRENT_RATE]
    fixed_metrics = metric_values(holdout[TARGET], holdout["자치구"].map(rate_2023).to_numpy(float))

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = clone(estimator).fit(labeled[selected_features], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[selected_features]), 0
    )
    joblib.dump({
        "version": VERSION,
        "model_name": f"Ridge_{selected_alpha:g}",
        "target": TARGET,
        "features": selected_features,
        "model": final_model,
    }, OUT / "교통사고_45차_예측모델.joblib")

    comparison = pd.DataFrame([
        {"구분": "45차", **holdout_metrics},
        {"구분": "44차", **metrics44},
        {"구분": "2023년 고정 초기모델", **fixed_metrics},
    ])
    with pd.ExcelWriter(OUT / "45차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="2025성능비교", index=False)
        results.to_excel(writer, sheet_name="점진조합탐색", index=False)
        pd.DataFrame({"최종변수": selected_features}).to_excel(writer, sheet_name="최종변수", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)

    metadata = {
        "모델명": "교통사고 45차 모델",
        "설계": "44차 기준 1개 추가·제거·교체 및 상위 변수쌍 추가 탐색",
        "평가조합수": len(results),
        "선정설계": selected["설계"],
        "alpha": selected_alpha,
        "최종변수수": len(selected_features),
        "최종변수": selected_features,
        "2021_2024_순차검증": {
            key: float(selected[key])
            for key in ["평균_MAE", "평균_RMSE", "평균_R2", "평균_WMAPE_pct"]
        },
        "2025_독립검증": holdout_metrics,
        "2025_44차": metrics44,
        "44차대비_MAE개선율_pct": (metrics44["MAE"] - holdout_metrics["MAE"]) / metrics44["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
