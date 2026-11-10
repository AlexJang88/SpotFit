from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, TARGET, build_panel
from ml.train_traffic_accident_v42 import (
    CV_YEARS, OUT, SEARCH_OUT, metric_values, pipeline,
)


def main() -> None:
    panel, _, _ = build_panel()
    subsets = pd.read_csv(SEARCH_OUT / "상위_1000개_조합.csv", encoding="utf-8-sig")
    candidates = [
        ("Ridge_0.1", pipeline(Ridge(alpha=0.1), True)),
        ("Ridge_1", pipeline(Ridge(alpha=1.0), True)),
        ("Ridge_10", pipeline(Ridge(alpha=10.0), True)),
        ("Lasso_0.01", pipeline(Lasso(alpha=0.01, max_iter=50000), True)),
        ("ElasticNet_0.01", pipeline(ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=50000), True)),
        ("RandomForest", pipeline(RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=1))),
        ("ExtraTrees", pipeline(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=1))),
        ("GradientBoosting", pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=42))),
    ]
    rows = []
    for _, subset_row in subsets.head(30).iterrows():
        subset = subset_row["변수"].split(" | ")
        for model_name, estimator in candidates:
            yearly = []
            for year in CV_YEARS:
                train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
                valid = panel[panel["예측연도"] == year]
                fitted = clone(estimator).fit(train[subset], train[TARGET])
                prediction = np.maximum(fitted.predict(valid[subset]), 0)
                yearly.append(metric_values(valid[TARGET], prediction))
            rows.append({
                "부분집합순위": int(subset_row["순위"]),
                "모델": model_name,
                "변수수": len(subset),
                "평균_MAE": np.mean([row["MAE"] for row in yearly]),
                "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
                "평균_R2": np.mean([row["R2"] for row in yearly]),
                "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
                "변수": " | ".join(subset),
            })
        print(f"상위 부분집합 {int(subset_row['순위'])}/30 알고리즘 비교 완료", flush=True)

    comparison = pd.DataFrame(rows).sort_values(["평균_MAE", "평균_RMSE", "변수수"])
    selected = comparison.iloc[0]
    selected_features = selected["변수"].split(" | ")
    selected_model_name = selected["모델"]
    selected_estimator = next(model for name, model in candidates if name == selected_model_name)

    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    holdout_model = clone(selected_estimator).fit(train[selected_features], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[selected_features]), 0)
    model_metrics = metric_values(holdout[TARGET], prediction)
    baseline_metrics = metric_values(holdout[TARGET], holdout[CURRENT_RATE].to_numpy())
    holdout_output = holdout[["연도", "예측연도", "자치구", TARGET, CURRENT_RATE]].copy()
    holdout_output["42차예측"] = prediction
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - prediction)

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = clone(selected_estimator).fit(labeled[selected_features], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[selected_features]), 0
    )
    joblib.dump({
        "version": 42,
        "model_name": selected_model_name,
        "target": TARGET,
        "target_display_name": "다음 해 어린이 1만 명당 교통사고 피해자 수",
        "features": selected_features,
        "model": final_model,
        "selection_cv_years": CV_YEARS,
        "final_holdout_year": 2025,
    }, OUT / "교통사고_42차_예측모델.joblib")

    performance = pd.DataFrame([
        {"구분": "42차", "모델": selected_model_name, **model_metrics},
        {"구분": "직전연도 유지 초기모델", "모델": "Persistence", **baseline_metrics},
    ])
    with pd.ExcelWriter(OUT / "42차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        performance.to_excel(writer, sheet_name="2025성능비교", index=False)
        comparison.to_excel(writer, sheet_name="알고리즘비교", index=False)
        subsets.to_excel(writer, sheet_name="상위변수조합", index=False)
        holdout_output.to_excel(writer, sheet_name="2025독립검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        pd.DataFrame({"최종변수": selected_features}).to_excel(writer, sheet_name="최종변수", index=False)

    metadata = {
        "모델명": "교통사고 42차 모델",
        "후보변수수": 38,
        "탐색고유조합수": 51794,
        "선정모델": selected_model_name,
        "최종변수수": len(selected_features),
        "최종변수": selected_features,
        "2021_2024_순차검증": {
            key: float(selected[key])
            for key in ["평균_MAE", "평균_RMSE", "평균_R2", "평균_WMAPE_pct"]
        },
        "2025_독립검증": model_metrics,
        "2025_초기모델": baseline_metrics,
        "2025_MAE개선율_pct": (baseline_metrics["MAE"] - model_metrics["MAE"]) / baseline_metrics["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
