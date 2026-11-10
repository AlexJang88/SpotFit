from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v41 import FEATURES as V41_FEATURES
from ml.train_traffic_accident_v42 import CV_YEARS, metric_values, pipeline


VERSION = 43
OUT = ROOT / "models" / "교통사고" / f"{VERSION}차"
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, all_features, _ = build_panel()
    v42_info = json.loads(
        (ROOT / "models" / "교통사고" / "42차" / "model_info.json").read_text(encoding="utf-8")
    )
    features = [feature for feature in v42_info["최종변수"] if feature not in V41_FEATURES]
    if len(features) != 15:
        raise ValueError(f"42차 전용 변수가 15개가 아닙니다: {len(features)}")
    missing = [feature for feature in features if feature not in all_features]
    if missing:
        raise ValueError(f"학습 패널에 없는 변수가 있습니다: {missing}")

    folds = []
    for year in CV_YEARS:
        train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
        valid = panel[panel["예측연도"] == year]
        xtr = train[features].to_numpy(float)
        xva = valid[features].to_numpy(float)
        median = np.nanmedian(xtr, axis=0)
        median = np.where(np.isnan(median), 0, median)
        xtr = np.where(np.isnan(xtr), median, xtr)
        xva = np.where(np.isnan(xva), median, xva)
        mean = xtr.mean(axis=0)
        scale = xtr.std(axis=0)
        scale[scale == 0] = 1
        xtr = (xtr - mean) / scale
        xva = (xva - mean) / scale
        ytr = train[TARGET].to_numpy(float)
        folds.append({
            "year": year,
            "gram": xtr.T @ xtr,
            "cross": xtr.T @ (ytr - ytr.mean()),
            "xva": xva,
            "yva": valid[TARGET].to_numpy(float),
            "ymean": ytr.mean(),
        })

    rows = []
    evaluated_fits = 0
    for size in range(1, len(features) + 1):
        for indices in itertools.combinations(range(len(features)), size):
            ix = np.fromiter(indices, dtype=int)
            for alpha in ALPHAS:
                yearly = []
                for fold in folds:
                    gram = fold["gram"][np.ix_(ix, ix)]
                    beta = np.linalg.solve(
                        gram + alpha * np.eye(len(ix)),
                        fold["cross"][ix],
                    )
                    prediction = np.maximum(fold["ymean"] + fold["xva"][:, ix] @ beta, 0)
                    yearly.append(metric_values(fold["yva"], prediction))
                    evaluated_fits += 1
                rows.append({
                    "변수수": len(ix),
                    "alpha": alpha,
                    "평균_MAE": np.mean([row["MAE"] for row in yearly]),
                    "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
                    "평균_R2": np.mean([row["R2"] for row in yearly]),
                    "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
                    "변수": " | ".join(features[i] for i in indices),
                })
        best_so_far = min(row["평균_MAE"] for row in rows)
        print(f"{size}개 변수 전수검사 완료: 현재 최고 CV MAE={best_so_far:.6f}", flush=True)

    subset_frame = pd.DataFrame(rows).sort_values(["평균_MAE", "평균_RMSE", "변수수"]).reset_index(drop=True)
    subset_frame.insert(0, "순위", range(1, len(subset_frame) + 1))
    subset_frame.head(1000).to_csv(OUT / "상위_1000개_부분집합.csv", index=False, encoding="utf-8-sig")

    candidates = [
        ("Ridge_0.1", pipeline(Ridge(alpha=0.1), True)),
        ("Ridge_1", pipeline(Ridge(alpha=1), True)),
        ("Ridge_10", pipeline(Ridge(alpha=10), True)),
        ("Lasso_0.01", pipeline(Lasso(alpha=0.01, max_iter=50000), True)),
        ("ElasticNet_0.01", pipeline(ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=50000), True)),
        ("RandomForest", pipeline(RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.7, random_state=43, n_jobs=1))),
        ("ExtraTrees", pipeline(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7, random_state=43, n_jobs=1))),
        ("GradientBoosting", pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=43))),
    ]
    algorithm_rows = []
    unique_subsets = subset_frame.drop_duplicates("변수").head(30)
    for _, subset_row in unique_subsets.iterrows():
        subset = subset_row["변수"].split(" | ")
        for model_name, estimator in candidates:
            yearly = []
            for year in CV_YEARS:
                train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
                valid = panel[panel["예측연도"] == year]
                fitted = clone(estimator).fit(train[subset], train[TARGET])
                prediction = np.maximum(fitted.predict(valid[subset]), 0)
                yearly.append(metric_values(valid[TARGET], prediction))
            algorithm_rows.append({
                "부분집합순위": int(subset_row["순위"]),
                "모델": model_name,
                "변수수": len(subset),
                "평균_MAE": np.mean([row["MAE"] for row in yearly]),
                "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
                "평균_R2": np.mean([row["R2"] for row in yearly]),
                "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
                "변수": " | ".join(subset),
            })
        print(f"상위 부분집합 {int(subset_row['순위'])} 알고리즘 비교 완료", flush=True)

    algorithm_frame = pd.DataFrame(algorithm_rows).sort_values(["평균_MAE", "평균_RMSE", "변수수"])
    selected = algorithm_frame.iloc[0]
    selected_features = selected["변수"].split(" | ")
    selected_name = selected["모델"]
    selected_estimator = next(model for name, model in candidates if name == selected_name)

    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    holdout_model = clone(selected_estimator).fit(train[selected_features], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[selected_features]), 0)
    model_metrics = metric_values(holdout[TARGET], prediction)

    rate_2023 = panel[panel["연도"] == 2023].set_index("자치구")[CURRENT_RATE]
    fixed_prediction = holdout["자치구"].map(rate_2023).to_numpy(float)
    fixed_metrics = metric_values(holdout[TARGET], fixed_prediction)
    persistence_metrics = metric_values(holdout[TARGET], holdout[CURRENT_RATE].to_numpy(float))

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = clone(selected_estimator).fit(labeled[selected_features], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[selected_features]), 0
    )
    joblib.dump({
        "version": VERSION,
        "model_name": selected_name,
        "target": TARGET,
        "features": selected_features,
        "model": final_model,
    }, OUT / "교통사고_43차_예측모델.joblib")

    holdout_output = holdout[["연도", "예측연도", "자치구", TARGET, CURRENT_RATE]].copy()
    holdout_output["43차예측"] = prediction
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - prediction)
    performance = pd.DataFrame([
        {"구분": "43차", "모델": selected_name, **model_metrics},
        {"구분": "2023년 고정 초기모델", "모델": "Fixed2023", **fixed_metrics},
        {"구분": "직전연도 유지모델", "모델": "Persistence", **persistence_metrics},
    ])
    with pd.ExcelWriter(OUT / "43차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        performance.to_excel(writer, sheet_name="2025성능비교", index=False)
        algorithm_frame.to_excel(writer, sheet_name="알고리즘비교", index=False)
        subset_frame.head(1000).to_excel(writer, sheet_name="상위부분집합", index=False)
        holdout_output.to_excel(writer, sheet_name="2025독립검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        pd.DataFrame({"최종변수": selected_features}).to_excel(writer, sheet_name="최종변수", index=False)

    metadata = {
        "모델명": "교통사고 43차 모델",
        "설계": "42차에만 존재하는 15개 컬럼의 모든 부분집합 전수검사",
        "후보변수수": len(features),
        "부분집합수": 2 ** len(features) - 1,
        "Ridge_순차검증학습횟수": evaluated_fits,
        "선정모델": selected_name,
        "최종변수수": len(selected_features),
        "최종변수": selected_features,
        "2021_2024_순차검증": {
            key: float(selected[key])
            for key in ["평균_MAE", "평균_RMSE", "평균_R2", "평균_WMAPE_pct"]
        },
        "2025_독립검증": model_metrics,
        "2025_2023년고정초기모델": fixed_metrics,
        "2025_직전연도유지모델": persistence_metrics,
        "초기모델대비_MAE개선율_pct": (fixed_metrics["MAE"] - model_metrics["MAE"]) / fixed_metrics["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
