from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v42 import CV_YEARS, metric_values, pipeline
from ml.train_traffic_accident_v44 import SCHOOL_FEATURE, prepare_panel


VERSION = 46
OUT = ROOT / "models" / "교통사고" / "41차~49차" / f"{VERSION}차"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    _, raw_features, _ = build_panel()
    info45 = json.loads((ROOT / "models" / "교통사고" / "41차~49차" / "45차" / "model_info.json").read_text(encoding="utf-8"))
    base_features = info45["최종변수"]
    candidate_features = list(dict.fromkeys([*raw_features, SCHOOL_FEATURE]))
    feature_index = {feature: index for index, feature in enumerate(candidate_features)}

    fold_data = []
    for year in CV_YEARS:
        train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
        valid = panel[panel["예측연도"] == year]
        xtr = train[candidate_features].to_numpy(float)
        xva = valid[candidate_features].to_numpy(float)
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
        fold_data.append({
            "gram": xtr.T @ xtr,
            "cross": xtr.T @ (ytr - ytr.mean()),
            "xva": xva,
            "yva": valid[TARGET].to_numpy(float),
            "ymean": ytr.mean(),
        })

    cache = {}

    def evaluate(features):
        key = tuple(features)
        if key in cache:
            return cache[key]
        indices = np.array([feature_index[feature] for feature in key])
        best = None
        for alpha in ALPHAS:
            yearly = []
            for fold in fold_data:
                gram = fold["gram"][np.ix_(indices, indices)]
                beta = np.linalg.solve(
                    gram + alpha * np.eye(len(indices)),
                    fold["cross"][indices],
                )
                prediction = np.maximum(fold["ymean"] + fold["xva"][:, indices] @ beta, 0)
                yearly.append(metric_values(fold["yva"], prediction))
            row = {
                "alpha": alpha,
                "평균_MAE": np.mean([item["MAE"] for item in yearly]),
                "평균_RMSE": np.mean([item["RMSE"] for item in yearly]),
                "평균_R2": np.mean([item["R2"] for item in yearly]),
                "평균_WMAPE_pct": np.mean([item["WMAPE_pct"] for item in yearly]),
            }
            if best is None or (row["평균_MAE"], row["평균_RMSE"]) < (best["평균_MAE"], best["평균_RMSE"]):
                best = row
        cache[key] = best
        return best

    selected = list(base_features)
    current = evaluate(selected)
    history = [{
        "단계": 0,
        "추가변수": "45차 기준",
        "변수수": len(selected),
        **current,
        "변수": " | ".join(selected),
    }]
    step = 0
    while True:
        remaining = [feature for feature in candidate_features if feature not in selected]
        trials = []
        for feature in remaining:
            result = evaluate([*selected, feature])
            trials.append((result["평균_MAE"], result["평균_RMSE"], feature, result))
        _, _, best_feature, best_result = min(trials)
        if best_result["평균_MAE"] >= current["평균_MAE"] - 1e-12:
            print(f"추가 개선 없음: 탐색 종료, 다음 후보={best_feature}", flush=True)
            break
        step += 1
        selected.append(best_feature)
        current = best_result
        history.append({
            "단계": step,
            "추가변수": best_feature,
            "변수수": len(selected),
            **current,
            "변수": " | ".join(selected),
        })
        print(f"{step}단계 추가: {best_feature}, CV MAE={current['평균_MAE']:.6f}", flush=True)

    alpha = float(current["alpha"])
    estimator = pipeline(Ridge(alpha=alpha), True)
    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    holdout_model = estimator.fit(train[selected], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[selected]), 0)
    holdout_metrics = metric_values(holdout[TARGET], prediction)
    metrics45 = info45["2025_독립검증"]

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = pipeline(Ridge(alpha=alpha), True).fit(labeled[selected], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[selected]), 0
    )
    joblib.dump({
        "version": VERSION,
        "model_name": f"Ridge_{alpha:g}",
        "target": TARGET,
        "features": selected,
        "model": final_model,
    }, OUT / "교통사고_46차_예측모델.joblib")

    comparison = pd.DataFrame([
        {"구분": "46차", **holdout_metrics},
        {"구분": "45차", **metrics45},
    ])
    with pd.ExcelWriter(OUT / "46차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="2025성능비교", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="단계별변수추가", index=False)
        pd.DataFrame({"최종변수": selected}).to_excel(writer, sheet_name="최종변수", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)

    metadata = {
        "모델명": "교통사고 46차 모델",
        "설계": "45차에서 순차검증 MAE가 개선되는 변수만 하나씩 전진 추가",
        "평가조합수": len(cache),
        "추가단계수": step,
        "alpha": alpha,
        "최종변수수": len(selected),
        "최종변수": selected,
        "단계별추가": [row["추가변수"] for row in history[1:]],
        "2021_2024_순차검증": current,
        "2025_독립검증": holdout_metrics,
        "2025_45차": metrics45,
        "45차대비_MAE개선율_pct": (metrics45["MAE"] - holdout_metrics["MAE"]) / metrics45["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
