from __future__ import annotations

import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v41 import FEATURES as V41_FEATURES


VERSION = 42
OUT = ROOT / "models" / "교통사고" / f"{VERSION}차"
SEARCH_OUT = OUT / "변수조합탐색"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
CV_YEARS = (2021, 2022, 2023, 2024)
RANDOM_ITERATIONS = 80_000
MAX_FEATURES = 25
SEED = 42


def metric_values(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, float)
    prediction = np.asarray(prediction, float)
    error = np.abs(actual - prediction)
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
        "WMAPE_pct": float(error.sum() / actual.sum() * 100),
    }


def pipeline(model, scaled: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_candidate_features() -> tuple[list[str], list[str]]:
    prior = (
        ROOT / "models" / "교통사고" / "41차" / "전체124개_부분집합탐색"
        / "상위_1000개_부분집합.csv"
    )
    row = pd.read_csv(prior, encoding="utf-8-sig").iloc[0]
    v35 = row["변수"].split(" | ")
    union = list(dict.fromkeys([*V41_FEATURES, *v35]))
    if len(union) != 38:
        raise ValueError(f"7개·35개 합집합이 38개가 아닙니다: {len(union)}")
    return union, v35


def main() -> None:
    started = time.time()
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    SEARCH_OUT.mkdir(parents=True, exist_ok=True)
    panel, all_features, _ = build_panel()
    features, v35 = load_candidate_features()
    missing = [feature for feature in features if feature not in all_features]
    if missing:
        raise ValueError(f"42차 후보 feature가 학습 패널에 없습니다: {missing}")

    # 각 순차검증 fold에서 학습기간만 이용해 결측 대체·표준화를 계산한다.
    fold_data = []
    for year in CV_YEARS:
        train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
        valid = panel[panel["예측연도"] == year]
        xtr = train[features].to_numpy(float)
        xva = valid[features].to_numpy(float)
        median = np.nanmedian(xtr, axis=0)
        median = np.where(np.isnan(median), 0.0, median)
        xtr = np.where(np.isnan(xtr), median, xtr)
        xva = np.where(np.isnan(xva), median, xva)
        mean = xtr.mean(axis=0)
        scale = xtr.std(axis=0)
        scale[scale == 0] = 1.0
        xtr = (xtr - mean) / scale
        xva = (xva - mean) / scale
        ytr = train[TARGET].to_numpy(float)
        yva = valid[TARGET].to_numpy(float)
        fold_data.append({
            "year": year,
            "gram": xtr.T @ xtr,
            "cross": xtr.T @ (ytr - ytr.mean()),
            "xva": xva,
            "yva": yva,
            "ymean": ytr.mean(),
        })

    cache: dict[tuple[int, ...], dict] = {}
    best_key: tuple[int, ...] = ()
    best_score = math.inf
    history = []
    stage = ["초기"]

    def evaluate(indices) -> dict:
        nonlocal best_key, best_score
        key = tuple(sorted(set(indices)))
        if not key or len(key) > MAX_FEATURES:
            return {"평균_MAE": math.inf}
        if key in cache:
            return cache[key]
        ix = np.fromiter(key, dtype=int)
        alpha_results = []
        for alpha in RIDGE_ALPHAS:
            yearly = []
            for fold in fold_data:
                g = fold["gram"][np.ix_(ix, ix)]
                c = fold["cross"][ix]
                beta = np.linalg.solve(g + alpha * np.eye(len(ix)), c)
                pred = np.maximum(fold["ymean"] + fold["xva"][:, ix] @ beta, 0)
                yearly.append(metric_values(fold["yva"], pred))
            alpha_results.append({
                "alpha": alpha,
                "평균_MAE": float(np.mean([row["MAE"] for row in yearly])),
                "평균_RMSE": float(np.mean([row["RMSE"] for row in yearly])),
                "평균_R2": float(np.mean([row["R2"] for row in yearly])),
                "평균_WMAPE_pct": float(np.mean([row["WMAPE_pct"] for row in yearly])),
                "연도별": yearly,
            })
        result = min(alpha_results, key=lambda row: (row["평균_MAE"], row["평균_RMSE"]))
        cache[key] = result
        if result["평균_MAE"] < best_score - 1e-12:
            best_score = result["평균_MAE"]
            best_key = key
            history.append({
                "평가조합수": len(cache),
                "단계": stage[0],
                "변수수": len(key),
                **{k: v for k, v in result.items() if k != "연도별"},
                "변수": " | ".join(features[i] for i in key),
            })
        return result

    # 기존 7개, 과적합 35개(상한 때문에 앞 25개), 단일 변수를 모두 시작점으로 평가한다.
    v41_key = tuple(features.index(feature) for feature in V41_FEATURES)
    evaluate(v41_key)
    evaluate(tuple(features.index(feature) for feature in v35[:MAX_FEATURES]))
    for i in range(len(features)):
        evaluate((i,))

    stage[0] = "전진선택"
    current = min(((evaluate((i,))["평균_MAE"], (i,)) for i in range(len(features))))[1]
    no_improvement = 0
    for size in range(2, MAX_FEATURES + 1):
        selected = set(current)
        options = []
        for candidate in range(len(features)):
            if candidate not in selected:
                key = tuple(sorted((*current, candidate)))
                options.append((evaluate(key)["평균_MAE"], key))
        score, proposed = min(options)
        old_score = evaluate(current)["평균_MAE"]
        current = proposed
        no_improvement = 0 if score < old_score - 1e-12 else no_improvement + 1
        print(f"전진선택 {size}개: CV MAE={score:.6f}, 전체최고={best_score:.6f}", flush=True)
        if no_improvement >= 4:
            break

    stage[0] = "부동선택"
    current = best_key
    while True:
        current_score = evaluate(current)["평균_MAE"]
        neighbors = []
        if len(current) > 1:
            for item in current:
                key = tuple(i for i in current if i != item)
                neighbors.append((evaluate(key)["평균_MAE"], key))
        if len(current) < MAX_FEATURES:
            selected = set(current)
            for item in range(len(features)):
                if item not in selected:
                    key = tuple(sorted((*current, item)))
                    neighbors.append((evaluate(key)["평균_MAE"], key))
        score, proposed = min(neighbors)
        if score >= current_score - 1e-12:
            break
        current = proposed

    # 여러 seed 주변을 simulated annealing으로 탐색한다.
    stage[0] = "확률교체"
    seeds = [best_key, v41_key]
    current = seeds[0]
    current_score = evaluate(current)["평균_MAE"]
    for iteration in range(1, RANDOM_ITERATIONS + 1):
        if iteration % 10_000 == 1:
            current = seeds[(iteration // 10_000) % len(seeds)] if iteration <= 20_001 else best_key
            current_score = evaluate(current)["평균_MAE"]
        proposal = set(current)
        move = rng.random()
        if move < 0.30 and len(proposal) < MAX_FEATURES:
            proposal.add(rng.randrange(len(features)))
        elif move < 0.55 and len(proposal) > 1:
            proposal.remove(rng.choice(tuple(proposal)))
        else:
            if proposal:
                proposal.remove(rng.choice(tuple(proposal)))
            proposal.add(rng.randrange(len(features)))
        key = tuple(sorted(proposal))
        score = evaluate(key)["평균_MAE"]
        temperature = max(0.001, 0.06 * (1 - iteration / RANDOM_ITERATIONS))
        if score < current_score or rng.random() < math.exp((current_score - score) / temperature):
            current, current_score = key, score
        if iteration % 10_000 == 0:
            print(f"확률탐색 {iteration:,}: 고유조합={len(cache):,}, 최고 CV MAE={best_score:.6f}", flush=True)

    ranked = sorted(cache.items(), key=lambda item: (item[1]["평균_MAE"], len(item[0])))
    subset_rows = []
    for rank, (key, result) in enumerate(ranked[:1000], 1):
        subset_rows.append({
            "순위": rank,
            "변수수": len(key),
            **{k: v for k, v in result.items() if k != "연도별"},
            "변수": " | ".join(features[i] for i in key),
        })
    pd.DataFrame(subset_rows).to_csv(SEARCH_OUT / "상위_1000개_조합.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(SEARCH_OUT / "최고성능_갱신이력.csv", index=False, encoding="utf-8-sig")

    # 상위 30개 조합을 여러 알고리즘으로 동일 순차검증한다.
    candidates = [
        ("Ridge_0.1", pipeline(Ridge(alpha=0.1), True)),
        ("Ridge_1", pipeline(Ridge(alpha=1.0), True)),
        ("Ridge_10", pipeline(Ridge(alpha=10.0), True)),
        ("Lasso_0.01", pipeline(Lasso(alpha=0.01, max_iter=50000), True)),
        ("ElasticNet_0.01", pipeline(ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=50000), True)),
        ("RandomForest", pipeline(RandomForestRegressor(n_estimators=600, min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1))),
        ("ExtraTrees", pipeline(ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1))),
        ("GradientBoosting", pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=42))),
        ("HistGradientBoosting", pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=42))),
    ]
    algorithm_rows = []
    for subset_rank, (key, _) in enumerate(ranked[:30], 1):
        subset = [features[i] for i in key]
        for model_name, estimator in candidates:
            yearly = []
            for year in CV_YEARS:
                train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
                valid = panel[panel["예측연도"] == year]
                fitted = clone(estimator).fit(train[subset], train[TARGET])
                prediction = np.maximum(fitted.predict(valid[subset]), 0)
                yearly.append(metric_values(valid[TARGET], prediction))
            algorithm_rows.append({
                "부분집합순위": subset_rank,
                "모델": model_name,
                "변수수": len(subset),
                "평균_MAE": np.mean([row["MAE"] for row in yearly]),
                "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
                "평균_R2": np.mean([row["R2"] for row in yearly]),
                "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
                "변수": " | ".join(subset),
            })
    algorithm_frame = pd.DataFrame(algorithm_rows).sort_values(["평균_MAE", "평균_RMSE", "변수수"])
    selected_row = algorithm_frame.iloc[0]
    selected_features = selected_row["변수"].split(" | ")
    selected_model_name = selected_row["모델"]
    selected_estimator = next(model for name, model in candidates if name == selected_model_name)

    # 여기서 처음으로 2025년 독립검증을 수행한다.
    train_2025 = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout_2025 = panel[panel["예측연도"] == 2025]
    holdout_model = clone(selected_estimator).fit(train_2025[selected_features], train_2025[TARGET])
    holdout_prediction = np.maximum(holdout_model.predict(holdout_2025[selected_features]), 0)
    holdout_metrics = metric_values(holdout_2025[TARGET], holdout_prediction)
    baseline_prediction = holdout_2025[CURRENT_RATE].to_numpy(float)
    baseline_metrics = metric_values(holdout_2025[TARGET], baseline_prediction)
    holdout_output = holdout_2025[["연도", "예측연도", "자치구", TARGET, CURRENT_RATE]].copy()
    holdout_output["42차예측"] = holdout_prediction
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - holdout_output["42차예측"])

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = clone(selected_estimator).fit(labeled[selected_features], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[selected_features]), 0
    )
    payload = {
        "version": VERSION,
        "model_name": selected_model_name,
        "target": TARGET,
        "target_display_name": "다음 해 어린이 1만 명당 교통사고 피해자 수",
        "features": selected_features,
        "model": final_model,
        "selection_cv_years": CV_YEARS,
        "final_holdout_year": 2025,
    }
    joblib.dump(payload, OUT / "교통사고_42차_예측모델.joblib")

    comparison = pd.DataFrame([
        {"구분": "42차", "모델": selected_model_name, **holdout_metrics},
        {"구분": "직전연도 유지 초기모델", "모델": "Persistence", **baseline_metrics},
    ])
    with pd.ExcelWriter(OUT / "42차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="2025성능비교", index=False)
        algorithm_frame.to_excel(writer, sheet_name="알고리즘비교", index=False)
        pd.DataFrame(subset_rows).to_excel(writer, sheet_name="상위변수조합", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="성능갱신이력", index=False)
        holdout_output.to_excel(writer, sheet_name="2025독립검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        pd.DataFrame({"최종변수": selected_features}).to_excel(writer, sheet_name="최종변수", index=False)

    metadata = {
        "모델명": "교통사고 42차 모델",
        "후보변수수": len(features),
        "탐색고유조합수": len(cache),
        "선정모델": selected_model_name,
        "최종변수수": len(selected_features),
        "최종변수": selected_features,
        "2021_2024_순차검증": {
            key: float(selected_row[key])
            for key in ["평균_MAE", "평균_RMSE", "평균_R2", "평균_WMAPE_pct"]
        },
        "2025_독립검증": holdout_metrics,
        "2025_초기모델": baseline_metrics,
        "2025_MAE개선율_pct": (baseline_metrics["MAE"] - holdout_metrics["MAE"]) / baseline_metrics["MAE"] * 100,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
