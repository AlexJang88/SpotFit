from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v41 import FEATURES as CORE_FEATURES, make_model as make_v41_model
from ml.train_traffic_accident_v42 import CV_YEARS, metric_values, pipeline


VERSION = 44
OUT = ROOT / "models" / "교통사고" / f"{VERSION}차"
SCHOOL_FEATURE = "어린이1만명당_유치원초등학교수"
ADDED_FEATURES = [
    "어린이1만명당_자동차등록대수",
    "승용차_낮_평균속도(km/h)",
    "자동차음주운전경험률(%)",
    SCHOOL_FEATURE,
]
FEATURES = [*CORE_FEATURES, *ADDED_FEATURES]


def prepare_panel() -> pd.DataFrame:
    panel, _, _ = build_panel()
    child_population = "어린이(13세 미만) 인구수"
    panel[SCHOOL_FEATURE] = (
        (panel["유치원수"] + panel["초등학교수"])
        / panel[child_population]
        * 10_000
    )
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    missing = [feature for feature in FEATURES if feature not in panel.columns]
    if missing:
        raise ValueError(f"44차 필수 변수가 없습니다: {missing}")

    candidates = [
        ("Ridge_0.1", pipeline(Ridge(alpha=0.1), True)),
        ("Ridge_1", pipeline(Ridge(alpha=1), True)),
        ("Ridge_10", pipeline(Ridge(alpha=10), True)),
        ("Ridge_100", pipeline(Ridge(alpha=100), True)),
        ("Lasso_0.01", pipeline(Lasso(alpha=0.01, max_iter=50000), True)),
        ("ElasticNet_0.01", pipeline(ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=50000), True)),
        ("Huber", pipeline(HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=5000), True)),
        ("RandomForest", pipeline(RandomForestRegressor(n_estimators=700, min_samples_leaf=4, max_features=0.7, random_state=44, n_jobs=1))),
        ("ExtraTrees", pipeline(ExtraTreesRegressor(n_estimators=700, min_samples_leaf=3, max_features=0.7, random_state=44, n_jobs=1))),
        ("GradientBoosting", pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=44))),
    ]

    rows = []
    yearly_rows = []
    for name, estimator in candidates:
        yearly = []
        for year in CV_YEARS:
            train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
            valid = panel[panel["예측연도"] == year]
            fitted = clone(estimator).fit(train[FEATURES], train[TARGET])
            prediction = np.maximum(fitted.predict(valid[FEATURES]), 0)
            result = metric_values(valid[TARGET], prediction)
            yearly.append(result)
            yearly_rows.append({"모델": name, "검증연도": year, **result})
        rows.append({
            "모델": name,
            "평균_MAE": np.mean([row["MAE"] for row in yearly]),
            "평균_RMSE": np.mean([row["RMSE"] for row in yearly]),
            "평균_R2": np.mean([row["R2"] for row in yearly]),
            "평균_WMAPE_pct": np.mean([row["WMAPE_pct"] for row in yearly]),
        })
        print(f"{name} 순차검증 완료", flush=True)

    comparison = pd.DataFrame(rows).sort_values(["평균_MAE", "평균_RMSE"])
    selected = comparison.iloc[0]
    selected_name = selected["모델"]
    selected_estimator = next(model for name, model in candidates if name == selected_name)

    # 모델 선택이 끝난 뒤 2025년을 한 번 평가한다.
    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    holdout_model = clone(selected_estimator).fit(train[FEATURES], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[FEATURES]), 0)
    holdout_metrics = metric_values(holdout[TARGET], prediction)

    rate_2023 = panel[panel["연도"] == 2023].set_index("자치구")[CURRENT_RATE]
    fixed_metrics = metric_values(
        holdout[TARGET],
        holdout["자치구"].map(rate_2023).to_numpy(float),
    )
    persistence_metrics = metric_values(holdout[TARGET], holdout[CURRENT_RATE].to_numpy(float))

    # 기존 41차를 같은 2025년 행에서 다시 계산해 직접 비교한다.
    model41 = make_v41_model().fit(train[CORE_FEATURES], train[TARGET])
    model41_metrics = metric_values(
        holdout[TARGET],
        np.maximum(model41.predict(holdout[CORE_FEATURES]), 0),
    )

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = clone(selected_estimator).fit(labeled[FEATURES], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(
        final_model.predict(future[FEATURES]), 0
    )
    joblib.dump({
        "version": VERSION,
        "model_name": selected_name,
        "target": TARGET,
        "features": FEATURES,
        "core_features": CORE_FEATURES,
        "added_features": ADDED_FEATURES,
        "model": final_model,
    }, OUT / "교통사고_44차_예측모델.joblib")

    holdout_output = holdout[["연도", "예측연도", "자치구", TARGET, CURRENT_RATE]].copy()
    holdout_output["44차예측"] = prediction
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - prediction)
    final_comparison = pd.DataFrame([
        {"구분": "44차", "모델": selected_name, **holdout_metrics},
        {"구분": "41차", "모델": "Ridge_0.1", **model41_metrics},
        {"구분": "2023년 고정 초기모델", "모델": "Fixed2023", **fixed_metrics},
        {"구분": "직전연도 유지모델", "모델": "Persistence", **persistence_metrics},
    ])
    feature_table = pd.DataFrame([
        *({"구분": "41차 핵심 고정", "변수": feature} for feature in CORE_FEATURES),
        *({"구분": "44차 영역 대표 추가", "변수": feature} for feature in ADDED_FEATURES),
    ])
    with pd.ExcelWriter(OUT / "44차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        final_comparison.to_excel(writer, sheet_name="2025최종비교", index=False)
        comparison.to_excel(writer, sheet_name="알고리즘순차검증", index=False)
        pd.DataFrame(yearly_rows).to_excel(writer, sheet_name="연도별순차검증", index=False)
        feature_table.to_excel(writer, sheet_name="11개변수", index=False)
        holdout_output.to_excel(writer, sheet_name="2025독립검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)

    metadata = {
        "모델명": "교통사고 44차 모델",
        "설계": "41차 핵심 7개 고정 + 영역 대표변수 4개",
        "선정모델": selected_name,
        "최종변수수": len(FEATURES),
        "핵심변수": CORE_FEATURES,
        "추가변수": ADDED_FEATURES,
        "최종변수": FEATURES,
        "2021_2024_순차검증": {
            key: float(selected[key])
            for key in ["평균_MAE", "평균_RMSE", "평균_R2", "평균_WMAPE_pct"]
        },
        "2025_독립검증": holdout_metrics,
        "2025_41차": model41_metrics,
        "2025_2023년고정초기모델": fixed_metrics,
        "초기모델대비_MAE개선율_pct": (fixed_metrics["MAE"] - holdout_metrics["MAE"]) / fixed_metrics["MAE"] * 100,
        "41차대비_MAE개선율_pct": (model41_metrics["MAE"] - holdout_metrics["MAE"]) / model41_metrics["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
