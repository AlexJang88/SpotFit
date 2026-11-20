"""교통사고 안전기준표 기반 자치구 상대점수 산출 및 1년 직접 예측."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "안전" / "교통사고" / "교통사고_안전기준표_2017_2025.xlsx"
CCTV_ROAD = ROOT / "data" / "interim" / "안전" / "교통사고안전지수" / "서울시_자치구_도로면적당_교통단속CCTV_2017_2025.csv"
INTERIM = ROOT / "data" / "interim" / "안전" / "교통사고안전지수"
OUT = ROOT / "artifacts" / "traffic_safety_forecast" / "standard_1y_2017_2025"
INDICATORS = {
    "기준1_인구1만명당_교통사고사망자수": {"구분": "위해", "원가중치": 50.0, "방향": "위험"},
    "기준2_도로면적당_교통단속CCTV대수": {"구분": "경감", "원가중치": 17.24, "방향": "보호"},
    "기준3_운전자안전벨트착용률": {"구분": "의식", "원가중치": 8.99, "방향": "보호"},
    "기준4_음주운전경험률": {"구분": "의식", "원가중치": 8.70, "방향": "위험"},
}
CATEGORY_ORIGINAL_WEIGHTS = {"위해": 50.0, "취약": 10.0, "경감": 20.0, "의식": 20.0}
EXCLUDED = [
    ("의식", "자동차등록대수당 교통법규위반건수", 2.31, "2017~2025 자료 없음"),
    ("취약", "인구1만명당 재난약자수", 7.91, "2017~2025 자료 없음"),
    ("취약", "인구1만명당 의료보장사업장수", 2.09, "2024년만 존재"),
    ("경감", "지역교통 안전환경개선사업 지출액", 2.76, "2017~2025 자료 없음"),
]


def adjusted_weights() -> dict[str, float]:
    available_categories = {spec["구분"] for spec in INDICATORS.values()}
    available_total = sum(CATEGORY_ORIGINAL_WEIGHTS[category] for category in available_categories)
    missing_weight = 100.0 - available_total
    category_weights = {
        category: CATEGORY_ORIGINAL_WEIGHTS[category]
        + missing_weight * CATEGORY_ORIGINAL_WEIGHTS[category] / available_total
        for category in available_categories
    }
    weights = {}
    for category in available_categories:
        category_indicators = {
            name: spec for name, spec in INDICATORS.items() if spec["구분"] == category
        }
        indicator_total = sum(spec["원가중치"] for spec in category_indicators.values())
        for name, spec in category_indicators.items():
            weights[name] = category_weights[category] * spec["원가중치"] / indicator_total
    return weights


def score_panel() -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_excel(INPUT, sheet_name="기준패널_2017_2025")
    cctv = pd.read_csv(CCTV_ROAD, encoding="utf-8-sig").rename(
        columns={"도로면적당_교통단속CCTV대수_km2": "기준2_도로면적당_교통단속CCTV대수"}
    )
    data = data.merge(
        cctv[["연도", "자치구", "기준2_도로면적당_교통단속CCTV대수"]],
        on=["연도", "자치구"], how="left", validate="one_to_one",
    )
    weights = adjusted_weights()
    if len(data) != 225 or data["자치구"].nunique() != 25 or data["연도"].nunique() != 9:
        raise ValueError("기준패널이 9개년×25개 구 구조가 아닙니다")
    contributions = []
    for column, spec in INDICATORS.items():
        if data.groupby("연도")[column].count().min() != 25:
            raise ValueError(f"완전하지 않은 지표: {column}")
        mean = data.groupby("연도")[column].transform("mean")
        std = data.groupby("연도")[column].transform(lambda x: x.std(ddof=0))
        z = (data[column] - mean) / std
        safety_z = -z if spec["방향"] == "위험" else z
        data[f"{column}_안전방향표준점수"] = safety_z
        data[f"{column}_가중기여도"] = safety_z * weights[column]
        contributions.append(data[f"{column}_가중기여도"])
    data["교통안전_대체점수"] = pd.concat(contributions, axis=1).sum(axis=1)
    data["서울25개구_순위"] = data.groupby("연도")["교통안전_대체점수"].rank(ascending=False, method="first").astype(int)
    data["상대안전등급"] = ((data["서울25개구_순위"] - 1) // 5 + 1).astype(int)
    data["상대안전등급_문자"] = data["상대안전등급"].map(dict(enumerate("ABCDE", 1)))
    return data, weights


def models() -> dict[str, object]:
    return {
        "Ridge_1": Ridge(alpha=1), "Ridge_10": Ridge(alpha=10), "Ridge_100": Ridge(alpha=100),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.03, random_state=42),
    }


def make_pipeline(estimator: object, numeric: list[str]) -> Pipeline:
    prep = ColumnTransformer([("district", OneHotEncoder(handle_unknown="ignore"), ["자치구"]),
                              ("numeric", StandardScaler(), numeric)])
    return Pipeline([("prepare", prep), ("model", estimator)])


def add_grade(frame: pd.DataFrame, score: str, prefix: str) -> None:
    frame[f"{prefix}_순위"] = frame[score].rank(ascending=False, method="first").astype(int)
    frame[f"{prefix}_등급"] = ((frame[f"{prefix}_순위"] - 1) // 5 + 1).astype(int)
    frame[f"{prefix}_등급문자"] = frame[f"{prefix}_등급"].map(dict(enumerate("ABCDE", 1)))


def evaluate(frame: pd.DataFrame, prediction: str, prefix: str) -> dict[str, float]:
    actual, pred = frame["실제점수"], frame[prediction]
    return {"MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(mean_squared_error(actual, pred) ** .5), "R2": float(r2_score(actual, pred)),
            "순위_Spearman": float(spearmanr(actual, pred).statistic),
            "등급_정확도": float((frame["실제_등급"] == frame[f"{prefix}_등급"]).mean()),
            "인접등급포함_정확도": float((frame["실제_등급"].sub(frame[f"{prefix}_등급"]).abs() <= 1).mean()),
            "A등급_적중률": float(((frame["실제_등급"] == 1) & (frame[f"{prefix}_등급"] == 1)).sum() / 5)}


def main() -> int:
    data, weights = score_panel()
    indicator_columns = list(INDICATORS)
    data = data.sort_values(["자치구", "연도"])
    delta_columns = []
    for column in indicator_columns:
        delta = f"{column}_전년차"
        data[delta] = data.groupby("자치구")[column].diff().fillna(0)
        delta_columns.append(delta)
    numeric = indicator_columns + delta_columns + ["기준연도"]
    features = ["자치구"] + numeric
    current = data.rename(columns={"연도": "기준연도"})
    future = data[["연도", "자치구", "교통안전_대체점수"] + indicator_columns].copy()
    future["기준연도"] = future["연도"] - 1
    future = future.drop(columns="연도").rename(columns={
        "교통안전_대체점수": "target_score", **{c: f"target_{c}" for c in indicator_columns}})
    pairs = current.merge(future, on=["기준연도", "자치구"], validate="one_to_one")
    pairs["목표연도"] = pairs["기준연도"] + 1
    train_val, validation = pairs[pairs["목표연도"] <= 2023], pairs[pairs["목표연도"] == 2024]
    final_train, test = pairs[pairs["목표연도"] <= 2024], pairs[pairs["목표연도"] == 2025].copy()

    rows = [{"모델": "Persistence", "검증_RMSE": mean_squared_error(validation.target_score, validation.교통안전_대체점수) ** .5}]
    for name, estimator in models().items():
        model = make_pipeline(estimator, numeric)
        model.fit(train_val[features], train_val.target_score)
        rows.append({"모델": name, "검증_RMSE": mean_squared_error(validation.target_score, model.predict(validation[features])) ** .5})
    comparison = pd.DataFrame(rows).sort_values("검증_RMSE")
    selected = comparison.iloc[0].모델
    best_ml = comparison[comparison.모델 != "Persistence"].iloc[0].모델
    predictions, fitted = {}, {}
    for name in {selected, best_ml}:
        if name == "Persistence":
            predictions[name] = test.교통안전_대체점수.to_numpy()
        else:
            model = make_pipeline(models()[name], numeric)
            model.fit(final_train[features], final_train.target_score)
            predictions[name] = model.predict(test[features])
            fitted[name] = model
    result = test[["목표연도", "자치구", "target_score", "교통안전_대체점수"]].rename(
        columns={"target_score": "실제점수", "교통안전_대체점수": "전년도점수"})
    result["선택모델_예측점수"] = predictions[selected]
    result["최우수ML_예측점수"] = predictions[best_ml]
    for score, prefix in [("실제점수", "실제"), ("전년도점수", "기준"),
                          ("선택모델_예측점수", "선택"), ("최우수ML_예측점수", "최우수ML")]:
        add_grade(result, score, prefix)
    metrics = {"선택모델명": selected, "최우수ML명": best_ml,
               "전년도유지": evaluate(result, "전년도점수", "기준"),
               "선택모델": evaluate(result, "선택모델_예측점수", "선택"),
               "최우수ML": evaluate(result, "최우수ML_예측점수", "최우수ML")}

    # 안전기준 지표별로 모델을 선택한 뒤 예측 지표를 같은 연도 내 표준화해 점수를 재계산한다.
    component_rows, component_predictions, component_models = [], {}, {}
    for indicator in indicator_columns:
        target = f"target_{indicator}"
        candidates = [("Persistence", mean_squared_error(validation[target], validation[indicator]) ** .5)]
        for name, estimator in models().items():
            model = make_pipeline(estimator, numeric)
            model.fit(train_val[features], train_val[target])
            candidates.append((name, mean_squared_error(validation[target], model.predict(validation[features])) ** .5))
        chosen, validation_rmse = min(candidates, key=lambda item: item[1])
        component_rows.extend({"예측지표": indicator, "모델": name, "검증_RMSE": rmse,
                               "선택여부": name == chosen} for name, rmse in candidates)
        if chosen == "Persistence":
            component_predictions[indicator] = test[indicator].to_numpy()
        else:
            model = make_pipeline(models()[chosen], numeric)
            model.fit(final_train[features], final_train[target])
            component_predictions[indicator] = model.predict(test[features])
            component_models[indicator] = model
    component = test[["자치구"]].copy()
    component["실제점수"] = test.target_score.to_numpy()
    score_parts = []
    for indicator, values in component_predictions.items():
        component[f"예측_{indicator}"] = values
        series = component[f"예측_{indicator}"]
        z = (series - series.mean()) / series.std(ddof=0)
        if INDICATORS[indicator]["방향"] == "위험":
            z = -z
        score_parts.append(z * weights[indicator])
    component["지표별모델_예측점수"] = pd.concat(score_parts, axis=1).sum(axis=1)
    add_grade(component, "실제점수", "실제")
    add_grade(component, "지표별모델_예측점수", "지표별")
    metrics["지표별예측후산출"] = evaluate(component, "지표별모델_예측점수", "지표별")

    INTERIM.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(INTERIM / "서울시_자치구_교통안전_대체지수_2017_2025.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"구분": INDICATORS[k]["구분"], "산출지표": k,
                   "원래가중치": INDICATORS[k]["원가중치"], "조정가중치": v,
                   "방향": INDICATORS[k]["방향"]} for k, v in weights.items()]).to_csv(
        INTERIM / "재설정_가중치_2017_2025.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(EXCLUDED, columns=["구분", "제외지표", "원래가중치", "제외사유"]).to_csv(
        INTERIM / "제외지표_2017_2025.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "모델비교_검증2024.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(component_rows).to_csv(OUT / "지표별_모델비교_검증2024.csv", index=False, encoding="utf-8-sig")
    result.sort_values("선택_순위").to_csv(OUT / "2025_예측대실제.csv", index=False, encoding="utf-8-sig")
    component.sort_values("지표별_순위").to_csv(OUT / "2025_지표별예측_예측대실제.csv", index=False, encoding="utf-8-sig")
    (OUT / "평가지표.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"selected": selected, "best_ml": best_ml, "models": fitted,
                 "component_models": component_models,
                 "features": features, "weights": weights}, OUT / "traffic_safety_1y_models.joblib")
    print(comparison.to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
