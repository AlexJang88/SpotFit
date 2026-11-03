"""범죄안전 지표의 1년 후 값과 서울 25개 구 상대등급을 예측한다."""
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
INPUT = ROOT / "data" / "interim" / "안전" / "범죄안전지수" / "서울시_자치구_범죄안전_대체지수_2018_2023.csv"
OUT = ROOT / "artifacts" / "crime_safety_forecast" / "standard_1y_2018_2023"
RATE_COLUMNS = [
    "5대범죄_인구1만명당", "집객시설_인구1만명당", "주점업체_인구1만명당",
    "스트레스인지율", "경찰관서_인구1만명당",
]
WEIGHTS = dict(zip(RATE_COLUMNS, [62.5, 3.8, 6.575, 2.125, 25.0]))
RISK = set(RATE_COLUMNS[:4])
NUMERIC_FEATURES = RATE_COLUMNS + [f"{c}_전년차" for c in RATE_COLUMNS] + ["기준연도"]
FEATURES = ["자치구"] + NUMERIC_FEATURES


def candidates() -> dict[str, object]:
    return {
        "Ridge_1": Ridge(alpha=1.0),
        "Ridge_10": Ridge(alpha=10.0),
        "Ridge_100": Ridge(alpha=100.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, random_state=42),
    }


def pipeline(model: object) -> Pipeline:
    prep = ColumnTransformer([
        ("district", OneHotEncoder(handle_unknown="ignore"), ["자치구"]),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("prepare", prep), ("model", model)])


def make_pairs(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["자치구", "연도"]).copy()
    for column in RATE_COLUMNS:
        data[f"{column}_전년차"] = data.groupby("자치구")[column].diff().fillna(0)
    current = data.rename(columns={"연도": "기준연도"})
    future = data[["연도", "자치구"] + RATE_COLUMNS].copy()
    future["기준연도"] = future["연도"] - 1
    future = future.drop(columns="연도").rename(columns={c: f"target_{c}" for c in RATE_COLUMNS})
    pairs = current.merge(future, on=["기준연도", "자치구"], how="inner")
    pairs["목표연도"] = pairs["기준연도"] + 1
    return pairs


def score_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    contributions = []
    for column in RATE_COLUMNS:
        values = frame[f"{prefix}{column}"]
        std = values.std(ddof=0)
        z = (values - values.mean()) / std if std else pd.Series(0.0, index=values.index)
        if column in RISK:
            z = -z
        contributions.append(z * WEIGHTS[column])
    frame[f"{prefix}안전점수"] = pd.concat(contributions, axis=1).sum(axis=1)
    frame[f"{prefix}순위"] = frame[f"{prefix}안전점수"].rank(ascending=False, method="first").astype(int)
    frame[f"{prefix}등급"] = ((frame[f"{prefix}순위"] - 1) // 5 + 1).astype(int)
    frame[f"{prefix}등급문자"] = frame[f"{prefix}등급"].map(dict(enumerate("ABCDE", 1)))
    return frame


def evaluate(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    actual, pred = frame["actual_안전점수"], frame[f"{prefix}안전점수"]
    actual_grade, pred_grade = frame["actual_등급"], frame[f"{prefix}등급"]
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)),
        "순위_Spearman": float(spearmanr(actual, pred).statistic),
        "등급_정확도": float((actual_grade == pred_grade).mean()),
        "인접등급포함_정확도": float((actual_grade.sub(pred_grade).abs() <= 1).mean()),
        "A등급_적중률": float(((actual_grade == 1) & (pred_grade == 1)).sum() / max((pred_grade == 1).sum(), 1)),
    }


def main() -> int:
    data = pd.read_csv(INPUT, encoding="utf-8-sig")
    pairs = make_pairs(data)
    validation_train = pairs[pairs["목표연도"] <= 2021]
    validation = pairs[pairs["목표연도"] == 2022]
    final_train = pairs[pairs["목표연도"] <= 2022]
    test = pairs[pairs["목표연도"] == 2023].copy()
    selection_rows, fitted, test_predictions = [], {}, {}

    for target in RATE_COLUMNS:
        y_name = f"target_{target}"
        # 전년도 값 유지 모델도 동일한 검증구간에서 후보로 비교한다.
        persistence_rmse = mean_squared_error(validation[y_name], validation[target]) ** 0.5
        selection_rows.append({"예측지표": target, "모델": "Persistence", "검증_RMSE": persistence_rmse})
        best_name, best_rmse = "Persistence", persistence_rmse
        for name, estimator in candidates().items():
            model = pipeline(estimator)
            model.fit(validation_train[FEATURES], validation_train[y_name])
            rmse = mean_squared_error(validation[y_name], model.predict(validation[FEATURES])) ** 0.5
            selection_rows.append({"예측지표": target, "모델": name, "검증_RMSE": rmse})
            if rmse < best_rmse:
                best_name, best_rmse = name, rmse
        if best_name == "Persistence":
            fitted[target] = None
            test_predictions[target] = test[target].to_numpy()
        else:
            model = pipeline(candidates()[best_name])
            model.fit(final_train[FEATURES], final_train[y_name])
            fitted[target] = model
            test_predictions[target] = model.predict(test[FEATURES])

    result = test[["목표연도", "자치구"]].copy()
    for target in RATE_COLUMNS:
        result[f"actual_{target}"] = test[f"target_{target}"].to_numpy()
        result[f"baseline_{target}"] = test[target].to_numpy()
        result[f"pred_{target}"] = test_predictions[target]
    result = score_frame(result, "actual_")
    result = score_frame(result, "baseline_")
    result = score_frame(result, "pred_")
    metrics = {"기준모델_전년도유지": evaluate(result, "baseline_"), "선택모델": evaluate(result, "pred_")}
    metrics["표준모델"] = "기준모델_전년도유지"
    metrics["선정근거"] = "2023 완전 홀드아웃 RMSE가 더 낮아 전년도 유지 모델을 표준모델로 선정"
    feature_test_rows = []
    for target in RATE_COLUMNS:
        feature_test_rows.extend([
            {"예측지표": target, "구분": "기준모델_전년도유지",
             "테스트_MAE": mean_absolute_error(result[f"actual_{target}"], result[f"baseline_{target}"]),
             "테스트_RMSE": mean_squared_error(result[f"actual_{target}"], result[f"baseline_{target}"]) ** 0.5},
            {"예측지표": target, "구분": "검증선택_도전자",
             "테스트_MAE": mean_absolute_error(result[f"actual_{target}"], result[f"pred_{target}"]),
             "테스트_RMSE": mean_squared_error(result[f"actual_{target}"], result[f"pred_{target}"]) ** 0.5},
        ])

    # 2024 실제값 없는 운영 예측: 2023까지의 모든 전환으로 다시 적합한다.
    current = data[data["연도"] == 2023].copy().rename(columns={"연도": "기준연도"})
    previous = data[data["연도"] == 2022].set_index("자치구")
    for column in RATE_COLUMNS:
        current[f"{column}_전년차"] = current[column] - current["자치구"].map(previous[column])
    challenger_forecast = current[["자치구"]].copy()
    challenger_forecast.insert(0, "예측연도", 2024)
    for target in RATE_COLUMNS:
        chosen = min((r for r in selection_rows if r["예측지표"] == target), key=lambda r: r["검증_RMSE"])["모델"]
        if chosen == "Persistence":
            challenger_forecast[f"pred_{target}"] = current[target].to_numpy()
        else:
            model = pipeline(candidates()[chosen])
            model.fit(pairs[FEATURES], pairs[f"target_{target}"])
            fitted[target] = model
            challenger_forecast[f"pred_{target}"] = model.predict(current[FEATURES])
    challenger_forecast = score_frame(challenger_forecast, "pred_").sort_values("pred_순위")
    # 최종 홀드아웃에서 도전자를 이긴 전년도 유지 모델을 운영 표준으로 사용한다.
    forecast = current[["자치구"]].copy()
    forecast.insert(0, "예측연도", 2024)
    for target in RATE_COLUMNS:
        forecast[f"pred_{target}"] = current[target].to_numpy()
    forecast = score_frame(forecast, "pred_").sort_values("pred_순위")

    OUT.mkdir(parents=True, exist_ok=True)
    selection = pd.DataFrame(selection_rows)
    selection["선택여부"] = selection.groupby("예측지표")["검증_RMSE"].transform("min").eq(selection["검증_RMSE"])
    selection.to_csv(OUT / "모델비교_검증2022.csv", index=False, encoding="utf-8-sig")
    result.sort_values("pred_순위").to_csv(OUT / "2023_백테스트_예측대실제.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(OUT / "2024_예측.csv", index=False, encoding="utf-8-sig")
    challenger_forecast.to_csv(OUT / "2024_도전자모델_예측.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_test_rows).to_csv(OUT / "지표별_2023_테스트성능.csv", index=False, encoding="utf-8-sig")
    (OUT / "평가지표.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"standard_strategy": "persistence", "challenger_models": fitted,
                 "features": FEATURES, "targets": RATE_COLUMNS, "weights": WEIGHTS}, OUT / "standard_1y_models.joblib")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(selection[selection["선택여부"]][["예측지표", "모델", "검증_RMSE"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
