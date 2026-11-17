"""Forecast next-year traffic-accident fatality risk following 안전기준표.png."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
SAFE = ROOT / "data" / "processed" / "안전"
OUT = ROOT / "artifacts" / "traffic_accident_safety_standard"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["연도", "자치구"]
TARGET = "다음연도_인구1만명당_교통사고사망자수"
RANDOM_STATE = 42


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SAFE / name, encoding="utf-8-sig")


def build_data():
    accidents = pd.read_excel(SAFE / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    drivers = csv("자치구별_운전자안전_2차.csv")
    population = csv("자치구별_인구활동량_1차.csv")
    for d in (accidents, drivers, population):
        d["연도"] = pd.to_numeric(d["연도"], errors="raise").astype(int)
        if d.duplicated(KEYS).any():
            raise ValueError("duplicate keys")
    frame = accidents.merge(drivers, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(population, on=KEYS, how="left", validate="one_to_one")
    frame = frame.sort_values(KEYS).reset_index(drop=True)

    pop = frame["생활인구_연평균"].replace(0, np.nan)
    frame["인구1만명당_교통사고사망자수"] = frame["사망자수"] / pop * 10_000
    frame["인구1만명당_자동차등록대수"] = frame["자동차등록대수"] / pop * 10_000
    frame["인구1만명당_CCTV전체누적"] = frame["CCTV전체누적"] / pop * 10_000
    frame["인구1만명당_과속신호카메라누적"] = frame["과속신호카메라누적"] / pop * 10_000
    frame["인구1만명당_교통단속CCTV누적"] = frame["교통단속CCTV누적"] / pop * 10_000

    g = frame.groupby("자치구", sort=False)
    frame["사망위험_1년전"] = g["인구1만명당_교통사고사망자수"].shift(1)
    frame["사망위험_최근3년평균"] = g["인구1만명당_교통사고사망자수"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    frame["사고건수_1년전"] = g["사고건수"].shift(1)
    frame["사고건수_최근3년평균"] = g["사고건수"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    frame[TARGET] = g["인구1만명당_교통사고사망자수"].shift(-1)
    frame["예측대상연도"] = frame["연도"] + 1

    criteria = [
        "연도", "인구1만명당_교통사고사망자수", "사망위험_1년전", "사망위험_최근3년평균",
        "생활인구_연평균", "운전자안전벨트착용률", "음주운전율",
        "인구1만명당_자동차등록대수", "인구1만명당_CCTV전체누적",
        "인구1만명당_과속신호카메라누적", "인구1만명당_교통단속CCTV누적",
    ]
    history = criteria + [
        "사고건수", "사고건수_1년전", "사고건수_최근3년평균", "사망자수", "부상자수",
        "어린이사고_사고건수", "어린이사고_사망자수", "어린이사고_부상자수",
        "생활인구_연중시간최대", "생활인구_월변동성",
    ]
    return frame, {"기준표대응": criteria, "기준표_사고이력확장": history}


def make_model(kind: str, features: list[str]) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "ridge":
        steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(steps), features),
        ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["자치구"]),
    ], verbose_feature_names_out=False)
    models = {
        "ridge": Ridge(alpha=30.0),
        "random_forest": RandomForestRegressor(
            n_estimators=700, min_samples_leaf=3, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=1),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700, min_samples_leaf=3, max_features=0.9,
            random_state=RANDOM_STATE, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=2,
            min_samples_leaf=4, loss="huber", random_state=RANDOM_STATE),
    }
    return Pipeline([("preprocessor", prep), ("model", models[kind])])


def metrics(y, pred):
    pred = np.maximum(pred, 0)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)),
    }


def main():
    frame, feature_sets = build_data()
    # Both predictor-year and target-year risk rates must be observable.
    labeled = frame[
        frame[TARGET].notna() & frame["인구1만명당_교통사고사망자수"].notna()
    ].copy()
    folds, prediction_rows = [], []
    for target_year in range(2021, 2026):
        train = labeled[labeled["예측대상연도"] < target_year]
        test = labeled[labeled["예측대상연도"] == target_year]
        if train.empty or test.empty:
            continue
        y = test[TARGET]
        baseline = test["인구1만명당_교통사고사망자수"].to_numpy()
        folds.append({"모델": "전년도유지_기준선", "예측연도": target_year, **metrics(y, baseline)})
        for district, actual, pred in zip(test["자치구"], y, baseline):
            prediction_rows.append({"모델":"전년도유지_기준선","예측연도":target_year,"자치구":district,"실제":actual,"예측":pred})
        for set_name, features in feature_sets.items():
            for kind in ("ridge", "random_forest", "extra_trees", "gradient_boosting"):
                name = f"{set_name}__{kind}"
                model = make_model(kind, features)
                model.fit(train[features + ["자치구"]], train[TARGET])
                pred = np.maximum(model.predict(test[features + ["자치구"]]), 0)
                folds.append({"모델": name, "예측연도": target_year, **metrics(y, pred)})
                for district, actual, value in zip(test["자치구"], y, pred):
                    prediction_rows.append({"모델":name,"예측연도":target_year,"자치구":district,"실제":actual,"예측":value})

    fold_df = pd.DataFrame(folds)
    comparison = fold_df.groupby("모델", as_index=False).agg(
        MAE=("MAE","mean"), RMSE=("RMSE","mean"), R2=("R2","mean")
    ).sort_values("MAE").reset_index(drop=True)
    winner = comparison.iloc[0]["모델"]
    ml_rows = comparison[comparison["모델"] != "전년도유지_기준선"]
    best_ml_name = ml_rows.iloc[0]["모델"]
    set_name, kind = best_ml_name.split("__", 1)
    features = feature_sets[set_name]
    model = make_model(kind, features)
    model.fit(labeled[features + ["자치구"]], labeled[TARGET])
    joblib.dump({"pipeline": model, "features": features, "target": TARGET,
                 "population_proxy": "생활인구_연평균", "model_name": best_ml_name}, OUT / "best_ml_model.joblib")

    future = frame[frame["연도"] == 2025].copy()
    ml_pred = np.maximum(model.predict(future[features + ["자치구"]]), 0)
    forecast = pd.DataFrame({
        "예측연도": 2026, "자치구": future["자치구"],
        "2025_인구1만명당_교통사고사망자수": future["인구1만명당_교통사고사망자수"],
        "최우수ML모델": best_ml_name, "ML예측": np.round(ml_pred, 4),
    })
    if winner == "전년도유지_기준선":
        forecast["최종선택모델"] = winner
        forecast["최종예측"] = forecast["2025_인구1만명당_교통사고사망자수"]
    else:
        forecast["최종선택모델"] = winner
        forecast["최종예측"] = forecast["ML예측"]

    names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.abs(estimator.coef_)
    importance = pd.DataFrame({"특성": names, "중요도": values}).sort_values("중요도", ascending=False)

    frame.to_csv(OUT/"modeling_dataset.csv",index=False,encoding="utf-8-sig")
    fold_df.to_csv(OUT/"metrics_by_year.csv",index=False,encoding="utf-8-sig")
    comparison.to_csv(OUT/"model_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(OUT/"walk_forward_predictions.csv",index=False,encoding="utf-8-sig")
    forecast.to_csv(OUT/"predictions_2026.csv",index=False,encoding="utf-8-sig")
    importance.to_csv(OUT/"feature_importance.csv",index=False,encoding="utf-8-sig")
    metadata = {
        "기준표_목표": "인구1만명당 교통사고 사망자수",
        "실제_목표": TARGET,
        "인구_대리변수": "생활인구_연평균",
        "검증연도": [2021,2022,2023,2024,2025],
        "전체최우수": winner,
        "최우수ML": best_ml_name,
        "전체최우수_MAE": float(comparison.iloc[0].MAE),
        "최우수ML_MAE": float(ml_rows.iloc[0].MAE),
        "학습행수": len(labeled),
        "제외": {
            "교차로수": "2026년 단면자료라 과거 예측에 사용 시 미래정보 누출",
            "도로면적": "processed/안전에 없음",
            "재난약자·숙박의료업소": "processed/안전에 없음",
            "대중교통": "기준표 직접 지표가 아니며 2024~2025 Q4만 존재",
        },
    }
    (OUT/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    report = f"""# 교통사고 안전기준 모델

- 목표: 다음 연도 인구 1만 명당 교통사고 사망자수
- 인구 대리변수: 생활인구 연평균(정확한 주민등록인구가 아니므로 해석 주의)
- 검증: 2021~2025년 walk-forward
- 전체 최우수: {winner}
- 최우수 ML: {best_ml_name}
- 전체 최우수 MAE: {comparison.iloc[0].MAE:.6f}
- 최우수 ML MAE: {ml_rows.iloc[0].MAE:.6f}

2026년 교차로 현황은 과거 학습에 넣지 않았으며, 결측값을 0으로 간주하지 않았습니다.
"""
    (OUT/"MODEL_REPORT.md").write_text(report,encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
