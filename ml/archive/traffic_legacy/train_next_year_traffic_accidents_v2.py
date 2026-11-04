"""Train a district-level next-year traffic accident forecasting model (v2)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "안전"
OUT = ROOT / "artifacts" / "traffic_accident_next_year_v2"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["연도", "자치구"]
RANDOM_STATE = 42


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def merge_optional(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    extra = extra.copy()
    extra["연도"] = pd.to_numeric(extra["연도"], errors="raise").astype(int)
    if extra.duplicated(KEYS).any():
        raise ValueError("duplicate year-district keys")
    return base.merge(extra, on=KEYS, how="left", validate="one_to_one")


def build_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_excel(DATA / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
    frame = frame.sort_values(KEYS).reset_index(drop=True)

    supplements = [
        read_csv("자치구별_운전자안전_2차.csv"),
        read_csv("자치구별_유흥단란주점_1차.csv"),
        read_csv("자치구별_소음진동_1차.csv"),
        read_csv("자치구별_인구활동량_1차.csv"),
        read_csv("자치구별_기상_1차.csv"),
    ]
    for extra in supplements:
        frame = merge_optional(frame, extra)

    grouped = frame.groupby("자치구", sort=False)
    frame["사고건수_1년전"] = grouped["사고건수"].shift(1)
    frame["사고건수_최근3년평균"] = grouped["사고건수"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    frame["사고건수_전년증감"] = frame["사고건수"] - frame["사고건수_1년전"]
    frame["자동차등록대수_전년증감률"] = grouped["자동차등록대수"].pct_change(fill_method=None)
    frame["인구활동자료_가용"] = frame["생활인구_연평균"].notna().astype(int)
    frame["기상자료_가용"] = frame["최고기온평균"].notna().astype(int)
    frame["단속자료_가용"] = frame[["CCTV전체누적", "음주단속적발건수"]].notna().any(axis=1).astype(int)
    frame["다음연도_사고건수"] = grouped["사고건수"].shift(-1)
    frame["예측대상연도"] = frame["연도"] + 1

    core = [
        "연도", "사고건수", "사고건수_1년전", "사고건수_최근3년평균", "사고건수_전년증감",
        "사망자수", "부상자수", "사상자수", "치사율(%)",
        "운전자안전벨트착용률", "음주운전율", "자동차등록대수", "자동차등록대수_전년증감률",
        "유흥주점및단란주점_연말기준_영업중업소수",
        "소음진동_전체민원건수", "교통소음진동_민원건수",
    ]
    augmented = core + [
        "어린이사고_사고건수", "어린이사고_사망자수", "어린이사고_부상자수",
        "생활인구_연평균", "생활인구_연중시간최대", "생활인구_월변동성",
        "상주인구_연평균", "유동인구_연평균",
        "폭염일수", "결빙일수", "최고기온평균", "최저기온평균",
        "CCTV전체누적", "과속신호카메라누적", "교통단속CCTV누적", "음주단속적발건수",
        "인구활동자료_가용", "기상자료_가용", "단속자료_가용",
    ]
    missing = set(augmented).difference(frame.columns)
    if missing:
        raise ValueError(f"missing model columns: {sorted(missing)}")
    return frame, core, augmented


def pipeline(model_name: str, numeric: list[str]) -> Pipeline:
    scale = model_name == "ridge"
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["자치구"]),
    ], verbose_feature_names_out=False)
    if model_name == "ridge":
        model = Ridge(alpha=100.0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.9,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocessor", pre), ("model", model)])


def score(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(pred, 0)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAPE_pct": float(mean_absolute_percentage_error(y, pred) * 100),
        "R2": float(r2_score(y, pred)),
    }


def evaluate(frame: pd.DataFrame, feature_sets: dict[str, list[str]]):
    labeled = frame[frame["다음연도_사고건수"].notna()].copy()
    test_years = list(range(2019, 2026))
    rows, predictions = [], []

    for target_year in test_years:
        train = labeled[labeled["예측대상연도"] < target_year]
        test = labeled[labeled["예측대상연도"] == target_year]
        if train.empty or test.empty:
            continue
        y_test = test["다음연도_사고건수"]
        baselines = {
            "baseline_persistence": test["사고건수"].to_numpy(),
            "baseline_recent3_mean": test["사고건수_최근3년평균"].to_numpy(),
        }
        for name, pred in baselines.items():
            met = score(y_test, pred)
            rows.append({"model": name, "target_year": target_year, **met})
            for idx, value in zip(test.index, pred):
                predictions.append({"model": name, "target_year": target_year,
                                    "자치구": frame.loc[idx, "자치구"], "actual": frame.loc[idx, "다음연도_사고건수"],
                                    "prediction": max(float(value), 0)})

        for set_name, features in feature_sets.items():
            for model_name in ("ridge", "random_forest", "extra_trees"):
                name = f"{set_name}_{model_name}"
                model = pipeline(model_name, features)
                model.fit(train[features + ["자치구"]], train["다음연도_사고건수"])
                pred = np.maximum(model.predict(test[features + ["자치구"]]), 0)
                met = score(y_test, pred)
                rows.append({"model": name, "target_year": target_year, **met})
                for idx, value in zip(test.index, pred):
                    predictions.append({"model": name, "target_year": target_year,
                                        "자치구": frame.loc[idx, "자치구"], "actual": frame.loc[idx, "다음연도_사고건수"],
                                        "prediction": float(value)})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def main() -> None:
    frame, core, augmented = build_dataset()
    feature_sets = {"core": core, "augmented": augmented}
    metrics, wf_predictions = evaluate(frame, feature_sets)
    comparison = metrics.groupby("model", as_index=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), MAPE_pct=("MAPE_pct", "mean"), R2=("R2", "mean")
    ).sort_values("MAE")
    selected = comparison.iloc[0]["model"]
    ml_comparison = comparison[~comparison.model.str.startswith("baseline")]
    best_ml_name = ml_comparison.iloc[0]["model"]
    set_name, model_name = best_ml_name.rsplit("_", 2)[0], best_ml_name.split("_")[-2] + "_" + best_ml_name.split("_")[-1] if best_ml_name.endswith(("random_forest", "extra_trees")) else best_ml_name.split("_")[-1]
    # rsplit above is ambiguous for underscored model names; resolve explicitly.
    for candidate in ("random_forest", "extra_trees", "ridge"):
        suffix = "_" + candidate
        if best_ml_name.endswith(suffix):
            set_name, model_name = best_ml_name[:-len(suffix)], candidate
            break
    features = feature_sets[set_name]
    labeled = frame[frame["다음연도_사고건수"].notna()].copy()
    best_ml = pipeline(model_name, features)
    best_ml.fit(labeled[features + ["자치구"]], labeled["다음연도_사고건수"])
    joblib.dump({"pipeline": best_ml, "features": features, "model_name": best_ml_name,
                 "target": "다음연도_사고건수"}, OUT / "best_ml_model.joblib")

    future = frame[frame["연도"] == 2025].copy()
    ml_pred = np.maximum(best_ml.predict(future[features + ["자치구"]]), 0)
    forecasts = pd.DataFrame({
        "예측연도": 2026, "자치구": future["자치구"], "2025_사고건수": future["사고건수"],
        "지속성기준선_예측": future["사고건수"], "최우수ML모델": best_ml_name,
        "ML_예측": np.round(ml_pred, 1),
    })
    if selected == "baseline_persistence":
        forecasts["최종선택모델"] = selected
        forecasts["최종예측"] = forecasts["지속성기준선_예측"]
    elif selected == "baseline_recent3_mean":
        forecasts["최종선택모델"] = selected
        forecasts["최종예측"] = np.round(future["사고건수_최근3년평균"], 1)
    else:
        forecasts["최종선택모델"] = selected
        # selected ML may differ from best_ml only if ties; best_ml is the top ML row.
        forecasts["최종예측"] = forecasts["ML_예측"]

    transformed_names = best_ml.named_steps["preprocessor"].get_feature_names_out()
    estimator = best_ml.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": transformed_names, "importance": estimator.feature_importances_})
    else:
        importance = pd.DataFrame({"feature": transformed_names, "importance": np.abs(estimator.coef_)})
    importance = importance.sort_values("importance", ascending=False)

    frame.to_csv(OUT / "modeling_dataset.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "metrics_by_year.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    wf_predictions.to_csv(OUT / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    forecasts.to_csv(OUT / "predictions_2026.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "target": "district accident count in year t+1", "validation_target_years": [2019, 2025],
        "selected_overall": selected, "best_ml_model": best_ml_name,
        "selected_mae": float(comparison.iloc[0].MAE),
        "best_ml_mae": float(ml_comparison.iloc[0].MAE), "training_rows": len(labeled),
        "excluded": ["대중교통 Q4: only 2024-2025 Q4", "도로시설: 2026 snapshot"],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
