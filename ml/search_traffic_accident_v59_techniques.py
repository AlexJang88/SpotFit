from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "58차"
OUTPUT = ROOT / "models" / "교통사고" / "59차"
REPORT = ROOT / "artifacts" / "reports" / "traffic_v59_techniques"


def scaled(model):
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)]
    )


def regressors():
    models = {}
    for alpha in (0.1, 1, 10, 100, 1000):
        models[f"Ridge_{alpha:g}"] = scaled(Ridge(alpha=alpha))
    for alpha in (0.001, 0.01, 0.1):
        models[f"ElasticNet_{alpha:g}"] = scaled(
            ElasticNet(alpha=alpha, l1_ratio=0.25, max_iter=10000, random_state=42)
        )
    for leaf in (2, 5, 10):
        models[f"ExtraTrees_{leaf}"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=leaf, max_features=0.7, random_state=42, n_jobs=1)),
            ]
        )
        models[f"RandomForest_{leaf}"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=350, min_samples_leaf=leaf, max_features=0.7, random_state=42, n_jobs=1)),
            ]
        )
    for depth in (1, 2, 3):
        models[f"GBR_{depth}"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=0.025, max_depth=depth, loss="huber", random_state=42)),
            ]
        )
    return models


def component_importance(component: dict) -> dict[str, float]:
    fitted = component["model"].named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(fitted.coef_, dtype=float)).mean(axis=0)
    values = values / (values.sum() or 1)
    return dict(zip(component["features"], values))


def feature_sets(bundle: dict) -> dict[str, list[str]]:
    scores = {}
    for component in (bundle["component_a"], bundle["component_b"]):
        for feature, value in component_importance(component).items():
            scores[feature] = scores.get(feature, 0.0) + float(value)
    ranked = [feature for feature, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    union = list(dict.fromkeys(bundle["component_a"]["features"] + bundle["component_b"]["features"]))
    sets = {f"top{size}": ranked[: min(size, len(ranked))] for size in (15, 20, 25, 30, 35, 40, 50)}
    sets["v58_union"] = union
    return {name: list(dict.fromkeys(features)) for name, features in sets.items()}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    distance = np.abs(actual - predicted)
    return float(np.mean(actual == predicted)), float(np.mean(distance <= 1)), float(np.mean(distance))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    definitions = regressors()
    winners = {}
    all_details = []
    all_summaries = []
    fitted_specs = {}

    for horizon in (2, 3):
        source_bundle = joblib.load(
            SOURCE / f"교통사고_58차_{2025 + horizon}년_{horizon}년후_모델.joblib"
        )
        sets = feature_sets(source_bundle)
        target_score = f"{horizon}년후_종합위험점수"
        target_grade = f"{horizon}년후_위험등급"
        target_year = f"{horizon}년후_목표연도"
        labeled = data[data[target_score].notna()].copy()
        predictions = {}
        rows = []

        # Forecast anchors are included so every learned model must beat legitimate simple baselines.
        anchors = {
            "anchor_current": "현재연도_종합위험점수",
            "anchor_mean2": "점수_평균2",
            "anchor_mean3": "점수_평균3",
        }
        for name, column in anchors.items():
            for year in CV_YEARS[horizon]:
                valid = labeled[labeled[target_year] == year]
                score = valid[column].to_numpy(dtype=float)
                predictions[(name, year)] = score
                actual = valid[target_grade].astype(int).to_numpy()
                exact, within1, error = metrics(actual, grades_from_risk(score))
                rows.append([horizon, name, year, exact, within1, error, 1])
            fitted_specs[name] = {"kind": "anchor", "column": column, "features": [column]}

        for set_name, features in sets.items():
            for model_name, estimator in definitions.items():
                for target_kind in ("score", "delta"):
                    fold_raw = {}
                    for year in CV_YEARS[horizon]:
                        train = labeled[labeled[target_year] < year]
                        valid = labeled[labeled[target_year] == year]
                        y = train[target_score].to_numpy(dtype=float)
                        if target_kind == "delta":
                            y = y - train["현재연도_종합위험점수"].to_numpy(dtype=float)
                        model = clone(estimator).fit(train[features], y)
                        raw = model.predict(valid[features])
                        if target_kind == "delta":
                            raw = raw + valid["현재연도_종합위험점수"].to_numpy(dtype=float)
                        fold_raw[year] = raw
                    for blend in (0.0, 0.2, 0.4, 0.6, 0.8):
                        config = f"{model_name}@{set_name}|{target_kind}|b{blend:g}"
                        for year in CV_YEARS[horizon]:
                            valid = labeled[labeled[target_year] == year]
                            current = valid["현재연도_종합위험점수"].to_numpy(dtype=float)
                            score = (1 - blend) * fold_raw[year] + blend * current
                            predictions[(config, year)] = score
                            actual = valid[target_grade].astype(int).to_numpy()
                            exact, within1, error = metrics(actual, grades_from_risk(score))
                            rows.append([horizon, config, year, exact, within1, error, len(features)])
                        fitted_specs[(horizon, config)] = {
                            "kind": "regression_rank",
                            "estimator": estimator,
                            "features": features,
                            "target_kind": target_kind,
                            "blend": blend,
                        }

        detail = pd.DataFrame(
            rows,
            columns=["예측간격", "모델", "검증목표연도", "정확등급정확도", "±1등급정확도", "평균등급오차", "변수수"],
        )
        summary = detail.groupby(["예측간격", "모델", "변수수"], as_index=False).agg(
            exact=("정확등급정확도", "mean"),
            within1=("±1등급정확도", "mean"),
            error=("평균등급오차", "mean"),
            min_exact=("정확등급정확도", "min"),
        )

        # Blend the strongest diverse score forecasts; ranking happens only after blending.
        top = list(summary.sort_values(["exact", "within1", "error"], ascending=[False, False, True]).head(30)["모델"])
        ensemble_rows = []
        for index, left in enumerate(top):
            for right in top[index + 1 :]:
                for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
                    name = f"ENS[{left}]+[{right}]|w{weight:g}"
                    for year in CV_YEARS[horizon]:
                        valid = labeled[labeled[target_year] == year]
                        score = weight * predictions[(left, year)] + (1 - weight) * predictions[(right, year)]
                        actual = valid[target_grade].astype(int).to_numpy()
                        exact, within1, error = metrics(actual, grades_from_risk(score))
                        ensemble_rows.append([horizon, name, year, exact, within1, error, -1])
                        predictions[(name, year)] = score
                    fitted_specs[(horizon, name)] = {"kind": "ensemble_rank", "left": left, "right": right, "weight": weight}
        detail = pd.concat([detail, pd.DataFrame(ensemble_rows, columns=detail.columns)], ignore_index=True)
        summary = detail.groupby(["예측간격", "모델", "변수수"], as_index=False).agg(
            exact=("정확등급정확도", "mean"), within1=("±1등급정확도", "mean"),
            error=("평균등급오차", "mean"), min_exact=("정확등급정확도", "min")
        ).sort_values(["exact", "within1", "error", "min_exact"], ascending=[False, False, True, False])
        winner = summary.iloc[0].to_dict()
        winners[horizon] = winner
        all_details.append(detail)
        all_summaries.append(summary)
        print(json.dumps({"horizon": horizon, "winner": winner}, ensure_ascii=False), flush=True)

        def fit_config(name: str):
            spec = fitted_specs.get((horizon, name), fitted_specs.get(name))
            if spec["kind"] == "anchor":
                return spec
            if spec["kind"] == "ensemble_rank":
                return {
                    "kind": "ensemble_rank",
                    "weight_a": spec["weight"],
                    "component_a": fit_config(spec["left"]),
                    "component_b": fit_config(spec["right"]),
                }
            y = labeled[target_score].to_numpy(dtype=float)
            if spec["target_kind"] == "delta":
                y = y - labeled["현재연도_종합위험점수"].to_numpy(dtype=float)
            fitted = clone(spec["estimator"]).fit(labeled[spec["features"]], y)
            return {key: value for key, value in spec.items() if key != "estimator"} | {"model": fitted}

        bundle = fit_config(str(winner["모델"]))
        bundle.update(version=59, horizon=horizon, forecast_year=2025 + horizon, metrics=winner)
        joblib.dump(bundle, OUTPUT / f"교통사고_59차_{2025 + horizon}년_{horizon}년후_모델.joblib")

    combined_detail = pd.concat(all_details, ignore_index=True)
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    winner_frame = pd.DataFrame([winners[2], winners[3]])
    with pd.ExcelWriter(OUTPUT / "59차_검증결과.xlsx", engine="openpyxl") as writer:
        winner_frame.to_excel(writer, sheet_name="성능요약", index=False)
        combined_summary.to_excel(writer, sheet_name="전체기법비교", index=False)
        combined_detail.to_excel(writer, sheet_name="연도별검증", index=False)
    info = {
        "모델명": "교통사고 59차 기법·경량변수 동시탐색 모델",
        "기반": "58차",
        "탐색": "미래점수/증감 회귀, 현재점수 혼합, 순위등급화, 상위모델 앙상블, 변수 15~전체",
        "성능": {
            "2년후": winners[2], "3년후": winners[3],
            "평균정확등급": float(np.mean([winners[2]["exact"], winners[3]["exact"]])),
        },
    }
    (OUTPUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
