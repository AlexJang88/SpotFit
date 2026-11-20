"""Exhaustive grouped-feature search for 1y/3y child crime exposure forecasts."""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "raw" / "아동범죄" / "external_features" / "아동지역범죄노출_외부확장피처결합_2015_2024.csv"
OUT = ROOT / "artifacts" / "child_crime_exposure_forecast"
MODEL_OUT = ROOT / "models" / "recommender" / "v1" / "child_crime_exposure"
TARGET = "Y_log1p_아동지역범죄노출률"
RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


FEATURE_GROUPS = {
    "현재노출수준": [TARGET],
    "노출추세": [
        "아동지역범죄노출률_lag1",
        "아동지역범죄노출률_lag2",
        "아동지역범죄노출률_lag3",
        "5대범죄_발생건수_전년변화율",
        "5대범죄_발생건수_3년변화율",
    ],
    "인구구조": [
        "아동인구_0_17세", "영유아인구_0_5세", "초등연령인구_6_12세", "청소년인구_13_17세",
        "인구수", "아동인구비율", "세대당인구", "남성비율",
        "아동인구_0_17세_전년변화율", "아동인구_0_17세_3년변화율",
        "인구수_전년변화율", "인구수_3년변화율", "세대수_전년변화율", "세대수_3년변화율",
    ],
    "유흥시설": [
        "유흥주점_연말_영업업소수", "단란주점_연말_영업업소수", "유흥_단란주점_합계",
        "유흥_단란주점_인구1만명당", "유흥_단란주점_아동1만명당",
        "유흥_단란주점_합계_전년변화율", "유흥_단란주점_합계_3년변화율",
    ],
    "고용구조": [
        "총종사자수", "기타종사자수", "상용근로자수", "임시_일용근로자수", "자영업주_무급가족수",
        "상용근로자수_비율", "임시_일용근로자수_비율", "자영업주_무급가족수_비율", "기타종사자수_비율",
        "총종사자수_전년변화율", "상용근로자수_전년변화율", "임시_일용근로자수_전년변화율",
        "자영업주_무급가족수_전년변화율", "기타종사자수_전년변화율",
    ],
    "고령인구": [
        "고령인구_내외국인", "고령인구비율_등록인구대비_근사",
        "고령인구_내외국인_전년변화율", "고령인구비율_등록인구대비_근사_전년변화율",
    ],
    "가구규모": ["가구_1인", "가구_2인", "가구_3인", "가구_4인", "가구_5인", "가구_계", "1인가구비율"],
    "연령별1인가구": [
        "1인가구_20세미만", "1인가구_20~24", "1인가구_25~29", "1인가구_30~34", "1인가구_35~39",
        "1인가구_40~44", "1인가구_45~49", "1인가구_50~54", "1인가구_55~59", "1인가구_60~64",
        "1인가구_65~69", "1인가구_70~74", "1인가구_75~79", "1인가구_80~84", "1인가구_85세이상", "1인가구_계",
    ],
    "가족지역변화": ["이혼건수", "조이혼율", "이혼건수_전년변화율", "가구증가율"],
}


def estimators() -> dict[str, object]:
    return {
        "Ridge_1": Ridge(alpha=1.0),
        "Ridge_10": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20_000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=4, max_features=0.8, random_state=RANDOM_STATE, n_jobs=1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=3, max_features=0.8, random_state=RANDOM_STATE, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.03, max_depth=2, min_samples_leaf=5,
            loss="huber", random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=3,
            random_state=RANDOM_STATE,
        ),
    }


def model_pipeline(estimator: object, numeric: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["자치구"]),
        ],
        sparse_threshold=0,
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def metrics(actual: np.ndarray, prediction: np.ndarray, current: np.ndarray) -> dict[str, float]:
    error = actual - prediction
    movement_actual = np.sign(actual - current)
    movement_pred = np.sign(prediction - current)
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
        "MAPE_pct": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-9)) * 100),
        "Spearman": float(spearmanr(actual, prediction).statistic),
        "방향정확도_pct": float(np.mean(movement_actual == movement_pred) * 100),
    }


def make_pairs(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    current = panel.copy().rename(columns={"연도": "기준연도"})
    future = panel[["연도", "자치구", TARGET]].rename(columns={TARGET: "target"})
    future["기준연도"] = future["연도"] - horizon
    future = future.rename(columns={"연도": "목표연도"})
    return current.merge(future, on=["기준연도", "자치구"], how="inner", validate="one_to_one")


def available_groups(panel: pd.DataFrame) -> dict[str, list[str]]:
    groups = {name: [column for column in columns if column in panel.columns] for name, columns in FEATURE_GROUPS.items()}
    missing = {name: sorted(set(columns) - set(groups[name])) for name, columns in FEATURE_GROUPS.items() if set(columns) - set(groups[name])}
    if missing:
        raise ValueError(f"Expected feature columns missing: {missing}")
    return groups


def fold_years(horizon: int) -> list[int]:
    return [2020, 2021, 2022, 2023] if horizon == 1 else [2021, 2022, 2023]


def evaluate_baseline(pairs: pd.DataFrame, horizon: int) -> dict[str, object]:
    rows = []
    for year in fold_years(horizon):
        valid = pairs[pairs["목표연도"].eq(year)]
        rows.append({"목표연도": year, **metrics(valid["target"].to_numpy(), valid[TARGET].to_numpy(), valid[TARGET].to_numpy())})
    return aggregate("Persistence", "현재노출수준", "기준모델", [TARGET], rows)


def aggregate(model: str, group_name: str, mode: str, features: list[str], rows: list[dict[str, float]]) -> dict[str, object]:
    result: dict[str, object] = {
        "모델": model, "예측방식": mode, "feature그룹": group_name, "feature수": len(features),
        "features": features, "folds": rows,
    }
    for key in ("MAE", "RMSE", "R2", "MAPE_pct", "Spearman", "방향정확도_pct"):
        result[f"CV_{key}"] = float(np.mean([row[key] for row in rows]))
    return result


def evaluate_candidate(
    pairs: pd.DataFrame,
    horizon: int,
    groups: tuple[str, ...],
    group_map: dict[str, list[str]],
    model_name: str,
    estimator: object,
    mode: str,
) -> dict[str, object]:
    numeric = list(dict.fromkeys(itertools.chain.from_iterable(group_map[group] for group in groups)))
    columns = ["자치구", *numeric]
    rows = []
    for year in fold_years(horizon):
        train = pairs[pairs["목표연도"].lt(year)]
        valid = pairs[pairs["목표연도"].eq(year)]
        model = model_pipeline(estimator, numeric)
        y_train = train["target"] if mode == "직접예측" else train["target"] - train[TARGET]
        model.fit(train[columns], y_train)
        raw = model.predict(valid[columns])
        prediction = raw if mode == "직접예측" else valid[TARGET].to_numpy() + raw
        rows.append({
            "목표연도": year,
            **metrics(valid["target"].to_numpy(), prediction, valid[TARGET].to_numpy()),
        })
    return aggregate(model_name, "+".join(groups), mode, numeric, rows)


def fit_final(best: dict[str, object], pairs: pd.DataFrame, group_map: dict[str, list[str]]) -> Pipeline:
    numeric = list(best["features"])
    model = model_pipeline(estimators()[str(best["모델"])], numeric)
    y = pairs["target"] if best["예측방식"] == "직접예측" else pairs["target"] - pairs[TARGET]
    model.fit(pairs[["자치구", *numeric]], y)
    return model


def run_horizon(panel: pd.DataFrame, horizon: int, group_map: dict[str, list[str]]) -> dict[str, object]:
    pairs = make_pairs(panel, horizon)
    development = pairs[pairs["목표연도"].lt(2024)].copy()
    holdout = pairs[pairs["목표연도"].eq(2024)].copy()
    baseline = evaluate_baseline(development, horizon)

    optional = [name for name in group_map if name != "현재노출수준"]
    # Phase 1: every one of the 2^8 meaningful feature-group combinations is
    # screened with regularized linear direct/residual models.
    results: list[dict[str, object]] = []
    combinations: list[tuple[str, ...]] = []
    for size in range(len(optional) + 1):
        for selected in itertools.combinations(optional, size):
            groups = ("현재노출수준", *selected)
            combinations.append(groups)
            for model_name in ("Ridge_1", "Ridge_10", "ElasticNet"):
                estimator = estimators()[model_name]
                for mode in ("직접예측", "Persistence잔차예측"):
                    results.append(evaluate_candidate(development, horizon, groups, group_map, model_name, estimator, mode))

    # Phase 2: run nonlinear methods on the overall top 30 combinations plus
    # the best combination at every group-count level, current-only, and full.
    screened = sorted(results, key=lambda row: (row["CV_RMSE"], row["CV_MAE"], row["feature수"]))
    candidate_group_names = {str(row["feature그룹"]) for row in screened[:30]}
    for group_count in range(1, len(group_map) + 1):
        level = [row for row in screened if len(str(row["feature그룹"]).split("+")) == group_count]
        candidate_group_names.add(str(level[0]["feature그룹"]))
    candidate_group_names.add("현재노출수준")
    candidate_group_names.add("+".join(group_map))
    nonlinear = ("RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting")
    for group_name in sorted(candidate_group_names):
        groups = tuple(group_name.split("+"))
        for model_name in nonlinear:
            estimator = estimators()[model_name]
            for mode in ("직접예측", "Persistence잔차예측"):
                results.append(evaluate_candidate(development, horizon, groups, group_map, model_name, estimator, mode))

    results.sort(key=lambda row: (row["CV_RMSE"], row["CV_MAE"], row["feature수"]))
    improved = [row for row in results if row["CV_RMSE"] < baseline["CV_RMSE"]]
    selected = improved[0] if improved else baseline

    # Record only new bests by feature-group complexity, starting at Persistence.
    records = [{"단계": 0, "구분": "최초기준", **{key: baseline[key] for key in baseline if key not in {"features", "folds"}}, "기준대비_RMSE개선_pct": 0.0}]
    best_so_far = float(baseline["CV_RMSE"])
    for group_count in range(1, len(group_map) + 1):
        level = [row for row in results if len(str(row["feature그룹"]).split("+")) == group_count]
        candidate = min(level, key=lambda row: (row["CV_RMSE"], row["CV_MAE"], row["feature수"]))
        if candidate["CV_RMSE"] < best_so_far:
            records.append({
                "단계": len(records), "구분": "성능갱신",
                **{key: candidate[key] for key in candidate if key not in {"features", "folds"}},
                "기준대비_RMSE개선_pct": (baseline["CV_RMSE"] - candidate["CV_RMSE"]) / baseline["CV_RMSE"] * 100,
            })
            best_so_far = float(candidate["CV_RMSE"])

    if selected["모델"] == "Persistence":
        holdout_prediction = holdout[TARGET].to_numpy()
        final_model = None
    else:
        development_model = fit_final(selected, development, group_map)
        raw = development_model.predict(holdout[["자치구", *selected["features"]]])
        holdout_prediction = raw if selected["예측방식"] == "직접예측" else holdout[TARGET].to_numpy() + raw
        final_model = fit_final(selected, pairs, group_map)
    holdout_final = metrics(holdout["target"].to_numpy(), holdout_prediction, holdout[TARGET].to_numpy())
    holdout_baseline = metrics(holdout["target"].to_numpy(), holdout[TARGET].to_numpy(), holdout[TARGET].to_numpy())

    horizon_out = OUT / f"{horizon}y"
    horizon_model = MODEL_OUT / f"{horizon}y"
    horizon_out.mkdir(parents=True, exist_ok=True)
    horizon_model.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if key not in {"features", "folds"}} for row in results])
    leaderboard["기준대비_RMSE개선_pct"] = (baseline["CV_RMSE"] - leaderboard["CV_RMSE"]) / baseline["CV_RMSE"] * 100
    leaderboard.to_csv(horizon_out / "전체_feature그룹조합_모델비교.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(horizon_out / "기준보다_개선된_기록.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected.get("folds", baseline["folds"])).to_csv(horizon_out / "최종선택_CV연도별성능.csv", index=False, encoding="utf-8-sig")
    holdout_result = holdout[["기준연도", "목표연도", "자치구", TARGET, "target"]].copy()
    holdout_result["예측값"] = holdout_prediction
    holdout_result.to_csv(horizon_out / "2024_홀드아웃_예측대실제.csv", index=False, encoding="utf-8-sig")

    report = {
        "horizon_years": horizon,
        "target_formula": "log(1 + 5대범죄 발생건수 / 0~17세 아동인구 * 10000)",
        "baseline": baseline,
        "selected": selected,
        "cv_rmse_improvement_pct": (baseline["CV_RMSE"] - selected["CV_RMSE"]) / baseline["CV_RMSE"] * 100,
        "holdout_2024_selected": holdout_final,
        "holdout_2024_baseline": holdout_baseline,
        "holdout_rmse_improvement_pct": (holdout_baseline["RMSE"] - holdout_final["RMSE"]) / holdout_baseline["RMSE"] * 100,
        "feature_groups": group_map,
        "algorithms": list(estimators()),
        "prediction_modes": ["직접예측", "Persistence잔차예측"],
        "configurations_compared": len(results),
        "feature_group_combinations_screened": len(combinations),
        "nonlinear_candidate_group_combinations": len(candidate_group_names),
        "development_target_years": sorted(development["목표연도"].unique().astype(int).tolist()),
        "selection_cv_target_years": fold_years(horizon),
        "untouched_holdout_target_year": 2024,
        "notes": [
            "개별 열 2^N 탐색은 통계적으로 무의미하므로 의미 기반 feature 그룹의 모든 2^8 조합을 정규화 선형모델로 전수 비교했다.",
            "비선형 모델은 전수검색 상위 30개와 각 그룹수별 최상, 현재값 단독, 전체 그룹 조합에 적용했다.",
            "모델 선택은 시간순 확대창 CV RMSE로만 수행했고 2024 홀드아웃은 선택 후 한 번 평가했다.",
            "회귀 문제이므로 단일 정답률 대신 RMSE, MAE, R2, MAPE, Spearman, 증감방향 정확도를 함께 제공한다.",
        ],
    }
    (horizon_out / "모델비교_설명.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (horizon_model / "model_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if final_model is not None:
        joblib.dump(
            {
                "model": final_model,
                "horizon_years": horizon,
                "target": TARGET,
                "target_formula": report["target_formula"],
                "features": ["자치구", *selected["features"]],
                "numeric_features": selected["features"],
                "model_name": selected["모델"],
                "prediction_mode": selected["예측방식"],
                "feature_groups": str(selected["feature그룹"]).split("+"),
                "training_target_years": sorted(pairs["목표연도"].unique().astype(int).tolist()),
            },
            horizon_model / f"child_crime_exposure_{horizon}y.joblib",
        )
    return report


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig").sort_values(["자치구", "연도"])
    panel["5대범죄_발생건수_전년변화율"] = panel.groupby("자치구")["5대범죄_발생건수"].pct_change(fill_method=None)
    panel["5대범죄_발생건수_3년변화율"] = panel.groupby("자치구")["5대범죄_발생건수"].pct_change(3, fill_method=None)
    groups = available_groups(panel)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    reports = [run_horizon(panel, horizon, groups) for horizon in (1, 3)]
    summary = []
    for report in reports:
        selected = report["selected"]
        baseline = report["baseline"]
        summary.append({
            "예측기간": f"{report['horizon_years']}년 뒤",
            "기준모델_CV_RMSE": baseline["CV_RMSE"],
            "최종모델": selected["모델"],
            "예측방식": selected["예측방식"],
            "feature그룹": selected["feature그룹"],
            "최종_CV_RMSE": selected["CV_RMSE"],
            "CV_RMSE_개선_pct": report["cv_rmse_improvement_pct"],
            "2024홀드아웃_RMSE_개선_pct": report["holdout_rmse_improvement_pct"],
            "비교설정수": report["configurations_compared"],
        })
    pd.DataFrame(summary).to_csv(OUT / "1년_3년_최종비교.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
