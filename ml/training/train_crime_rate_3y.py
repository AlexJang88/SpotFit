"""서울 자치구 5대 범죄율의 3년 후 예측모델을 선택하고 최종 모델만 저장한다."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "안전" / "범죄"
SAFETY_PANEL = ROOT / "data" / "interim" / "안전" / "범죄안전지수" / "서울시_자치구_범죄안전_대체지수_2018_2023.csv"
CCTV_PANEL = ROOT / "data" / "raw" / "안전" / "교통사고" / "파생" / "서울시_CCTV_설치연도기반_자치구_연도별_추정_2003_2026.csv"
OUT = ROOT / "artifacts" / "crime_rate_3y"
MODEL_OUT = ROOT / "models" / "recommender" / "v1" / "crime_rate_3y"
RANDOM_STATE = 42


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"CSV encoding not recognized: {path}")


def make_panel() -> pd.DataFrame:
    crime = read_csv(RAW / "범죄_2014_2025.csv")
    crime["연도"] = pd.to_numeric(crime["연도"], errors="raise").astype(int)
    population = read_csv(RAW / "서울시_자치구_연도별_등록인구.csv").rename(
        columns={"통계연도": "연도", "구명": "자치구"}
    )
    population = population[["연도", "자치구", "세대수", "인구수", "남자인구수", "여자인구수"]]
    bars = read_csv(RAW / "파생" / "유흥_단란주점_자치구_연도별_2010_2026.csv")
    facilities = read_csv(RAW / "파생" / "집객시설_자치구_연도별_2018_2023.csv")[["연도", "자치구", "집객시설수"]]
    safety = read_csv(SAFETY_PANEL)[["연도", "자치구", "스트레스인지율", "경찰관서수"]]
    cctv = read_csv(CCTV_PANEL)
    cctv = cctv[cctv["구분"].eq("전체")][["연도", "자치구", "폐기미반영_누적추정카메라수"]]

    frame = crime.merge(population, on=["연도", "자치구"], how="left", validate="one_to_one")
    for extra in (bars, facilities, safety, cctv):
        frame = frame.merge(extra, on=["연도", "자치구"], how="left", validate="one_to_one")

    occurrence = [c for c in crime.columns if c.endswith("_발생건")]
    arrests = [c for c in crime.columns if c.endswith("_검거건")]
    divisor = frame["인구수"] / 10_000
    for column in occurrence:
        frame[column.replace("_발생건", "_발생률_1만명당")] = frame[column] / divisor
    for column in arrests:
        occurrence_column = column.replace("_검거건", "_발생건")
        frame[column.replace("_검거건", "_검거율")] = np.where(
            frame[occurrence_column] > 0, frame[column] / frame[occurrence_column] * 100, 100
        )
    frame["범죄율_1만명당"] = frame["소계_발생건"] / divisor
    frame["세대당인구"] = frame["인구수"] / frame["세대수"]
    frame["남성비율"] = frame["남자인구수"] / frame["인구수"] * 100
    frame["주점_1만명당"] = frame["유흥_단란주점_합계"] / divisor
    frame["집객시설_1만명당"] = frame["집객시설수"] / divisor
    frame["경찰관서_1만명당"] = frame["경찰관서수"] / divisor
    frame["CCTV_1만명당"] = frame["폐기미반영_누적추정카메라수"] / divisor
    frame = frame.sort_values(["자치구", "연도"])
    for column in ("범죄율_1만명당", "인구수", "세대수", "주점_1만명당", "집객시설_1만명당", "CCTV_1만명당"):
        frame[f"{column}_전년변화율"] = frame.groupby("자치구")[column].pct_change(fill_method=None) * 100
    frame["범죄율_1만명당_전년차"] = frame.groupby("자치구")["범죄율_1만명당"].diff()
    return frame


def make_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    current = panel[panel["연도"].between(2018, 2023)].copy().rename(columns={"연도": "기준연도"})
    target = panel[["연도", "자치구", "범죄율_1만명당"]].copy()
    target["기준연도"] = target["연도"] - 3
    target = target.rename(columns={"연도": "목표연도", "범죄율_1만명당": "target_범죄율_1만명당"})
    return current.merge(target, on=["기준연도", "자치구"], how="left", validate="one_to_one")


FEATURE_GROUPS = {
    "현재범죄수준": ["범죄율_1만명당"],
    "범죄추세": ["범죄율_1만명당_전년차", "범죄율_1만명당_전년변화율"],
    "범죄유형구성": [
        "살인_발생률_1만명당", "강도_발생률_1만명당", "강간·강제추행_발생률_1만명당",
        "절도_발생률_1만명당", "폭력_발생률_1만명당",
    ],
    "검거역량": ["소계_검거율", "살인_검거율", "강도_검거율", "강간·강제추행_검거율", "절도_검거율", "폭력_검거율"],
    "인구가구": ["인구수", "세대수", "세대당인구", "남성비율", "인구수_전년변화율", "세대수_전년변화율"],
    "위험시설": ["주점_1만명당", "집객시설_1만명당", "주점_1만명당_전년변화율", "집객시설_1만명당_전년변화율"],
    "안전자원건강": ["경찰관서_1만명당", "CCTV_1만명당", "CCTV_1만명당_전년변화율", "스트레스인지율"],
}


def estimators() -> dict[str, object]:
    return {
        "Ridge_1": Ridge(alpha=1.0),
        "Ridge_10": Ridge(alpha=10.0),
        "Ridge_100": Ridge(alpha=100.0),
        "ElasticNet": ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=20_000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=4, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=3, random_state=RANDOM_STATE),
    }


def pipeline(estimator: object, numeric: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["자치구"]),
    ], sparse_threshold=0)
    return Pipeline([("prep", prep), ("model", estimator)])


def measure(actual: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
        "MAPE_pct": float(np.mean(np.abs((actual.to_numpy() - prediction) / actual.to_numpy())) * 100),
        "Spearman": float(spearmanr(actual, prediction).statistic),
    }


def rolling_cv(pairs: pd.DataFrame, groups: tuple[str, ...], model_name: str, estimator: object) -> dict[str, object] | None:
    numeric = list(dict.fromkeys(itertools.chain.from_iterable(FEATURE_GROUPS[g] for g in groups)))
    features = ["자치구", *numeric]
    fold_rows = []
    for target_year in (2022, 2023, 2024):
        train = pairs[pairs["목표연도"] < target_year].dropna(subset=["target_범죄율_1만명당"])
        valid = pairs[pairs["목표연도"] == target_year].dropna(subset=["target_범죄율_1만명당"])
        if train.empty or valid.empty:
            continue
        model = pipeline(estimator, numeric)
        model.fit(train[features], train["target_범죄율_1만명당"])
        metrics = measure(valid["target_범죄율_1만명당"], model.predict(valid[features]))
        fold_rows.append({"목표연도": target_year, **metrics})
    if len(fold_rows) != 3:
        return None
    return {
        "모델": model_name, "feature그룹": "+".join(groups), "feature수": len(numeric),
        "features": numeric, "CV_MAE": np.mean([x["MAE"] for x in fold_rows]),
        "CV_RMSE": np.mean([x["RMSE"] for x in fold_rows]),
        "CV_R2": np.mean([x["R2"] for x in fold_rows]),
        "CV_MAPE_pct": np.mean([x["MAPE_pct"] for x in fold_rows]),
        "CV_Spearman": np.mean([x["Spearman"] for x in fold_rows]), "folds": fold_rows,
    }


def main() -> int:
    panel = make_panel()
    pairs = make_pairs(panel)
    development = pairs[pairs["목표연도"].between(2021, 2024)].copy()
    holdout = pairs[pairs["목표연도"].eq(2025)].dropna(subset=["target_범죄율_1만명당"]).copy()

    baseline_folds = []
    for year in (2022, 2023, 2024):
        valid = development[development["목표연도"].eq(year)]
        baseline_folds.append({"목표연도": year, **measure(valid["target_범죄율_1만명당"], valid["범죄율_1만명당"].to_numpy())})
    baseline = {
        "모델": "Persistence_현재범죄율유지", "feature그룹": "현재범죄수준", "feature수": 1,
        "features": ["범죄율_1만명당"], "CV_MAE": np.mean([x["MAE"] for x in baseline_folds]),
        "CV_RMSE": np.mean([x["RMSE"] for x in baseline_folds]), "CV_R2": np.mean([x["R2"] for x in baseline_folds]),
        "CV_MAPE_pct": np.mean([x["MAPE_pct"] for x in baseline_folds]),
        "CV_Spearman": np.mean([x["Spearman"] for x in baseline_folds]), "folds": baseline_folds,
    }

    group_names = list(FEATURE_GROUPS)
    results = []
    # 현재 범죄수준은 모든 학습모델에 포함하고, 나머지 여섯 그룹의 모든 조합을 비교한다.
    optional = group_names[1:]
    for size in range(0, len(optional) + 1):
        for selected in itertools.combinations(optional, size):
            groups = ("현재범죄수준", *selected)
            for name, estimator in estimators().items():
                result = rolling_cv(development, groups, name, estimator)
                if result:
                    results.append(result)
    leaderboard = pd.DataFrame([{k: v for k, v in r.items() if k not in ("features", "folds")} for r in results])
    leaderboard = leaderboard.sort_values(["CV_RMSE", "CV_MAE", "feature수"]).reset_index(drop=True)
    best_row = leaderboard.iloc[0]
    best = next(r for r in results if r["모델"] == best_row["모델"] and r["feature그룹"] == best_row["feature그룹"])

    # 평가 순서에서 기준모델 대비 최저 RMSE 기록이 갱신된 이력만 남긴다.
    ordered = sorted(results, key=lambda r: (len(r["feature그룹"].split("+")), r["feature그룹"], r["모델"]))
    record, best_so_far = [], baseline["CV_RMSE"]
    record.append({"단계": 0, "구분": "최초기준", **{k: baseline[k] for k in ("모델", "feature그룹", "feature수", "CV_MAE", "CV_RMSE", "CV_R2", "CV_MAPE_pct", "CV_Spearman")}, "직전기록대비_RMSE개선_pct": 0.0})
    for candidate in ordered:
        if candidate["CV_RMSE"] < best_so_far - 1e-12:
            improvement = (best_so_far - candidate["CV_RMSE"]) / best_so_far * 100
            record.append({"단계": len(record), "구분": "성능갱신", **{k: candidate[k] for k in ("모델", "feature그룹", "feature수", "CV_MAE", "CV_RMSE", "CV_R2", "CV_MAPE_pct", "CV_Spearman")}, "직전기록대비_RMSE개선_pct": improvement})
            best_so_far = candidate["CV_RMSE"]

    numeric = best["features"]
    features = ["자치구", *numeric]
    final_estimator = estimators()[best["모델"]]
    model = pipeline(final_estimator, numeric)
    train_to_2024 = pairs[pairs["목표연도"].between(2021, 2024)].dropna(subset=["target_범죄율_1만명당"])
    model.fit(train_to_2024[features], train_to_2024["target_범죄율_1만명당"])
    holdout_prediction = model.predict(holdout[features])
    holdout_metrics = measure(holdout["target_범죄율_1만명당"], holdout_prediction)
    holdout_baseline = measure(holdout["target_범죄율_1만명당"], holdout["범죄율_1만명당"].to_numpy())

    # 최종 운영모델은 2025 정답까지 포함해 재학습하고, 2023 기준으로 2026을 예측한다.
    full_train = pairs[pairs["목표연도"].between(2021, 2025)].dropna(subset=["target_범죄율_1만명당"])
    operational = pipeline(estimators()[best["모델"]], numeric)
    operational.fit(full_train[features], full_train["target_범죄율_1만명당"])
    forecast_source = pairs[pairs["기준연도"].eq(2023)].copy()
    forecast_source["예측범죄율_1만명당"] = operational.predict(forecast_source[features])

    all_attempted = list(dict.fromkeys(itertools.chain.from_iterable(FEATURE_GROUPS.values())))
    excluded = [
        {"자료": "자율방범대", "사유": "자치구별 관측연도가 2015·2023 중심이라 3년 순차검증 패널이 불완전"},
        {"자료": "2025 단일시점 경찰 지구대·파출소 주소", "사유": "과거 연도 상태를 복원할 수 없어 목표연도 정보 누수 가능"},
        {"자료": "2026 상권·인구·부동산 단일시점 자료", "사유": "2018~2022 기준연도의 과거 값을 동일 정의로 확보할 수 없음"},
    ]
    report = {
        "목표": "기준연도에서 3년 뒤 서울 자치구 5대 범죄율(인구 1만명당 발생건수)",
        "선택기준": "2022~2024 목표연도 확대창 순차검증 평균 RMSE 최소; 2025 홀드아웃은 선택 후 1회 평가",
        "최초기준모델": baseline,
        "최종모델": best,
        "기준대비_CV_RMSE개선_pct": (baseline["CV_RMSE"] - best["CV_RMSE"]) / baseline["CV_RMSE"] * 100,
        "2025홀드아웃_최종모델": holdout_metrics,
        "2025홀드아웃_기준모델": holdout_baseline,
        "2025홀드아웃_RMSE개선_pct": (holdout_baseline["RMSE"] - holdout_metrics["RMSE"]) / holdout_baseline["RMSE"] * 100,
        "시도한_전체_feature": all_attempted,
        "feature그룹": FEATURE_GROUPS,
        "제외자료": excluded,
        "시도한_알고리즘": list(estimators()),
        "비교한_학습모델수": len(results),
        "주의": "회귀모델의 정답률은 단일 비율이 아니므로 RMSE·MAE·MAPE·R²·Spearman으로 평가한다.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(OUT / "전체_feature조합_모델비교.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(record).to_csv(OUT / "기준모델부터_성능개선기록.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best["folds"]).to_csv(OUT / "최종모델_연도별_CV성능.csv", index=False, encoding="utf-8-sig")
    holdout_result = holdout[["목표연도", "자치구", "target_범죄율_1만명당", "범죄율_1만명당"]].copy()
    holdout_result["최종모델_예측범죄율_1만명당"] = holdout_prediction
    holdout_result.to_csv(OUT / "2025_홀드아웃_예측대실제.csv", index=False, encoding="utf-8-sig")
    forecast_source[["자치구", "예측범죄율_1만명당"]].sort_values("예측범죄율_1만명당", ascending=False).to_csv(
        OUT / "2026_자치구별_3년후_범죄율예측.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "모델설명.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({
        "model": operational, "features": features, "numeric_features": numeric,
        "target": "3년후_5대범죄율_인구1만명당", "model_name": best["모델"],
        "feature_groups": best["feature그룹"].split("+"), "training_target_years": [2021, 2022, 2023, 2024, 2025],
    }, MODEL_OUT / "crime_rate_3y.joblib")
    (MODEL_OUT / "model_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_model": best["모델"], "best_groups": best["feature그룹"], "best_features": best["features"],
        "cv_rmse": best["CV_RMSE"], "baseline_cv_rmse": baseline["CV_RMSE"],
        "cv_improvement_pct": report["기준대비_CV_RMSE개선_pct"],
        "holdout": holdout_metrics, "holdout_baseline": holdout_baseline,
        "holdout_improvement_pct": report["2025홀드아웃_RMSE개선_pct"], "models_compared": len(results),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
