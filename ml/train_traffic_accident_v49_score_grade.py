from __future__ import annotations

import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_best_child_traffic_accident_rate import ROOT, TARGET
from ml.train_traffic_accident_v44 import prepare_panel


BASE_VERSION = 49
VERSION = 51
BASE_MODEL_DIR = ROOT / "models" / "교통사고" / "41차~49차" / f"{BASE_VERSION}차"
MODEL_DIR = ROOT / "models" / "교통사고" / f"{VERSION}차"
PROCESSED = ROOT / "data" / "processed" / f"교통사고_{VERSION}차_최종모델데이터.xlsx"
SCORE_MODEL = MODEL_DIR / "교통사고_51차_가중치5분위_점수등급모델.joblib"
SCORE_REPORT = MODEL_DIR / "51차_가중치5분위_점수등급평가.xlsx"
CV_YEARS = (2021, 2022, 2023, 2024, 2025)


def quintile_score(series: pd.Series, positive: bool) -> pd.Series:
    # 정방향은 큰 값, 역방향은 작은 값이 높은 위험점수가 된다.
    percentile = series.rank(method="average", ascending=not positive, pct=True)
    # 위험순위 1위는 percentile이 작으므로 100점, 하위권은 20점.
    return (120 - np.ceil(percentile * 5).clip(1, 5) * 20).astype(int)


def composite_grade(series: pd.Series) -> pd.Series:
    percentile = series.rank(method="average", ascending=False, pct=True)
    return np.ceil(percentile * 5).clip(1, 5).astype(int)


def estimator_candidates():
    def ridge(alpha):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    return {
        **{f"Ridge_{alpha:g}": ridge(alpha) for alpha in (0.1, 1, 10, 100, 300)},
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.03,
                                                 loss="huber", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=4,
                                             max_features=.7, random_state=42, n_jobs=1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4,
                                           max_features=.7, random_state=42, n_jobs=1)),
        ]),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads((BASE_MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    features = info["최종변수"]
    bundle = joblib.load(BASE_MODEL_DIR / "교통사고_49차_예측모델.joblib")
    coefficients = bundle["model"].named_steps["model"].coef_.astype(float)
    weights = np.abs(coefficients) / np.abs(coefficients).sum()

    weight_table = pd.DataFrame({
        "변수": features,
        "표준화계수": coefficients,
        "방향": np.where(coefficients >= 0, "정방향", "역방향"),
        "가중치_pct": weights * 100,
    }).sort_values("가중치_pct", ascending=False).reset_index(drop=True)

    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    base = panel[["연도", "예측연도", "자치구", TARGET, *features]].copy()
    base.insert(0, "서울시", "서울특별시")
    base.insert(4, "데이터구분", np.select(
        [base[TARGET].isna(), base["예측연도"] == 2025],
        ["2026예측입력", "2025독립검증"], default="모델개발_순차검증",
    ))

    score_columns = []
    contribution_columns = []
    for feature, coefficient, weight in zip(features, coefficients, weights):
        score_col = f"점수_{feature}"
        contribution_col = f"가중점수_{feature}"
        base[score_col] = base.groupby("연도", group_keys=False)[feature].apply(
            lambda s: quintile_score(s, coefficient >= 0)
        )
        base[contribution_col] = base[score_col] * weight
        score_columns.append(score_col)
        contribution_columns.append(contribution_col)

    base["현재연도_종합위험점수"] = base[contribution_columns].sum(axis=1)
    base["현재연도_위험등급"] = base.groupby("연도")["현재연도_종합위험점수"].transform(composite_grade)
    district = base.groupby("자치구", sort=False)
    base["다음연도_종합위험점수"] = district["현재연도_종합위험점수"].shift(-1)
    base["다음연도_위험등급"] = district["현재연도_위험등급"].shift(-1)

    model_features = [*features, *score_columns, "현재연도_종합위험점수"]
    labeled = base[base["다음연도_종합위험점수"].notna()].copy()
    evaluation_rows = []
    best_name, best_mae, best_estimator = None, float("inf"), None
    for name, estimator in estimator_candidates().items():
        fold_rows = []
        for year in CV_YEARS:
            train = labeled[labeled["예측연도"] < year]
            valid = labeled[labeled["예측연도"] == year]
            if train.empty or valid.empty:
                continue
            fitted = deepcopy(estimator).fit(train[model_features], train["다음연도_종합위험점수"])
            pred = np.clip(fitted.predict(valid[model_features]), 20, 100)
            temp = valid[["자치구"]].copy()
            temp["pred"] = pred
            pred_grade = composite_grade(temp["pred"])
            fold_rows.append({
                "검증연도": year,
                "MAE": mean_absolute_error(valid["다음연도_종합위험점수"], pred),
                "RMSE": mean_squared_error(valid["다음연도_종합위험점수"], pred) ** .5,
                "등급정확도": accuracy_score(valid["다음연도_위험등급"].astype(int), pred_grade),
                "등급1칸이내정확도": np.mean(np.abs(valid["다음연도_위험등급"].astype(int).to_numpy() - pred_grade.to_numpy()) <= 1),
            })
        mean_mae = float(np.mean([row["MAE"] for row in fold_rows]))
        evaluation_rows.extend([{"모델": name, **row} for row in fold_rows])
        if mean_mae < best_mae:
            best_name, best_mae, best_estimator = name, mean_mae, estimator

    summary = pd.DataFrame(evaluation_rows).groupby("모델", as_index=False).agg(
        평균_MAE=("MAE", "mean"), 평균_RMSE=("RMSE", "mean"),
        평균_등급정확도=("등급정확도", "mean"), 평균_등급1칸이내정확도=("등급1칸이내정확도", "mean"),
    ).sort_values(["평균_MAE", "평균_RMSE"])

    final_model = deepcopy(best_estimator).fit(labeled[model_features], labeled["다음연도_종합위험점수"])
    future = base[base["예측연도"] == 2026].copy()
    future["2026년_예측_종합위험점수"] = np.clip(final_model.predict(future[model_features]), 20, 100)
    future["2026년_예측_위험등급"] = composite_grade(future["2026년_예측_종합위험점수"])
    forecast_columns = ["자치구", "현재연도_종합위험점수", "현재연도_위험등급",
                        "2026년_예측_종합위험점수", "2026년_예측_위험등급"]

    joblib.dump({
        "version": VERSION,
        "model_name": best_name,
        "raw_features": features,
        "score_features": score_columns,
        "model_features": model_features,
        "weights": dict(zip(features, weights)),
        "directions": dict(zip(features, np.where(coefficients >= 0, "정방향", "역방향"))),
        "model": final_model,
    }, SCORE_MODEL)

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        base.to_excel(writer, sheet_name="51차점수등급데이터", index=False)
        weight_table.to_excel(writer, sheet_name="변수가중치", index=False)
        future[forecast_columns].sort_values("2026년_예측_종합위험점수", ascending=False).to_excel(
            writer, sheet_name="2026점수등급예측", index=False
        )
        summary.to_excel(writer, sheet_name="점수예측모델비교", index=False)

    with pd.ExcelWriter(SCORE_REPORT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="모델비교", index=False)
        pd.DataFrame(evaluation_rows).to_excel(writer, sheet_name="연도별검증", index=False)
        weight_table.to_excel(writer, sheet_name="가중치", index=False)
        future[forecast_columns].sort_values("2026년_예측_종합위험점수", ascending=False).to_excel(
            writer, sheet_name="2026예측", index=False
        )

    print(json.dumps({
        "가중치합_pct": float(weight_table["가중치_pct"].sum()),
        "선정모델": best_name,
        "순차검증_MAE": best_mae,
        "변수수": len(features),
        "processed": str(PROCESSED),
        "model": str(SCORE_MODEL),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
