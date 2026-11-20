from __future__ import annotations

import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.train_best_child_traffic_accident_rate import ROOT


BASE_VERSION = 49
VERSION = 51
BASE_MODEL_DIR = ROOT / "models" / "교통사고" / "41차~49차" / "49차"
MODEL_DIR = ROOT / "models" / "교통사고" / "51차"
PROCESSED = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
OUT_MODEL = MODEL_DIR / "교통사고_51차_개선등급예측모델.joblib"
OUT_REPORT = MODEL_DIR / "51차_개선등급예측평가.xlsx"
CV_YEARS = (2021, 2022, 2023, 2024, 2025)


def equal_quintile_grade(risk: np.ndarray) -> np.ndarray:
    # risk가 클수록 위험: 25개 자치구를 5개씩 1~5등급으로 배정한다.
    order = np.argsort(-np.asarray(risk), kind="stable")
    grade = np.empty(len(order), dtype=int)
    grade[order] = np.ceil((np.arange(len(order)) + 1) / (len(order) / 5)).astype(int)
    return np.clip(grade, 1, 5)


def scaled(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def candidates():
    result = {}
    for c in (.01, .1, 1, 10, 100):
        result[f"Logistic_C{c:g}"] = scaled(LogisticRegression(
            C=c, max_iter=5000, class_weight="balanced", random_state=42
        ))
    for alpha in (.1, 1, 10, 100, 300):
        result[f"RidgeClassifier_{alpha:g}"] = scaled(RidgeClassifier(alpha=alpha, class_weight="balanced"))
    for c in (.1, 1, 10):
        result[f"SVC_linear_C{c:g}"] = scaled(SVC(C=c, kernel="linear", probability=True,
                                                    class_weight="balanced", random_state=42))
        result[f"SVC_rbf_C{c:g}"] = scaled(SVC(C=c, kernel="rbf", probability=True,
                                                 class_weight="balanced", random_state=42))
    result["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=.7,
                                          class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    result["ExtraTrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=.7,
                                        class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    result["GradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=150, learning_rate=.03,
                                              max_depth=2, random_state=42)),
    ])
    for k in (5, 9, 15):
        result[f"KNN_{k}"] = scaled(KNeighborsClassifier(n_neighbors=k, weights="distance"))
    return result


def risk_from_model(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = model.classes_.astype(float)
        expected_grade = probabilities @ classes
        return -expected_grade
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        classes = model.classes_.astype(float)
        if decision.ndim == 1:
            return decision
        shifted = decision - decision.max(axis=1, keepdims=True)
        probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        return -(probability @ classes)
    return -model.predict(x).astype(float)


def add_history(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["자치구", "연도"]).copy()
    group = data.groupby("자치구", sort=False)["현재연도_종합위험점수"]
    data["종합점수_최근2년평균"] = group.transform(lambda s: s.rolling(2, min_periods=1).mean())
    data["종합점수_최근3년평균"] = group.transform(lambda s: s.rolling(3, min_periods=1).mean())
    data["종합점수_최근3년변동성"] = group.transform(lambda s: s.rolling(3, min_periods=2).std(ddof=0)).fillna(0)
    data["종합점수_전년증감"] = group.diff().fillna(0)
    data["전년_위험등급"] = data.groupby("자치구", sort=False)["현재연도_위험등급"].shift(1).fillna(
        data["현재연도_위험등급"]
    )
    return data.sort_values(["연도", "자치구"]).reset_index(drop=True)


def main() -> None:
    data = pd.read_excel(PROCESSED, sheet_name="51차점수등급데이터")
    data = add_history(data)
    raw_features = json.loads((BASE_MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))["최종변수"]
    score_features = [f"점수_{feature}" for feature in raw_features]
    history_features = [
        "현재연도_종합위험점수", "현재연도_위험등급", "종합점수_최근2년평균",
        "종합점수_최근3년평균", "종합점수_최근3년변동성", "종합점수_전년증감", "전년_위험등급",
    ]
    model_features = [*raw_features, *score_features, *history_features]
    labeled = data[data["다음연도_위험등급"].notna()].copy()

    rows = []
    model_defs = candidates()
    # 현재 등급 유지 기준
    for year in CV_YEARS:
        valid = labeled[labeled["예측연도"] == year]
        actual = valid["다음연도_위험등급"].astype(int).to_numpy()
        pred = valid["현재연도_위험등급"].astype(int).to_numpy()
        rows.append({"모델": "현재등급유지", "방식": "직접", "검증연도": year,
                     "등급정확도": accuracy_score(actual, pred),
                     "1칸이내정확도": np.mean(np.abs(actual - pred) <= 1),
                     "평균등급오차": np.mean(np.abs(actual - pred))})

    for name, estimator in model_defs.items():
        for year in CV_YEARS:
            train = labeled[labeled["예측연도"] < year]
            valid = labeled[labeled["예측연도"] == year]
            fitted = deepcopy(estimator).fit(train[model_features], train["다음연도_위험등급"].astype(int))
            actual = valid["다음연도_위험등급"].astype(int).to_numpy()
            direct = fitted.predict(valid[model_features]).astype(int)
            ranked = equal_quintile_grade(risk_from_model(fitted, valid[model_features]))
            for method, pred in (("직접", direct), ("순위보정", ranked)):
                rows.append({"모델": name, "방식": method, "검증연도": year,
                             "등급정확도": accuracy_score(actual, pred),
                             "1칸이내정확도": np.mean(np.abs(actual - pred) <= 1),
                             "평균등급오차": np.mean(np.abs(actual - pred))})

    # 등급을 연속값으로 학습하는 순서형 근사도 비교한다.
    ordinal_defs = {f"OrdinalRidge_{a:g}": scaled(Ridge(alpha=a)) for a in (.1, 1, 10, 100, 300)}
    for name, estimator in ordinal_defs.items():
        for year in CV_YEARS:
            train = labeled[labeled["예측연도"] < year]
            valid = labeled[labeled["예측연도"] == year]
            fitted = deepcopy(estimator).fit(train[model_features], train["다음연도_위험등급"].astype(int))
            continuous = fitted.predict(valid[model_features])
            actual = valid["다음연도_위험등급"].astype(int).to_numpy()
            direct = np.clip(np.rint(continuous), 1, 5).astype(int)
            ranked = equal_quintile_grade(-continuous)
            for method, pred in (("직접", direct), ("순위보정", ranked)):
                rows.append({"모델": name, "방식": method, "검증연도": year,
                             "등급정확도": accuracy_score(actual, pred),
                             "1칸이내정확도": np.mean(np.abs(actual - pred) <= 1),
                             "평균등급오차": np.mean(np.abs(actual - pred))})

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["모델", "방식"], as_index=False).agg(
        평균_등급정확도=("등급정확도", "mean"),
        평균_1칸이내정확도=("1칸이내정확도", "mean"),
        평균_등급오차=("평균등급오차", "mean"),
    ).sort_values(["평균_등급정확도", "평균_1칸이내정확도", "평균_등급오차"],
                  ascending=[False, False, True]).reset_index(drop=True)
    winner = summary.iloc[0]
    winner_name, winner_method = winner["모델"], winner["방식"]

    if winner_name.startswith("OrdinalRidge_"):
        alpha = float(winner_name.split("_")[1])
        final = deepcopy(ordinal_defs[winner_name]).fit(labeled[model_features], labeled["다음연도_위험등급"].astype(int))
        prediction_kind = "ordinal"
    else:
        final = deepcopy(model_defs[winner_name]).fit(labeled[model_features], labeled["다음연도_위험등급"].astype(int))
        prediction_kind = "classifier"

    future = data[data["예측연도"] == 2026].copy()
    if prediction_kind == "ordinal":
        continuous = final.predict(future[model_features])
        future_grade = (equal_quintile_grade(-continuous) if winner_method == "순위보정"
                        else np.clip(np.rint(continuous), 1, 5).astype(int))
        risk = -continuous
    else:
        direct = final.predict(future[model_features]).astype(int)
        risk = risk_from_model(final, future[model_features])
        future_grade = equal_quintile_grade(risk) if winner_method == "순위보정" else direct

    forecast = future[["자치구", "현재연도_종합위험점수", "현재연도_위험등급"]].copy()
    forecast["2026년_개선예측_위험등급"] = future_grade
    forecast["모델위험순위"] = pd.Series(risk).rank(ascending=False, method="min").astype(int).to_numpy()
    forecast = forecast.sort_values(["2026년_개선예측_위험등급", "모델위험순위"])

    joblib.dump({
        "version": VERSION, "model_name": winner_name, "method": winner_method,
        "model_features": model_features, "model": final,
    }, OUT_MODEL)
    with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="모델비교", index=False)
        detail.to_excel(writer, sheet_name="연도별검증", index=False)
        forecast.to_excel(writer, sheet_name="2026개선등급예측", index=False)
    with pd.ExcelWriter(PROCESSED, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary.to_excel(writer, sheet_name="개선등급모델비교", index=False)
        forecast.to_excel(writer, sheet_name="개선2026등급예측", index=False)

    print(json.dumps({
        "선정모델": winner_name, "방식": winner_method,
        "등급정확도_pct": float(winner["평균_등급정확도"] * 100),
        "1칸이내정확도_pct": float(winner["평균_1칸이내정확도"] * 100),
        "평균등급오차": float(winner["평균_등급오차"]),
        "기존정확도_pct": 44.0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
