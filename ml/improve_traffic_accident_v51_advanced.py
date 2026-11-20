from __future__ import annotations

"""51차 안전등급 모델을 순차 연도검증으로 확장 탐색한다."""

import json
import re
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
MODEL_DIR = ROOT / "models" / "교통사고" / "51차"
REPORT = MODEL_DIR / "51차_고도화등급예측평가.xlsx"
MODEL = MODEL_DIR / "교통사고_51차_고도화등급예측모델.joblib"
CV_YEARS = (2021, 2022, 2023, 2024, 2025)


def grades_from_risk(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(len(order)) // 5 + 1
    return out


def scaled(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df.sort_values(["자치구", "연도"]).copy()
    raw = list(pd.read_excel(DATA, sheet_name="변수가중치")["변수"])
    score = [f"점수_{c}" for c in raw]
    grp = df.groupby("자치구", sort=False)
    for lag in (1, 2, 3):
        df[f"위험점수_lag{lag}"] = grp["현재연도_종합위험점수"].shift(lag)
        df[f"위험등급_lag{lag}"] = grp["현재연도_위험등급"].shift(lag)
    df["점수_증감1"] = df["현재연도_종합위험점수"] - df["위험점수_lag1"]
    df["점수_증감2"] = df["현재연도_종합위험점수"] - df["위험점수_lag2"]
    df["점수_평균2"] = df[["현재연도_종합위험점수", "위험점수_lag1"]].mean(axis=1)
    df["점수_평균3"] = df[["현재연도_종합위험점수", "위험점수_lag1", "위험점수_lag2"]].mean(axis=1)
    df["점수_변동3"] = df[["현재연도_종합위험점수", "위험점수_lag1", "위험점수_lag2"]].std(axis=1).fillna(0)
    district_dummies = pd.get_dummies(df["자치구"], prefix="구", dtype=float)
    df = pd.concat([df, district_dummies], axis=1)
    history = ["현재연도_종합위험점수", "현재연도_위험등급", "위험점수_lag1", "위험점수_lag2", "위험점수_lag3",
               "위험등급_lag1", "위험등급_lag2", "위험등급_lag3", "점수_증감1", "점수_증감2", "점수_평균2", "점수_평균3", "점수_변동3"]
    sets = {
        "history": history,
        "scores_history": score + history,
        "raw_history": raw + history,
        "all": raw + score + history,
        "all_district": raw + score + history + list(district_dummies.columns),
    }
    return df.sort_values(["연도", "자치구"]).reset_index(drop=True), sets


def regressors():
    models = {}
    for alpha in (0.01, .1, 1, 3, 10, 30, 100, 300, 1000):
        models[f"Ridge_{alpha:g}"] = scaled(Ridge(alpha=alpha))
    for depth in (1, 2, 3):
        models[f"GBR_d{depth}"] = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=.025, max_depth=depth, loss="huber", random_state=42))])
    return models


def evaluate(data: pd.DataFrame, feature_sets: dict[str, list[str]]):
    labeled = data[data["다음연도_위험등급"].notna()].copy()
    rows, predictions = [], {}
    # Persistence/smoothing baselines are legitimate forecasts and useful ensemble anchors.
    anchors = {"current": "현재연도_종합위험점수", "mean2": "점수_평균2", "mean3": "점수_평균3"}
    for year in CV_YEARS:
        va = labeled[labeled["예측연도"] == year]
        actual = va["다음연도_위험등급"].astype(int).to_numpy()
        for name, col in anchors.items():
            pred = grades_from_risk(va[col].to_numpy())
            rows.append([name, "anchor", "rank", year, accuracy_score(actual, pred), np.mean(np.abs(actual-pred)<=1), np.mean(np.abs(actual-pred))])
            predictions[(name, "anchor", year)] = va[col].to_numpy()

    for fs_name, features in feature_sets.items():
        for model_name, estimator in regressors().items():
            for target_kind in ("score", "delta"):
                for year in CV_YEARS:
                    tr = labeled[labeled["예측연도"] < year]
                    va = labeled[labeled["예측연도"] == year]
                    y = tr["다음연도_종합위험점수"].to_numpy()
                    if target_kind == "delta":
                        y = y - tr["현재연도_종합위험점수"].to_numpy()
                    fitted = deepcopy(estimator).fit(tr[features], y)
                    risk = fitted.predict(va[features])
                    if target_kind == "delta":
                        risk = va["현재연도_종합위험점수"].to_numpy() + risk
                    actual = va["다음연도_위험등급"].astype(int).to_numpy()
                    for blend in (0, .2, .4, .6, .8):
                        blended = (1-blend)*risk + blend*va["현재연도_종합위험점수"].to_numpy()
                        pred = grades_from_risk(blended)
                        key = f"{model_name}|{target_kind}|b{blend:g}"
                        rows.append([key, fs_name, "rank", year, accuracy_score(actual,pred), np.mean(np.abs(actual-pred)<=1), np.mean(np.abs(actual-pred))])
                        predictions[(key, fs_name, year)] = blended
    detail = pd.DataFrame(rows, columns=["모델", "변수세트", "방식", "검증연도", "정확등급정확도", "±1등급정확도", "평균등급오차"])
    summary = detail.groupby(["모델","변수세트","방식"], as_index=False).agg(
        mean_exact=("정확등급정확도","mean"), min_exact=("정확등급정확도","min"),
        mean_within1=("±1등급정확도","mean"), mean_error=("평균등급오차","mean")).rename(columns={
            "mean_exact":"평균_정확등급정확도", "min_exact":"최저연도_정확도",
            "mean_within1":"평균_±1등급정확도", "mean_error":"평균등급오차"})
    summary = summary.sort_values(["평균_정확등급정확도","평균_±1등급정확도","평균등급오차"], ascending=[False,False,True]).reset_index(drop=True)
    # 상위 단일모델들의 순위를 섞어 경계 인근의 흔들림을 줄인다.
    top_keys = [(r["모델"], r["변수세트"]) for _, r in summary.head(35).iterrows()]
    ensemble_rows = []
    for i, key_a in enumerate(top_keys):
        for key_b in top_keys[i+1:]:
            for weight_a in (.25, .5, .75):
                name = f"ENS[{key_a[0]}@{key_a[1]}]+[{key_b[0]}@{key_b[1]}]|w{weight_a:g}"
                for year in CV_YEARS:
                    risk_a = predictions.get((key_a[0], key_a[1], year))
                    risk_b = predictions.get((key_b[0], key_b[1], year))
                    if risk_a is None or risk_b is None: continue
                    va = labeled[labeled["예측연도"] == year]
                    actual = va["다음연도_위험등급"].astype(int).to_numpy()
                    pred = grades_from_risk(weight_a*risk_a + (1-weight_a)*risk_b)
                    ensemble_rows.append([name,"ensemble","rank",year,accuracy_score(actual,pred),np.mean(np.abs(actual-pred)<=1),np.mean(np.abs(actual-pred))])
    if ensemble_rows:
        detail = pd.concat([detail, pd.DataFrame(ensemble_rows, columns=detail.columns)], ignore_index=True)
        summary = detail.groupby(["모델","변수세트","방식"], as_index=False).agg(
            mean_exact=("정확등급정확도","mean"), min_exact=("정확등급정확도","min"),
            mean_within1=("±1등급정확도","mean"), mean_error=("평균등급오차","mean")).rename(columns={
                "mean_exact":"평균_정확등급정확도", "min_exact":"최저연도_정확도", "mean_within1":"평균_±1등급정확도", "mean_error":"평균등급오차"})
    summary["선정점수"] = summary["평균_정확등급정확도"] + .10*summary["최저연도_정확도"] + .03*summary["평균_±1등급정확도"] - .01*summary["평균등급오차"]
    summary = summary.sort_values(["평균_정확등급정확도","평균_±1등급정확도","평균등급오차"], ascending=[False,False,True]).reset_index(drop=True)
    return labeled, detail, summary


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(DATA, sheet_name="51차점수등급데이터")
    data, feature_sets = add_features(raw)
    labeled, detail, summary = evaluate(data, feature_sets)
    best = summary.iloc[0]
    model_name, fs_name = best["모델"], best["변수세트"]
    def fit_spec(spec: str, selected_fs: str):
        if selected_fs == "anchor":
            return {"kind":"anchor", "column":{"current":"현재연도_종합위험점수","mean2":"점수_평균2","mean3":"점수_평균3"}[spec]}
        base, target_kind, blend_text = spec.split("|")
        y = labeled["다음연도_종합위험점수"].to_numpy().copy()
        if target_kind == "delta": y -= labeled["현재연도_종합위험점수"].to_numpy()
        fitted = regressors()[base].fit(labeled[feature_sets[selected_fs]], y)
        return {"kind":"regression_rank", "model_name":base, "target_kind":target_kind,
                "blend":float(blend_text[1:]), "feature_set":selected_fs,
                "features":feature_sets[selected_fs], "model":fitted}

    if fs_name == "ensemble":
        match = re.fullmatch(r"ENS\[(.+)@([^@]+)\]\+\[(.+)@([^@]+)\]\|w([0-9.]+)", model_name)
        if not match: raise ValueError(f"앙상블 이름 해석 실패: {model_name}")
        spec_a, fs_a, spec_b, fs_b, weight = match.groups()
        bundle = {"kind":"ensemble_rank", "weight_a":float(weight),
                  "component_a":fit_spec(spec_a, fs_a), "component_b":fit_spec(spec_b, fs_b)}
    elif fs_name == "anchor":
        bundle = {"kind":"anchor", "column": {"current":"현재연도_종합위험점수","mean2":"점수_평균2","mean3":"점수_평균3"}[model_name]}
    else:
        bundle = fit_spec(model_name, fs_name)
    joblib.dump(bundle, MODEL)
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="모델비교", index=False)
        detail.to_excel(writer, sheet_name="연도별검증", index=False)
        pd.DataFrame({"변수세트":list(feature_sets), "변수":[", ".join(v) for v in feature_sets.values()]}).to_excel(writer, sheet_name="변수세트", index=False)
    info_path = MODEL_DIR / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({"고도화모델": str(model_name), "고도화변수세트": str(fs_name),
                 "고도화_2021_2025_평균_정확등급정확도_pct": round(float(best["평균_정확등급정확도"]*100),1),
                 "고도화_2021_2025_평균_1칸이내정확도_pct": round(float(best["평균_±1등급정확도"]*100),1),
                 "고도화_평균등급오차": round(float(best["평균등급오차"]),3)})
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
