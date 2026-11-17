from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ml.traffic_model_estimators import CurrentAccidentPersistenceRegressor
from ml.train_traffic_accident_1y_best import DATA_PATH, INDICATORS, SCORE, TARGET, build_safety_features

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "교통사고"
SUMMARY_PATH = MODEL_ROOT / "모델_1차_8차_성능비교.xlsx"
RANDOM_STATE = 42
CV_TARGET_YEARS = [2021, 2022, 2023, 2024]

CURRENT = "현재연도_교통사고건수"
TYPE_FEATURES = [
    "보행어린이 교통사고 건수",
    "어린이 교통사고 건수",
    "어린이 보호구역 내 보행어린이 교통사고 건수",
    "음주운전 사고 건수",
    "자전거 피해 교통사고 건수",
]
TREND_FEATURES = [
    "사고건수_전년증감",
    "사고건수_전년증감률",
    "사고건수_최근3년평균",
    "사고건수_최근3년추세",
    "사고건수_최근3년변동성",
]

MODEL_SPECS = {
    2: {"name": "현재연도 교통사고 건수", "features": [CURRENT]},
    3: {"name": "자동차 등록대수", "features": ["자동차등록대수"]},
    4: {"name": "도로연장", "features": ["도로연장_km"]},
    5: {"name": "총 인구수", "features": ["총인구수"]},
    6: {"name": "최근 사고 추세", "features": TREND_FEATURES},
    7: {"name": "사고 유형별 건수", "features": TYPE_FEATURES},
    8: {"name": "어린이·노인 인구", "features": ["어린이_0_12세_인구수", "노인_65세이상_인구수"]},
}


@dataclass
class Candidate:
    name: str
    family: str
    estimator: BaseEstimator


def metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    error = np.abs(actual - predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
        "WMAPE_pct": float(error.sum() / actual.sum() * 100),
        "오차기반정확도_pct": float(max(0, 100 - error.sum() / actual.sum() * 100)),
        "sMAPE_pct": float(np.mean(2 * error / np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)) * 100),
    }


def pipe(model, scaled=False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidates(include_persistence=False) -> list[Candidate]:
    result = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 300.0]:
        result.append(Candidate(f"Ridge_a{alpha:g}", "Ridge", pipe(Ridge(alpha=alpha), scaled=True)))
    for alpha, ratio in [(0.001, .1), (.001, .5), (.01, .1), (.01, .5), (.1, .1), (.1, .5)]:
        result.append(Candidate(f"ElasticNet_a{alpha:g}_l{ratio:g}", "ElasticNet",
                                pipe(ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=30000), scaled=True)))
    for c, eps, gamma in [(1,.1,"scale"),(10,.1,"scale"),(100,.1,"scale"),(300,.1,"scale"),
                          (10,.01,"scale"),(100,.01,"scale"),(100,.1,.1),(300,.1,.05)]:
        result.append(Candidate(f"SVR_c{c}_e{eps}_g{gamma}", "SVR",
                                pipe(SVR(C=c, epsilon=eps, gamma=gamma, kernel="rbf"), scaled=True)))
    tree_specs = [
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("ExtraTrees", ExtraTreesRegressor(n_estimators=500, max_features=.7, min_samples_leaf=4, random_state=42, n_jobs=1)),
        ("RandomForest", RandomForestRegressor(n_estimators=500, max_features=1.0, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("RandomForest", RandomForestRegressor(n_estimators=500, max_features=.7, min_samples_leaf=4, random_state=42, n_jobs=1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=300, learning_rate=.03, max_depth=2, min_samples_leaf=4, loss="huber", random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=180, learning_rate=.04, max_depth=1, min_samples_leaf=5, loss="huber", random_state=42)),
    ]
    for i, (family, estimator) in enumerate(tree_specs, 1):
        result.append(Candidate(f"{family}_{i}", family, pipe(estimator)))
    if XGBRegressor is not None:
        settings = [
            dict(n_estimators=300,max_depth=2,learning_rate=.03,min_child_weight=3,subsample=.8,colsample_bytree=.9,reg_alpha=0,reg_lambda=10),
            dict(n_estimators=250,max_depth=2,learning_rate=.04,min_child_weight=5,subsample=.9,colsample_bytree=.9,reg_alpha=1,reg_lambda=10),
            dict(n_estimators=400,max_depth=1,learning_rate=.03,min_child_weight=3,subsample=.9,colsample_bytree=1,reg_alpha=0,reg_lambda=5),
        ]
        for i, params in enumerate(settings, 1):
            result.append(Candidate(f"XGBoost_{i}", "XGBoost", pipe(XGBRegressor(
                objective="reg:squarederror", eval_metric="mae", random_state=42, n_jobs=1, verbosity=0, **params))))
    if include_persistence:
        result.append(Candidate("현재값유지", "Persistence", CurrentAccidentPersistenceRegressor(CURRENT)))
    return result


def add_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values(["자치구", "기준연도"]).reset_index(drop=True)
    g = out.groupby("자치구", sort=False)[CURRENT]
    prior = g.shift(1)
    out["사고건수_전년증감"] = out[CURRENT] - prior
    out["사고건수_전년증감률"] = (out[CURRENT] / prior.replace(0, np.nan) - 1) * 100
    out["사고건수_최근3년평균"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    out["사고건수_최근3년추세"] = g.transform(lambda s: s.rolling(3, min_periods=2).apply(
        lambda a: np.polyfit(np.arange(len(a)), a, 1)[0], raw=True))
    out["사고건수_최근3년변동성"] = g.transform(lambda s: s.rolling(3, min_periods=2).std(ddof=0))

    source = pd.read_csv(ROOT / "data" / "interim" / "안전" / "교통사고" /
                         "자치구별_교통사고_통합_2017_2025_최종.csv", encoding="utf-8-sig")
    source = source.rename(columns={"연도": "기준연도"})
    if source.duplicated(["기준연도", "자치구"]).any():
        raise ValueError("통합 교통사고 자료에 연도·자치구 중복이 있습니다.")
    out = out.merge(source[["기준연도", "자치구", *TYPE_FEATURES]],
                    on=["기준연도", "자치구"], how="left", validate="one_to_one")
    return out


def train_one(version: int, frame: pd.DataFrame) -> dict:
    spec = MODEL_SPECS[version]
    features = [SCORE, *spec["features"]]
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"{version}차 필수 특성이 없습니다: {missing}")
    labeled = frame[frame[TARGET].notna()].copy()
    future = frame[frame[TARGET].isna()].copy()
    pool = candidates(include_persistence=version == 2)
    fold_rows, cv_predictions = [], []
    for candidate in pool:
        for year in CV_TARGET_YEARS:
            train = labeled[labeled["예측연도"] < year]
            valid = labeled[labeled["예측연도"] == year]
            model = clone(candidate.estimator).fit(train[features], train[TARGET])
            pred = np.maximum(model.predict(valid[features]), 0)
            fold_rows.append({"후보명":candidate.name,"알고리즘":candidate.family,"검증연도":year,
                              "학습행수":len(train),**metrics(valid[TARGET],pred)})
            part = valid[["기준연도","예측연도","자치구",TARGET]].copy()
            part["후보명"], part["예측값"] = candidate.name, pred
            cv_predictions.append(part)
    folds = pd.DataFrame(fold_rows)
    comparison = folds.groupby(["후보명","알고리즘"],as_index=False).agg(
        평균_MAE=("MAE","mean"),평균_RMSE=("RMSE","mean"),평균_R2=("R2","mean"),
        평균_WMAPE_pct=("WMAPE_pct","mean"),평균_오차기반정확도_pct=("오차기반정확도_pct","mean")
    ).sort_values(["평균_MAE","평균_RMSE"]).reset_index(drop=True)
    selected_name = comparison.iloc[0]["후보명"]
    selected = next(c for c in pool if c.name == selected_name)

    train_2024 = labeled[labeled["예측연도"] <= 2024]
    holdout = labeled[labeled["예측연도"] == 2025]
    holdout_model = clone(selected.estimator).fit(train_2024[features], train_2024[TARGET])
    holdout_pred = np.maximum(holdout_model.predict(holdout[features]), 0)
    holdout_metrics = metrics(holdout[TARGET], holdout_pred)
    holdout_out = holdout[["기준연도","예측연도","서울시","자치구",TARGET,*features]].copy()
    holdout_out["예측값"] = np.round(holdout_pred,1)
    holdout_out["절대오차"] = np.abs(holdout_out[TARGET]-holdout_out["예측값"])

    final_model = clone(selected.estimator).fit(labeled[features], labeled[TARGET])
    future_pred = np.maximum(final_model.predict(future[features]), 0)
    forecast = future[["기준연도","예측연도","서울시","자치구",*features]].copy()
    forecast["예측_다음연도_교통사고건수"] = np.round(future_pred,1)

    importance = permutation_importance(holdout_model,holdout[features],holdout[TARGET],
                                        scoring="neg_mean_absolute_error",n_repeats=50,
                                        random_state=42,n_jobs=1)
    imp = pd.DataFrame({"특성":features,"순열중요도_MAE증가":importance.importances_mean,
                        "표준편차":importance.importances_std}).sort_values("순열중요도_MAE증가",ascending=False)

    out = MODEL_ROOT / f"{version}차"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model_type":f"SafetyStandardPlus{version}","version":version,"horizon_years":1,
               "target":TARGET,"model":final_model,"features":features,"selected_candidate":selected_name,
               "base_safety_indicators":INDICATORS,"added_group":spec["name"],
               "data_sha256":hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()}
    joblib.dump(payload,out/f"교통사고_{version}차_예측모델.joblib")
    metadata = {"차수":version,"설계":f"1차 안전기준표 + {spec['name']}","예측대상":TARGET,
                "입력특성":features,"추가특성":spec["features"],"선정모델":selected_name,
                "교차검증최고":comparison.iloc[0].to_dict(),"2025_홀드아웃성능":holdout_metrics,
                "학습행수":len(labeled),"2026_예측행수":len(forecast)}
    (out/"model_info.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    with pd.ExcelWriter(out/"모델평가_및_2026예측.xlsx",engine="openpyxl") as writer:
        pd.DataFrame({"구분":["1차 공통","추가"],"특성":[SCORE,", ".join(spec["features"])]}).to_excel(writer,sheet_name="사용변수",index=False)
        comparison.to_excel(writer,sheet_name="후보모델비교",index=False)
        folds.to_excel(writer,sheet_name="연도별교차검증",index=False)
        pd.concat(cv_predictions,ignore_index=True).to_excel(writer,sheet_name="교차검증예측",index=False)
        holdout_out.to_excel(writer,sheet_name="2025홀드아웃검증",index=False)
        forecast.to_excel(writer,sheet_name="2026예측",index=False)
        imp.to_excel(writer,sheet_name="특성중요도",index=False)
    (out/"README.md").write_text(
        f"# 교통사고 {version}차 모델\n\n- 설계: 1차 안전기준표 + {spec['name']}\n"
        f"- 선정 모델: {selected_name}\n- 2025 홀드아웃 MAE: {holdout_metrics['MAE']:.2f}건\n"
        f"- WMAPE: {holdout_metrics['WMAPE_pct']:.2f}%\n- 오차 기반 참고 정확도: {holdout_metrics['오차기반정확도_pct']:.2f}%\n"
        f"- R²: {holdout_metrics['R2']:.4f}\n",encoding="utf-8")
    print(f"{version}차 완료: {selected_name}, holdout MAE={holdout_metrics['MAE']:.2f}",flush=True)
    return {"차수":f"{version}차","추가변수군":spec["name"],"선정모델":selected_name,
            **{f"홀드아웃_{k}":v for k,v in holdout_metrics.items()},"입력특성수":len(features)}


def main():
    raw = pd.read_excel(DATA_PATH,sheet_name="모델데이터")
    frame = add_external_features(build_safety_features(raw))
    summaries = [train_one(version,frame) for version in MODEL_SPECS]
    first_info = json.loads((MODEL_ROOT/"1차"/"model_info.json").read_text(encoding="utf-8"))
    first_metrics = first_info["2025_홀드아웃성능"]
    first = {"차수":"1차","추가변수군":"없음(안전기준표만)","선정모델":first_info["최종후보"],
             **{f"홀드아웃_{k}":v for k,v in first_metrics.items()},
             "홀드아웃_오차기반정확도_pct": 100 - first_metrics["WMAPE_pct"], "입력특성수":1}
    summary = pd.DataFrame([first,*summaries])
    with pd.ExcelWriter(SUMMARY_PATH,engine="openpyxl") as writer:
        summary.to_excel(writer,sheet_name="성능비교",index=False)
        pd.DataFrame([{"차수":f"{k}차","설계":f"1차 안전기준표 + {v['name']}","추가특성":", ".join(v["features"])}
                      for k,v in MODEL_SPECS.items()]).to_excel(writer,sheet_name="차수별설계",index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
