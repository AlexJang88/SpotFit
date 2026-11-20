from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "교통사고_초기모델.xlsx"
OUT_DIR = ROOT / "models" / "교통사고" / "1차"
TARGET = "다음연도_교통사고건수"
RANDOM_STATE = 42
CV_TARGET_YEARS = [2021, 2022, 2023, 2024]

# 의료보장사업장수 2.09는 재난약자에, 교통안전예산 2.76은 CCTV에 승계했다.
INDICATORS = {
    "인구1만명당_교통사고사망자수": {"weight": 50.00, "direction": "risk", "section": "위해"},
    "인구1만명당_재난약자수": {"weight": 10.00, "direction": "risk", "section": "취약"},
    "도로면적당_전체CCTV누적대수_대체지표": {"weight": 20.00, "direction": "safe", "section": "경감"},
    "운전자안전벨트착용률_pct": {"weight": 8.99, "direction": "safe", "section": "의식"},
    "자동차음주운전경험률_pct": {"weight": 8.70, "direction": "risk", "section": "의식"},
    "자동차1만대당_법규위반다발지역발생건수_대체지표": {"weight": 2.31, "direction": "risk", "section": "의식"},
}
CONTRIBUTIONS = [f"점수기여_{name}" for name in INDICATORS]
SCORE = "안전기준표점수_100점"


@dataclass
class Candidate:
    name: str
    family: str
    feature_set: str
    features: list[str]
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
        "sMAPE_pct": float(np.mean(2 * error / np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)) * 100),
    }


def minmax_within_year(frame: pd.DataFrame, column: str) -> pd.Series:
    def scale(series: pd.Series) -> pd.Series:
        low, high = series.min(), series.max()
        if pd.isna(low) or pd.isna(high) or high <= low:
            return pd.Series(0.5, index=series.index)
        return (series - low) / (high - low)
    return frame.groupby("기준연도", group_keys=False)[column].apply(scale)


def build_safety_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if abs(sum(v["weight"] for v in INDICATORS.values()) - 100) > 1e-9:
        raise ValueError("안전기준표 가중치 합이 100이 아닙니다.")
    for name, spec in INDICATORS.items():
        out[name] = pd.to_numeric(out[name], errors="raise")
        normalized = minmax_within_year(out, name)
        safe_value = normalized if spec["direction"] == "safe" else 1 - normalized
        out[f"점수기여_{name}"] = safe_value * spec["weight"]
    out[SCORE] = out[CONTRIBUTIONS].sum(axis=1)
    return out


def pipeline(model: BaseEstimator, scaled=False, polynomial=False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if polynomial:
        steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False)))
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidate_models() -> list[Candidate]:
    feature_sets = {"가중합점수": [SCORE], "가중구성요소": CONTRIBUTIONS}
    result = []
    for set_name, features in feature_sets.items():
        for alpha in [0.1, 1.0, 10.0, 100.0, 300.0]:
            result.append(Candidate(f"Ridge_{set_name}_a{alpha:g}", "Ridge", set_name, features,
                                    pipeline(Ridge(alpha=alpha), scaled=True)))
        if set_name == "가중합점수":
            for alpha in [0.1, 1.0, 10.0, 100.0]:
                result.append(Candidate(f"PolynomialRidge_{set_name}_a{alpha:g}", "PolynomialRidge", set_name,
                                        features, pipeline(Ridge(alpha=alpha), scaled=True, polynomial=True)))
        tree_models = [
            ("ExtraTrees", ExtraTreesRegressor(n_estimators=800, max_features=1.0, min_samples_leaf=3, random_state=42, n_jobs=1)),
            ("ExtraTrees", ExtraTreesRegressor(n_estimators=800, max_features=0.8, min_samples_leaf=5, random_state=42, n_jobs=1)),
            ("RandomForest", RandomForestRegressor(n_estimators=800, max_features=1.0, min_samples_leaf=3, random_state=42, n_jobs=1)),
            ("RandomForest", RandomForestRegressor(n_estimators=800, max_features=0.8, min_samples_leaf=5, random_state=42, n_jobs=1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=42)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=180, learning_rate=0.04, max_depth=1, min_samples_leaf=5, loss="huber", random_state=42)),
        ]
        for i, (family, model) in enumerate(tree_models, 1):
            result.append(Candidate(f"{family}_{set_name}_{i}", family, set_name, features, pipeline(model)))
        if XGBRegressor is not None:
            settings = [
                dict(n_estimators=300, max_depth=2, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=10.0),
                dict(n_estimators=220, max_depth=1, learning_rate=0.04, min_child_weight=5, subsample=0.9, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=10.0),
            ]
            for i, params in enumerate(settings, 1):
                model = XGBRegressor(objective="reg:squarederror", eval_metric="mae", random_state=42, n_jobs=1, verbosity=0, **params)
                result.append(Candidate(f"XGBoost_{set_name}_{i}", "XGBoost", set_name, features, pipeline(model)))
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(DATA_PATH, sheet_name="모델데이터")
    required = ["기준연도", "예측연도", "서울시", "자치구", TARGET, *INDICATORS]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    if raw.duplicated(["기준연도", "자치구"]).any():
        raise ValueError("기준연도·자치구 중복이 있습니다.")
    frame = build_safety_features(raw)
    labeled = frame[frame[TARGET].notna()].copy()
    future = frame[frame[TARGET].isna()].copy()
    if len(labeled) != 200 or len(future) != 25:
        raise ValueError(f"예상 행 수와 다릅니다: 학습 {len(labeled)}, 미래 {len(future)}")

    candidates = candidate_models()
    fold_rows, prediction_rows = [], []
    for candidate in candidates:
        for target_year in CV_TARGET_YEARS:
            train = labeled[labeled["예측연도"] < target_year]
            valid = labeled[labeled["예측연도"] == target_year]
            fitted = clone(candidate.estimator).fit(train[candidate.features], train[TARGET])
            pred = np.maximum(fitted.predict(valid[candidate.features]), 0)
            fold_rows.append({"후보명": candidate.name, "알고리즘": candidate.family,
                              "특성세트": candidate.feature_set, "검증연도": target_year,
                              "학습행수": len(train), **metrics(valid[TARGET], pred)})
            temp = valid[["기준연도", "예측연도", "자치구", TARGET]].copy()
            temp["후보명"], temp["예측값"] = candidate.name, pred
            prediction_rows.append(temp)

    folds = pd.DataFrame(fold_rows)
    comparison = folds.groupby(["후보명", "알고리즘", "특성세트"], as_index=False).agg(
        평균_MAE=("MAE", "mean"), 평균_RMSE=("RMSE", "mean"), 평균_R2=("R2", "mean"),
        전체_WMAPE_pct=("WMAPE_pct", "mean"), 전체_sMAPE_pct=("sMAPE_pct", "mean"),
    ).sort_values(["평균_MAE", "평균_RMSE"]).reset_index(drop=True)
    selected_name = comparison.iloc[0]["후보명"]
    selected = next(c for c in candidates if c.name == selected_name)

    # 2025년은 모델 선택에 사용하지 않는 최종 홀드아웃이다.
    train_2024 = labeled[labeled["예측연도"] <= 2024]
    holdout = labeled[labeled["예측연도"] == 2025]
    holdout_model = clone(selected.estimator).fit(train_2024[selected.features], train_2024[TARGET])
    holdout_pred = np.maximum(holdout_model.predict(holdout[selected.features]), 0)
    holdout_metrics = metrics(holdout[TARGET], holdout_pred)
    holdout_output = holdout[["기준연도", "예측연도", "서울시", "자치구", SCORE, TARGET]].copy()
    holdout_output["예측_다음연도_교통사고건수"] = np.round(holdout_pred, 1)
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - holdout_output["예측_다음연도_교통사고건수"])

    deployment_model = clone(selected.estimator).fit(labeled[selected.features], labeled[TARGET])
    future_pred = np.maximum(deployment_model.predict(future[selected.features]), 0)
    prediction_output = future[["기준연도", "예측연도", "서울시", "자치구", SCORE]].copy()
    prediction_output["예측_다음연도_교통사고건수"] = np.round(future_pred, 1)

    importance = permutation_importance(holdout_model, holdout[selected.features], holdout[TARGET],
                                        scoring="neg_mean_absolute_error", n_repeats=50,
                                        random_state=RANDOM_STATE, n_jobs=1)
    importance_output = pd.DataFrame({"모델입력특성": selected.features,
                                      "순열중요도_MAE증가": importance.importances_mean,
                                      "표준편차": importance.importances_std}).sort_values("순열중요도_MAE증가", ascending=False)

    indicator_audit = pd.DataFrame([
        {"기준표영역": spec["section"], "사용지표": name, "최종가중치": spec["weight"],
         "방향": "높을수록 안전" if spec["direction"] == "safe" else "높을수록 위험",
         "상태": "대체지표 사용" if "대체지표" in name else "사용"}
        for name, spec in INDICATORS.items()
    ] + [
        {"기준표영역": "취약", "사용지표": "인구1만명당_의료보장사업장수", "최종가중치": 0,
         "방향": "높을수록 위험", "상태": "미확보로 제외; 2.09를 재난약자에 승계"},
        {"기준표영역": "경감", "사용지표": "지역교통_안전환경개선사업_지출액", "최종가중치": 0,
         "방향": "높을수록 안전", "상태": "미확보로 제외; 2.76을 CCTV에 승계"},
    ])

    payload = {"model_type": "SafetyStandardOnly", "horizon_years": 1, "target": TARGET,
               "model": deployment_model, "features": selected.features,
               "selected_candidate": selected.name, "safety_indicators": INDICATORS,
               "score_formula": "연도별 25개 자치구 min-max 정규화 → 방향 변환 → 가중합",
               "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()}
    joblib.dump(payload, OUT_DIR / "교통사고_1차_안전기준표모델.joblib")

    metadata = {"모델명": "교통사고 1차 비교모델",
                "원칙": "안전기준표 교통사고 변수와 그 계산 결과만 사용",
                "예측대상": "다음 연도 자치구별 전체 교통사고 건수",
                "최종후보": selected.name, "최종입력특성": selected.features,
                "원지표": INDICATORS, "기준표외_X변수": [],
                "교차검증최고": comparison.iloc[0].to_dict(),
                "2025_홀드아웃성능": holdout_metrics,
                "학습행수": len(labeled), "2026_예측행수": len(prediction_output)}
    (OUT_DIR / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(OUT_DIR / "모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        indicator_audit.to_excel(writer, sheet_name="사용변수검사", index=False)
        comparison.to_excel(writer, sheet_name="후보모델비교", index=False)
        folds.to_excel(writer, sheet_name="연도별교차검증", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_excel(writer, sheet_name="교차검증예측", index=False)
        holdout_output.to_excel(writer, sheet_name="2025홀드아웃검증", index=False)
        prediction_output.to_excel(writer, sheet_name="2026예측", index=False)
        importance_output.to_excel(writer, sheet_name="특성중요도", index=False)

    report = f"""# 교통사고 1차 비교모델

## 원칙
- 안전기준표의 교통사고 지표와 그 지표로 계산한 점수만 X로 사용합니다.
- 자치구명, 연도, 현재 사고 건수, 사고 이력, 총인구수 등 기준표 밖 변수는 X로 사용하지 않습니다.
- Y는 다음 연도 자치구별 전체 교통사고 건수입니다.

## 가중치
- 사망위험 50.00, 재난약자 10.00, CCTV 20.00
- 안전벨트 8.99, 음주운전 8.70, 법규위반 2.31

## 결과
- 선택 후보: {selected.name}
- 2025년 홀드아웃 MAE: {holdout_metrics['MAE']:.2f}건
- RMSE: {holdout_metrics['RMSE']:.2f}건
- R²: {holdout_metrics['R2']:.4f}
- WMAPE: {holdout_metrics['WMAPE_pct']:.2f}%

이 모델은 이후 기준표 밖 변수를 추가했을 때 성능이 얼마나 개선되는지 비교하기 위한 기준 모델입니다.
"""
    (OUT_DIR / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
