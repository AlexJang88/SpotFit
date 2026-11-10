from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "안전"
TRAFFIC = INTERIM / "교통사고"
HOTSPOT = INTERIM / "사고다발지역"
SCHOOL_PATH = INTERIM / "자치구별_유초중고_학교현황_2017_2025_최종.xlsx"
MODEL_VERSION = 41
OUT = ROOT / "models" / "교통사고" / f"{MODEL_VERSION}차"
TARGET = "다음연도_어린이교통사고피해자율_어린이1만명당"
CURRENT_RATE = "현재연도_어린이교통사고피해자율_어린이1만명당"
CV_YEARS = [2021, 2022, 2023, 2024]


@dataclass
class Candidate:
    name: str
    mode: str
    feature_group: str
    estimator: Pipeline


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_one(left: pd.DataFrame, path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    right = read_csv(path)
    keys = ["연도", "자치구"]
    if columns is not None:
        right = right[keys + columns]
    if right.duplicated(keys).any():
        raise ValueError(f"연도·자치구 중복: {path.name}")
    return left.merge(right, on=keys, how="left", validate="one_to_one")


def hotspot_features(path: Path, prefix: str) -> pd.DataFrame:
    data = read_csv(path)
    numeric = ["발생건수", "사상자수", "사망자수", "중상자수", "경상자수", "부상신고자수"]
    grouped = data.groupby(["연도", "자치구"], as_index=False).agg(
        **{f"{prefix}_지점수": ("사고다발지역ID", "size")},
        **{f"{prefix}_{column}합계": (column, "sum") for column in numeric},
    )
    return grouped


def build_panel() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    base_path = TRAFFIC / "자치구별_교통사고_통합_2017_2025_최종.csv"
    panel = read_csv(base_path)
    if panel.duplicated(["연도", "자치구"]).any() or len(panel) != 225:
        raise ValueError("통합 교통사고 자료의 키 또는 행 수를 확인하세요.")

    for path in [
        INTERIM / "자치구별_도로현황_2017_2025_최종.csv",
        INTERIM / "자치구별_범죄예방CCTV_2017_2025_최종.csv",
        INTERIM / "자치구별_승용차통행속도_2017_2025_최종.csv",
        INTERIM / "자치구별_전체CCTV_2017_2025_최종.csv",
    ]:
        panel = merge_one(panel, path, [c for c in read_csv(path).columns if c not in ["연도", "서울시", "자치구"]])

    school = pd.read_excel(SCHOOL_PATH, sheet_name="자치구별_모델용")
    school = school.drop(columns=["서울시"], errors="ignore")
    if len(school) != 225 or school.duplicated(["연도", "자치구"]).any():
        raise ValueError("학교현황 자료가 225개 고유 연도·자치구 행이 아닙니다.")
    panel = panel.merge(school, on=["연도", "자치구"], how="left", validate="one_to_one")

    behavior_path = TRAFFIC / "자치구별_교통안전행태_2017_2025_최종.csv"
    behavior = read_csv(behavior_path)
    behavior = behavior[behavior["연령구간"].str.startswith("전체", na=False)].copy()
    behavior_columns = [column for column in behavior.columns if column.endswith("률(%)")]
    behavior = behavior[["연도", "자치구", *behavior_columns]]
    behavior[behavior_columns] = behavior[behavior_columns].apply(pd.to_numeric, errors="coerce")
    if len(behavior) != 225 or behavior.duplicated(["연도", "자치구"]).any():
        raise ValueError("교통안전행태 전체 연령 자료가 225개 자치구·연도 행이 아닙니다.")
    panel = panel.merge(behavior, on=["연도", "자치구"], how="left", validate="one_to_one")

    hotspot_sources = [
        ("자치구별_교통사고다발지역_2017_2025.csv", "전체사고다발"),
        ("자치구별_법규위반별교통사고다발지역_2017_2025.csv", "법규위반다발"),
        ("자치구별_보행어린이교통사고다발지역_2017_2025.csv", "보행어린이다발"),
        ("자치구별_어린이보호구역내어린이교통사고다발지역_2017_2025.csv", "보호구역어린이다발"),
        ("자치구별_자전거교통사고다발지역_2017_2025.csv", "자전거다발"),
    ]
    hotspot_columns = []
    for filename, prefix in hotspot_sources:
        aggregated = hotspot_features(HOTSPOT / filename, prefix)
        added = [c for c in aggregated.columns if c not in ["연도", "자치구"]]
        hotspot_columns.extend(added)
        panel = panel.merge(aggregated, on=["연도", "자치구"], how="left", validate="one_to_one")
        panel[added] = panel[added].fillna(0)

    child_deaths = "어린이 교통사고 사망자 수"
    child_injuries = "어린이 교통사고 부상자 수"
    child_population = "어린이(13세 미만) 인구수"
    if (panel[child_population] <= 0).any():
        raise ValueError("어린이 인구가 0 이하인 행은 피해자율을 계산할 수 없습니다.")
    panel["어린이교통사고피해자수"] = panel[child_deaths] + panel[child_injuries]
    panel[CURRENT_RATE] = panel["어린이교통사고피해자수"] / panel[child_population] * 10000
    panel["보행어린이교통사고율_어린이1만명당"] = panel["보행어린이 교통사고 건수"] / panel[child_population] * 10000
    panel["보호구역보행어린이교통사고율_어린이1만명당"] = (
        panel["어린이 보호구역 내 보행어린이 교통사고 건수"] / panel[child_population] * 10000
    )
    panel["어린이부상률_어린이1만명당"] = panel["어린이 교통사고 부상자 수"] / panel[child_population] * 10000
    panel["어린이인구비율_pct"] = panel[child_population] / panel["총인구수"] * 100
    panel["자동차1만대당_전체교통사고건수"] = panel["교통사고 건수"] / panel["자동차 등록대수 합계"] * 10000
    panel["어린이1만명당_자동차등록대수"] = panel["자동차 등록대수 합계"] / panel[child_population] * 10000
    panel["도로km당_전체교통사고건수"] = panel["교통사고 건수"] / panel["도로연장(km)"]
    panel["도로면적당_전체CCTV"] = panel["전체_CCTV_현존기준_설치연도누적대수"] / panel["도로면적(km²)"]

    panel = panel.sort_values(["자치구", "연도"]).reset_index(drop=True)
    group = panel.groupby("자치구", sort=False)[CURRENT_RATE]
    panel["어린이피해자율_전년증감"] = group.diff()
    panel["어린이피해자율_최근3년평균"] = group.transform(lambda s: s.rolling(3, min_periods=2).mean())
    panel["어린이피해자율_최근3년추세"] = group.transform(
        lambda s: s.rolling(3, min_periods=2).apply(lambda a: np.polyfit(np.arange(len(a)), a, 1)[0], raw=True)
    )
    panel["어린이피해자율_최근3년변동성"] = group.transform(lambda s: s.rolling(3, min_periods=2).std(ddof=0))
    panel[TARGET] = group.shift(-1)
    panel["예측연도"] = panel["연도"] + 1

    excluded = {"연도", "예측연도", "자치구", TARGET}
    numeric_features = [c for c in panel.select_dtypes(include=[np.number]).columns if c not in excluded]
    # 다음 연도 정보나 식별용 코드는 사용하지 않는다.
    numeric_features = [c for c in numeric_features if c not in ["행정구역코드", "자치구코드"]]
    audit = pd.DataFrame({
        "변수": numeric_features,
        "결측수": [int(panel[c].isna().sum()) for c in numeric_features],
        "고유값수": [int(panel[c].nunique(dropna=True)) for c in numeric_features],
        "사용시점": "현재연도",
    })
    return panel, numeric_features, audit


def pipe(model, scaled=False, k: int | None = None) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if k is not None:
        steps.append(("selector", SelectKBest(f_regression, k=k)))
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidates(feature_count: int) -> list[Candidate]:
    result = []
    for k in [10, 20, 30, 45, 60, 75, feature_count]:
        k = min(k, feature_count)
        label = "전체" if k == feature_count else f"상위{k}"
        for alpha in [1, 10, 100]:
            result.append(Candidate(f"Ridge_{label}_a{alpha}", "direct", label, pipe(Ridge(alpha=alpha), True, None if k == feature_count else k)))
            result.append(Candidate(f"DeltaRidge_{label}_a{alpha}", "delta", label, pipe(Ridge(alpha=alpha), True, None if k == feature_count else k)))
    for alpha in [.001, .01, .1]:
        result.append(Candidate(f"ElasticNet_전체_a{alpha}", "direct", "전체", pipe(ElasticNet(alpha=alpha, l1_ratio=.2, max_iter=50000), True)))
        result.append(Candidate(f"DeltaElasticNet_전체_a{alpha}", "delta", "전체", pipe(ElasticNet(alpha=alpha, l1_ratio=.2, max_iter=50000), True)))
    trees = [
        ("ExtraTrees_leaf3", ExtraTreesRegressor(n_estimators=800, min_samples_leaf=3, max_features=.7, random_state=42, n_jobs=1)),
        ("ExtraTrees_leaf5", ExtraTreesRegressor(n_estimators=800, min_samples_leaf=5, max_features=.7, random_state=42, n_jobs=1)),
        ("RandomForest_leaf4", RandomForestRegressor(n_estimators=800, min_samples_leaf=4, max_features=.7, random_state=42, n_jobs=1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=300, learning_rate=.03, max_depth=2, min_samples_leaf=5, loss="huber", random_state=42)),
    ]
    for name, model in trees:
        result.append(Candidate(name, "direct", "전체", pipe(model)))
        result.append(Candidate("Delta" + name, "delta", "전체", pipe(clone(model))))
    unique = {}
    for candidate in result:
        unique[candidate.name] = candidate
    return list(unique.values())


def predict(candidate: Candidate, model, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict(data[features])
    if candidate.mode == "delta":
        raw = data[CURRENT_RATE].to_numpy() + raw
    return np.maximum(raw, 0)


def metric_values(actual, prediction) -> dict:
    actual = np.asarray(actual, float); prediction = np.asarray(prediction, float)
    error = np.abs(actual - prediction)
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** .5),
        "R2": float(r2_score(actual, prediction)),
        "WMAPE_pct": float(error.sum() / actual.sum() * 100),
        "참고정확도_pct": float(100 - error.sum() / actual.sum() * 100),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, features, audit = build_panel()
    labeled = panel[panel[TARGET].notna()].copy()
    future = panel[panel[TARGET].isna()].copy()
    if len(labeled) != 200 or len(future) != 25:
        raise ValueError(f"행 수 오류: 학습 {len(labeled)}, 미래 {len(future)}")

    pool = candidates(len(features))
    fold_rows, prediction_rows = [], []
    for candidate in pool:
        for year in CV_YEARS:
            train = labeled[labeled["예측연도"] < year]
            valid = labeled[labeled["예측연도"] == year]
            y_train = train[TARGET] - train[CURRENT_RATE] if candidate.mode == "delta" else train[TARGET]
            model = clone(candidate.estimator).fit(train[features], y_train)
            prediction = predict(candidate, model, valid, features)
            fold_rows.append({"후보명": candidate.name, "예측방식": candidate.mode, "검증연도": year, **metric_values(valid[TARGET], prediction)})
            part = valid[["연도", "예측연도", "자치구", CURRENT_RATE, TARGET]].copy()
            part["후보명"] = candidate.name; part["예측값"] = prediction
            prediction_rows.append(part)
    folds = pd.DataFrame(fold_rows)
    comparison = folds.groupby(["후보명", "예측방식"], as_index=False).agg(
        평균_MAE=("MAE", "mean"), 평균_RMSE=("RMSE", "mean"), 평균_R2=("R2", "mean"), 평균_WMAPE_pct=("WMAPE_pct", "mean")
    ).sort_values(["평균_MAE", "평균_RMSE"]).reset_index(drop=True)
    selected = next(c for c in pool if c.name == comparison.iloc[0]["후보명"])

    train = labeled[labeled["예측연도"] <= 2024]
    holdout = labeled[labeled["예측연도"] == 2025]
    y_train = train[TARGET] - train[CURRENT_RATE] if selected.mode == "delta" else train[TARGET]
    holdout_model = clone(selected.estimator).fit(train[features], y_train)
    holdout_prediction = predict(selected, holdout_model, holdout, features)
    holdout_metrics = metric_values(holdout[TARGET], holdout_prediction)
    baseline_prediction = holdout[CURRENT_RATE].to_numpy()
    baseline_metrics = metric_values(holdout[TARGET], baseline_prediction)

    final_y = labeled[TARGET] - labeled[CURRENT_RATE] if selected.mode == "delta" else labeled[TARGET]
    final_model = clone(selected.estimator).fit(labeled[features], final_y)
    future_prediction = predict(selected, final_model, future, features)

    selector = final_model.named_steps.get("selector")
    selected_features = (
        [feature for feature, keep in zip(features, selector.get_support()) if keep]
        if selector is not None else features.copy()
    )

    holdout_output = holdout[["연도", "예측연도", "자치구", CURRENT_RATE, TARGET]].copy()
    holdout_output["모델예측"] = holdout_prediction
    holdout_output["현재율유지예측"] = baseline_prediction
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측어린이교통사고피해자율_어린이1만명당"] = future_prediction

    importance = permutation_importance(holdout_model, holdout[features], holdout[TARGET] if selected.mode == "direct" else holdout[TARGET] - holdout[CURRENT_RATE],
                                        scoring="neg_mean_absolute_error", n_repeats=50, random_state=42, n_jobs=1)
    importance_output = pd.DataFrame({"변수": features, "순열중요도_MAE증가": importance.importances_mean, "표준편차": importance.importances_std}).sort_values("순열중요도_MAE증가", ascending=False)

    cv_all = pd.concat(prediction_rows, ignore_index=True)
    rolling_selected = cv_all[cv_all["후보명"] == selected.name].copy()
    holdout_for_rolling = holdout_output[["연도", "예측연도", "자치구", CURRENT_RATE, TARGET, "모델예측"]].rename(columns={"모델예측": "예측값"})
    holdout_for_rolling["후보명"] = selected.name
    rolling_selected = pd.concat([rolling_selected, holdout_for_rolling], ignore_index=True)
    rolling_model_metrics = metric_values(rolling_selected[TARGET], rolling_selected["예측값"])
    rolling_baseline_metrics = metric_values(rolling_selected[TARGET], rolling_selected[CURRENT_RATE])
    rolling_summary = pd.DataFrame([
        {"구분": "선정 모델", "모델": selected.name, **rolling_model_metrics},
        {"구분": "현재 어린이사고율 유지 기준", "모델": "Persistence", **rolling_baseline_metrics},
    ])
    yearly_rows = []
    for year, part in rolling_selected.groupby("예측연도"):
        yearly_rows.append({"예측연도": year, "구분": "선정 모델", **metric_values(part[TARGET], part["예측값"])})
        yearly_rows.append({"예측연도": year, "구분": "현재율 유지 기준", **metric_values(part[TARGET], part[CURRENT_RATE])})
    yearly_summary = pd.DataFrame(yearly_rows)

    payload = {"model": final_model, "model_name": selected.name, "prediction_mode": selected.mode,
               "features": features, "selected_features": selected_features,
               "current_rate_feature": CURRENT_RATE, "target": TARGET,
               "model_version": MODEL_VERSION,
               "target_definition": "(다음 연도 어린이 교통사고 사망자 수 + 부상자 수) / 다음 연도 0~12세 인구 × 10,000",
               "manual_weights": None, "horizon_years": 1}
    joblib.dump(payload, OUT / f"교통사고_{MODEL_VERSION}차_예측모델.joblib")

    summary = pd.DataFrame([
        {"구분": "선정 모델", "모델": selected.name, **holdout_metrics},
        {"구분": "현재 어린이사고율 유지 기준", "모델": "Persistence", **baseline_metrics},
    ])
    with pd.ExcelWriter(OUT / "모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="최종성능", index=False)
        rolling_summary.to_excel(writer, sheet_name="2021_2025기준비교", index=False)
        yearly_summary.to_excel(writer, sheet_name="연도별기준비교", index=False)
        comparison.to_excel(writer, sheet_name="후보모델비교", index=False)
        folds.to_excel(writer, sheet_name="롤링교차검증", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_excel(writer, sheet_name="교차검증예측", index=False)
        holdout_output.to_excel(writer, sheet_name="2025홀드아웃", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        importance_output.to_excel(writer, sheet_name="변수중요도", index=False)
        pd.DataFrame({"선택변수": selected_features}).to_excel(writer, sheet_name="최종선택변수", index=False)
        audit.to_excel(writer, sheet_name="입력변수감사", index=False)
        panel.to_excel(writer, sheet_name="모델데이터", index=False)

    metadata = {"모델명": f"교통사고 {MODEL_VERSION}차 어린이 피해자율 1년 뒤 모델",
                "차수": MODEL_VERSION,
                "Y정의": "(다음 연도 어린이 교통사고 사망자 수 + 부상자 수) / 다음 연도 0~12세 인구 × 10,000",
                "선정후보": selected.name,
                "예측방식": selected.mode, "전체후보변수수": len(features), "전체후보변수": features,
                "최종선택변수수": len(selected_features), "최종선택변수": selected_features,
                "2025_홀드아웃": holdout_metrics, "2025_기준모델": baseline_metrics,
                "2021_2025_롤링": rolling_model_metrics, "2021_2025_기준모델": rolling_baseline_metrics,
                "교차검증최고": comparison.iloc[0].to_dict(), "학습행수": len(labeled), "2026예측행수": len(forecast),
                "학교데이터": str(SCHOOL_PATH.relative_to(ROOT)),
                "data_sha256": hashlib.sha256((TRAFFIC / "자치구별_교통사고_통합_2017_2025_최종.csv").read_bytes()).hexdigest(),
                "school_data_sha256": hashlib.sha256(SCHOOL_PATH.read_bytes()).hexdigest()}
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "README.md").write_text(
        f"# 교통사고 {MODEL_VERSION}차 모델\n\n- Y: {TARGET}\n"
        f"- Y 정의: (다음 연도 어린이 교통사고 사망자 수 + 부상자 수) / 다음 연도 0~12세 인구 × 10,000\n"
        f"- 선정 모델: {selected.name}\n"
        f"- 전체 후보 변수: {len(features)}개 / 최종 선택 변수: {len(selected_features)}개\n- 2025 홀드아웃 MAE: {holdout_metrics['MAE']:.4f}\n"
        f"- 2025 참고 정확도: {holdout_metrics['참고정확도_pct']:.2f}%\n- 2025 R²: {holdout_metrics['R2']:.4f}\n"
        f"- 2025 현재율 유지 기준 MAE: {baseline_metrics['MAE']:.4f}\n"
        f"- 2021~2025 롤링 MAE: {rolling_model_metrics['MAE']:.4f} / 기준 MAE: {rolling_baseline_metrics['MAE']:.4f}\n",
        encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
