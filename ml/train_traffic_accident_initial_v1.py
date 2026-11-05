from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "안전" / "교통사고"
OUT = ROOT / "models" / "안전" / "교통사고" / "1~12차" / "1차"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

FEATURES = [
    "인구만명당 교통사고 사망자수",
    "인구만명당 의료보장 사업장수",
    "운전자안전벨트 착용률",
    "음주운전 경험률",
]
TARGET = "다음연도 교통사고 건수"


def load_data() -> pd.DataFrame:
    accident_path = next(INTERIM.glob("자치구별_교통사고_2차.xlsx"))
    accidents = pd.read_excel(accident_path, sheet_name=0)
    drivers = pd.read_csv(INTERIM / "자치구별_운전자안전_2차.csv")
    workplaces = pd.read_csv(INTERIM / "자치구별_인구만명당_의료보장사업장수_2017_2024.csv")
    population = pd.read_csv(INTERIM / "자치구별_주민등록인구_2017_2025.csv")

    keys = ["연도", "자치구"]
    for frame in (accidents, drivers, workplaces, population):
        frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
        if frame.duplicated(keys).any():
            raise ValueError("연도·자치구 중복이 있습니다.")

    base = accidents[accidents["연도"].between(2017, 2025)][
        keys + ["사고건수", "사망자수", "부상자수", "어린이사고_사고건수"]
    ].copy()
    base = base.merge(population[keys + ["주민등록인구_계"]], on=keys, validate="one_to_one")
    base = base.merge(
        workplaces[keys + ["인구만명당_의료보장사업장수"]], on=keys, how="left", validate="one_to_one"
    )
    base = base.merge(
        drivers[keys + ["운전자안전벨트착용률", "음주운전율"]], on=keys, how="left", validate="one_to_one"
    )
    base["인구만명당 교통사고 사망자수"] = base["사망자수"] / base["주민등록인구_계"] * 10_000
    base = base.rename(
        columns={
            "인구만명당_의료보장사업장수": "인구만명당 의료보장 사업장수",
            "운전자안전벨트착용률": "운전자안전벨트 착용률",
            "음주운전율": "음주운전 경험률",
        }
    )
    base = base.sort_values(["자치구", "연도"]).reset_index(drop=True)
    base[TARGET] = base.groupby("자치구")["사고건수"].shift(-1)
    base["예측대상연도"] = base["연도"] + 1
    return base


def add_safety_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    weights = {
        "인구만명당 교통사고 사망자수": 62.5,
        "인구만명당 의료보장 사업장수": 12.5,
        "운전자안전벨트 착용률": 8.99 / (8.99 + 8.70) * 25.0,
        "음주운전 경험률": 8.70 / (8.99 + 8.70) * 25.0,
    }
    higher_is_safer = {
        "인구만명당 교통사고 사망자수": False,
        "인구만명당 의료보장 사업장수": False,
        "운전자안전벨트 착용률": True,
        "음주운전 경험률": False,
    }
    contribution_cols = []
    for feature, weight in weights.items():
        normalized = out.groupby("연도")[feature].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else pd.Series(0.5, index=s.index)
        )
        safe = normalized if higher_is_safer[feature] else 1 - normalized
        col = f"점수기여_{feature}"
        out[col] = safe * weight
        contribution_cols.append(col)
    out["안전점수_100점"] = out[contribution_cols].sum(axis=1)
    out["연도내_안전백분위"] = out.groupby("연도")["안전점수_100점"].rank(pct=True, ascending=False)
    out["안전등급"] = pd.cut(
        out["연도내_안전백분위"], bins=[0, 0.10, 0.35, 0.65, 0.90, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True
    ).astype("Int64")
    return out


def candidates() -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=800, min_samples_leaf=4, max_features=0.8,
                                             random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=800, min_samples_leaf=4, max_features=1.0,
                                           random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2,
                                                 min_samples_leaf=5, loss="huber", random_state=RANDOM_STATE)),
        ]),
    }


def metric_row(actual: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0)
    abs_error = np.abs(actual - prediction)
    wmape = abs_error.sum() / actual.sum() * 100
    smape = np.mean(2 * abs_error / np.maximum(np.abs(actual) + np.abs(prediction), 1e-9)) * 100
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
        "WMAPE(%)": float(wmape),
        "오차기반정확도(%)": float(max(0, 100 - wmape)),
        "sMAPE(%)": float(smape),
    }


def main() -> None:
    frame = add_safety_score(load_data())
    labeled = frame[
        frame["연도"].between(2017, 2024) & frame[TARGET].notna() & frame[FEATURES].notna().all(axis=1)
    ].copy()
    if len(labeled) != 200:
        raise ValueError(f"예상 학습·검증 행 200개와 다릅니다: {len(labeled)}")

    final_train = labeled[labeled["예측대상연도"] <= 2024].copy()
    final_test = labeled[labeled["예측대상연도"] == 2025].copy()
    if len(final_train) != 175 or len(final_test) != 25:
        raise ValueError("최종 학습 175행·검증 25행 분리가 깨졌습니다.")

    fold_rows = []
    for name, model in candidates().items():
        for target_year in range(2021, 2025):
            train = final_train[final_train["예측대상연도"] < target_year]
            valid = final_train[final_train["예측대상연도"] == target_year]
            model.fit(train[FEATURES], train[TARGET])
            pred = model.predict(valid[FEATURES])
            fold_rows.append({"모델": name, "검증대상연도": target_year, "학습행": len(train), **metric_row(valid[TARGET], pred)})
    folds = pd.DataFrame(fold_rows)
    comparison = folds.groupby("모델", as_index=False).agg(
        내부검증_MAE=("MAE", "mean"), 내부검증_RMSE=("RMSE", "mean"), 내부검증_R2=("R2", "mean"),
        내부검증_WMAPE=("WMAPE(%)", "mean"),
    ).sort_values(["내부검증_MAE", "내부검증_RMSE"]).reset_index(drop=True)
    selected_name = comparison.iloc[0]["모델"]
    selected = candidates()[selected_name]
    selected.fit(final_train[FEATURES], final_train[TARGET])
    final_pred = np.maximum(selected.predict(final_test[FEATURES]), 0)
    final_metrics = metric_row(final_test[TARGET], final_pred)

    validation = final_test[["연도", "예측대상연도", "자치구", TARGET, "안전점수_100점", "안전등급"]].copy()
    validation["예측 교통사고 건수"] = np.round(final_pred, 1)
    validation["예측오차"] = validation["예측 교통사고 건수"] - validation[TARGET]
    validation["절대오차"] = validation["예측오차"].abs()
    validation = validation.sort_values("절대오차").reset_index(drop=True)

    importance = permutation_importance(
        selected, final_test[FEATURES], final_test[TARGET], scoring="neg_mean_absolute_error",
        n_repeats=50, random_state=RANDOM_STATE, n_jobs=1,
    )
    importance_df = pd.DataFrame({
        "변수": FEATURES,
        "순열중요도_MAE증가": importance.importances_mean,
        "표준편차": importance.importances_std,
    }).sort_values("순열중요도_MAE증가", ascending=False)

    model_payload = {
        "pipeline": selected,
        "features": FEATURES,
        "target": TARGET,
        "input_years": [2017, 2023],
        "target_years": [2018, 2024],
        "validation_input_year": 2024,
        "validation_target_year": 2025,
        "selected_model": selected_name,
    }
    joblib.dump(model_payload, OUT / "교통사고_1차_초기모델.joblib")
    folds.to_csv(OUT / "내부_순차검증_결과.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "후보모델_비교.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT / "2025년_최종검증_예측결과.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUT / "변수중요도.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "모델구분": "안전-교통사고-1차 초기모델",
        "예측목표": TARGET,
        "예측간격": "1년 후",
        "선정모델": selected_name,
        "입력변수": FEATURES,
        "제외": {
            "경감지표": "세모 자료이므로 사용자 요청에 따라 전체 제외",
            "인구만명당 재난약자수": "미확보",
            "자동차등록대수당 교통법규위반건수": "교통법규위반건수 미확보",
        },
        "학습": "2017~2023년 X → 2018~2024년 y, 175행",
        "최종검증": "2024년 X → 2025년 y, 25행",
        "최종검증지표": final_metrics,
        "안전점수가중치": {
            "위해": 62.5,
            "취약": 12.5,
            "의식": 25.0,
            "경감": 0.0,
            "설명": "경감 20% 제외 후 나머지 80%를 100%로 재정규화; 의식 내 확보 항목은 원 가중치 8.99:8.70 비율로 배분",
        },
        "주의": "R2×100은 분산설명률이지 적중 정확도가 아님. 오차기반정확도는 100-WMAPE로 별도 제시.",
    }
    (OUT / "모델정보.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 안전-교통사고-1차 초기모델

## 결과

- 선정 모델: {selected_name}
- 최종 검증: 2024년 기준표 X로 2025년 교통사고 건수 예측(서울 25개 자치구)
- MAE: {final_metrics['MAE']:.2f}건
- RMSE: {final_metrics['RMSE']:.2f}건
- R²: {final_metrics['R2']:.4f} (분산설명률 {final_metrics['R2'] * 100:.2f}%)
- WMAPE: {final_metrics['WMAPE(%)']:.2f}%
- 오차 기반 정확도(100-WMAPE): {final_metrics['오차기반정확도(%)']:.2f}%

## 설계

- 학습: 2017~2023년 X → 2018~2024년 y, 175행
- 최종 검증: 2024년 X → 2025년 y, 25행
- 후보 모델 선택: 2021~2024년 확대창 순차검증 평균 MAE 최소
- 사용 변수: {', '.join(FEATURES)}
- 경감지표는 세모 자료이므로 완전히 제외했으며 대리지표를 사용하지 않음
- 자치구명, 연도, 현재 사고건수, 교통량 등 안전기준표 밖 변수는 X로 사용하지 않음

## 안전점수

- 위해 62.5%, 취약 12.5%, 의식 25.0%로 재정규화
- 각 연도 25개 자치구 안에서 지표 방향을 반영해 0~100점으로 계산
- 안전등급은 연도 내 상대 백분위에 따라 1~5등급으로 표시

## 해석 주의

- R²는 정확도 자체가 아니라 자치구별 사고 건수 차이를 설명한 비율임
- 오차 기반 정확도는 전체 실제 건수 대비 절대오차 비율을 100에서 뺀 참고값임
- 안전기준표 항목만 사용한 초기 기준 모델이므로 최종 확장 모델보다 성능이 낮을 수 있음
"""
    (OUT / "모델결과보고서.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\n후보모델 비교\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
