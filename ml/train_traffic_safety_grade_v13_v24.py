"""기존 1~12차 변수 구성을 안전기준표 기반 다음연도 안전등급 모델 13~24차로 재구성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from train_traffic_accident_incremental_models import add_incremental_features
from train_traffic_accident_initial_v1 import RANDOM_STATE, candidates, load_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"
TRACKING_ROOT = ROOT / "artifacts" / "mlflow_traffic_safety_grade"
EXPERIMENT = "traffic-safety-grade-v13-v24"
TARGET_SCORE = "다음연도 안전점수"
TARGET_GRADE = "다음연도 안전등급"
FOLD_TARGET_YEARS = range(2021, 2025)

CATEGORY = {
    "인구만명당 교통사고 사망자수": "위해",
    "인구만명당 의료보장 사업장수": "취약",
    "운전자안전벨트 착용률": "의식",
    "음주운전 경험률": "의식",
    "현재연도 교통사고 건수": "위해",
    "사고건수 1년전": "위해",
    "최근 3년 사고건수 평균": "위해",
    "사고건수 전년 증감": "위해",
    "최근 3년 사고건수 추세": "위해",
    "최근 3년 사고건수 변동성": "위해",
    "자동차 등록 대수": "취약",
    "생활인구 연평균": "취약",
    "생활인구 연중시간최대": "취약",
    "생활인구 월변동성": "취약",
    "사고당 부상자수": "위해",
    "어린이 사고 비율": "위해",
    "인구만명당 연령기반 재난약자수": "취약",
    "차량통행속도 연평균": "취약",
}

HIGHER_IS_SAFER = {
    "운전자안전벨트 착용률": True,
}

OFFICIAL_AWARENESS_UNITS = {
    "운전자안전벨트 착용률": 8.99,
    "음주운전 경험률": 8.70,
}

ORIGINAL_CATEGORY_WEIGHTS = {"위해": 50.0, "취약": 10.0, "경감": 20.0, "의식": 20.0}
NO_MITIGATION_WEIGHTS = {"위해": 62.5, "취약": 12.5, "경감": 0.0, "의식": 25.0}


def load_version_features() -> dict[int, list[str]]:
    result = {}
    for old_version in range(1, 13):
        metadata = json.loads(
            (MODEL_ROOT / f"{old_version}차" / "모델정보.json").read_text(encoding="utf-8")
        )
        result[old_version + 12] = metadata["입력변수"]
    result[25] = [
        *result[19],
        "인구만명당 연령기반 재난약자수",
    ]
    result[26] = [
        *result[19],
        "차량통행속도 연평균",
    ]
    return result


def add_age_vulnerability_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """연령자료로 공식 재난약자 지표의 대리변수를 만든다.

    실제 재학생 수가 없으므로 3~12세 주민등록인구와 65세 이상 인구를 사용한다.
    """
    path = ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_연령별인구현황_2017_2025.csv"
    age = pd.read_csv(path)
    keys = ["연도", "자치구"]
    if age.duplicated(keys).any():
        raise ValueError("연령자료에 연도·자치구 중복이 있습니다.")
    vulnerable_columns = [f"{value}세" for value in range(3, 13)] + [
        f"{value}세" for value in range(65, 100)
    ] + ["100세 이상"]
    age = age.copy()
    age["연령기반 재난약자수"] = age[vulnerable_columns].sum(axis=1)
    age["인구만명당 연령기반 재난약자수"] = (
        age["연령기반 재난약자수"] / age["총인구수"] * 10_000
    )
    return frame.merge(
        age[keys + ["인구만명당 연령기반 재난약자수"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def add_traffic_speed(frame: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_차량통행속도_2017_2025.csv"
    speed = pd.read_csv(path).rename(
        columns={"차량통행속도_연평균_kmh": "차량통행속도 연평균"}
    )
    keys = ["연도", "자치구"]
    if speed.duplicated(keys).any():
        raise ValueError("차량통행속도 자료에 연도·자치구 중복이 있습니다.")
    return frame.merge(
        speed[keys + ["차량통행속도 연평균"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def grade_by_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    percentile = frame.groupby("연도")[score_col].rank(method="first", ascending=False, pct=True)
    return pd.cut(
        percentile,
        bins=[0, 0.10, 0.35, 0.65, 0.90, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)


def internal_weights(features: list[str]) -> pd.DataFrame:
    rows = []
    for category, category_weight in NO_MITIGATION_WEIGHTS.items():
        members = [feature for feature in features if CATEGORY[feature] == category]
        if not members:
            continue
        if category == "의식" and all(feature in OFFICIAL_AWARENESS_UNITS for feature in members):
            units = np.array([OFFICIAL_AWARENESS_UNITS[feature] for feature in members], dtype=float)
        else:
            units = np.ones(len(members), dtype=float)
        for feature, weight in zip(members, category_weight * units / units.sum()):
            rows.append({
                "변수": feature,
                "영역": category,
                "영역_원래가중치_pct": ORIGINAL_CATEGORY_WEIGHTS[category],
                "영역_적용가중치_pct": category_weight,
                "변수_최종가중치_pct": float(weight),
                "높을수록안전": bool(HIGHER_IS_SAFER.get(feature, False)),
                "배분근거": "기준표 8.99:8.70" if category == "의식" else "영역 내 균등 배분",
            })
    weights = pd.DataFrame(rows)
    if not np.isclose(weights["변수_최종가중치_pct"].sum(), 100.0):
        raise ValueError("최종 변수 가중치 합계가 100%가 아닙니다.")
    return weights


def add_historical_score(frame: pd.DataFrame, features: list[str], weights: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    contribution_cols = []
    for row in weights.itertuples(index=False):
        normalized = out.groupby("연도")[row.변수].transform(
            lambda values: (
                (values - values.min()) / (values.max() - values.min())
                if values.notna().any() and values.max() > values.min()
                else pd.Series(0.5, index=values.index)
            )
        )
        safe = normalized if row.높을수록안전 else 1.0 - normalized
        col = f"점수기여::{row.변수}"
        out[col] = safe * row.변수_최종가중치_pct
        contribution_cols.append(col)
    out["안전점수"] = out[contribution_cols].sum(axis=1, min_count=len(contribution_cols))
    valid = out["안전점수"].notna()
    out.loc[valid, "안전등급"] = grade_by_rank(out.loc[valid], "안전점수")
    out["안전등급"] = out["안전등급"].astype("Int64")
    out = out.sort_values(["자치구", "연도"]).reset_index(drop=True)
    out[TARGET_SCORE] = out.groupby("자치구")["안전점수"].shift(-1)
    out[TARGET_GRADE] = out.groupby("자치구")["안전등급"].shift(-1).astype("Int64")
    out["예측대상연도"] = out["연도"] + 1
    return out


def assign_predicted_grades(target_years: pd.Series, scores: np.ndarray) -> np.ndarray:
    temp = pd.DataFrame({"연도": np.asarray(target_years, dtype=int), "예측점수": scores})
    return grade_by_rank(temp, "예측점수").to_numpy(dtype=int)


def metrics(actual_score, predicted_score, actual_grade, predicted_grade) -> dict[str, float]:
    actual_score = np.asarray(actual_score, dtype=float)
    predicted_score = np.asarray(predicted_score, dtype=float)
    actual_grade = np.asarray(actual_grade, dtype=int)
    predicted_grade = np.asarray(predicted_grade, dtype=int)
    return {
        "score_mae": float(mean_absolute_error(actual_score, predicted_score)),
        "score_rmse": float(mean_squared_error(actual_score, predicted_score) ** 0.5),
        "score_r2": float(r2_score(actual_score, predicted_score)),
        "grade_accuracy": float(accuracy_score(actual_grade, predicted_grade)),
        "grade_within1_accuracy": float(np.mean(np.abs(actual_grade - predicted_grade) <= 1)),
        "grade_macro_f1": float(f1_score(actual_grade, predicted_grade, average="macro", zero_division=0)),
        "grade_quadratic_kappa": float(cohen_kappa_score(actual_grade, predicted_grade, weights="quadratic")),
    }


def cv_candidate(frame: pd.DataFrame, features: list[str], model_name: str) -> tuple[dict, list[dict]]:
    fold_rows = []
    for target_year in FOLD_TARGET_YEARS:
        train = frame[frame["예측대상연도"] < target_year]
        valid = frame[frame["예측대상연도"] == target_year]
        model = candidates()[model_name]
        model.fit(train[features], train[TARGET_SCORE])
        prediction = np.clip(model.predict(valid[features]), 0, 100)
        predicted_grade = assign_predicted_grades(valid["예측대상연도"], prediction)
        fold_rows.append({
            "검증대상연도": target_year,
            "학습행": len(train),
            "검증행": len(valid),
            **metrics(valid[TARGET_SCORE], prediction, valid[TARGET_GRADE], predicted_grade),
        })
    folds = pd.DataFrame(fold_rows)
    summary = {f"cv_{column}": float(folds[column].mean()) for column in folds.columns if column not in {
        "검증대상연도", "학습행", "검증행"
    }}
    summary["cv_grade_accuracy_std"] = float(folds["grade_accuracy"].std(ddof=0))
    return summary, fold_rows


def log_candidate(version: int, model_name: str, features: list[str], weights: pd.DataFrame,
                  summary: dict, fold_rows: list[dict], save_model: bool, frame: pd.DataFrame) -> str:
    with mlflow.start_run(run_name=f"v{version}_{model_name}", nested=True) as run:
        mlflow.log_params({
            "version": version,
            "algorithm": model_name,
            "feature_count": len(features),
            "features": json.dumps(features, ensure_ascii=False),
            "target": TARGET_GRADE,
            "mitigation_policy": "none_available_proportional_redistribution",
            "category_weights": json.dumps(NO_MITIGATION_WEIGHTS, ensure_ascii=False),
        })
        mlflow.log_metrics(summary)
        mlflow.log_dict({"features": features}, "features.json")
        mlflow.log_dict(fold_rows, "fold_results.json")
        mlflow.log_dict(weights.to_dict(orient="records"), "feature_category_weights.json")
        if save_model:
            fitted = candidates()[model_name]
            fitted.fit(frame[features], frame[TARGET_SCORE])
            mlflow.sklearn.log_model(
                fitted,
                name="model",
                serialization_format="cloudpickle",
                input_example=frame[features].head(3),
            )
        return run.info.run_id


def train_version(version: int, base_frame: pd.DataFrame, features: list[str]) -> dict:
    out_dir = MODEL_ROOT / f"{version}차"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = internal_weights(features)
    scored = add_historical_score(base_frame, features, weights)
    labeled = scored[
        scored["연도"].between(2017, 2024)
        & scored[features].notna().all(axis=1)
        & scored[TARGET_SCORE].notna()
        & scored[TARGET_GRADE].notna()
    ].copy()
    development = labeled[labeled["예측대상연도"] <= 2024].copy()
    final_test = labeled[labeled["예측대상연도"] == 2025].copy()
    if len(final_test) != 25:
        raise ValueError(f"{version}차 최종검증 표본이 25개 자치구가 아닙니다: {len(final_test)}")

    comparisons = []
    fold_records = []
    best_accuracy = -np.inf
    record_number = 0
    for model_name in candidates():
        summary, fold_rows = cv_candidate(development, features, model_name)
        is_record = summary["cv_grade_accuracy"] > best_accuracy
        if is_record:
            best_accuracy = summary["cv_grade_accuracy"]
            record_number += 1
        run_id = log_candidate(version, model_name, features, weights, summary, fold_rows,
                               is_record, development)
        comparisons.append({"알고리즘": model_name, "run_id": run_id, **summary})
        fold_records.extend({"알고리즘": model_name, **row} for row in fold_rows)
        if is_record:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.set_tags({"record_breaker": "true", "record_number": str(record_number)})

    comparison = pd.DataFrame(comparisons).sort_values(
        ["cv_grade_accuracy", "cv_grade_quadratic_kappa", "cv_score_mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    winner = comparison.iloc[0]
    selected = candidates()[winner["알고리즘"]]
    selected.fit(development[features], development[TARGET_SCORE])
    test_score = np.clip(selected.predict(final_test[features]), 0, 100)
    test_grade = assign_predicted_grades(final_test["예측대상연도"], test_score)
    final_metrics = metrics(final_test[TARGET_SCORE], test_score, final_test[TARGET_GRADE], test_grade)

    future_input = scored[scored["연도"] == 2025].copy()
    if len(future_input) != 25 or future_input[features].isna().any().any():
        raise ValueError(f"{version}차 2026년 예측 입력이 완전하지 않습니다.")
    future_score = np.clip(selected.predict(future_input[features]), 0, 100)
    future_grade = assign_predicted_grades(pd.Series(np.repeat(2026, 25)), future_score)

    payload = {
        "pipeline": selected,
        "features": features,
        "feature_categories": {feature: CATEGORY[feature] for feature in features},
        "feature_weights_pct": dict(zip(weights["변수"], weights["변수_최종가중치_pct"])),
        "category_weights_pct": NO_MITIGATION_WEIGHTS,
        "target": TARGET_SCORE,
        "final_output": "안전등급",
        "version": version,
        "selected_model": winner["알고리즘"],
    }
    joblib.dump(payload, out_dir / f"교통사고_안전등급_{version}차_모델.joblib")

    validation = final_test[["연도", "예측대상연도", "자치구", TARGET_SCORE, TARGET_GRADE]].copy()
    validation["예측 안전점수"] = np.round(test_score, 4)
    validation["예측 안전등급"] = test_grade
    validation["등급일치"] = validation[TARGET_GRADE].astype(int) == test_grade
    validation["등급오차"] = test_grade - validation[TARGET_GRADE].astype(int)
    validation.to_csv(out_dir / "2025년_안전등급_최종검증.csv", index=False, encoding="utf-8-sig")

    future = future_input[["연도", "자치구"]].copy()
    future["예측대상연도"] = 2026
    future["예측 안전점수"] = np.round(future_score, 4)
    future["예측 안전등급"] = future_grade
    future.sort_values(["예측 안전등급", "예측 안전점수"], ascending=[True, False]).to_csv(
        out_dir / "2026년_안전등급_예측결과.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(out_dir / "후보모델_비교.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_records).to_csv(out_dir / "내부_순차검증_결과.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(out_dir / "변수영역_가중치.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "모델구분": f"안전-교통사고-{version}차",
        "체계": "안전등급 모델 13~26차",
        "기존대응차수": version - 12,
        "최종출력": "다음연도 안전등급(1~5)",
        "중간출력": "다음연도 안전점수",
        "선정모델": winner["알고리즘"],
        "입력변수": features,
        "변수영역": {feature: CATEGORY[feature] for feature in features},
        "원래영역가중치_pct": ORIGINAL_CATEGORY_WEIGHTS,
        "적용영역가중치_pct": NO_MITIGATION_WEIGHTS,
        "경감지표처리": "완전한 경감지표 없음; 모든 행에 동일하게 위해 62.5·취약 12.5·의식 25로 비례 재배분",
        "의료보장_2025처리": "2024년 최근 공표값을 2025년에 이월",
        "연령기반_재난약자_정의": (
            "3~12세와 65세 이상 주민등록인구의 합계를 총인구수로 나눈 인구만명당 대리지표; "
            "실제 유치원생·초등학생 재학생 수를 사용한 공식 지표는 아님"
            if "인구만명당 연령기반 재난약자수" in features else None
        ),
        "차량통행속도_정의": (
            "서울시 영업용택시 GPS 기반 자치구별 월평균 통행속도의 연평균; 취약영역에 배분"
            if "차량통행속도 연평균" in features else None
        ),
        "학습행": len(development),
        "최종검증행": len(final_test),
        "CV성능": {key: float(winner[key]) for key in winner.index if key.startswith("cv_")},
        "2025최종검증": final_metrics,
        "mlflow_run_id": winner["run_id"],
        "주의": "경감지표 제외 환산 안전등급이며 공식 전체지표 등급이 아님",
    }
    (out_dir / "모델정보.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with mlflow.start_run(run_id=winner["run_id"], nested=True):
        mlflow.set_tags({
            "selected_for_version": "true",
            "selection_status": "exploratory_provisional_grade",
            "model_database_status": "stored_by_mlflow_tracking_backend",
        })
        if mlflow.MlflowClient().get_run(winner["run_id"]).data.tags.get("record_breaker") != "true":
            mlflow.sklearn.log_model(
                selected,
                name="model",
                serialization_format="cloudpickle",
                input_example=development[features].head(3),
            )
        mlflow.log_dict(metadata, "model_info.json")
        mlflow.log_artifact(str(out_dir / "변수영역_가중치.csv"))

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", type=int, choices=range(13, 27), default=list(range(13, 25)))
    args = parser.parse_args()
    frame = add_traffic_speed(add_age_vulnerability_proxy(add_incremental_features(load_data())))
    frame["인구만명당 의료보장 사업장수"] = frame.groupby("자치구")[
        "인구만명당 의료보장 사업장수"
    ].ffill()
    version_features = load_version_features()

    TRACKING_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (TRACKING_ROOT / "tracking.db").as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(TRACKING_ROOT / "runs").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    source_files = [
        ROOT / "docs" / "안전기준표.png",
        ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_교통사고_2차.xlsx",
        ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_운전자안전_2차.csv",
        ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_인구활동량_1차.csv",
        ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_연령별인구현황_2017_2025.csv",
        ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_차량통행속도_2017_2025.csv",
    ]
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    selected_label = "_".join(map(str, args.versions))
    with mlflow.start_run(run_name=f"traffic_safety_grade_v{selected_label}_{stamp}"):
        mlflow.log_params({
            "validation": "expanding_window_2021_2024",
            "final_test": "2025",
            "future_prediction": "2026",
            "seed": RANDOM_STATE,
            "grade_ratio": "10:25:30:25:10",
            "original_category_weights": json.dumps(ORIGINAL_CATEGORY_WEIGHTS, ensure_ascii=False),
            "applied_category_weights": json.dumps(NO_MITIGATION_WEIGHTS, ensure_ascii=False),
            **{f"sha256_{index}": hashlib.sha256(path.read_bytes()).hexdigest()
               for index, path in enumerate(source_files)},
        })
        summaries = [train_version(version, frame, version_features[version]) for version in args.versions]
        mlflow.log_dict(summaries, "versions_summary.json")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
