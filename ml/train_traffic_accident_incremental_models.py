"""1차 교통사고 모델을 기준으로 2~5차 독립 확장 모델을 학습한다."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train_traffic_accident_initial_v1 import (
    FEATURES as INITIAL_FEATURES,
    INTERIM,
    RANDOM_STATE,
    TARGET,
    add_safety_score,
    candidates,
    load_data,
    metric_row,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"
TRACKING_ROOT = ROOT / "artifacts" / "mlflow_traffic_accident"
EXPERIMENT = "traffic-accident-incremental-models"
FOLD_TARGET_YEARS = range(2021, 2025)

CURRENT = "현재연도 교통사고 건수"
HISTORY_FEATURES = [
    "사고건수 1년전",
    "최근 3년 사고건수 평균",
    "사고건수 전년 증감",
    "최근 3년 사고건수 추세",
    "최근 3년 사고건수 변동성",
]
VEHICLES = "자동차 등록 대수"
LIVING_FEATURES = ["생활인구 연평균", "생활인구 연중시간최대", "생활인구 월변동성"]


def add_incremental_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values(["자치구", "연도"]).reset_index(drop=True)
    grouped = out.groupby("자치구", sort=False)["사고건수"]
    out[CURRENT] = out["사고건수"]
    out[HISTORY_FEATURES[0]] = grouped.shift(1)
    out[HISTORY_FEATURES[1]] = grouped.transform(lambda s: s.rolling(3, min_periods=3).mean())
    out[HISTORY_FEATURES[2]] = out["사고건수"] - out[HISTORY_FEATURES[0]]
    out[HISTORY_FEATURES[3]] = grouped.transform(
        lambda s: s.rolling(3, min_periods=3).apply(lambda v: np.polyfit(np.arange(3), v, 1)[0], raw=True)
    )
    out[HISTORY_FEATURES[4]] = grouped.transform(lambda s: s.rolling(3, min_periods=3).std(ddof=0))

    driver = pd.read_csv(INTERIM / "자치구별_운전자안전_2차.csv")
    driver = driver[["연도", "자치구", "자동차등록대수"]].rename(columns={"자동차등록대수": VEHICLES})
    living = pd.read_csv(INTERIM / "자치구별_인구활동량_1차.csv")
    living = living[["연도", "자치구", "생활인구_연평균", "생활인구_연중시간최대", "생활인구_월변동성"]].rename(
        columns={"생활인구_연평균": LIVING_FEATURES[0], "생활인구_연중시간최대": LIVING_FEATURES[1],
                 "생활인구_월변동성": LIVING_FEATURES[2]}
    )
    for extra in (driver, living):
        if extra.duplicated(["연도", "자치구"]).any():
            raise ValueError("추가 자료에 연도·자치구 중복이 있습니다.")
        out = out.merge(extra, on=["연도", "자치구"], how="left", validate="one_to_one")
    return out


def feature_subsets(features: list[str]) -> list[list[str]]:
    return [list(c) for size in range(1, len(features) + 1) for c in combinations(features, size)]


def cv_result(frame: pd.DataFrame, features: list[str], model_name: str) -> tuple[dict, list[dict]]:
    rows = []
    for target_year in FOLD_TARGET_YEARS:
        train = frame[frame["예측대상연도"] < target_year]
        valid = frame[frame["예측대상연도"] == target_year]
        model = candidates()[model_name]
        model.fit(train[features], train[TARGET])
        score = metric_row(valid[TARGET], model.predict(valid[features]))
        rows.append({"검증대상연도": target_year, "학습행": len(train), "검증행": len(valid), **score})
    folds = pd.DataFrame(rows)
    summary = {
        "cv_mae": float(folds.MAE.mean()), "cv_rmse": float(folds.RMSE.mean()),
        "cv_r2": float(folds.R2.mean()), "cv_wmape": float(folds["WMAPE(%)"].mean()),
    }
    return summary, rows


def log_candidate(version: int, kind: str, model_name: str, features: list[str], summary: dict,
                  fold_rows: list[dict], record: bool, record_number: int, frame: pd.DataFrame) -> str:
    with mlflow.start_run(run_name=f"v{version}_{kind}_{model_name}_{len(features)}f", nested=True) as run:
        mlflow.log_params({"version": version, "candidate_kind": kind, "algorithm": model_name,
                           "feature_count": len(features), "features": json.dumps(features, ensure_ascii=False)})
        mlflow.log_metrics(summary)
        mlflow.log_dict({"features": features}, "features.json")
        mlflow.log_dict(fold_rows, "fold_results.json")
        if record:
            fitted = candidates()[model_name]
            fitted.fit(frame[features], frame[TARGET])
            mlflow.set_tags({"record_breaker": "true", "record_number": str(record_number),
                             "model_database_status": "stored_by_mlflow_tracking_backend"})
            mlflow.sklearn.log_model(fitted, name="model", serialization_format="cloudpickle",
                                     input_example=frame[features].head(3))
        return run.info.run_id


def train_version(version: int, title: str, added_pool: list[str], frame: pd.DataFrame) -> dict:
    out = MODEL_ROOT / f"{version}차"
    out.mkdir(parents=True, exist_ok=True)
    required = INITIAL_FEATURES + added_pool
    labeled = frame[
        frame["연도"].between(2017, 2024) & frame[TARGET].notna() & frame[required].notna().all(axis=1)
    ].copy()
    final_train = labeled[labeled["예측대상연도"] <= 2024].copy()
    final_test = labeled[labeled["예측대상연도"] == 2025].copy()
    if len(final_test) != 25:
        raise ValueError(f"{version}차 2025년 검증 표본이 25개 자치구가 아닙니다: {len(final_test)}")

    comparison_rows, all_fold_rows = [], []
    baseline_best = float("inf")
    for model_name in candidates():
        summary, folds = cv_result(final_train, INITIAL_FEATURES, model_name)
        run_id = log_candidate(version, "initial_baseline", model_name, INITIAL_FEATURES, summary, folds,
                               False, 0, final_train)
        comparison_rows.append({"후보구분": "1차_동일표본기준", "알고리즘": model_name,
                                "추가변수": "없음", "전체변수": " | ".join(INITIAL_FEATURES),
                                "run_id": run_id, **summary})
        all_fold_rows.extend({"후보구분": "1차_동일표본기준", "알고리즘": model_name,
                              "추가변수": "없음", **row} for row in folds)
        baseline_best = min(baseline_best, summary["cv_mae"])

    incumbent = baseline_best
    record_number = 0
    expanded = []
    for added in feature_subsets(added_pool):
        features = INITIAL_FEATURES + added
        for model_name in candidates():
            summary, folds = cv_result(final_train, features, model_name)
            record = summary["cv_mae"] < incumbent
            if record:
                incumbent = summary["cv_mae"]
                record_number += 1
            run_id = log_candidate(version, "expanded", model_name, features, summary, folds,
                                   record, record_number, final_train)
            row = {"후보구분": "확장", "알고리즘": model_name, "추가변수": " | ".join(added),
                   "전체변수": " | ".join(features), "run_id": run_id, **summary}
            comparison_rows.append(row); expanded.append(row)
            all_fold_rows.extend({"후보구분": "확장", "알고리즘": model_name,
                                  "추가변수": " | ".join(added), **fold} for fold in folds)

    comparison = pd.DataFrame(comparison_rows).sort_values(["cv_mae", "cv_rmse"]).reset_index(drop=True)
    expanded_table = pd.DataFrame(expanded).sort_values(["cv_mae", "cv_rmse"]).reset_index(drop=True)
    winner = expanded_table.iloc[0]
    winner_features = winner["전체변수"].split(" | ")
    selected = candidates()[winner["알고리즘"]]
    selected.fit(final_train[winner_features], final_train[TARGET])
    prediction = np.maximum(selected.predict(final_test[winner_features]), 0)
    final_metrics = metric_row(final_test[TARGET], prediction)
    improved = float(winner.cv_mae) < baseline_best

    payload = {"pipeline": selected, "features": winner_features, "target": TARGET,
               "version": version, "selected_model": winner["알고리즘"], "status": "improved" if improved else "no_improvement"}
    joblib.dump(payload, out / f"교통사고_{version}차_모델.joblib")
    validation = final_test[["연도", "예측대상연도", "자치구", TARGET, "안전점수_100점", "안전등급"]].copy()
    validation["예측 교통사고 건수"] = np.round(prediction, 1)
    validation["예측오차"] = validation["예측 교통사고 건수"] - validation[TARGET]
    validation["절대오차"] = validation["예측오차"].abs()
    validation.to_csv(out / "2025년_최종검증_예측결과.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out / "후보모델_비교.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_fold_rows).to_csv(out / "내부_순차검증_결과.csv", index=False, encoding="utf-8-sig")
    importance = permutation_importance(selected, final_test[winner_features], final_test[TARGET],
                                        scoring="neg_mean_absolute_error", n_repeats=50,
                                        random_state=RANDOM_STATE, n_jobs=1)
    pd.DataFrame({"변수": winner_features, "순열중요도_MAE증가": importance.importances_mean,
                  "표준편차": importance.importances_std}).sort_values("순열중요도_MAE증가", ascending=False).to_csv(
        out / "변수중요도.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "모델구분": f"안전-교통사고-{version}차", "실험정의": title, "예측목표": TARGET,
        "선정모델": winner["알고리즘"], "입력변수": winner_features, "추가후보변수": added_pool,
        "학습행": len(final_train), "최종검증행": len(final_test), "동일표본_1차최고_CV_MAE": baseline_best,
        "선정모델_CV_MAE": float(winner.cv_mae), "CV_MAE_개선": baseline_best - float(winner.cv_mae),
        "1차대비개선여부": improved, "최종검증지표": final_metrics,
        "안전점수": "추가 예측변수에는 가중치를 주지 않고 1차 기준표 산식을 유지",
        "mlflow_run_id": winner.run_id,
    }
    (out / "모델정보.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 교통사고 {version}차 모델\n\n- 실험: {title}\n- 선정 모델: {winner['알고리즘']}\n- 선정 추가변수: {winner['추가변수']}\n- 동일 표본 1차 최고 CV MAE: {baseline_best:.2f}건\n- {version}차 최고 CV MAE: {winner.cv_mae:.2f}건\n- 개선 폭: {baseline_best-float(winner.cv_mae):.2f}건\n- 개선 판정: {'개선' if improved else '개선 없음'}\n- 2025 최종 MAE: {final_metrics['MAE']:.2f}건\n- 2025 최종 RMSE: {final_metrics['RMSE']:.2f}건\n- 2025 최종 R²: {final_metrics['R2']:.4f}\n- 2025 최종 WMAPE: {final_metrics['WMAPE(%)']:.2f}%\n\n추가 변수는 사고 건수 예측용이며 안전지수 가중치에는 포함하지 않았습니다.\n"""
    (out / "모델결과보고서.md").write_text(report, encoding="utf-8")
    with mlflow.start_run(run_id=winner.run_id, nested=True):
        mlflow.set_tags({"selection_status": "improved" if improved else "no_improvement",
                         "selected_for_version": "true"})
        if mlflow.MlflowClient().get_run(winner.run_id).data.tags.get("record_breaker") != "true":
            mlflow.sklearn.log_model(selected, name="model", serialization_format="cloudpickle",
                                     input_example=final_train[winner_features].head(3))
        mlflow.log_dict(metadata, "model_info.json")
    return metadata


def main() -> None:
    frame = add_safety_score(add_incremental_features(load_data()))
    TRACKING_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (TRACKING_ROOT / "tracking.db").as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(TRACKING_ROOT / "runs").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    definitions = [
        (2, "1차 + 현재연도 교통사고 건수", [CURRENT]),
        (3, "1차 + 사고 이력 파생변수 조합", HISTORY_FEATURES),
        (4, "1차 + 자동차 등록 대수", [VEHICLES]),
        (5, "1차 + 생활인구 변수 조합", LIVING_FEATURES),
    ]
    with mlflow.start_run(run_name=f"traffic_incremental_v2_v5_{stamp}"):
        source_files = [INTERIM / "자치구별_교통사고_2차.xlsx", INTERIM / "자치구별_운전자안전_2차.csv",
                        INTERIM / "자치구별_인구활동량_1차.csv"]
        mlflow.log_params({"validation": "expanding_window_2021_2024", "final_test": "2025",
                           "seed": RANDOM_STATE, "base_features": json.dumps(INITIAL_FEATURES, ensure_ascii=False),
                           **{f"sha256_{i}": hashlib.sha256(p.read_bytes()).hexdigest() for i, p in enumerate(source_files)}})
        summaries = [train_version(version, title, pool, frame) for version, title, pool in definitions]
        mlflow.log_dict(summaries, "versions_summary.json")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
