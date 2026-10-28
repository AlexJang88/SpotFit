"""서울 자치구별 보육·교육 인프라 만족도 모델과 feature 조합 이력을 만든다.

두 집단을 별도 학습한다.
* preschool: 미취학 자녀 가구의 보육시설 만족도
* school_age: 전체 가구의 교육환경 만족도(학령기 만족도의 proxy)

2021~2023년을 개발 자료로 사용하고 2024년은 마지막까지 보지 않는 시간 홀드아웃이다.
모든 전진선택 후보 조합은 MLflow에 기록하고, CV MAE가 갱신될 때마다 모델 artifact를 남긴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training.train_child_population_growth import CORE, load_population, merge_features  # noqa: E402

SURVEY_DIR = ROOT / "data/raw/아동/만족도 조사"
GROUP_CONFIG = {
    "preschool": {
        "label": "미취학아동(0~5세)",
        "target": "childcare_facility_satisfaction",
        "target_definition": "미취학 자녀가 실제 이용한 보육시설 만족도(1~5점)",
    },
    "school_age": {
        "label": "학령기(초·중·고, 6~17세)",
        "target": "education_environment_satisfaction",
        "target_definition": "서울서베이 생활환경 중 교육환경 만족도(1~5점, 학령기 proxy)",
    },
}


def read_survey(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"지원하지 않는 CSV 인코딩: {path}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def build_satisfaction_panel(code_to_gu: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    files = {int(path.name[:4]): path for path in SURVEY_DIR.rglob("*.csv") if path.name[:4].isdigit()}
    for year in range(2021, 2025):
        frame = read_survey(files[year])
        gu_col = "구코드"
        weight_col = "표준화가중치" if "표준화가중치" in frame else "원가중치"
        frame["gu_code"] = frame[gu_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        frame["gu"] = frame["gu_code"].map(code_to_gu)
        frame["weight"] = numeric(frame[weight_col])

        education_col = next(c for c in frame if "생활환경 만족도_교육환경" in c)
        frame["education_environment_satisfaction"] = numeric(frame[education_col]).where(lambda s: s.between(1, 5))

        facility_cols = [c for c in frame if "보육시설 만족도" in c]
        facility = frame[facility_cols].apply(numeric).where(lambda x: x.ge(1) & x.le(5))
        # 자녀가 여러 명이어도 한 가구가 과도하게 가중되지 않도록 먼저 가구 평균을 낸다.
        frame["childcare_facility_satisfaction"] = facility.mean(axis=1)

        for gu, part in frame.dropna(subset=["gu"]).groupby("gu"):
            for group, target in (("preschool", "childcare_facility_satisfaction"),
                                  ("school_age", "education_environment_satisfaction")):
                valid = part[target].notna() & part["weight"].notna() & part["weight"].gt(0)
                rows.append({
                    "year": year, "gu": gu, "group": group,
                    "target": weighted_mean(part[target], part["weight"]),
                    "survey_n": int(valid.sum()), "survey_weight_sum": float(part.loc[valid, "weight"].sum()),
                })
    result = pd.DataFrame(rows)
    if result.duplicated(["year", "gu", "group"]).any():
        raise ValueError("만족도 자치구-연도-집단 키가 중복되었습니다.")
    return result


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    return frame[column].sum(min_count=1) if column in frame else np.nan


def build_district_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    population, population_audit = load_population(CORE)
    panel, feature_audit = merge_features(population)
    panel = panel[panel["year"].between(2021, 2024)].copy()

    sum_columns = [
        "total_population", "children_infant", "children_elementary", "children_middle", "children_high",
        "observed_apartment_households_by_area", "총_가구_수", "아파트_가구_수", "교육_지출_총금액",
        "지하철_역_수", "버스_정거장_수", "daycare_count_reconstructed",
        "children_park_count_reconstructed", "hospital_count_reconstructed", "cctv_location_count_reconstructed",
        "emergency_bell_count_reconstructed", "school_count_유치원", "student_count_유치원",
        "teacher_count_유치원", "class_count_유치원", "school_count_초등학교", "student_count_초등학교",
        "teacher_count_초등학교", "class_count_초등학교", "school_count_중학교", "student_count_중학교",
        "teacher_count_중학교", "class_count_중학교", "school_count_고등학교", "student_count_고등학교",
        "teacher_count_고등학교", "class_count_고등학교",
    ]
    district_rows = []
    for (year, gu), part in panel.groupby(["year", "gu"]):
        row = {"year": year, "gu": gu}
        row.update({c: safe_sum(part, c) for c in sum_columns})
        # 자치구 자료가 행정동마다 반복된 열은 합계가 아니라 중앙값으로 집계한다.
        for c in ["academy_survivor_count", "exam_academy_survivor_count", "language_academy_survivor_count",
                  "four_year_college_progression_rate", "special_autonomous_high_school_count",
                  "special_autonomous_supply_per_100_middle_graduates", "아파트_평균_면적", "아파트_평균_시가"]:
            row[c] = pd.to_numeric(part[c], errors="coerce").median() if c in part else np.nan
        district_rows.append(row)
    result = pd.DataFrame(district_rows)
    result["school_age_population"] = result[["children_elementary", "children_middle", "children_high"]].sum(axis=1)

    def per1000(numerator: str, denominator: str, output: str) -> None:
        result[output] = result[numerator].div(result[denominator].replace(0, np.nan)) * 1000

    per1000("daycare_count_reconstructed", "children_infant", "daycare_per_1000_preschool")
    per1000("school_count_유치원", "children_infant", "kindergarten_per_1000_preschool")
    per1000("children_park_count_reconstructed", "children_infant", "children_parks_per_1000_preschool")
    per1000("hospital_count_reconstructed", "total_population", "hospitals_per_1000_residents")
    per1000("observed_apartment_households_by_area", "total_population", "apartment_households_per_1000")
    per1000("지하철_역_수", "total_population", "subway_stations_per_1000")
    per1000("버스_정거장_수", "total_population", "bus_stops_per_1000")
    result["education_spend_per_household"] = result["교육_지출_총금액"].div(result["총_가구_수"].replace(0, np.nan))
    result["kindergarten_students_per_class"] = result["student_count_유치원"].div(result["class_count_유치원"].replace(0, np.nan))
    for level, prefix in (("초등학교", "elementary"), ("중학교", "middle"), ("고등학교", "high")):
        per1000(f"school_count_{level}", "school_age_population", f"{prefix}_schools_per_1000_school_age")
        per1000(f"teacher_count_{level}", "school_age_population", f"{prefix}_teachers_per_1000_school_age")
        result[f"{prefix}_students_per_class"] = result[f"student_count_{level}"].div(
            result[f"class_count_{level}"].replace(0, np.nan))
    audit = pd.concat([population_audit, feature_audit], ignore_index=True, sort=False)
    return result, audit


def candidate_features(group: str) -> list[str]:
    common = [
        "apartment_households_per_1000", "아파트_평균_면적", "아파트_평균_시가",
        "education_spend_per_household", "subway_stations_per_1000", "bus_stops_per_1000",
        "hospitals_per_1000_residents",
    ]
    if group == "preschool":
        return [
            "daycare_per_1000_preschool", "kindergarten_per_1000_preschool",
            "kindergarten_students_per_class", "children_parks_per_1000_preschool", *common,
        ]
    return [
        "elementary_schools_per_1000_school_age", "middle_schools_per_1000_school_age",
        "high_schools_per_1000_school_age", "elementary_students_per_class", "middle_students_per_class",
        "high_students_per_class", "elementary_teachers_per_1000_school_age",
        "middle_teachers_per_1000_school_age", "high_teachers_per_1000_school_age",
        "academy_survivor_count", "exam_academy_survivor_count", "language_academy_survivor_count",
        "four_year_college_progression_rate", "special_autonomous_high_school_count", *common,
    ]


def estimator() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def candidate_estimators() -> dict[str, Pipeline]:
    """작은 패널에서 선형성과 비선형성을 동일 분할로 비교한다."""
    return {
        "dummy_mean": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "ridge_alpha_1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "ridge_alpha_10": estimator(),
        "elastic_net": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=.01, l1_ratio=.2, max_iter=10000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                              max_features=.8, random_state=42, n_jobs=-1)),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5,
                                            max_features=.8, random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=8,
                                                      min_samples_leaf=10, l2_regularization=2.0,
                                                      random_state=42)),
        ]),
    }


def metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual)
    return {
        "mae": float(mean_absolute_error(actual_array, predicted)),
        "rmse": float(mean_squared_error(actual_array, predicted) ** .5),
        "r2": float(r2_score(actual_array, predicted)),
        "within_0_5_accuracy": float((np.abs(actual_array - predicted) <= .5).mean()),
        "rounded_accuracy": float((np.rint(actual_array) == np.rint(predicted)).mean()),
    }


def cv_metrics(data: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    splitter = GroupKFold(n_splits=5)
    predictions = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(data, groups=data["gu"]), start=1):
        train, valid = data.iloc[train_index], data.iloc[valid_index]
        model = estimator().fit(train[features], train["target"])
        pred = model.predict(valid[features])
        part = valid[["year", "gu", "target"]].copy()
        part["prediction"] = pred
        part["fold"] = fold
        predictions.append(part)
    prediction = pd.concat(predictions, ignore_index=True)
    return metrics(prediction["target"], prediction["prediction"].to_numpy()), prediction


def evaluate_estimator(data: pd.DataFrame, features: list[str], model: Pipeline) -> dict[str, float]:
    predictions = pd.Series(index=data.index, dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train_index, valid_index in splitter.split(data, groups=data["gu"]):
        train, valid = data.iloc[train_index], data.iloc[valid_index]
        fitted = clone(model).fit(train[features], train["target"])
        predictions.loc[valid.index] = fitted.predict(valid[features])
    return metrics(data["target"], predictions.loc[data.index].to_numpy())


def bh_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    adjusted = (valid * len(valid) / np.arange(1, len(valid) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    output.loc[valid.index] = adjusted
    return output


def md_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=np.number):
        printable[column] = printable[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    printable = printable.fillna("").astype(str)
    return "\n".join([
        "| " + " | ".join(printable.columns) + " |",
        "| " + " | ".join(["---"] * len(printable.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in printable.to_numpy()],
    ])


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env.dev", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/child_satisfaction_scores")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="child-satisfaction-feature-combinations")
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--min-improvement", type=float, default=.001)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output / stamp
    models_dir = output / "models"
    models_dir.mkdir(parents=True)

    district, audit = build_district_features()
    code_to_gu = (load_population(CORE)[0].assign(
        gu_code=lambda x: x["dong_code"].str[:5].astype(int).astype(str).str[-3:].str.zfill(3)
    ).drop_duplicates("gu_code").set_index("gu_code")["gu"].to_dict())
    satisfaction = build_satisfaction_panel(code_to_gu)
    panel = satisfaction.merge(district, on=["year", "gu"], how="left", validate="many_to_one")
    panel.to_csv(output / "district_satisfaction_training_panel.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "feature_source_audit.csv", index=False, encoding="utf-8-sig")

    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'artifacts/mlflow_child_satisfaction.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    attempts, checkpoints, algorithm_comparisons = [], [], []
    correlations, holdout_predictions, selections = [], [], {}

    with mlflow.start_run(run_name=f"child_satisfaction_{stamp}") as parent:
        mlflow.log_params({
            "groups": "preschool,school_age", "development_years": "2021-2023",
            "holdout_year": 2024, "selection_metric": "5-fold district-grouped CV MAE",
            "feature_selection_model": "Ridge(alpha=10)", "max_features": args.max_features,
            "algorithm_selection": "lowest 5-fold district-grouped CV MAE",
        })
        mlflow.set_tags({
            "spatial_unit": "Seoul district", "causal_claim": "none",
            "db_write_method": "MLflow Tracking API; metadata is persisted by MLflow backend store",
        })

        for group, config in GROUP_CONFIG.items():
            work = panel[panel["group"].eq(group)].replace([np.inf, -np.inf], np.nan).dropna(subset=["target"]).copy()
            development, holdout = work[work["year"].le(2023)].copy(), work[work["year"].eq(2024)].copy()
            candidates = [c for c in candidate_features(group)
                          if c in development and development[c].notna().mean() >= .7
                          and development[c].nunique(dropna=True) > 1]
            if len(development) < 50 or len(holdout) < 20 or not candidates:
                raise ValueError(f"{group}: 학습/홀드아웃/feature 자료가 부족합니다.")

            for feature in candidates:
                pair = development[[feature, "target"]].dropna()
                rho, p = stats.spearmanr(pair[feature], pair["target"])
                correlations.append({"group": group, "feature": feature, "n": len(pair),
                                     "spearman_rho": rho, "p_nominal": p})

            dummy = Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("model", DummyRegressor(strategy="mean"))])
            dummy_predictions = pd.Series(index=development.index, dtype=float)
            splitter = GroupKFold(n_splits=5)
            for train_index, valid_index in splitter.split(development, groups=development["gu"]):
                train, valid = development.iloc[train_index], development.iloc[valid_index]
                dummy.fit(train[[candidates[0]]], train["target"])
                dummy_predictions.loc[valid.index] = dummy.predict(valid[[candidates[0]]])
            baseline_mae = mean_absolute_error(development["target"], dummy_predictions.loc[development.index])

            selected: list[str] = []
            best_mae = float("inf")
            checkpoint_number = 0
            group_feature_stages: list[dict] = []
            rejected_example: dict | None = None
            while len(selected) < min(args.max_features, len(candidates)):
                round_results = []
                for added in [c for c in candidates if c not in selected]:
                    features = [*selected, added]
                    score, cv_prediction = cv_metrics(development, features)
                    row = {"group": group, "round": len(features), "candidate_added": added,
                           "features": json.dumps(features, ensure_ascii=False), "feature_count": len(features),
                           **{f"cv_{k}": v for k, v in score.items()}}
                    attempts.append(row)
                    round_results.append((score["mae"], added, features, score, cv_prediction))
                    with mlflow.start_run(run_name=f"{group}_r{len(features)}_{added}", nested=True):
                        mlflow.log_params({
                            "group": group, "round": len(features), "candidate_added": added,
                            "feature_count": len(features), "features": json.dumps(features, ensure_ascii=False),
                            "target": config["target"], "algorithm": "ridge_alpha_10",
                        })
                        mlflow.log_metrics({f"cv_{k}": v for k, v in score.items()})
                        mlflow.log_dict({"features": features}, "features.json")
                candidate_mae, added, features, score, cv_prediction = min(round_results, key=lambda x: x[0])
                if selected and best_mae - candidate_mae < args.min_improvement:
                    # 보고서에는 단순히 임계값 미달인 조합보다, 직전 최적보다 실제 MAE가
                    # 나빠진 조합 중 하락 폭이 가장 큰 한 건을 반례로 남긴다.
                    degraded = [x for x in round_results if x[0] > best_mae]
                    chosen_rejection = max(degraded or round_results, key=lambda x: x[0])
                    rejected_example = {
                        "status": "rejected_degradation", "features": chosen_rejection[2],
                        "ridge_cv_mae": chosen_rejection[0], "previous_cv_mae": best_mae,
                    }
                    break
                selected, best_mae = features, candidate_mae
                checkpoint_number += 1
                model = estimator().fit(development[selected], development["target"])
                model_path = models_dir / f"{group}_checkpoint_{checkpoint_number}.joblib"
                joblib.dump({"model": model, "features": selected, "target": config["target"],
                             "target_definition": config["target_definition"]}, model_path)
                checkpoint = {"group": group, "checkpoint": checkpoint_number,
                              "features": json.dumps(selected, ensure_ascii=False),
                              "relative_mae_improvement_vs_mean": 1 - best_mae / baseline_mae,
                              **{f"cv_{k}": v for k, v in score.items()}}
                checkpoints.append(checkpoint)
                group_feature_stages.append({"status": f"checkpoint_{checkpoint_number}",
                                             "features": selected.copy(), "ridge_cv_mae": best_mae})
                with mlflow.start_run(run_name=f"{group}_record_{checkpoint_number}", nested=True):
                    mlflow.set_tag("record_breaker", "true")
                    mlflow.log_params({"group": group, "checkpoint": checkpoint_number,
                                       "features": json.dumps(selected, ensure_ascii=False),
                                       "feature_count": len(selected), "target": config["target"]})
                    mlflow.log_metrics({k: v for k, v in checkpoint.items() if isinstance(v, (int, float))})
                    mlflow.log_dict({"features": selected, "target_definition": config["target_definition"]}, "features.json")
                    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

            if rejected_example is not None:
                group_feature_stages.append(rejected_example)

            # 개선 체크포인트 조합과 성능 하락 반례 조합 각각에 동일 학습기법을 비교한다.
            # 2024 홀드아웃 성능은 기록만 하고 최종 기법 선택에는 사용하지 않는다.
            fitted_candidates: dict[str, Pipeline] = {}
            for stage in group_feature_stages:
                stage_features = stage["features"]
                for algorithm, candidate_model in candidate_estimators().items():
                    cv_score = evaluate_estimator(development, stage_features, candidate_model)
                    fitted = clone(candidate_model).fit(development[stage_features], development["target"])
                    algorithm_holdout_prediction = fitted.predict(holdout[stage_features])
                    algorithm_holdout_score = metrics(holdout["target"], algorithm_holdout_prediction)
                    comparison = {
                        "group": group, "feature_stage": stage["status"], "algorithm": algorithm,
                        "features": json.dumps(stage_features, ensure_ascii=False),
                        "feature_count": len(stage_features),
                        "previous_ridge_cv_mae": stage.get("previous_cv_mae", np.nan),
                        **{f"cv_{k}": v for k, v in cv_score.items()},
                        **{f"holdout_{k}": v for k, v in algorithm_holdout_score.items()},
                    }
                    algorithm_comparisons.append(comparison)
                    if stage_features == selected:
                        fitted_candidates[algorithm] = fitted
                    with mlflow.start_run(run_name=f"{group}_{stage['status']}_{algorithm}", nested=True):
                        mlflow.log_params({"group": group, "feature_stage": stage["status"],
                                           "algorithm": algorithm, "target": config["target"],
                                           "features": json.dumps(stage_features, ensure_ascii=False),
                                           "feature_count": len(stage_features)})
                        mlflow.log_metrics({k: v for k, v in comparison.items() if isinstance(v, (int, float))
                                            and np.isfinite(v)})
                        mlflow.log_dict({"features": stage_features,
                                         "target_definition": config["target_definition"]}, "features.json")

            final_stage = f"checkpoint_{checkpoint_number}"
            group_comparisons = [x for x in algorithm_comparisons
                                 if x["group"] == group and x["feature_stage"] == final_stage]
            eligible = [x for x in group_comparisons if x["algorithm"] != "dummy_mean"]
            winning_comparison = min(eligible, key=lambda x: x["cv_mae"])
            winning_algorithm = str(winning_comparison["algorithm"])
            winner = fitted_candidates[winning_algorithm]
            holdout_prediction = winner.predict(holdout[selected])
            holdout_score = metrics(holdout["target"], holdout_prediction)
            pred = holdout[["year", "gu", "target", "survey_n"]].copy()
            pred["group"] = group
            pred["prediction"] = holdout_prediction
            pred["absolute_error"] = np.abs(pred["target"] - pred["prediction"])
            holdout_predictions.append(pred)

            # 운영용 모델은 홀드아웃 평가가 끝난 뒤 2024년까지 전부 다시 학습한다.
            final_model = clone(candidate_estimators()[winning_algorithm]).fit(work[selected], work["target"])
            final_path = models_dir / f"{group}_final_all_2021_2024.joblib"
            joblib.dump({"model": final_model, "features": selected, "target": config["target"],
                         "target_definition": config["target_definition"], "algorithm": winning_algorithm,
                         "holdout_metrics": holdout_score}, final_path)
            selections[group] = {
                **config, "features": selected, "algorithm": winning_algorithm,
                "algorithm_selection_reason": "2021~2023 자치구 GroupKFold 5분할 CV MAE가 가장 낮은 비기준 모델",
                "development_rows": len(development), "holdout_rows": len(holdout),
                "baseline_cv_mae": baseline_mae, "feature_selection_cv_mae": best_mae,
                "best_cv_mae": winning_comparison["cv_mae"],
                "relative_cv_mae_improvement": 1 - winning_comparison["cv_mae"] / baseline_mae,
                "holdout": holdout_score,
            }
            with mlflow.start_run(run_name=f"{group}_final", nested=True):
                mlflow.set_tags({"final_model": "true", "validation_year": "2024"})
                mlflow.log_params({"group": group, "target": config["target"], "feature_count": len(selected),
                                   "features": json.dumps(selected, ensure_ascii=False),
                                   "algorithm": winning_algorithm,
                                   "selection_reason": "lowest_grouped_cv_mae_on_2021_2023"})
                mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_score.items()})
                mlflow.log_dict({"features": selected, "target_definition": config["target_definition"]}, "features.json")
                mlflow.sklearn.log_model(final_model, name="model", serialization_format="cloudpickle")

        correlation_df = pd.DataFrame(correlations)
        correlation_df["bh_q"] = correlation_df.groupby("group")["p_nominal"].transform(bh_adjust)
        attempt_df = pd.DataFrame(attempts)
        checkpoint_df = pd.DataFrame(checkpoints)
        algorithm_df = pd.DataFrame(algorithm_comparisons)
        prediction_df = pd.concat(holdout_predictions, ignore_index=True)
        summary_rows = []
        for (group, stage), stage_data in algorithm_df.groupby(["group", "feature_stage"], sort=False):
            candidates_only = stage_data[stage_data["algorithm"].ne("dummy_mean")]
            best = candidates_only.loc[candidates_only["cv_mae"].idxmin()]
            dummy_row = stage_data[stage_data["algorithm"].eq("dummy_mean")].iloc[0]
            previous_mae = best["previous_ridge_cv_mae"]
            summary_rows.append({
                "group": group, "feature_stage": stage, "features": best["features"],
                "selected_algorithm": best["algorithm"], "cv_mae": best["cv_mae"],
                "cv_r2": best["cv_r2"], "cv_within_0_5_accuracy": best["cv_within_0_5_accuracy"],
                "improvement_vs_dummy_pct": (1 - best["cv_mae"] / dummy_row["cv_mae"]) * 100,
                "mae_change_vs_previous": np.nan,
                "holdout_mae_reference_only": best["holdout_mae"],
                "selection_reason": "동일 자치구 GroupKFold에서 CV MAE가 가장 낮음",
            })
        combination_summary_df = pd.DataFrame(summary_rows)
        # 각 조합에서 선택된 최적 알고리즘끼리 순서대로 비교한다.
        combination_summary_df["mae_change_vs_previous"] = combination_summary_df.groupby(
            "group", sort=False
        )["cv_mae"].diff()
        correlation_df.to_csv(output / "feature_correlations.csv", index=False, encoding="utf-8-sig")
        attempt_df.to_csv(output / "all_feature_combination_attempts.csv", index=False, encoding="utf-8-sig")
        checkpoint_df.to_csv(output / "record_breaking_models.csv", index=False, encoding="utf-8-sig")
        algorithm_df.to_csv(output / "algorithm_comparison.csv", index=False, encoding="utf-8-sig")
        combination_summary_df.to_csv(output / "feature_combination_algorithm_summary.csv", index=False,
                                      encoding="utf-8-sig")
        prediction_df.to_csv(output / "holdout_2024_predictions.csv", index=False, encoding="utf-8-sig")
        (output / "selected_models.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

        sections = []
        for group, selected in selections.items():
            sections.append(f"""### {selected['label']}

- Y: `{selected['target']}` — {selected['target_definition']}
- 최종 feature: {', '.join(f'`{x}`' for x in selected['features'])}
- 최종 학습기법: `{selected['algorithm']}`
- 선택 이유: {selected['algorithm_selection_reason']}
- 개발 CV MAE: {selected['best_cv_mae']:.4f} (평균 예측 대비 {selected['relative_cv_mae_improvement']:.1%} 개선)
- 2024 홀드아웃: MAE {selected['holdout']['mae']:.4f}, RMSE {selected['holdout']['rmse']:.4f}, R² {selected['holdout']['r2']:.4f}, ±0.5점 적중률 {selected['holdout']['within_0_5_accuracy']:.1%}
""")
        report = f"""# 자치구별 육아·교육 인프라 만족도 모델

- 생성 시각: {datetime.now().astimezone().isoformat()}
- 공간 단위: 서울 25개 자치구
- feature/모델 선택: 2021~2023년, 자치구 GroupKFold 5분할
- 최종 일반화 검증: 2024년 25개 자치구(조합 선택에 사용하지 않음)
- feature 전진선택 알고리즘: 표준화 Ridge(alpha=10)
- 최종 학습기법 후보: 평균 기준모델, Ridge 2종, ElasticNet, Random Forest, Extra Trees, Histogram Gradient Boosting
- 최종 학습기법 선택 기준: 2021~2023 자치구 GroupKFold CV MAE 최솟값

## 최종 선택 결과

{''.join(sections)}

## 정확도의 의미

만족도는 1~5점 연속값이므로 분류 정확도가 아니다. 주 지표는 MAE이고, 이해를 돕기 위해 실제 자치구 만족도의 ±0.5점 안에 들어온 비율을 `within_0_5_accuracy`로 함께 기록했다. R²가 0 이하면 2024년 평균 예측보다 못하다는 뜻이다.

## 학습기법 비교와 최종 선택 근거

2024년 성능을 보고 모델을 고르는 누수를 막기 위해 최종 학습기법은 개발기간 CV MAE만으로 선택했다. `dummy_mean`은 모델이 단순 평균보다 실제로 나은지 확인하는 기준선이며 최종 후보에서는 제외했다.

{md_table(algorithm_df.sort_values(['group', 'cv_mae']))}

## Feature 조합별 최적 기법과 성능 변화 요약

`checkpoint`는 feature 추가 후 CV MAE가 개선된 조합이다. `rejected_degradation`은 변수를 더 추가했지만 직전 조합보다 성능이 하락한 예시다. `mae_change_vs_previous`가 양수면 오차가 증가한 것이다.

{md_table(combination_summary_df)}

## 변수 조합 탐색

전진선택의 매 단계에서 남은 feature를 하나씩 추가한 모든 후보를 평가했다. `all_feature_combination_attempts.csv`에는 모든 시도, `record_breaking_models.csv`에는 개발 CV MAE가 실제로 개선된 조합만 있다. 개선 조합은 모두 joblib과 MLflow model artifact로 저장했다. 모든 MLflow run의 `features` 파라미터와 `features.json`에 정확한 입력 변수가 기록된다.

{md_table(checkpoint_df)}

## 연관도 분석

아래 값은 개발기간의 단순 Spearman 연관도이며 인과효과가 아니다. `bh_q`는 집단별 다중검정 보정값이다.

{md_table(correlation_df.sort_values(['group', 'spearman_rho'], key=lambda x: x.abs() if x.name == 'spearman_rho' else x, ascending=False))}

## 중요한 한계

- 미취학 Y는 실제 시설 이용 가구의 보육시설 만족도다. 2022년부터 생긴 ‘전반적 돌봄·양육환경 만족도’와 문항이 달라 합치지 않았다.
- 학령기 Y는 초·중·고 자녀 가구만의 응답이 아니라 서울서베이 전체 가구의 교육환경 만족도이므로 학령기 만족도의 proxy다.
- 확보된 인프라의 공통 연도 때문에 2010~2020 만족도는 이번 학습에 넣지 못했다. 과거 만족도만 늘리고 당시 인프라를 현재 값으로 채우면 데이터 누수가 된다.
- 3개 개발연도와 25개 구로 조합을 탐색하므로 과적합 가능성이 크다. 2024 홀드아웃 결과를 최종 판단 기준으로 봐야 한다.
- 상관과 예측은 인과관계를 입증하지 않는다. 시설 배치의 역인과, 주택·소득·선호의 누락변수가 남는다.

## DB 및 MLflow 저장

직접 SQL을 실행하지 않았다. MLflow Tracking API가 실험·run·parameter·metric·tag를 백엔드 DB에 기록하고 모델/CSV/보고서는 artifact 저장소에 기록한다. 이는 프로젝트의 ‘MLflow DB 직접 수정 금지’ 규칙을 따른다.
"""
        (output / "report.md").write_text(report, encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="evidence")
        for path in models_dir.iterdir():
            mlflow.log_artifact(str(path), artifact_path="models")
        mlflow.log_dict(selections, "selected_models.json")
        print(json.dumps({"output": str(output), "parent_run_id": parent.info.run_id,
                          "tracking_uri": tracking_uri, "selected": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
