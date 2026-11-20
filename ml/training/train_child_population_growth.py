"""핵심 아동 인구 원본과 연도별 지역 특징으로 다음 연도 증감률을 분석·학습한다."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "data/raw/아동/아동 데이터/핵심"
GROUPS = {
    "infant": (0, 5, "영유아(0~5세)"),
    "elementary": (6, 11, "초등 학령기(6~11세)"),
    "middle": (12, 14, "중등 학령기(12~14세)"),
    "high": (15, 17, "고등 학령기(15~17세)"),
}
SCHOOL_LEVEL_BY_GROUP = {
    "elementary": "초등학교",
    "middle": "중학교",
    "high": "고등학교",
}
SECONDARY_REQUIRED_FEATURES = {
    "middle": [
        "special_autonomous_high_school_count",
        "special_autonomous_supply_per_100_middle_graduates",
        "academy_survivor_count",
        "exam_academy_survivor_count",
        "language_academy_survivor_count",
    ],
    "high": [
        "four_year_college_progression_rate",
        "academy_survivor_count",
        "exam_academy_survivor_count",
        "language_academy_survivor_count",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_population(core: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, audit = [], []
    for path in sorted(core.glob("*연령별인구현황19.csv")):
        year_match = re.search(r"(20\d{2})", path.name)
        if not year_match:
            continue
        year = int(year_match.group(1))
        frame = pd.read_csv(path, encoding="cp949", dtype=str)
        extracted = frame["행정구역"].str.extract(r"^(.*?)\s*\((\d{10})\)\s*$")
        frame["region_name"] = extracted[0].str.strip()
        frame["dong_code"] = extracted[1]
        # 서울 행정동만 사용한다. 시·구 합계행은 코드 끝 5자리가 00000이다.
        frame = frame[frame["dong_code"].str.startswith("11", na=False)
                      & ~frame["dong_code"].str.endswith("00000", na=False)].copy()
        names = frame["region_name"].str.split()
        frame["gu"] = names.str[-2]
        frame["dong"] = names.str[-1]
        frame["year"] = year
        frame["total_population"] = numeric(frame[f"{year}년_계_총인구수"])
        for group, (start, end, _) in GROUPS.items():
            cols = [f"{year}년_계_{age}세" for age in range(start, end + 1)]
            frame[f"children_{group}"] = frame[cols].apply(numeric).sum(axis=1, min_count=len(cols))
            # t+1년의 해당 학령인구가 되는 t년 코호트. 0세는 아직 태어나지 않아
            # 영유아에는 만들 수 없고 초·중·고에만 누수 없는 선행지표로 사용한다.
            if start > 0:
                cohort_cols = [f"{year}년_계_{age}세" for age in range(start - 1, end)]
                frame[f"entrant_cohort_{group}"] = frame[cohort_cols].apply(numeric).sum(
                    axis=1, min_count=len(cohort_cols)
                )
        keep = ["dong_code", "year", "gu", "dong", "total_population",
                *[f"children_{g}" for g in GROUPS],
                *[f"entrant_cohort_{g}" for g in SCHOOL_LEVEL_BY_GROUP]]
        frames.append(frame[keep])
        audit.append({"source": str(path.relative_to(ROOT)), "year": year, "sha256": sha256(path),
                      "raw_rows": len(pd.read_csv(path, encoding="cp949", usecols=[0])),
                      "seoul_dong_rows": len(frame), "role": "target_population_included"})
    population = pd.concat(frames, ignore_index=True).sort_values(["dong_code", "year"])
    if population.duplicated(["dong_code", "year"]).any():
        raise ValueError("핵심 인구 원본에 행정동·연도 중복이 있습니다.")
    for group in GROUPS:
        current = population[f"children_{group}"]
        future = population.groupby("dong_code")[f"children_{group}"].shift(-1)
        consecutive = population.groupby("dong_code")["year"].shift(-1).eq(population["year"] + 1)
        population[f"future_children_{group}"] = future.where(consecutive)
        population[f"target_log_growth_{group}"] = (
            np.log1p(future) - np.log1p(current)
        ).where(consecutive & current.ge(30))
        population[f"target_growth_pct_{group}"] = (
            (future - current).div(current.where(current.gt(0))) * 100
        ).where(consecutive)
    return population, pd.DataFrame(audit)


def merge_features(population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = population.copy()
    audits = []
    boundaries = json.loads((ROOT / "frontend/public/seoul-dong.geojson").read_text(encoding="utf-8"))
    boundary_names = pd.DataFrame([
        {"admin_code": str(feature["properties"]["adm_cd"]),
         "gu": feature["properties"]["sggnm"],
         "dong": feature["properties"]["adm_nm"].split()[-1]}
        for feature in boundaries["features"]
    ]).drop_duplicates("admin_code")

    housing_path = ROOT / "data/interim/child_housing_observed/observed_apartment_panel_2021_2025.csv"
    housing = pd.read_csv(housing_path, dtype={"dong_code": str})
    housing_features = [c for c in housing if c in {
        "아파트_단지_수", "아파트_평균_면적", "아파트_평균_시가",
        "observed_apartment_households_by_area", "observed_apartment_households_by_price",
    }]
    data = data.merge(housing[["dong_code", "year", *housing_features]], on=["dong_code", "year"], how="left", validate="one_to_one")
    audits.append({"dataset": "서울 상권분석서비스 아파트-행정동", "status": "included_observed",
                   "years": "2021-2025 Q4", "features": ", ".join(housing_features),
                   "reason": "행정동·연도별 실제 관측"})

    # 계절 차이를 섞지 않도록 행정동별 상권분석 자료도 매년 4분기만 사용한다.
    local_zone_specs = {
        "길단위인구": ["총_유동인구_수"],
        "상주인구": ["총_상주인구_수", "총_가구_수", "아파트_가구_수"],
        "소비": ["지출_총금액", "식료품_지출_총금액", "의료비_지출_총금액", "교육_지출_총금액"],
        "직장인구": ["총_직장_인구_수"],
        "집객시설": ["집객시설_수", "관공서_수", "은행_수", "종합병원_수", "일반_병원_수", "약국_수",
                 "유치원_수", "초등학교_수", "중학교_수", "고등학교_수", "슈퍼마켓_수", "지하철_역_수", "버스_정거장_수"],
    }
    for dataset, wanted in local_zone_specs.items():
        path = ROOT / f"data/raw/local_zone/서울시 상권분석서비스({dataset}-행정동).csv"
        frame = pd.read_csv(path, encoding="cp949", dtype=str)
        frame = frame[frame["기준_년분기_코드"].str.match(r"^202[1-5]4$", na=False)].copy()
        frame["year"] = frame["기준_년분기_코드"].str[:4].astype(int)
        frame["dong_code"] = frame["행정동_코드"].str.zfill(8) + "00"
        available = [column for column in wanted if column in frame]
        for column in available:
            frame[column] = numeric(frame[column])
        if frame.duplicated(["dong_code", "year"]).any():
            raise ValueError(f"{dataset}: 행정동·연도 중복")
        data = data.merge(frame[["dong_code", "year", *available]], on=["dong_code", "year"], how="left", validate="one_to_one")
        audits.append({"dataset": f"서울 상권분석서비스 {dataset}-행정동", "status": "included_observed",
                       "years": "2021-2025 Q4", "features": ", ".join(available),
                       "reason": "동일 계절의 행정동·연도별 실제 관측"})

    school_path = ROOT / "data/interim/child_school_observed/observed_school_panel_2021_2026.csv"
    school = pd.read_csv(school_path, dtype={"admin_code": str})
    # 학교/시설 패널의 adm_cd는 통계경계 8자리 코드이고 인구의 주민등록 10자리
    # 코드와 단순 접두·접미 관계가 아니다. 경계의 공식 자치구·행정동명으로 연결한다.
    school = school.drop(columns=["admin_name"], errors="ignore").merge(
        boundary_names, on="admin_code", how="left", validate="many_to_one"
    )
    school_features = [c for c in school if c.startswith(("school_count_", "student_count_", "teacher_count_", "class_count_"))]
    for level in SCHOOL_LEVEL_BY_GROUP.values():
        students, classes = f"student_count_{level}", f"class_count_{level}"
        school[f"students_per_class_{level}"] = school[students].div(school[classes].replace(0, np.nan))
    school_features.extend([f"students_per_class_{level}" for level in SCHOOL_LEVEL_BY_GROUP.values()])
    data = data.merge(school[["gu", "dong", "year", *school_features]], on=["gu", "dong", "year"], how="left", validate="one_to_one")
    # 학교 패널은 학교가 있는 동만 행으로 만들어진다. 원본 연도·서울 경계가 완전하므로
    # 매핑된 전체 격자에서 행이 없는 경우는 미관측이 아니라 학교 0개다.
    data[school_features] = data[school_features].fillna(0)
    audits.append({"dataset": "교육통계 학교별 일람표", "status": "included_observed",
                   "years": "2021-2025 4월", "features": f"{len(school_features)}개 학교 수·학생·교원·학급",
                   "reason": "같은 계절의 연도별 실제 관측"})

    reconstructed_path = ROOT / "data/interim/child_infrastructure_reconstructed/reconstructed_infrastructure_panel_2021_2026.csv"
    reconstructed = pd.read_csv(reconstructed_path, dtype={"admin_code": str})
    reconstructed = reconstructed.drop(columns=["admin_name"], errors="ignore").merge(
        boundary_names, on="admin_code", how="left", validate="many_to_one"
    )
    reconstructed_features = [c for c in reconstructed if c.endswith(("_reconstructed", "_observed_event"))]
    # 현재 진료과목을 과거에 소급한 소아과 컬럼은 누수 위험 때문에 제외한다.
    data = data.merge(reconstructed[["gu", "dong", "year", *reconstructed_features]], on=["gu", "dong", "year"], how="left", validate="one_to_one")
    data[reconstructed_features] = data[reconstructed_features].fillna(0)
    audits.append({"dataset": "보육·병원·공원·CCTV·비상벨", "status": "included_reconstructed",
                   "years": "2021-2025", "features": f"{len(reconstructed_features)}개",
                   "reason": "현재 목록의 인허가·설치·폐업 사건일로 복원; 실제 과거 스냅샷보다 신뢰도 낮음"})

    safety_path = ROOT / "data/interim/child_safety_zone/safety_zone_snapshots.csv"
    safety = pd.read_csv(safety_path)
    safety = safety[safety["month"].eq(12)].copy()
    safety_features = [c for c in safety if c.startswith("safety_zone_")]
    data = data.merge(safety[["year", "gu", "dong", *safety_features]], on=["year", "gu", "dong"], how="left", validate="one_to_one")
    audits.append({"dataset": "어린이보호구역 연말 통계", "status": "included_limited",
                   "years": "2023-2025", "features": f"{len(safety_features)}개",
                   "reason": "실제 연말 관측이나 2021-2022 부재; 결측 대체 후 민감도 주의"})

    secondary_path = ROOT / "data/interim/secondary_education/secondary_education_panel_2021_2025.csv"
    if secondary_path.exists():
        secondary = pd.read_csv(secondary_path)
        secondary_features = [
            feature for feature_list in SECONDARY_REQUIRED_FEATURES.values()
            for feature in feature_list if feature in secondary.columns
        ]
        secondary_features = list(dict.fromkeys(secondary_features))
        data = data.merge(
            secondary[["gu", "year", *secondary_features]], on=["gu", "year"],
            how="left", validate="many_to_one",
        )
        audits.append({"dataset": "중고등 진학성과·학원 패널", "status": "included_mixed_quality",
                       "years": "2021-2025", "features": ", ".join(secondary_features),
                       "reason": "4년제 진학률은 공식 관측; 특목자사고는 공급 대리변수; 학원은 현재 생존시설 복원 하한"})

    audits.extend([
        {"dataset": "2026 단일 스냅샷 시설", "status": "excluded", "years": "주로 2026", "features": "-",
         "reason": "과거 연도에 복제하면 미래정보 누수"},
        {"dataset": "상권 공간단위 자료", "status": "excluded", "years": "다년", "features": "-",
         "reason": "상권 경계는 행정동 목표와 공간 단위가 다르고 중첩 가중 변환이 없음"},
        {"dataset": "점포-행정동", "status": "excluded", "years": "2025-2026", "features": "-",
         "reason": "개발기간 2021-2023 관측이 없어 다음 연도 학습 불가"},
        {"dataset": "성별·아동전체·주민등록 보조 인구", "status": "excluded", "years": "2021-2025", "features": "-",
         "reason": "핵심 1세 단위 인구와 중복 집계 방지"},
        {"dataset": "2026년 자료", "status": "excluded_from_training", "years": "2026", "features": "-",
         "reason": "2027년 목표 인구가 없어 학습 정답 생성 불가"},
    ])
    return data, pd.DataFrame(audits)


def model_candidates() -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))]),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=.8, random_state=42, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, max_features=.8, random_state=42, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=42))]),
    }


def metric_values(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    actual_pct = np.expm1(y.to_numpy()) * 100
    predicted_pct = np.expm1(pred) * 100
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** .5), "r2": float(r2_score(y, pred)),
            "mae_pct_point": float(mean_absolute_error(actual_pct, predicted_pct)),
            "rmse_pct_point": float(mean_squared_error(actual_pct, predicted_pct) ** .5),
            "direction_accuracy": float((np.sign(actual_pct) == np.sign(predicted_pct)).mean())}


def markdown_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=np.number):
        printable[column] = printable[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    printable = printable.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in printable.to_numpy()]
    return "\n".join([header, separator, *rows])


def required_features(group: str) -> list[str]:
    """연령군의 코호트 기준선과 같은 학교급 교육 인프라를 강제로 보존한다."""
    level = SCHOOL_LEVEL_BY_GROUP.get(group)
    if level is None:
        return []
    return [
        f"entrant_cohort_{group}",
        f"school_count_{level}",
        f"student_count_{level}",
        f"teacher_count_{level}",
        f"class_count_{level}",
        f"students_per_class_{level}",
        *SECONDARY_REQUIRED_FEATURES.get(group, []),
    ]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env.dev", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=CORE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/child_population_growth")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="child-population-next-year-growth")
    parser.add_argument("--age-groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS),
                        help="학습할 연령군(기본: 전체)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / stamp
    models_dir = out / "models"
    models_dir.mkdir(parents=True)
    population, source_audit = load_population(args.core_dir)
    panel, feature_audit = merge_features(population)
    panel.to_csv(out / "training_panel.csv", index=False, encoding="utf-8-sig")
    source_audit.to_csv(out / "population_source_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(out / "feature_inclusion_audit.csv", index=False, encoding="utf-8-sig")

    excluded = {"year", "gu", "dong"}
    target_prefixes = ("children_", "future_children_", "target_")
    candidates = [c for c in panel.select_dtypes(include=np.number).columns
                  if c not in excluded and not c.startswith(target_prefixes)]
    correlation_rows, model_rows, prediction_frames, selections = [], [], [], {}

    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'artifacts/mlflow_child_population_growth.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"child_growth_{stamp}") as parent:
        parent_id = parent.info.run_id
        mlflow.log_params({"target_horizon": "one_year", "target": "log_growth", "train_years": "2021,2022,2023",
                           "holdout_year": 2024, "minimum_current_children": 30,
                           "feature_selection": "required_age_school_features_plus_top_8_abs_spearman_train_only",
                           "age_groups": ",".join(args.age_groups)})
        mlflow.set_tags({"data_source": str(args.core_dir), "validation": "leave_one_year_out_then_2024_holdout",
                         "deployment_status": "exploratory_not_production"})
        for group in args.age_groups:
            _, _, label = GROUPS[group]
            target = f"target_log_growth_{group}"
            work = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
            train = work[work.year.le(2023)].copy()
            test = work[work.year.eq(2024)].copy()
            # 현재 아동 수는 중요한 기준선이지만 목표 계산 컬럼과 구분해 명시적으로 추가한다.
            work_feature = f"current_children_{group}"
            train[work_feature] = train[f"children_{group}"]
            test[work_feature] = test[f"children_{group}"]
            usable = [c for c in candidates
                      if train[c].notna().mean() >= .6
                      and train[c].nunique(dropna=True) > 1
                      and train[c].fillna(0).ne(0).mean() >= .05]
            ranking = []
            for feature in usable:
                pair = train[[feature, target]].dropna()
                rho, p = stats.spearmanr(pair[feature], pair[target]) if len(pair) >= 20 else (np.nan, np.nan)
                pearson = stats.pearsonr(pair[feature], pair[target]).statistic if len(pair) >= 20 and pair[feature].nunique() > 1 else np.nan
                ranking.append({"age_group": group, "age_label": label, "feature": feature, "n": len(pair),
                                "spearman": rho, "spearman_p_nominal": p, "pearson": pearson})
            rank = pd.DataFrame(ranking).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
            correlation_rows.extend(rank.to_dict("records"))
            mandatory = [feature for feature in required_features(group) if feature in usable]
            selected = [feature for feature in rank.feature if feature not in mandatory][:8]
            features = [work_feature, "total_population", *mandatory, *selected]
            # 중복 피처 제거
            features = list(dict.fromkeys(features))
            selections[group] = features
            folds = sorted(train.year.unique())
            fitted = {}
            baseline_fold_rmse: np.ndarray | None = None
            for name, model in model_candidates().items():
                fold_scores = []
                for validation_year in folds:
                    fold_train, fold_valid = train[train.year.ne(validation_year)], train[train.year.eq(validation_year)]
                    model.fit(fold_train[features], fold_train[target])
                    fold_scores.append(metric_values(fold_valid[target], model.predict(fold_valid[features])))
                cv_rmse = np.array([x["rmse"] for x in fold_scores])
                if name == "dummy_median":
                    baseline_fold_rmse = cv_rmse.copy()
                if baseline_fold_rmse is None:
                    raise RuntimeError("기준 모델은 후보 목록의 첫 번째여야 합니다.")
                model.fit(train[features], train[target])
                holdout = metric_values(test[target], model.predict(test[features]))
                relative_improvement = 1 - cv_rmse.mean() / baseline_fold_rmse.mean()
                fold_wins = int((cv_rmse < baseline_fold_rmse).sum())
                row = {"age_group": group, "model": name, "feature_count": len(features),
                       "cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std(ddof=1),
                       "baseline_cv_rmse": baseline_fold_rmse.mean(),
                       "relative_improvement": relative_improvement, "fold_wins": fold_wins,
                       **{f"holdout_{k}": v for k, v in holdout.items()}}
                model_rows.append(row)
                fitted[name] = model
                with mlflow.start_run(run_name=f"{group}_{name}", nested=True):
                    mlflow.log_params({"age_group": group, "algorithm": name, "features": json.dumps(features, ensure_ascii=False),
                                       "train_rows": len(train), "holdout_rows": len(test)})
                    mlflow.log_metrics({k: float(v) for k, v in row.items() if k.startswith(("cv_", "holdout_"))})
            group_table = pd.DataFrame([r for r in model_rows if r["age_group"] == group])
            winner = str(group_table.sort_values(["cv_rmse_mean", "cv_rmse_std"]).iloc[0].model)
            winner_row = group_table[group_table.model.eq(winner)].iloc[0]
            eligible = bool(winner_row.relative_improvement >= .05 and winner_row.fold_wins >= int(np.ceil(.8 * len(folds))))
            selections[group] = {"model": winner, "features": features,
                                 "decision": "exploratory_candidate" if eligible else "selection_deferred",
                                 "decision_reason": (f"기준 대비 CV RMSE {winner_row.relative_improvement:.1%} 개선, "
                                                     f"{int(winner_row.fold_wins)}/{len(folds)} 연도 폴드 승리")}
            winner_model = fitted[winner]
            model_path = models_dir / f"child_growth_{group}.joblib"
            joblib.dump({"model": winner_model, "features": features, "target": target}, model_path)
            pred = test[["dong_code", "year", "gu", "dong", target, f"target_growth_pct_{group}"]].copy()
            pred["age_group"] = group
            pred["model"] = winner
            pred["prediction_log_growth"] = winner_model.predict(test[features])
            pred["prediction_growth_pct"] = np.expm1(pred["prediction_log_growth"]) * 100
            prediction_frames.append(pred)

        correlations = pd.DataFrame(correlation_rows)
        comparison = pd.DataFrame(model_rows).sort_values(["age_group", "cv_rmse_mean"])
        predictions = pd.concat(prediction_frames, ignore_index=True)
        correlations.to_csv(out / "correlations.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(out / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
        (out / "selected_models.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

        sns.set_theme(style="whitegrid", font="Malgun Gothic")
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(12, 6))
        sns.barplot(comparison, x="age_group", y="holdout_rmse", hue="model")
        plt.title("2024→2025 시간 외 테스트 RMSE")
        plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=160); plt.close()
        top = correlations.sort_values("spearman", key=lambda s: s.abs(), ascending=False).groupby("age_group").head(8)
        grid = sns.catplot(top, x="spearman", y="feature", col="age_group", kind="bar", col_wrap=2, sharex=True, sharey=False, height=4)
        grid.fig.suptitle("학습기간(2021~2023) 특징과 다음 연도 로그 증감률 Spearman", y=1.02)
        grid.savefig(out / "correlation_top_features.png", dpi=160); plt.close("all")

        winners = comparison[comparison.apply(lambda row: row.model == selections[row.age_group]["model"], axis=1)]
        target_summary = pd.DataFrame([{"age_group": g, "rows": int(panel[f"target_log_growth_{g}"].notna().sum()),
                                        "median_growth_pct": float(panel[f"target_growth_pct_{g}"].median())}
                                       for g in args.age_groups])
        report = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>아동 인구 증감 분석·예측</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;max-width:1200px;margin:35px auto;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px}}.warn{{background:#fff3cd;padding:14px;border-left:5px solid #d39e00}}img{{max-width:100%}}code{{background:#eee;padding:2px 4px}}</style>
<h1>행정동별 다음 연도 아동 인구 증감률 분석 및 예측</h1>
<h2>1. 분석 목적과 정답 정의</h2><p>지정된 <code>data/raw/아동/아동 데이터/핵심</code>의 2021~2025년 1세 단위 주민등록인구만 정답 산출에 사용했습니다. 연령군은 영유아 0~5세, 초등 6~11세, 중등 12~14세, 고등 15~17세이며 실제 재학 상태가 아니라 만 나이 구간입니다.</p>
<p>학습 목표는 <code>log((t+1년 아동수+1)/(t년 아동수+1))</code>입니다. 양수는 증가, 0은 유지, 음수는 감소입니다. 일반 백분율도 해석용으로 보존하되 작은 분모가 손실을 지배하지 않도록 현재 아동 30명 미만은 해당 연령군 학습에서 제외했습니다.</p>
<h2>2. 자료 선별</h2>{feature_audit.to_html(index=False, border=0)}
<div class='warn'><b>중요:</b> reconstructed 피처는 현재 시설목록에 남은 사건일로 복원한 값이며 완전한 과거 스냅샷이 아닙니다. 상관은 인과효과가 아니고, 아동 증감에는 출생·연령 이동·전입·전출이 함께 들어갑니다.</div>
<h2>3. 검증 설계</h2><p>2021~2023년을 개발기간으로 두고 각 연도 하나씩 제외하는 leave-one-year-out CV로 모델을 선택했습니다. 2024년 특징으로 2025년 증감률을 예측하는 표본은 최종 시간 외 테스트로만 사용했습니다. 특징 순위도 개발기간에서만 계산했습니다.</p>
<h2>4. 목표 분포</h2>{target_summary.to_html(index=False, border=0, float_format=lambda x:f'{x:.4f}')}
<h2>5. 연관도 분석</h2><p>Spearman은 단조 관계, Pearson은 선형 관계입니다. 표본 수와 명목 p값을 함께 기록했지만 공간적으로 인접한 행정동의 의존성과 다중 비교가 있으므로 확정적 유의성 판정으로 사용하지 않습니다.</p>{top.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}<img src='correlation_top_features.png'>
<h2>6. 모델 비교와 선정</h2><p>Dummy median, Ridge, Random Forest, Extra Trees, HistGradientBoosting을 같은 피처와 같은 시간 폴드에서 비교했습니다. 연령군별 CV RMSE 평균이 가장 낮은 모델을 고정한 뒤 2024→2025 결과를 한 번 평가했습니다. 기준 모델 대비 CV RMSE 5% 이상 개선하고 개발기간 3개 연도 폴드 모두에서 승리해야 탐색 후보로 유지하며, 그렇지 않으면 <code>selection_deferred</code>입니다.</p>
{comparison.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}<img src='model_comparison.png'>
<h2>7. 최종 선정 모델</h2>{winners.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}
<h2>8. 해석과 사용 제한</h2><ul><li>연도는 4개 전환뿐이므로 행정동 행 수가 많아도 독립적인 시간 변화 정보는 적습니다.</li><li>R²가 0 이하라면 같은 테스트셋의 평균 예측보다 설명력이 낮다는 뜻이며 운영 배포하면 안 됩니다.</li><li>현재 인구는 강한 기준 피처지만 증감 원인을 설명하지 않습니다.</li><li>2026년 말 인구와 추가 과거 시설 스냅샷이 확보되면 다시 학습해야 합니다.</li><li>이 결과는 탐색 모델이며 정책 효과나 시설 설치의 인과효과로 해석하지 않습니다.</li></ul>
</html>"""
        (out / "report.html").write_text(report, encoding="utf-8")
        detail_sections = []
        for group in args.age_groups:
            selected = selections[group]
            feature_rows = []
            mandatory = set(required_features(group))
            for feature in selected["features"]:
                role = (
                    "공식 관측 진학성과" if feature == "four_year_college_progression_rate" else
                    "특목·자사고 지역 공급 대리지표" if feature.startswith("special_autonomous_") else
                    "현재 생존 학원 역산 민감도 지표" if "academy_survivor_count" in feature else
                    "필수 교육·코호트" if feature in mandatory else
                    "현재 연령 인구 기준선" if feature.startswith("current_children_") else
                    "전체 인구 기준선" if feature == "total_population" else "학습기간 연관 상위 보조 feature"
                )
                feature_rows.append({"feature": feature, "selection_role": role})
            age_corr = correlations[
                correlations.age_group.eq(group)
                & correlations.feature.isin(required_features(group))
            ][["feature", "n", "spearman", "spearman_p_nominal", "pearson"]]
            winner_row = winners[winners.age_group.eq(group)][[
                "model", "cv_rmse_mean", "relative_improvement", "fold_wins",
                "holdout_mae_pct_point", "holdout_rmse_pct_point", "holdout_r2",
                "holdout_direction_accuracy",
            ]]
            start, end, label = GROUPS[group]
            detail_sections.append(f"""## {label}

- Target: `target_log_growth_{group} = log((t+1년 {start}~{end}세 인구+1)/(t년 {start}~{end}세 인구+1))`
- 해석용 Target: `target_growth_pct_{group}`, 즉 t→t+1년 {start}~{end}세 인구증가율(%)
- 현재 인구 30명 미만 행정동은 로그 증감률 학습에서 제외

### 사용 feature

{markdown_table(pd.DataFrame(feature_rows))}

### 해당 학교급 인프라와 Target의 학습기간 연관

{markdown_table(age_corr)}

### 선택 모델과 2024→2025 시간 외 성능

{markdown_table(winner_row)}
""")
        detailed_report = f"""# 초·중·고 학령인구와 교육 인프라 분석·예측

- 생성시각: {datetime.now().astimezone().isoformat()}
- 공간 단위: 서울 행정동
- 자료 기간: 2021~2025년
- 개발기간: 2021~2023년 feature로 각각 다음 해 예측
- 최종 검증: 2024년 feature → 2025년 인구증가율
- 분석 대상: {', '.join(GROUPS[g][2] for g in args.age_groups)}

## 분석 방법

1. 주민등록 1세별 인구에서 연령군별 현재 인구와 다음 해 인구증가율을 만들었다.
2. t+1년에 해당 학령기가 되는 t년 진입 코호트와 같은 학교급의 학교·학생·교원·학급·학급당 학생 수를 필수 feature로 보존했다.
3. 나머지 실제 관측·복원 후보 중 개발기간 Spearman 절대값 상위 8개만 추가했다. 2024→2025 정답은 feature 선택에 사용하지 않았다.
4. 연도 하나씩 제외하는 3개 시간 폴드로 Dummy median, Ridge, Random Forest, Extra Trees, HistGradientBoosting을 비교했다.
5. CV RMSE가 가장 낮은 모델을 고정한 뒤 2024→2025를 한 번 평가했다.

## 추가 요청 지표의 확보·사용 상태

| 요청 지표 | 확보 상태 | 모델 사용 방식 |
| --- | --- | --- |
| 4년제 진학률 | 2021~2025 학교별 교육통계의 졸업자·국내대학·국외대학 진학자로 산출 | 고등 모델 필수 feature |
| 학원 수 | 2026년 현재 운영시설 25,479건과 개설일 확보 | 폐원시설이 빠진 생존시설 역산값을 중·고등 민감도 feature로 사용 |
| 특목·자사고 진학률 | 공식 서울시 자치구 통계의 존재는 확인했으나 데이터 허브가 실제 행을 반환하지 않음 | 진학률 대신 자치구 소재 특목·자사고 입학자/중학교 졸업자 공급 대리지표 사용 |
| 서울대 진학률 | 공식 공개 교육통계는 개별 대학명을 제공하지 않음 | 누락값이나 사설 자료로 임의 대체하지 않고 미사용 |

`special_autonomous_supply_per_100_middle_graduates`는 출신 중학교 졸업생의 실제 진학률이 아니다. 같은 자치구에 위치한 특목·자사고 입학 규모를 중학교 졸업자 수로 나눈 지역 공급 대리지표다.
학원 역산값은 2026년까지 살아남은 시설만 포함하므로 폐원시설 누락에 따른 생존편향과 미래정보 참조 가능성이 있다. 주 모델의 확정 근거가 아니라 민감도 실험으로만 해석한다.

{''.join(detail_sections)}
## 공통 해석 제한

- 상관계수와 예측력은 교육 인프라의 인과효과가 아니다.
- 학교가 있는 행정동의 시설은 인접 동 학생도 이용하므로 행정동 경계와 실제 학군이 일치하지 않는다.
- 독립적인 연간 전환이 네 번뿐이라 시간 변화에 대한 근거가 약하다.
- 자치구 진학률·학원 지표를 같은 자치구의 여러 행정동에 반복 부여했으므로 표의 상관 p값은 유효 표본 수를 과대평가한다.
- 학원 수는 현재 생존시설을 개설일로 역산한 민감도 지표이므로 운영 모델에는 사용할 수 없다.
- 서울대 진학률과 출신 중학교 기준 특목·자사고 진학률은 공식 공개 원자료 부재로 모델에 넣지 않았다.
- `holdout_r2`가 0 이하이거나 선정 상태가 `selection_deferred`이면 운영 모델로 배포하지 않는다.
"""
        (out / "report.md").write_text(detailed_report, encoding="utf-8")
        for path in out.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="evidence")
        for path in models_dir.iterdir():
            mlflow.log_artifact(str(path), artifact_path="models")
    print(comparison.to_string(index=False))
    print(f"MLflow parent run: {parent_id}")
    print(f"결과: {out.resolve()}")


if __name__ == "__main__":
    main()
