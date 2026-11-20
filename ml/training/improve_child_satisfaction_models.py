"""기존 만족도 우승 모델에 신규 raw feature를 추가해 증분 개선 모델만 보존한다."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.base import clone


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.training.train_child_population_growth import CORE, load_population  # noqa: E402
from ml.training.train_child_satisfaction_scores import (  # noqa: E402
    SURVEY_DIR,
    build_district_features,
    build_satisfaction_panel,
    candidate_estimators,
    evaluate_estimator,
    metrics,
    numeric,
    read_survey,
    weighted_mean,
)


NEW_FEATURES = {
    "preschool": [
        "household_income_band_weighted_mean",
        "preschool_household_sample_share_pct",
        "crime_total_per_1000_residents",
        "sexual_violence_per_1000_residents",
        "violence_per_1000_residents",
    ],
    "school_age": [
        "household_income_band_weighted_mean",
        "crime_total_per_1000_residents",
        "sexual_violence_per_1000_residents",
        "violence_per_1000_residents",
        "middle_to_high_school_rate",
        "middle_to_special_purpose_high_rate",
        "middle_to_autonomous_high_rate",
        "community_child_centers_per_1000_school_age",
    ],
}

PROCESSED_FEATURES = {
    "preschool": [
        "processed_preschool_net_migration_per_1000",
        "processed_kindergartens_per_1000_preschool",
        "processed_kindergarten_students_per_class",
        "processed_kindergarten_teachers_per_1000_preschool",
        "processed_public_kindergarten_share",
        "processed_safety_zones_per_1000_preschool",
    ],
    "school_age": [
        "processed_school_age_net_migration_per_1000",
        "processed_elementary_net_migration_per_1000",
        "processed_middle_net_migration_per_1000",
        "processed_high_net_migration_per_1000",
        "processed_community_child_centers_per_1000_school_age",
        "processed_safety_zones_per_1000_school_age",
    ],
}

SETTLEMENT_FEATURES = {
    "preschool": [
        "harmful_entertainment_facilities_per_1000_residents",
        "lag1_social_environment_satisfaction",
    ],
    "school_age": [
        "harmful_entertainment_facilities_per_1000_residents",
        "lag1_social_environment_satisfaction",
    ],
}

SCHOOL_PROXIMITY_FEATURES = [
    "school_harmful_exposure_200m_student_weighted",
    "school_harmful_exposure_500m_student_weighted",
    "school_harmful_any_200m_student_share",
    "school_harmful_any_500m_student_share",
]

# LOCALDATA 개방자치단체코드. 주소 문자열은 원 API 인코딩 상태에 영향을 받을 수 있어
# 행정기관 코드를 자치구 키로 사용한다.
LOCALDATA_TEAM_TO_GU = {
    str(3000000 + index * 10000): gu for index, gu in enumerate([
        "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
        "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
        "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
    ])
}


def build_settlement_candidate_features(district: pd.DataFrame) -> pd.DataFrame:
    """유해업소 연혁과 전년도 사회환경 만족도를 자치구×연도 feature로 만든다.

    도서관·체육 프로그램 API는 현재 스냅샷뿐이므로 과거 행에 복제하지 않는다.
    사회환경 만족도는 동일 설문·동일 연도 누수를 피하기 위해 반드시 1년 시차를 둔다.
    """
    keys = ["year", "gu"]
    collected_root = ROOT / "data/raw/child_infrastructure_collected"
    raw_dir = max(
        (path for path in collected_root.iterdir()
         if path.is_dir() and (path / "danran_bar.json").is_file()
         and (path / "entertainment_bar.json").is_file()),
        key=lambda path: path.name,
    )
    facility_rows: list[dict] = []
    for dataset in ("danran_bar", "entertainment_bar"):
        payload = json.loads((raw_dir / f"{dataset}.json").read_text(encoding="utf-8-sig"))
        for row in payload["rows"]:
            facility_rows.append({
                "dataset": dataset,
                "facility_id": row.get("MGTNO"),
                "gu": LOCALDATA_TEAM_TO_GU.get(str(row.get("OPNSFTEAMCODE", "")).strip()),
                "permit_date": pd.to_datetime(row.get("APVPERMYMD"), errors="coerce"),
                "close_date": pd.to_datetime(row.get("DCBYMD"), errors="coerce"),
            })
    facilities = pd.DataFrame(facility_rows).dropna(subset=["facility_id", "gu", "permit_date"])
    facilities = facilities.drop_duplicates(["dataset", "facility_id"], keep="last")
    active_rows: list[dict] = []
    for year in range(2021, 2025):
        year_end = pd.Timestamp(year=year, month=12, day=31)
        active = facilities[
            facilities["permit_date"].le(year_end)
            & (facilities["close_date"].isna() | facilities["close_date"].gt(year_end))
        ]
        counts = active.groupby("gu").size()
        active_rows.extend({"year": year, "gu": gu, "harmful_entertainment_facility_count": int(count)}
                           for gu, count in counts.items())
    harmful = pd.DataFrame(active_rows)

    population = district[[*keys, "total_population"]].copy()
    result = population.merge(harmful, on=keys, how="left", validate="one_to_one")
    result["harmful_entertainment_facility_count"] = result[
        "harmful_entertainment_facility_count"].fillna(0)
    result["harmful_entertainment_facilities_per_1000_residents"] = result[
        "harmful_entertainment_facility_count"].div(result["total_population"].replace(0, np.nan)) * 1000

    code_to_gu = (load_population(CORE)[0].assign(
        gu_code=lambda x: x["dong_code"].str[:5].astype(int).astype(str).str[-3:].str.zfill(3)
    ).drop_duplicates("gu_code").set_index("gu_code")["gu"].to_dict())
    lagged_rows: list[dict] = []
    survey_files = {int(path.name[:4]): path for path in SURVEY_DIR.rglob("*.csv") if path.name[:4].isdigit()}
    for source_year in range(2020, 2024):
        frame = read_survey(survey_files[source_year])
        frame["gu_code"] = frame["구코드"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        frame["gu"] = frame["gu_code"].map(code_to_gu)
        weight_col = "표준화가중치" if "표준화가중치" in frame else "원가중치"
        frame["weight"] = numeric(frame[weight_col])
        social_col = next(column for column in frame if "생활환경 만족도_사회환경" in column)
        frame["social"] = numeric(frame[social_col]).where(lambda values: values.between(1, 5))
        for gu, part in frame.dropna(subset=["gu"]).groupby("gu"):
            lagged_rows.append({
                "year": source_year + 1,
                "gu": gu,
                "lag1_social_environment_satisfaction": weighted_mean(part["social"], part["weight"]),
            })
    lagged = pd.DataFrame(lagged_rows)
    result = result.merge(lagged, on=keys, how="left", validate="one_to_one")
    return result.drop(columns=["total_population"])


def build_school_harmful_proximity_features() -> tuple[pd.DataFrame, dict]:
    """연도별 운영 학교 주변 200m·500m 유해시설 노출을 학생 수로 가중 집계한다."""
    collected_root = ROOT / "data/raw/child_infrastructure_collected"
    raw_dir = max(
        (path for path in collected_root.iterdir()
         if path.is_dir() and (path / "danran_bar.json").is_file()
         and (path / "entertainment_bar.json").is_file()),
        key=lambda path: path.name,
    )
    facility_rows: list[dict] = []
    for dataset in ("danran_bar", "entertainment_bar"):
        payload = json.loads((raw_dir / f"{dataset}.json").read_text(encoding="utf-8-sig"))
        for row in payload["rows"]:
            facility_rows.append({
                "dataset": dataset,
                "facility_id": row.get("MGTNO"),
                "permit_date": pd.to_datetime(row.get("APVPERMYMD"), errors="coerce"),
                "close_date": pd.to_datetime(row.get("DCBYMD"), errors="coerce"),
                "x": pd.to_numeric(str(row.get("X", "")).strip(), errors="coerce"),
                "y": pd.to_numeric(str(row.get("Y", "")).strip(), errors="coerce"),
            })
    facilities = pd.DataFrame(facility_rows).dropna(
        subset=["facility_id", "permit_date", "x", "y"])
    facilities = facilities.drop_duplicates(["dataset", "facility_id"], keep="last")

    school_detail_path = ROOT / "data/interim/child_school_observed/school_address_mapping_2021_2026.csv"
    schools = pd.read_csv(school_detail_path, encoding="utf-8-sig")
    schools = schools[
        schools["year"].between(2021, 2024)
        & schools["학교급"].isin(["초등학교", "중학교", "고등학교"])
        & ~schools["상태"].astype(str).str.contains("폐", na=False)
    ].copy()
    schools["student_count"] = pd.to_numeric(schools["student_count"], errors="coerce")

    location_path = (ROOT / "data/processed/child_population/20261022_131634/mapped/"
                     "전국초중등학교위치표준데이터.csv")
    locations = pd.read_csv(location_path, encoding="utf-8-sig", low_memory=False)
    locations = locations[
        locations["소재지도로명주소"].astype(str).str.contains("서울", na=False)
        & locations["학교급구분"].isin(["초등학교", "중학교", "고등학교"])
    ].copy()
    locations["latitude"] = pd.to_numeric(locations["위도"], errors="coerce")
    locations["longitude"] = pd.to_numeric(locations["경도"], errors="coerce")
    locations = locations.dropna(subset=["학교명", "latitude", "longitude"])
    # 이름이 중복된 경우 같은 학교 주소와 가장 가까운 점을 고르기 위한 좌표가 학교 패널에는 없으므로
    # 공간 노출을 과대 복제하지 않도록 중복 학교명은 보수적으로 제외한다.
    duplicate_names = set(locations.loc[locations["학교명"].duplicated(keep=False), "학교명"])
    locations = locations[~locations["학교명"].isin(duplicate_names)].drop_duplicates("학교명")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5174", always_xy=True)
    locations["school_x"], locations["school_y"] = transformer.transform(
        locations["longitude"].to_numpy(), locations["latitude"].to_numpy())
    schools = schools.merge(
        locations[["학교명", "school_x", "school_y"]], on="학교명", how="left", validate="many_to_one")
    usable = schools.dropna(subset=["school_x", "school_y", "student_count"]).copy()
    usable = usable[usable["student_count"].gt(0)]

    rows: list[dict] = []
    for year, year_schools in usable.groupby("year"):
        year_end = pd.Timestamp(year=int(year), month=12, day=31)
        active = facilities[
            facilities["permit_date"].le(year_end)
            & (facilities["close_date"].isna() | facilities["close_date"].gt(year_end))
        ]
        tree = cKDTree(active[["x", "y"]].to_numpy())
        points = year_schools[["school_x", "school_y"]].to_numpy()
        year_schools = year_schools.copy()
        for radius in (200, 500):
            year_schools[f"harmful_{radius}m"] = [
                len(matches) for matches in tree.query_ball_point(points, r=radius)]
        for gu, part in year_schools.groupby("행정구"):
            weights = part["student_count"]
            row = {"year": int(year), "gu": gu, "school_count_with_coordinates": int(len(part)),
                   "school_students_with_coordinates": float(weights.sum())}
            for radius in (200, 500):
                counts = part[f"harmful_{radius}m"]
                row[f"school_harmful_exposure_{radius}m_student_weighted"] = float(
                    np.average(counts, weights=weights))
                row[f"school_harmful_any_{radius}m_student_share"] = float(
                    weights[counts.gt(0)].sum() / weights.sum())
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(schools)
    quality = {
        "school_rows_2021_2024": expected,
        "school_rows_with_coordinates_and_students": len(usable),
        "mapping_rate": len(usable) / expected if expected else np.nan,
        "excluded_duplicate_location_names": len(duplicate_names),
        "facility_rows_with_coordinates": len(facilities),
        "facility_coordinate_rate": len(facilities) / len(facility_rows) if facility_rows else np.nan,
        "school_location_source_date": "2026-10-01",
        "school_location_assumption": "학교 주소 위치는 분석기간 중 불변으로 간주; 연도별 운영 여부와 학생 수는 연도 자료 사용",
    }
    return result, quality


def build_processed_candidate_features(district: pd.DataFrame) -> pd.DataFrame:
    """processed의 이동·유치원·지역아동센터·보호구역을 자치구×연도로 집계한다."""
    keys = ["year", "gu"]
    migration_path = (ROOT / "data/processed/child_population/admin_dong_age_migration/"
                      "admin_dong_age_migration_annual_2021_2025.csv")
    migration = pd.read_csv(migration_path, dtype={"dong_code": str})
    migration_grouped = migration.groupby([*keys, "age_group"], as_index=False).agg(
        net_migration=("net_migration", "sum"), age_population=("age_population_year_end", "sum"))
    migration_grouped["rate"] = migration_grouped["net_migration"].div(
        migration_grouped["age_population"].replace(0, np.nan)) * 1000
    migration_wide = migration_grouped.pivot_table(
        index=keys, columns="age_group", values="rate", aggfunc="first").reset_index()
    migration_wide.columns.name = None
    migration_wide = migration_wide.rename(columns={
        "preschool": "processed_preschool_net_migration_per_1000",
        "elementary": "processed_elementary_net_migration_per_1000",
        "middle": "processed_middle_net_migration_per_1000",
        "high": "processed_high_net_migration_per_1000",
    })
    school_migration = migration[migration["age_group"].isin(["elementary", "middle", "high"])].groupby(
        keys, as_index=False).agg(net_migration=("net_migration", "sum"),
                                  age_population=("age_population_year_end", "sum"))
    school_migration["processed_school_age_net_migration_per_1000"] = school_migration[
        "net_migration"].div(school_migration["age_population"].replace(0, np.nan)) * 1000
    result = migration_wide.merge(
        school_migration[[*keys, "processed_school_age_net_migration_per_1000"]],
        on=keys, how="outer", validate="one_to_one")

    kindergarten_path = ROOT / "data/processed/행정동별_유치원_최종.xlsx"
    kindergarten = pd.read_excel(kindergarten_path, sheet_name="상반기_분석").rename(columns={
        "기준연도": "year", "자치구": "gu", "학교명": "school_name", "설립구분": "founder",
        "학급수": "classes", "원아수_계": "students", "교원수_계": "teachers",
    })
    kindergarten = kindergarten[kindergarten["year"].between(2021, 2025)].copy()
    for column in ("classes", "students", "teachers"):
        kindergarten[column] = pd.to_numeric(kindergarten[column], errors="coerce").fillna(0)
    kindergarten["active"] = kindergarten[["classes", "students"]].max(axis=1).gt(0)
    kindergarten["public_active"] = kindergarten["active"] & kindergarten["founder"].eq("공립")
    kindergarten_rows = kindergarten.groupby(keys, as_index=False).agg(
        processed_kindergarten_count=("active", "sum"),
        processed_public_kindergarten_count=("public_active", "sum"),
        processed_kindergarten_classes=("classes", "sum"),
        processed_kindergarten_students=("students", "sum"),
        processed_kindergarten_teachers=("teachers", "sum"),
    )
    result = result.merge(kindergarten_rows, on=keys, how="outer", validate="one_to_one")

    center_path = ROOT / "data/processed/행정동별_지역아동센터_최종.csv"
    centers = pd.read_csv(center_path, encoding="utf-8-sig").rename(columns={"기준연도": "year", "자치구": "gu"})
    centers = centers.groupby(keys, as_index=False).size().rename(columns={"size": "processed_community_child_center_count"})
    result = result.merge(centers, on=keys, how="outer", validate="one_to_one")

    safety_path = ROOT / "data/processed/행정동별_어린이보호구역_최종.csv"
    safety = pd.read_csv(safety_path, encoding="utf-8-sig")
    safety["date"] = pd.to_datetime(safety["기준일"])
    safety["year"] = safety["date"].dt.year
    latest = safety.groupby("year")["date"].transform("max")
    safety = safety[safety["date"].eq(latest)].rename(columns={"자치구": "gu", "계": "processed_safety_zone_count"})
    safety = safety.groupby(keys, as_index=False)["processed_safety_zone_count"].sum()
    result = result.merge(safety, on=keys, how="outer", validate="one_to_one")

    result = district[[*keys, "children_infant", "school_age_population"]].merge(
        result, on=keys, how="left", validate="one_to_one")
    infant = result["children_infant"].replace(0, np.nan)
    school_age = result["school_age_population"].replace(0, np.nan)
    result["processed_kindergartens_per_1000_preschool"] = result["processed_kindergarten_count"].div(infant) * 1000
    result["processed_kindergarten_students_per_class"] = result["processed_kindergarten_students"].div(
        result["processed_kindergarten_classes"].replace(0, np.nan))
    result["processed_kindergarten_teachers_per_1000_preschool"] = result[
        "processed_kindergarten_teachers"].div(infant) * 1000
    result["processed_public_kindergarten_share"] = result["processed_public_kindergarten_count"].div(
        result["processed_kindergarten_count"].replace(0, np.nan))
    result["processed_community_child_centers_per_1000_school_age"] = result[
        "processed_community_child_center_count"].div(school_age) * 1000
    result["processed_safety_zones_per_1000_preschool"] = result["processed_safety_zone_count"].div(infant) * 1000
    result["processed_safety_zones_per_1000_school_age"] = result["processed_safety_zone_count"].div(school_age) * 1000
    return result.drop(columns=["children_infant", "school_age_population"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_combinations(base: list[str], additions: list[str], max_added: int) -> list[list[str]]:
    result = [base]
    for size in range(1, min(max_added, len(additions)) + 1):
        result.extend([base + list(values) for values in itertools.combinations(additions, size)])
    return result


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env.dev", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "artifacts/child_satisfaction_scores/20261028_142519/selected_models.json")
    parser.add_argument("--incumbent-summary", type=Path,
                        default=ROOT / "artifacts/child_satisfaction_improvement/20261028_175858/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/child_satisfaction_improvement")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="child-satisfaction-incremental-improvement")
    parser.add_argument("--max-added-features", type=int, default=3)
    parser.add_argument("--min-mae-improvement", type=float, default=.001)
    parser.add_argument("--feature-set", choices=("raw_added", "processed", "settlement", "school_proximity", "all"), default="all")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output / stamp
    models_dir = output / "models"
    models_dir.mkdir(parents=True)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if args.incumbent_summary.is_file():
        incumbent = json.loads(args.incumbent_summary.read_text(encoding="utf-8"))
        for improvement in incumbent.get("improvements", []):
            group = improvement["group"]
            baseline[group]["features"] = json.loads(improvement["features"])
            baseline[group]["algorithm"] = improvement["algorithm"]
            baseline[group]["holdout"] = {
                key.removeprefix("holdout_"): value for key, value in improvement.items()
                if key.startswith("holdout_")
            }

    district, audit = build_district_features()
    processed = build_processed_candidate_features(district)
    district = district.merge(processed, on=["year", "gu"], how="left", validate="one_to_one")
    settlement = build_settlement_candidate_features(district)
    district = district.merge(settlement, on=["year", "gu"], how="left", validate="one_to_one")
    proximity, proximity_quality = build_school_harmful_proximity_features()
    district = district.merge(proximity, on=["year", "gu"], how="left", validate="one_to_one")
    code_to_gu = (load_population(CORE)[0].assign(
        gu_code=lambda x: x["dong_code"].str[:5].astype(int).astype(str).str[-3:].str.zfill(3)
    ).drop_duplicates("gu_code").set_index("gu_code")["gu"].to_dict())
    satisfaction = build_satisfaction_panel(code_to_gu)
    panel = satisfaction.merge(district, on=["year", "gu"], how="left", validate="many_to_one")
    panel.to_csv(output / "training_panel.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "feature_source_audit.csv", index=False, encoding="utf-8-sig")

    raw_sources = [
        ROOT / "data/raw/가구소득/가구소득_2015_2025.csv",
        ROOT / "data/raw/범죄/범죄_2014_2025.csv",
        ROOT / "data/raw/중학교_고등학교_진학률_2011_2025.csv",
        *sorted((ROOT / "data/raw/지역아동센터").glob("20??_*.xlsx")),
        ROOT / "data/processed/child_population/admin_dong_age_migration/admin_dong_age_migration_annual_2021_2025.csv",
        ROOT / "data/processed/행정동별_유치원_최종.xlsx",
        ROOT / "data/processed/행정동별_지역아동센터_최종.csv",
        ROOT / "data/processed/행정동별_어린이보호구역_최종.csv",
        ROOT / "data/raw/child_infrastructure_collected/20261029/danran_bar.json",
        ROOT / "data/raw/child_infrastructure_collected/20261029/entertainment_bar.json",
        ROOT / "data/interim/child_school_observed/school_address_mapping_2021_2026.csv",
        ROOT / "data/processed/child_population/20261022_131634/mapped/전국초중등학교위치표준데이터.csv",
        *sorted(SURVEY_DIR.rglob("202?_*.csv")),
    ]
    source_manifest = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in raw_sources]
    (output / "new_raw_source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "school_proximity_quality.json").write_text(
        json.dumps(proximity_quality, ensure_ascii=False, indent=2), encoding="utf-8")

    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'artifacts/mlflow_child_satisfaction.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    comparisons: list[dict] = []
    improvements: list[dict] = []
    algorithms = {name: model for name, model in candidate_estimators().items() if name != "dummy_mean"}

    with mlflow.start_run(run_name=f"incremental_comparison_{stamp}") as parent:
        mlflow.log_params({
            "groups": "preschool,school_age", "development_years": "2021-2023",
            "holdout_year": 2024, "selection_metric": "district-grouped-5-fold-cv-mae",
            "max_added_features": args.max_added_features,
            "min_mae_improvement": args.min_mae_improvement,
            "feature_set": args.feature_set,
            "save_policy": "only sequential CV MAE record breakers are logged as models",
        })
        mlflow.set_tags({
            "comparison_design": "feature_combination_x_algorithm",
            "db_write_method": "MLflow Tracking API",
            "causal_claim": "none",
        })

        for group in ("preschool", "school_age"):
            work = panel[panel["group"].eq(group)].replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])
            development = work[work["year"].le(2023)].copy()
            holdout = work[work["year"].eq(2024)].copy()
            base_features = list(baseline[group]["features"])
            feature_pool = {
                "raw_added": NEW_FEATURES[group],
                "processed": PROCESSED_FEATURES[group],
                "settlement": SETTLEMENT_FEATURES[group],
                "school_proximity": SCHOOL_PROXIMITY_FEATURES if group == "school_age" else [],
                "all": [*NEW_FEATURES[group], *PROCESSED_FEATURES[group], *SETTLEMENT_FEATURES[group],
                        *(SCHOOL_PROXIMITY_FEATURES if group == "school_age" else [])],
            }[args.feature_set]
            additions = [feature for feature in feature_pool if feature not in base_features
                         and feature in development and development[feature].notna().any()
                         and development[feature].nunique(dropna=True) > 1]
            combos = feature_combinations(base_features, additions, args.max_added_features)
            if args.feature_set == "school_proximity" and group == "school_age":
                broad_density = "harmful_entertainment_facilities_per_1000_residents"
                without_broad = [feature for feature in base_features if feature != broad_density]
                combos.extend(feature_combinations(without_broad, additions, args.max_added_features)[1:])
                combos = list(dict.fromkeys(tuple(combo) for combo in combos))
                combos = [list(combo) for combo in combos]

            baseline_algorithm = baseline[group]["algorithm"]
            baseline_cv = evaluate_estimator(development, base_features, algorithms[baseline_algorithm])
            incumbent_mae = baseline_cv["mae"]
            comparison_candidates = []
            for features in combos:
                added = [feature for feature in features if feature not in base_features]
                for algorithm, estimator in algorithms.items():
                    cv_score = evaluate_estimator(development, features, estimator)
                    fitted = clone(estimator).fit(development[features], development["target"])
                    holdout_score = metrics(holdout["target"], fitted.predict(holdout[features]))
                    row = {
                        "group": group, "algorithm": algorithm,
                        "features": json.dumps(features, ensure_ascii=False),
                        "added_features": json.dumps(added, ensure_ascii=False),
                        "feature_count": len(features), "added_feature_count": len(added),
                        **{f"cv_{key}": value for key, value in cv_score.items()},
                        **{f"holdout_{key}": value for key, value in holdout_score.items()},
                        "historical_baseline_cv_mae": baseline_cv["mae"],
                    }
                    comparisons.append(row)
                    if added:
                        comparison_candidates.append((cv_score["mae"], features, algorithm, fitted, row))

            # 조합 크기와 무관하게 MAE가 낮은 순서로 보되, 기존 우승보다 개선된 기록만 모델로 남긴다.
            for candidate_mae, features, algorithm, fitted, row in sorted(comparison_candidates, key=lambda x: x[0]):
                if incumbent_mae - candidate_mae < args.min_mae_improvement:
                    continue
                previous_mae = incumbent_mae
                incumbent_mae = candidate_mae
                checkpoint = len(improvements) + 1
                model_path = models_dir / f"{group}_improvement_{checkpoint}.joblib"
                payload = {
                    "model": fitted, "features": features, "target": baseline[group]["target"],
                    "algorithm": algorithm, "cv_metrics": {k: v for k, v in row.items() if k.startswith("cv_")},
                }
                joblib.dump(payload, model_path)
                improvement = {
                    **row, "checkpoint": checkpoint, "previous_best_cv_mae": previous_mae,
                    "absolute_mae_improvement": previous_mae - candidate_mae,
                    "relative_mae_improvement_pct": (1 - candidate_mae / previous_mae) * 100,
                }
                improvements.append(improvement)
                with mlflow.start_run(run_name=f"{group}_improved_{checkpoint}", nested=True):
                    mlflow.log_params({
                        "group": group, "algorithm": algorithm,
                        "target": baseline[group]["target"], "features": json.dumps(features, ensure_ascii=False),
                        "added_features": row["added_features"], "feature_count": len(features),
                    })
                    mlflow.log_metrics({key: float(value) for key, value in improvement.items()
                                        if isinstance(value, (int, float)) and np.isfinite(value)})
                    mlflow.set_tags({
                        "improved_vs_previous": "true", "model_saved": "true",
                        "selection_status": "exploratory_candidate", "champion_promoted": "false",
                        "holdout_warning": (
                            "2024 MAE worsened versus historical model"
                            if row["holdout_mae"] > baseline[group]["holdout"]["mae"]
                            else "none"
                        ),
                    })
                    mlflow.log_dict({"features": features, "source_manifest": source_manifest}, "features.json")
                    mlflow.sklearn.log_model(fitted, name="model", serialization_format="cloudpickle",
                                             input_example=development[features].head(3))
                    mlflow.log_artifact(str(model_path), artifact_path="joblib")

        comparison_df = pd.DataFrame(comparisons).sort_values(["group", "cv_mae"])
        improvement_df = pd.DataFrame(improvements)
        comparison_df.to_csv(output / "feature_algorithm_comparison.csv", index=False, encoding="utf-8-sig")
        improvement_df.to_csv(output / "saved_improved_models.csv", index=False, encoding="utf-8-sig")
        summary = {
            "parent_run_id": parent.info.run_id, "tracking_uri": tracking_uri,
            "attempted_models": len(comparison_df), "saved_improved_models": len(improvement_df),
            "improvements": improvements,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output / "feature_algorithm_comparison.csv"), artifact_path="evidence")
        mlflow.log_artifact(str(output / "saved_improved_models.csv"), artifact_path="evidence")
        mlflow.log_artifact(str(output / "new_raw_source_manifest.json"), artifact_path="evidence")
        mlflow.log_artifact(str(output / "school_proximity_quality.json"), artifact_path="evidence")
        mlflow.log_dict(summary, "summary.json")

    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
