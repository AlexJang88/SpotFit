"""현재 인프라 스냅샷과 최신 인구를 결합해 비교 가능한 행정동 학습 특징을 만든다."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def safe_rate(numerator: pd.Series, denominator: pd.Series, multiplier: float) -> pd.Series:
    return numerator.div(denominator.where(denominator.gt(0))).mul(multiplier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infrastructure", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/child_infrastructure_features")
    args = parser.parse_args()
    infra_path = args.infrastructure if args.infrastructure.is_absolute() else ROOT / args.infrastructure
    pop_path = args.population if args.population.is_absolute() else ROOT / args.population
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output if args.output.is_absolute() else ROOT / args.output) / stamp
    out.mkdir(parents=True, exist_ok=False)

    infra = pd.read_csv(infra_path, dtype={"adm_cd": str})
    pop = pd.read_csv(pop_path, dtype={"dong_code": str})
    latest_year = int(pop["year"].max())
    pop = pop[pop["year"].eq(latest_year)].copy()
    pop["region_key"] = pop["gu"].astype(str).str.strip() + "|" + pop["dong"].astype(str).str.strip()
    if pop["region_key"].duplicated().any():
        raise ValueError("최신 인구에 중복 자치구·행정동 조합이 있습니다.")

    boundaries = gpd.read_file(ROOT / "frontend/public/seoul-dong.geojson").to_crs("EPSG:5179")
    area = pd.DataFrame({"adm_nm": boundaries["adm_nm"].astype(str), "area_km2": boundaries.geometry.area / 1_000_000})
    infra["region_key"] = infra["district"].astype(str).str.strip() + "|" + infra["administrative_dong"].astype(str).str.strip()
    data = infra.merge(pop.drop(columns=["gu", "dong"], errors="ignore"), on="region_key", how="left", validate="one_to_one", indicator=True)
    data[data["_merge"].ne("both")].to_csv(out / "unmatched_population.csv", index=False, encoding="utf-8-sig")
    data = data.drop(columns="_merge").merge(area, on="adm_nm", how="left", validate="one_to_one")
    data["children_total_0_17"] = data[["children_infant", "children_elementary", "children_middle", "children_high"]].sum(axis=1, min_count=4)

    data["daycare_per_1000_infant"] = safe_rate(data["daycare_count"], data["children_infant"], 1000)
    data["daycare_capacity_per_1000_infant"] = safe_rate(data["daycare_capacity"], data["children_infant"], 1000)
    data["public_daycare_share"] = data["public_daycare_count"].div(data["daycare_count"].where(data["daycare_count"].gt(0)))
    data["elementary_school_per_1000_children"] = safe_rate(data["elementary_school_count"], data["children_elementary"], 1000)
    data["middle_school_per_1000_children"] = safe_rate(data["middle_school_count"], data["children_middle"], 1000)
    data["high_school_per_1000_children"] = safe_rate(data["high_school_count"], data["children_high"], 1000)
    data["child_center_per_1000_elementary"] = safe_rate(data["child_center_count"], data["children_elementary"], 1000)
    data["child_center_capacity_per_1000_elementary"] = safe_rate(data["child_center_capacity"], data["children_elementary"], 1000)
    data["park_area_m2_per_resident"] = safe_rate(data["park_area_m2"], data["total_population"], 1)
    data["children_park_area_m2_per_infant"] = safe_rate(data["children_park_area_m2"], data["children_infant"], 1)
    data["pediatric_hospital_per_1000_children"] = safe_rate(data["pediatric_hospital_count"], data["children_total_0_17"], 1000)
    data["public_library_per_10000_residents"] = safe_rate(data["public_library_count"], data["total_population"], 10000)
    data["kindergarten_per_1000_infant"] = safe_rate(data["kindergarten_count"], data["children_infant"], 1000)
    data["kindergarten_children_per_1000_infant"] = safe_rate(data["kindergarten_children"], data["children_infant"], 1000)
    data["public_kindergarten_share"] = data["public_kindergarten_count"].div(data["kindergarten_count"].where(data["kindergarten_count"].gt(0)))
    data["child_sports_program_per_1000_children"] = safe_rate(data["child_sports_program_count"], data["children_total_0_17"], 1000)
    data["apartment_households_per_1000_residents"] = safe_rate(data["apartment_household_count"], data["total_population"], 1000)
    data["apartment_parking_per_household"] = data["apartment_parking_count"].div(data["apartment_household_count"].where(data["apartment_household_count"].gt(0)))
    for source, target in [("cctv_camera_count", "cctv_per_km2"), ("emergency_bell_count", "emergency_bell_per_km2"), ("park_count", "park_count_per_km2")]:
        data[target] = data[source].div(data["area_km2"].where(data["area_km2"].gt(0)))

    rate_columns = [
        "daycare_per_1000_infant", "daycare_capacity_per_1000_infant", "public_daycare_share",
        "elementary_school_per_1000_children", "middle_school_per_1000_children",
        "high_school_per_1000_children", "child_center_per_1000_elementary",
        "child_center_capacity_per_1000_elementary", "park_area_m2_per_resident",
        "children_park_area_m2_per_infant", "pediatric_hospital_per_1000_children",
        "public_library_per_10000_residents",
        "kindergarten_per_1000_infant", "kindergarten_children_per_1000_infant",
        "public_kindergarten_share", "child_sports_program_per_1000_children",
        "apartment_households_per_1000_residents", "apartment_parking_per_household",
        "cctv_per_km2", "emergency_bell_per_km2", "park_count_per_km2",
    ]
    quality = pd.DataFrame({"feature": rate_columns, "missing": [int(data[c].isna().sum()) for c in rate_columns], "p01": [data[c].quantile(.01) for c in rate_columns], "median": [data[c].median() for c in rate_columns], "p99": [data[c].quantile(.99) for c in rate_columns]})
    data.to_csv(out / "current_child_infrastructure_features_by_dong.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(out / "feature_quality.csv", index=False, encoding="utf-8-sig")
    print(f"인구 기준연도={latest_year}, 행정동={len(data)}, 파생 특징={len(rate_columns)}")
    print(quality.to_string(index=False))
    print(f"저장 위치: {out}")


if __name__ == "__main__":
    main()
