"""보유 중인 아동 인프라 원자료를 서울 행정동 현재시점 특징표로 결합한다.

현재 스냅샷이므로 과거 연도 학습 행에 복제하지 않는다. 각 원본의 좌표계와 운영상태를
명시적으로 처리하며, 매핑되지 않은 행은 품질 보고서에 남긴다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/아동"


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"인코딩을 판별할 수 없습니다: {path}")


def spatial_aggregate(frame: pd.DataFrame, boundaries: gpd.GeoDataFrame, lon: str, lat: str,
                      metrics: dict[str, tuple[str, str]], source_crs: str = "EPSG:4326") -> tuple[pd.DataFrame, dict]:
    work = frame.copy()
    work[lon] = pd.to_numeric(work[lon], errors="coerce")
    work[lat] = pd.to_numeric(work[lat], errors="coerce")
    valid = work[lon].notna() & work[lat].notna()
    points = gpd.GeoDataFrame(work.loc[valid].copy(), geometry=gpd.points_from_xy(work.loc[valid, lon], work.loc[valid, lat]), crs=source_crs).to_crs("EPSG:4326")
    joined = gpd.sjoin(points, boundaries[["adm_cd", "adm_nm", "geometry"]], how="left", predicate="within")
    mapped = joined.dropna(subset=["adm_cd"])
    result = mapped.groupby(["adm_cd", "adm_nm"]).agg(**metrics).reset_index()
    quality = {"input_rows": len(frame), "coordinate_rows": int(valid.sum()), "mapped_rows": int(joined["adm_cd"].notna().sum())}
    return result, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daycare", type=Path, required=True, help="preprocess_daycare.py가 만든 행정동 집계 CSV")
    parser.add_argument("--collected-dir", type=Path, required=True, help="수집기가 만든 날짜별 raw 폴더")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/child_infrastructure_snapshot")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output if args.output.is_absolute() else ROOT / args.output) / stamp
    out.mkdir(parents=True, exist_ok=False)

    boundaries = gpd.read_file(ROOT / "frontend/public/seoul-dong.geojson").to_crs("EPSG:4326")
    panel = boundaries[["adm_cd", "adm_nm"]].copy()
    panel["district"] = panel["adm_nm"].str.split().str[1]
    panel["administrative_dong"] = panel["adm_nm"].str.split().str[2]
    quality_rows: list[dict] = []

    daycare_path = args.daycare if args.daycare.is_absolute() else ROOT / args.daycare
    collected_dir = args.collected_dir if args.collected_dir.is_absolute() else ROOT / args.collected_dir
    daycare = pd.read_csv(daycare_path, dtype={"adm_cd": str})
    daycare["adm_cd"] = daycare["adm_cd"].str.zfill(8)
    panel = panel.merge(daycare.drop(columns=["adm_nm", "district", "administrative_dong"], errors="ignore"), on="adm_cd", how="left", validate="one_to_one")

    library_payload = json.loads((collected_dir / "public_library.json").read_text(encoding="utf-8-sig"))
    library = pd.DataFrame(library_payload["rows"]).assign(one=1)
    # 서울 도서관 API의 XCNTS는 위도, YDNTS는 경도입니다. 일반적인 X/Y 명칭과 반대라 명시합니다.
    agg, q = spatial_aggregate(library, boundaries, "YDNTS", "XCNTS", {"public_library_count": ("one", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one")
    quality_rows.append({"dataset": "public_library", **q})

    kindergarten = pd.read_csv(collected_dir / "kindergarten_geocoded.csv", encoding="utf-8-sig")
    kindergarten = kindergarten[kindergarten["status"].eq("OK")].copy()
    kindergarten["one"] = 1
    kindergarten["public_kindergarten"] = kindergarten["FNDN_TYPE"].astype(str).str.contains("공립").astype(int)
    kindergarten["kindergarten_classes"] = kindergarten[["AGE_3_CLAS_CNT", "AGE_4_CLAS_CNT", "AGE_5_CLAS_CNT", "MIX_CLAS_CNT", "SPCL_CLAS_CNT"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    kindergarten["kindergarten_children"] = kindergarten[["AGE_3_TDL_CNT", "AGE_4_TDL_CNT", "AGE_5_TDL_CNT", "MIX_TDL_CNT", "SPCL_TDL_CNT"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    agg, q = spatial_aggregate(kindergarten, boundaries, "longitude", "latitude", {
        "kindergarten_count": ("one", "sum"), "public_kindergarten_count": ("public_kindergarten", "sum"),
        "kindergarten_class_count": ("kindergarten_classes", "sum"), "kindergarten_children": ("kindergarten_children", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one")
    quality_rows.append({"dataset": "kindergarten", **q})

    sports_payload = json.loads((collected_dir / "sports_program.json").read_text(encoding="utf-8-sig"))
    sports = pd.DataFrame(sports_payload["rows"])
    sports["district"] = sports["AR_CD_NAME"].astype(str).str.strip().map(lambda value: value if value.endswith("구") else value + "구")
    sports["is_child_program"] = sports[["BD_CONTENT", "BD_FIELD_2", "BD_FIELD_11"]].fillna("").astype(str).agg(" ".join, axis=1).str.contains("유아|어린이|아동|청소년")
    sports_by_district = sports.groupby("district").agg(
        sports_program_count=("BD_IDX", "size"), child_sports_program_count=("is_child_program", "sum")
    ).reset_index()
    # 좌표 없는 프로그램을 행정동에 복제하지 않고 자치구 보조 자료로 별도 보존합니다.
    sports_by_district.to_csv(out / "sports_program_by_district.csv", index=False, encoding="utf-8-sig")
    sports_geo = pd.read_csv(collected_dir / "sports_program_geocoded.csv", encoding="utf-8-sig")
    sports_geo = sports_geo[sports_geo["status"].eq("OK")].copy()
    sports_geo["child_program"] = sports_geo[["BD_CONTENT", "BD_FIELD_2", "BD_FIELD_11"]].fillna("").astype(str).agg(" ".join, axis=1).str.contains("유아|어린이|아동|청소년").astype(int)
    sports_geo["one"] = 1
    agg, q = spatial_aggregate(sports_geo, boundaries, "longitude", "latitude", {
        "sports_program_count": ("one", "sum"), "child_sports_program_count": ("child_program", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one")
    quality_rows.append({"dataset": "sports_program", **q})

    apartment = pd.DataFrame(json.loads((collected_dir / "apartment.json").read_text(encoding="utf-8-sig"))["rows"])
    apartment = apartment[apartment["USE_YN"].eq("Y")].copy()
    apartment["one"] = 1
    for column in ["TNOHSH", "WHOL_DONG_CNT", "GFA", "PRK_CNTOM"]:
        apartment[column] = pd.to_numeric(apartment[column], errors="coerce")
    agg, q = spatial_aggregate(apartment, boundaries, "XCRD", "YCRD", {
        "apartment_complex_count": ("one", "sum"), "apartment_household_count": ("TNOHSH", "sum"),
        "apartment_building_count": ("WHOL_DONG_CNT", "sum"), "apartment_gross_area_m2": ("GFA", "sum"),
        "apartment_parking_count": ("PRK_CNTOM", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one")
    quality_rows.append({"dataset": "apartment", **q})

    school = read_csv(RAW / "아동 데이터/전국초중등학교위치표준데이터.csv")
    school = school[school["소재지도로명주소"].fillna(school["소재지지번주소"]).astype(str).str.contains("서울") & school["운영상태"].eq("운영")]
    for level, label in [("초등학교", "elementary_school_count"), ("중학교", "middle_school_count"), ("고등학교", "high_school_count")]:
        part = school[school["학교급구분"].eq(level)].assign(one=1)
        agg, q = spatial_aggregate(part, boundaries, "경도", "위도", {label: ("one", "sum")})
        panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one")
        quality_rows.append({"dataset": label, **q})

    center = read_csv(RAW / "아동 데이터/전국지역아동센터표준데이터.csv")
    center = center[center["시도명"].astype(str).str.contains("서울")].assign(one=1)
    center["정원수"] = pd.to_numeric(center["정원수"], errors="coerce")
    agg, q = spatial_aggregate(center, boundaries, "경도", "위도", {"child_center_count": ("one", "sum"), "child_center_capacity": ("정원수", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one"); quality_rows.append({"dataset": "child_center", **q})

    park = read_csv(RAW / "기타/전국도시공원정보표준데이터.csv")
    address = park["소재지도로명주소"].fillna(park["소재지지번주소"]).astype(str)
    park = park[address.str.contains("서울")].assign(one=1)
    park["공원면적"] = pd.to_numeric(park["공원면적"], errors="coerce")
    park["child_park"] = park["공원구분"].astype(str).str.contains("어린이").astype(int)
    park["child_park_area"] = park["공원면적"].where(park["child_park"].eq(1), 0)
    agg, q = spatial_aggregate(park, boundaries, "경도", "위도", {"park_count": ("one", "sum"), "park_area_m2": ("공원면적", "sum"), "children_park_count": ("child_park", "sum"), "children_park_area_m2": ("child_park_area", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one"); quality_rows.append({"dataset": "park", **q})

    cctv = read_csv(RAW / "기타/CCTV정보_서울특별시.csv")
    cctv["카메라대수"] = pd.to_numeric(cctv["카메라대수"], errors="coerce").fillna(1)
    cctv["one"] = 1
    agg, q = spatial_aggregate(cctv, boundaries, "WGS84경도", "WGS84위도", {"cctv_location_count": ("one", "sum"), "cctv_camera_count": ("카메라대수", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one"); quality_rows.append({"dataset": "cctv", **q})

    bell = read_csv(RAW / "기타/안전비상벨위치정보_서울특별시.csv").assign(one=1)
    agg, q = spatial_aggregate(bell, boundaries, "WGS84경도", "WGS84위도", {"emergency_bell_count": ("one", "sum")})
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one"); quality_rows.append({"dataset": "emergency_bell", **q})

    hospital = read_csv(RAW / "병원/서울특별시_병원.csv")
    hospital = hospital[hospital["영업상태명"].eq("영업/정상")].copy()
    hospital["one"] = 1
    hospital["pediatric"] = hospital["진료과목내용명"].astype(str).str.contains("소아청소년과").astype(int)
    agg, q = spatial_aggregate(hospital, boundaries, "좌표정보(X)", "좌표정보(Y)", {"hospital_count": ("one", "sum"), "pediatric_hospital_count": ("pediatric", "sum")}, source_crs="EPSG:5174")
    panel = panel.merge(agg.drop(columns="adm_nm"), on="adm_cd", how="left", validate="one_to_one"); quality_rows.append({"dataset": "hospital", **q})

    feature_columns = [c for c in panel.columns if c not in {"adm_cd", "adm_nm", "district", "administrative_dong"}]
    # 위 원본들은 서울 전체 목록이므로, 매핑 시설이 없는 행정동은 결측이 아닌 0개로 해석합니다.
    panel[feature_columns] = panel[feature_columns].fillna(0)
    panel.to_csv(out / "child_infrastructure_snapshot_by_dong.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_rows).to_csv(out / "source_mapping_quality.csv", index=False, encoding="utf-8-sig")
    print(f"행정동 {len(panel)}개, 특징 {len(feature_columns)}개")
    print(pd.DataFrame(quality_rows).to_string(index=False))
    print(f"저장 위치: {out}")


if __name__ == "__main__":
    main()
