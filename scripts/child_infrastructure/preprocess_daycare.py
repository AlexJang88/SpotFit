"""수집한 어린이집 스냅샷을 정제하고 서울 행정동별 공급 지표로 집계한다."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, default=ROOT / "frontend/public/seoul-dong.geojson")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/child_infrastructure")
    args = parser.parse_args()

    source = args.input if args.input.is_absolute() else ROOT / args.input
    boundaries_path = args.boundaries if args.boundaries.is_absolute() else ROOT / args.boundaries
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output if args.output.is_absolute() else ROOT / args.output) / stamp
    output.mkdir(parents=True, exist_ok=False)

    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(payload["rows"])
    frame["latitude"] = pd.to_numeric(frame["LA"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["LO"], errors="coerce")
    frame["capacity"] = pd.to_numeric(frame["CRCAPAT"], errors="coerce")
    frame["children"] = pd.to_numeric(frame["CRCHCNT"], errors="coerce")
    frame["is_active"] = frame["CRSTATUSNAME"].isin(["정상", "재개"])

    # 서울 경계의 대략적인 bbox로 명백한 오좌표를 우선 판별합니다.
    frame["coordinate_in_seoul_bbox"] = (
        frame["longitude"].between(126.70, 127.20) & frame["latitude"].between(37.40, 37.72)
    )
    coordinate_frequency = frame.groupby(["longitude", "latitude"])["CRNAME"].transform("size")
    # 서로 다른 주소가 같은 대표좌표에 몰린 사례를 배제하기 위한 보수적 품질 플래그입니다.
    frame["coordinate_repeated_20_plus"] = coordinate_frequency.ge(20)
    frame["coordinate_usable"] = (
        frame["longitude"].notna()
        & frame["latitude"].notna()
        & frame["coordinate_in_seoul_bbox"]
        & ~frame["coordinate_repeated_20_plus"]
    )

    active = frame.loc[frame["is_active"]].copy()
    points = gpd.GeoDataFrame(
        active.loc[active["coordinate_usable"]].copy(),
        geometry=gpd.points_from_xy(
            active.loc[active["coordinate_usable"], "longitude"],
            active.loc[active["coordinate_usable"], "latitude"],
        ),
        crs="EPSG:4326",
    )
    boundaries = gpd.read_file(boundaries_path).to_crs("EPSG:4326")
    keep = [column for column in ["adm_cd", "adm_nm", "name", "geometry"] if column in boundaries.columns]
    joined = gpd.sjoin(points, boundaries[keep], how="left", predicate="within")
    joined["district"] = joined.get("adm_nm", "").astype(str).str.split().str[1]
    joined["administrative_dong"] = joined.get("adm_nm", "").astype(str).str.split().str[2]

    group_columns = [column for column in ["adm_cd", "adm_nm", "district", "administrative_dong"] if column in joined]
    aggregate = joined.dropna(subset=["adm_nm"]).groupby(group_columns, dropna=False).agg(
        daycare_count=("CRNAME", "size"),
        daycare_capacity=("capacity", "sum"),
        daycare_children=("children", "sum"),
        public_daycare_count=("CRTYPENAME", lambda values: values.astype(str).str.contains("국공립").sum()),
    ).reset_index()
    aggregate["daycare_occupancy_rate"] = aggregate["daycare_children"].div(
        aggregate["daycare_capacity"].where(aggregate["daycare_capacity"].gt(0))
    )

    frame.drop(columns=[], errors="ignore").to_csv(output / "daycare_quality_all.csv", index=False, encoding="utf-8-sig")
    joined.drop(columns="geometry").to_csv(output / "daycare_active_mapped.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(output / "daycare_by_administrative_dong.csv", index=False, encoding="utf-8-sig")
    quality = pd.DataFrame([{
        "all_rows": len(frame),
        "active_rows": len(active),
        "active_usable_coordinates": int(active["coordinate_usable"].sum()),
        "active_mapped_rows": int(joined["adm_nm"].notna().sum()),
        "active_unmapped_or_bad_coordinate_rows": int(len(active) - joined["adm_nm"].notna().sum()),
        "administrative_dongs_with_daycare": len(aggregate),
    }])
    quality.to_csv(output / "quality_summary.csv", index=False, encoding="utf-8-sig")
    print(quality.to_string(index=False))
    print(f"저장 위치: {output}")


if __name__ == "__main__":
    main()
