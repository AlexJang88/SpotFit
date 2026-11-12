"""MDIS 국내인구이동 원자료를 서울 행정동·연령대별로 집계한다.

입력 한 행은 전입 세대이며 세대원은 최대 10명이다. 행을 세대 1명으로 세면 안 되고,
각 세대원의 만나이를 읽어 전입지와 전출지에 각각 한 명씩 더한다.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/raw/아동/인구이동_행정동_연령별"
DEFAULT_POPULATION = ROOT / "data/raw/아동/아동 데이터/핵심"
DEFAULT_OUTPUT = ROOT / "data/processed/child_population/admin_dong_age_migration"

AGE_BANDS = (
    (0, 5, "preschool", "0-5세"),
    (6, 11, "elementary", "6-11세"),
    (12, 14, "middle", "12-14세"),
    (15, 17, "high", "15-17세"),
)


def age_band(age: int) -> tuple[str, str] | None:
    for low, high, key, label in AGE_BANDS:
        if low <= age <= high:
            return key, label
    return None


def code(parts: list[bytes], start: int) -> str:
    return b"".join(parts[start : start + 3]).decode("ascii").zfill(10)


def aggregate_file(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    monthly: dict[tuple[int, int, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    rows = people = child_people = malformed = 0

    with path.open("rb") as source:
        for raw_line in source:
            rows += 1
            fields = raw_line.rstrip(b"\r\n").split(b",")
            if len(fields) < 41:
                malformed += 1
                continue
            try:
                year = int(fields[3])
                month = int(fields[4])
                destination = code(fields, 0)
                origin = code(fields, 6)
            except (ValueError, UnicodeDecodeError):
                malformed += 1
                continue

            destination_seoul = destination.startswith("11")
            origin_seoul = origin.startswith("11")
            if not (destination_seoul or origin_seoul):
                continue

            for person_start in range(10, 40, 3):
                raw_age = fields[person_start + 1]
                if not raw_age:
                    continue
                people += 1
                try:
                    band = age_band(int(raw_age))
                except ValueError:
                    continue
                if band is None:
                    continue
                child_people += 1
                band_key, band_label = band
                if destination_seoul:
                    monthly[(year, month, destination, band_key, band_label)][0] += 1
                if origin_seoul:
                    monthly[(year, month, origin, band_key, band_label)][1] += 1

    records = []
    for (year, month, dong_code, band_key, band_label), (inflow, outflow) in monthly.items():
        records.append(
            {
                "year": year,
                "month": month,
                "dong_code": dong_code,
                "age_group": band_key,
                "age_band": band_label,
                "inflow": inflow,
                "outflow": outflow,
                "net_migration": inflow - outflow,
            }
        )
    audit = {
        "source_rows": rows,
        "seoul_people_all_ages": people,
        "seoul_child_people_0_17": child_people,
        "malformed_rows": malformed,
    }
    return pd.DataFrame.from_records(records), audit


def load_population(population_dir: Path) -> pd.DataFrame:
    frames = []
    code_pattern = re.compile(r"\((\d{10})\)")
    for path in sorted(population_dir.glob("20*연령별인구현황19.csv")):
        match = re.match(r"(20\d{2})", path.name)
        if not match:
            continue
        year = int(match.group(1))
        frame = pd.read_csv(path, encoding="cp949", dtype=str)
        area = frame.iloc[:, 0].fillna("")
        frame["dong_code"] = area.str.extract(code_pattern, expand=False)
        frame = frame[
            frame["dong_code"].str.startswith("11", na=False)
            & ~frame["dong_code"].str.endswith("00000", na=False)
        ].copy()
        frame["area_name"] = area.str.replace(r"\s*\(\d{10}\)\s*$", "", regex=True).str.strip()
        name_parts = frame["area_name"].str.split()
        frame["gu"] = name_parts.str[1]
        frame["dong"] = name_parts.str[-1]
        for low, high, key, label in AGE_BANDS:
            columns = [f"{year}년_계_{age}세" for age in range(low, high + 1)]
            values = frame[columns].apply(
                lambda column: pd.to_numeric(column.str.replace(",", "", regex=False), errors="coerce")
            )
            part = frame[["dong_code", "gu", "dong"]].copy()
            part["year"] = year
            part["age_group"] = key
            part["age_band"] = label
            part["age_population_year_end"] = values.sum(axis=1, min_count=1)
            frames.append(part)
    if not frames:
        raise FileNotFoundError(f"연령별 행정동 인구 파일을 찾지 못했습니다: {population_dir}")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["year", "dong_code", "age_group"]).any():
        raise ValueError("행정동 인구에 연도·행정동·연령대 중복이 있습니다.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--population-dir", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files = sorted(args.raw_dir.rglob("20*_인구관련연간자료_*.csv"))
    if not files:
        raise FileNotFoundError(f"MDIS CSV를 찾지 못했습니다: {args.raw_dir}")

    monthly_frames = []
    audits = []
    for path in files:
        print(f"집계 중: {path.name}", flush=True)
        monthly, audit = aggregate_file(path)
        audit["file"] = path.name
        audits.append(audit)
        monthly_frames.append(monthly)

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = monthly.groupby(
        ["year", "month", "dong_code", "age_group", "age_band"], as_index=False
    )[["inflow", "outflow", "net_migration"]].sum()
    annual = monthly.groupby(
        ["year", "dong_code", "age_group", "age_band"], as_index=False
    )[["inflow", "outflow", "net_migration"]].sum()

    population = load_population(args.population_dir)
    annual = annual.merge(
        population,
        on=["year", "dong_code", "age_group", "age_band"],
        how="left",
        validate="one_to_one",
    )
    denominator = annual["age_population_year_end"].where(annual["age_population_year_end"] > 0)
    annual["inflow_per_1000_year_end_population"] = annual["inflow"] / denominator * 1000
    annual["outflow_per_1000_year_end_population"] = annual["outflow"] / denominator * 1000
    annual["net_migration_per_1000_year_end_population"] = annual["net_migration"] / denominator * 1000

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.sort_values(["year", "month", "dong_code", "age_group"]).to_csv(
        args.output_dir / "admin_dong_age_migration_monthly_2021_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annual.sort_values(["year", "dong_code", "age_group"]).to_csv(
        args.output_dir / "admin_dong_age_migration_annual_2021_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(audits).to_csv(args.output_dir / "source_audit.csv", index=False, encoding="utf-8-sig")

    coverage = annual.groupby("year").agg(
        dong_codes=("dong_code", "nunique"),
        rows=("dong_code", "size"),
        population_matched=("age_population_year_end", "count"),
        inflow=("inflow", "sum"),
        outflow=("outflow", "sum"),
    )
    coverage.to_csv(args.output_dir / "coverage_by_year.csv", encoding="utf-8-sig")
    print(coverage.to_string(), flush=True)
    print(f"저장 완료: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
