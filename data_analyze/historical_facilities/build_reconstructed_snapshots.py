from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "raw" / "1년치만 있는 데이터"
OUTPUT_DIR = ROOT / "data" / "processed" / "historical_facilities" / "reconstructed_2021_2025"
YEARS = range(2021, 2026)


def read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    path = SOURCE_DIR / name
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                return list(reader.fieldnames or []), rows
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"지원 인코딩으로 읽을 수 없습니다: {path}")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    if not value or value in {"-", "0", "000000", "00000000"}:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    formats = []
    if len(digits) >= 8:
        formats.append((digits[:8], "%Y%m%d"))
    elif len(digits) == 6:
        formats.append((digits + "01", "%Y%m%d"))
    elif len(digits) == 4:
        formats.append((digits + "0101", "%Y%m%d"))
    for candidate, fmt in formats:
        try:
            parsed = datetime.strptime(candidate, fmt).date()
            if 1900 <= parsed.year <= 2100:
                return parsed
        except ValueError:
            pass
    return None


def earliest(*values: date | None) -> date | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def write_snapshot(
    category: str,
    year: int,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    basis: str,
    quality_flag: str,
) -> Path:
    target_dir = OUTPUT_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{year}.csv"
    extra = ["snapshot_year", "reconstruction_basis", "quality_flag"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "snapshot_year": year,
                    "reconstruction_basis": basis,
                    "quality_flag": quality_flag,
                }
            )
    return path


def build_cctv() -> list[dict[str, object]]:
    fields, rows = read_csv("서울시_CCTV정보.csv")
    summary = []
    for year in YEARS:
        cutoff = date(year, 12, 31)
        selected = [row for row in rows if (parse_date(row.get("설치연월")) or date.max) <= cutoff]
        write_snapshot(
            "cctv",
            year,
            fields,
            selected,
            "현재 목록의 설치연월이 연말 이전인 기록",
            "reconstructed_current_survivors; removals_unknown",
        )
        camera_count = sum(int(float(row.get("카메라대수") or 0)) for row in selected)
        summary.append({"category": "cctv", "year": year, "rows": len(selected), "metric_value": camera_count, "metric": "카메라대수합"})
    return summary


def restaurant_active(row: dict[str, str], cutoff: date) -> bool:
    designated = parse_date(row.get("지정일자"))
    redesignated = parse_date(row.get("재지정일자"))
    started = earliest(designated, redesignated)
    if started is None or started > cutoff:
        return False
    cancelled = earliest(parse_date(row.get("지정취소일자")), parse_date(row.get("폐업일자")))
    if cancelled is None or cancelled > cutoff:
        return True
    return redesignated is not None and cancelled < redesignated <= cutoff


def build_restaurants() -> list[dict[str, object]]:
    fields, rows = read_csv("서울시_모범음식점정보.csv")
    summary = []
    for year in YEARS:
        cutoff = date(year, 12, 31)
        selected = [row for row in rows if restaurant_active(row, cutoff)]
        write_snapshot(
            "model_restaurants",
            year,
            fields,
            selected,
            "지정일/재지정일과 지정취소일/폐업일로 연말 상태 재구성",
            "reconstructed_event_dates; malformed_dates_excluded; designation_intervals_approximate",
        )
        summary.append({"category": "model_restaurants", "year": year, "rows": len(selected), "metric_value": len(selected), "metric": "업소수"})
    return summary


def hospital_active(row: dict[str, str], cutoff: date) -> bool:
    opened = parse_date(row.get("인허가일자"))
    if opened is None or opened > cutoff:
        return False
    ended = earliest(parse_date(row.get("인허가취소일자")), parse_date(row.get("폐업일자")))
    if ended is not None and ended <= cutoff:
        return False
    pause_start = parse_date(row.get("휴업시작일자"))
    pause_end = parse_date(row.get("휴업종료일자"))
    if pause_start is not None and pause_start <= cutoff and (pause_end is None or pause_end > cutoff):
        return False
    return True


def build_hospitals() -> list[dict[str, object]]:
    fields, rows = read_csv("서울시_병원.csv")
    summary = []
    for year in YEARS:
        cutoff = date(year, 12, 31)
        selected = [row for row in rows if hospital_active(row, cutoff)]
        write_snapshot(
            "hospitals",
            year,
            fields,
            selected,
            "인허가일과 취소/폐업/휴업일로 연말 운영 상태 재구성",
            "reconstructed_event_dates; source_is_hospital_licenses_not_all_clinics",
        )
        summary.append({"category": "hospitals", "year": year, "rows": len(selected), "metric_value": len(selected), "metric": "병원수"})
    return summary


def build_current_survivor_history(
    filename: str,
    category: str,
    start_column: str,
    basis: str,
) -> list[dict[str, object]]:
    fields, rows = read_csv(filename)
    summary = []
    for year in YEARS:
        cutoff = date(year, 12, 31)
        selected = [row for row in rows if (parse_date(row.get(start_column)) or date.max) <= cutoff]
        write_snapshot(
            category,
            year,
            fields,
            selected,
            basis,
            "reconstructed_current_survivors; closures_or_boundary_changes_unknown",
        )
        summary.append({"category": category, "year": year, "rows": len(selected), "metric_value": len(selected), "metric": "기록수"})
    return summary


def build_survey_records() -> list[dict[str, object]]:
    fields, rows = read_csv("서울시_통행불편지역.csv")
    summary = []
    for year in YEARS:
        selected = [row for row in rows if (parsed := parse_date(row.get("조사일자"))) is not None and parsed.year == year]
        write_snapshot(
            "passage_inconvenience_surveys",
            year,
            fields,
            selected,
            "조사일자가 해당 연도인 기록",
            "annual_survey_records_not_year_end_inventory; invalid_dates_excluded",
        )
        summary.append({"category": "passage_inconvenience_surveys", "year": year, "rows": len(selected), "metric_value": len(selected), "metric": "조사기록수"})
    return summary


def write_summary(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "year", "rows", "metric_value", "metric"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summary: list[dict[str, object]] = []
    summary += build_cctv()
    summary += build_restaurants()
    summary += build_hospitals()
    summary += build_current_survivor_history(
        "전국도시공원정보표준데이터.csv",
        "parks",
        "지정고시일",
        "현재 공원 목록 중 지정고시일이 연말 이전인 기록",
    )
    summary += build_current_survivor_history(
        "전국초중등학교위치표준데이터.csv",
        "schools",
        "설립일자",
        "현재 운영 학교 중 설립일자가 연말 이전인 기록",
    )
    summary += build_current_survivor_history(
        "전국초등학교통학구역표준데이터.csv",
        "school_zones",
        "생성일자",
        "현재 통학구역 목록 중 생성일자가 연말 이전인 기록",
    )
    summary += build_survey_records()
    write_summary(summary)
    print(f"생성 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
