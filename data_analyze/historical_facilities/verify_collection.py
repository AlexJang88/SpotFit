from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "processed" / "historical_facilities"
RECONSTRUCTED = BASE / "reconstructed_2021_2025"
OFFICIAL = BASE / "official_2021_2025"
RAW = ROOT / "data" / "raw" / "5년치_시설"
YEARS = list(range(2021, 2026))

RECONSTRUCTED_CATEGORIES = [
    "cctv",
    "model_restaurants",
    "hospitals",
    "passage_inconvenience_surveys",
    "parks",
    "school_zones",
    "schools",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_annual_csvs(directory: Path, allow_empty: bool = False) -> None:
    for year in YEARS:
        path = directory / f"{year}.csv"
        assert path.is_file(), f"누락 파일: {path}"
        rows = csv_rows(path)
        if not allow_empty:
            assert rows, f"빈 파일: {path}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    for category in RECONSTRUCTED_CATEGORIES:
        verify_annual_csvs(RECONSTRUCTED / category, allow_empty=category == "passage_inconvenience_surveys")

    verify_annual_csvs(OFFICIAL / "child_centers")
    child_summary = csv_rows(OFFICIAL / "child_centers" / "summary.csv")
    assert [int(row["year"]) for row in child_summary] == YEARS
    assert all(int(row["national_centers"]) > 0 for row in child_summary)

    tourism_rows = csv_rows(OFFICIAL / "tourism_visitors" / "seoul_2021_2025_long.csv")
    assert sorted({int(row["year"]) for row in tourism_rows}) == YEARS
    assert all(row["spot_code"] and row["spot_name"] for row in tourism_rows)
    tourism_summary = csv_rows(OFFICIAL / "tourism_visitors" / "summary.csv")
    assert all(int(row["reporting_spots"]) > 0 for row in tourism_summary)

    for year in YEARS:
        child_file = next((RAW / "지역아동센터").glob(f"{year}_*.xlsx"))
        assert zipfile.is_zipfile(child_file), f"유효하지 않은 XLSX: {child_file}"
        tourism_pdf = next((RAW / "관광지점입장객_대체지표").glob(f"{year}_*.pdf"))
        assert tourism_pdf.read_bytes()[:5] == b"%PDF-", f"유효하지 않은 PDF: {tourism_pdf}"

    manifest = csv_rows(Path(__file__).with_name("official_download_manifest.csv"))
    for item in manifest:
        path = ROOT / item["relative_path"]
        assert path.is_file(), f"매니페스트 파일 누락: {path}"
        assert path.stat().st_size == int(item["bytes"]), f"크기 불일치: {path}"
        assert sha256(path) == item["sha256"], f"해시 불일치: {path}"

    source_manifest = csv_rows(Path(__file__).with_name("source_manifest.csv"))
    inventory = csv_rows(Path(__file__).with_name("collection_inventory.csv"))
    assert len(source_manifest) == 9, "출처 매니페스트는 9개 카테고리를 포함해야 합니다."
    assert len(inventory) == 9, "수집 인벤토리는 9개 카테고리를 포함해야 합니다."
    assert {row["category"] for row in source_manifest} == {row["category"] for row in inventory}

    print("검증 완료: 9개 카테고리, 2021~2025년 자료와 출처 매니페스트가 모두 존재합니다.")


if __name__ == "__main__":
    main()
