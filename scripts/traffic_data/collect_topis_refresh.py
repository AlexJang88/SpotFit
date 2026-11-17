"""Collect TOPIS monthly traffic workbooks without changing the v94 snapshot.

This is an ingestion stage for a future annual traffic-model release. A partial
year is expected; completeness is checked separately before training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://topis.seoul.go.kr"
RAW_ROOT = ROOT / "data" / "raw" / "traffic_model_refresh" / "topis"
MONTH_PATTERN = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])\s*월")


def post(path: str, fields: dict[str, str]) -> bytes:
    request = urllib.request.Request(
        BASE_URL + path,
        data=urllib.parse.urlencode(fields).encode("utf-8"),
        headers={"User-Agent": "SpotFit-data-pipeline/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def monthly_entries(year: int) -> dict[int, dict]:
    payload = json.loads(post("/refroom/selectRefRoomListASC.do", {
        "blbdDivCd": "08", "bdwrDivCd": str(year), "mainBdwrRowNum": "24",
    }))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("TOPIS listing did not contain a rows array")
    entries: dict[int, dict] = {}
    for row in rows:
        filename = str(row.get("apndFileNm") or "")
        match = MONTH_PATTERN.search(filename)
        if not match:
            continue
        month = int(match.group(1))
        if month in entries:
            raise ValueError(f"TOPIS published more than one workbook for {year}-{month:02d}")
        if not filename.lower().endswith(".xlsx"):
            raise ValueError(f"TOPIS month {month} is not an XLSX workbook: {filename}")
        entries[month] = row
    return entries


def _write_new(path: Path, content: bytes) -> None:
    """Never replace an existing raw file, even when TOPIS revises a month."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"TOPIS revised an existing raw month; review manually: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        # Another run may have completed while the download was in progress.
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"TOPIS raw month changed concurrently: {path}")
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(year: int, output_root: Path = RAW_ROOT) -> dict:
    if not 2017 <= year <= 2100:
        raise ValueError("year must be between 2017 and 2100")
    entries = monthly_entries(year)
    year_dir = output_root / str(year)
    records = []
    for month, entry in sorted(entries.items()):
        fields = {key: str(entry[key]) for key in
                  ("apndFileNm", "apndFilePathNm", "bdwrSeq", "blbdDivCd")}
        content = post("/downloadFileRefRoom.do", fields)
        if not content.startswith(b"PK") or not zipfile.is_zipfile(io.BytesIO(content)):
            raise ValueError(f"TOPIS {year}-{month:02d} response is not an XLSX file")
        filename = f"{month:02d}_topis_traffic_{year}.xlsx"
        path = year_dir / filename
        _write_new(path, content)
        records.append({
            "month": month, "file": filename, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source_name": str(entry["apndFileNm"]),
        })
    result = {
        "source": BASE_URL + "/refRoom/openRefRoom_2.do",
        "year": year, "months_collected": len(records),
        "complete_year": len(records) == 12,
        "records": records,
    }
    year_dir.mkdir(parents=True, exist_ok=True)
    manifest = year_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=year_dir,
                                     suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2)
    try:
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.year), ensure_ascii=False))


if __name__ == "__main__":
    main()
