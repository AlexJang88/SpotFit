"""2015~2024 서울 아파트 실거래와 학교별 학생·학급 원자료 수집기.

공식 웹 서비스가 공개하는 응답을 수정하지 않고 data/raw에 보존한다. 같은 파일이
이미 있으면 기본적으로 다시 내려받지 않으며 SHA-256 manifest를 작성한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
APARTMENT_OUT = ROOT / "data/raw/부동산/국토부/아파트_실거래_2015_2024"
SCHOOL_OUT = ROOT / "data/raw/school/학교알리미_학생학급_2015_2024"
YEARS = range(2015, 2025)
SCHOOL_KINDS = {"02": "elementary", "03": "middle", "04": "high"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_with_retry(session: requests.Session, method: str, url: str, *, retries: int = 3, **kwargs):
    for attempt in range(retries):
        try:
            response = session.request(method, url, timeout=kwargs.pop("timeout", 180), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def collect_apartments(force: bool) -> list[dict]:
    """국토교통부 실거래가 공개시스템의 서울 아파트 매매 CSV를 연도별 보존한다."""
    APARTMENT_OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "SpotFit-historical-data-collector/1.0"})
    landing = request_with_retry(session, "GET", "https://rt.molit.go.kr/pt/xls/xls.do", timeout=30)
    session_ids = re.findall(r"jsessionid=([A-Za-z0-9_.-]+)", landing.text)
    if not session_ids:
        raise RuntimeError("국토부 자료제공 페이지에서 세션 식별자를 찾지 못했습니다.")
    session_id = session_ids[0]
    records: list[dict] = []
    for year in YEARS:
        target = APARTMENT_OUT / f"seoul_apartment_sale_{year}.csv"
        if not target.exists() or force:
            params = {
                "srhThingNo": "A", "srhDelngSecd": "1", "srhAddrGbn": "1", "srhLfstsSecd": "1",
                "srhFromDt": f"{year}-01-01", "srhToDt": f"{year}-12-31",
                "srhSidoCd": "11000", "srhSggCd": "", "srhEmdCd": "", "srhHsmpCd": "",
                "srhArea": "", "srhFromAmount": "", "srhToAmount": "",
                "sidoNm": "서울특별시", "sggNm": "전체", "emdNm": "전체", "hsmpNm": "전체",
                "areaNm": "전체", "loadNm": "전체", "mobileAt": "",
            }
            check = request_with_retry(
                session, "POST", f"https://rt.molit.go.kr/pt/xls/ptXlsDownDataCheck.do;jsessionid={session_id}",
                data=params, timeout=60).json()
            expected_rows = int(check.get("cnt", 0))
            if expected_rows <= 0:
                raise RuntimeError(f"{year}년 아파트 실거래 조회 건수가 0입니다.")
            response = request_with_retry(
                session, "POST", f"https://rt.molit.go.kr/pt/xls/ptXlsCSVDown.do;jsessionid={session_id}",
                data=params, timeout=300)
            if len(response.content) < 100 or response.content.lstrip().startswith(b"<"):
                raise RuntimeError(f"{year}년 아파트 CSV 응답 형식이 올바르지 않습니다.")
            decoded = response.content.decode("cp949")
            lines = decoded.splitlines()
            required_metadata = (f"계약일자 : {year}-01-01 ~ {year}-12-31", "실거래구분 : 아파트(매매)")
            if not all(any(value in line for line in lines) for value in required_metadata):
                raise RuntimeError(f"{year}년 아파트 CSV 내부 검색조건이 요청과 다릅니다.")
            header_index = next((index for index, line in enumerate(lines) if "거래금액(만원)" in line), None)
            if header_index is None or header_index + 1 >= len(lines) or "서울특별시" not in lines[header_index + 1]:
                raise RuntimeError(f"{year}년 아파트 CSV가 서울 거래자료가 아닙니다.")
            actual_rows = len(lines) - header_index - 1
            target.write_bytes(response.content)
            status = "collected"
        else:
            expected_rows = None
            decoded = target.read_bytes().decode("cp949")
            lines = decoded.splitlines()
            header_index = next(index for index, line in enumerate(lines) if "거래금액(만원)" in line)
            actual_rows = len(lines) - header_index - 1
            status = "existing"
        records.append({
            "dataset": "molit_apartment_sale", "year": year, "school_kind": "",
            "status": status, "reported_rows": actual_rows, "bytes": target.stat().st_size,
            "sha256": sha256(target), "path": str(target.relative_to(ROOT)),
            "source": "https://rt.molit.go.kr/pt/xls/xls.do",
        })
        print(f"[아파트] {year}: {status}, {target.stat().st_size:,} bytes")
    return records


def collect_schools(force: bool) -> list[dict]:
    """학교알리미 학년별·학급별 학생수 공개 API 응답을 학교급·연도별 보존한다."""
    SCHOOL_OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "SpotFit-historical-data-collector/1.0"})
    records: list[dict] = []
    for year in YEARS:
        for code, label in SCHOOL_KINDS.items():
            target = SCHOOL_OUT / f"schoolinfo_students_classes_{year}_{label}.json"
            if not target.exists() or force:
                response = request_with_retry(session, "POST", "https://www.schoolinfo.go.kr/openData.do", data={
                    "APIKEY": "schoolinfo2020", "APITYPE": "09", "DEPTHNO": "0",
                    "SCHULKNDCODE": code, "PBANYR": str(year), "LCTNSCCODE": "01",
                }, timeout=120)
                payload = response.json()
                rows = payload.get("list")
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError(f"학교알리미 {year} {label} 응답에 자료가 없습니다: {payload.get('resultMsg')}")
                document = {
                    "metadata": {
                        "title": "학교알리미 학년별·학급별 학생수", "year": year,
                        "school_kind_code": code, "school_kind": label, "education_office_code": "01",
                        "source": "https://www.schoolinfo.go.kr/ng/go/pnnggo_a01_l2.do",
                        "collected_at": datetime.now().astimezone().isoformat(),
                    },
                    "rows": rows,
                }
                target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
                status = "collected"
            else:
                document = json.loads(target.read_text(encoding="utf-8-sig"))
                rows = document["rows"]
                status = "existing"
            records.append({
                "dataset": "schoolinfo_students_classes", "year": year, "school_kind": label,
                "status": status, "reported_rows": len(rows), "bytes": target.stat().st_size,
                "sha256": sha256(target), "path": str(target.relative_to(ROOT)),
                "source": "https://www.schoolinfo.go.kr/ng/go/pnnggo_a01_l2.do",
            })
            print(f"[학교] {year} {label}: {status}, {len(rows):,} rows")
    return records


def write_manifest(records: list[dict]) -> Path:
    target = ROOT / "data/raw/historical_apartment_school_manifest.csv"
    fields = ["dataset", "year", "school_kind", "status", "reported_rows", "bytes", "sha256", "path", "source"]
    merged: dict[tuple[str, str, str], dict] = {}
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                merged[(row["dataset"], row["year"], row["school_kind"])] = row
    for row in records:
        merged[(str(row["dataset"]), str(row["year"]), str(row["school_kind"]))] = row
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged.values())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "apartment", "school"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    records: list[dict] = []
    if args.only in ("all", "apartment"):
        records.extend(collect_apartments(args.force))
    if args.only in ("all", "school"):
        records.extend(collect_schools(args.force))
    manifest = write_manifest(records)
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
