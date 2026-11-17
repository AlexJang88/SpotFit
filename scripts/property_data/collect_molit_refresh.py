"""Download Seoul apartment sale and jeonse/wolse transactions from the MOLIT open API (data.go.kr).

  매매   : https://www.data.go.kr/data/15126469/openapi.do  (RTMSDataSvcAptTradeDev)
  전월세 : https://www.data.go.kr/data/15126474/openapi.do  (RTMSDataSvcAptRent)

Key: environment variable DATA_GO_KR_API_KEY (apply for both services with the same account).
Each month is saved as one CSV in data/raw/부동산/refresh/<trade|rent>/YYYYMM.csv exactly as the API returns it.
Months are overwritten on purpose: late reports and cancellations change a month after it is first published.
A month is only written when the whole month was downloaded, so a failure never leaves a half month behind.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from scripts.property_data.common import (
    REFETCH_MONTHS, REFRESH, SEOUL_SGG, last_complete_month, month_index, month_range, month_str, require_env, write_atomic)

SERVICES = {
    "trade": "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev",
    "rent": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
}
PAGE_SIZE = 1000
OK_CODES = {"00", "000"}


class ApiError(RuntimeError):
    pass


def _get(url: str, params: dict, retries: int = 3) -> bytes:
    # The portal hands out the key both URL-encoded and decoded; normalise so either can be pasted.
    params = {**params, "serviceKey": urllib.parse.unquote(params["serviceKey"])}
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "SpotFit-data-pipeline/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries - 1:
                raise ApiError(f"API 호출 실패: {error}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def parse_page(body: bytes) -> tuple[list[dict], int]:
    """Return (rows, totalCount). Raises ApiError for anything but a normal answer (bad key, quota, ...)."""
    root = ET.fromstring(body)
    if root.tag == "OpenAPI_ServiceResponse":                       # gateway-level error (wrong key, quota exceeded, ...)
        raise ApiError(f"게이트웨이 오류: {root.findtext('.//returnAuthMsg')} / {root.findtext('.//returnReasonCode')}")
    code = (root.findtext(".//resultCode") or "").strip()
    if code not in OK_CODES:
        raise ApiError(f"API 오류 {code}: {root.findtext('.//resultMsg')}")
    total = int(root.findtext(".//totalCount") or 0)
    rows = [{child.tag: (child.text or "").strip() for child in item} for item in root.iter("item")]
    return rows, total


def fetch_district_month(kind: str, key: str, sgg: str, ym: str) -> list[dict]:
    rows, page = [], 1
    while True:
        body = _get(SERVICES[kind], {"serviceKey": key, "LAWD_CD": sgg, "DEAL_YMD": ym,
                                     "pageNo": page, "numOfRows": PAGE_SIZE})
        batch, total = parse_page(body)
        rows += batch
        if len(rows) >= total or not batch:
            return rows
        page += 1


def to_csv_bytes(rows: list[dict]) -> bytes:
    columns = sorted({name for row in rows for name in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def collect_month(kind: str, ym: str, key: str, output=REFRESH) -> dict:
    rows, per_district = [], {}
    for sgg in SEOUL_SGG:
        part = fetch_district_month(kind, key, sgg, ym)
        per_district[sgg] = len(part)
        rows += part
    content = to_csv_bytes(rows) if rows else b""
    if rows:
        write_atomic(output / kind / f"{ym}.csv", content)
    return {"kind": kind, "month": ym, "rows": len(rows), "districts": per_district,
            "sha256": hashlib.sha256(content).hexdigest() if rows else None}


def months_to_collect(base_end: str | None, today=None) -> list[str]:
    """Every month after what the raw file already has, plus the latest REFETCH_MONTHS confirmed months."""
    end = last_complete_month(today)
    refetch_start = month_str(month_index(end) - REFETCH_MONTHS + 1)
    start = refetch_start if base_end is None else min(refetch_start, month_str(month_index(base_end) + 1))
    return month_range(start, end) if month_index(start) <= month_index(end) else []


def collect(months: list[str], kinds=("trade", "rent"), output=REFRESH) -> dict:
    key = require_env("DATA_GO_KR_API_KEY")
    records = []
    for kind in kinds:
        for ym in months:
            record = collect_month(kind, ym, key, output)
            if record["rows"] == 0:
                raise ApiError(f"{kind} {ym}: 25개 자치구 모두 거래가 0건입니다. 아직 공개되지 않은 달이거나 활용신청이 승인되지 않았습니다.")
            records.append(record)
            print(f"{kind} {ym}: {record['rows']:,}건", flush=True)
    manifest = {"collected_at": datetime.now(timezone.utc).isoformat(), "records": records}
    write_atomic(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", help="YYYYMM (기본: 확정 월 중 최근 3개월)")
    parser.add_argument("--to", dest="end", help="YYYYMM (기본: 오늘 기준 확정된 마지막 달)")
    args = parser.parse_args()
    end = args.end or last_complete_month()
    months = month_range(args.start, end) if args.start else months_to_collect(None)
    print(json.dumps(collect(months), ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
