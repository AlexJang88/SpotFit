"""Download the monthly 주택담보대출 금리 (예금은행 신규취급액 기준) from the Bank of Korea ECOS API.

Key: environment variable ECOS_API_KEY (free, https://ecos.bok.or.kr/api/ -> 인증키 신청). Separate from DATA_GO_KR_API_KEY.
Statistic 121Y006 = 예금은행 대출금리(신규취급액 기준). The item is checked by name, so a wrong item code can never be used silently.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from scripts.property_data.common import REFRESH, require_env, write_atomic

STAT_CODE = "121Y006"
ITEM_CODE = "BECBLA0302"
ITEM_NAME_PART = "주택담보대출"


def parse(payload: dict) -> dict[str, float]:
    if "StatisticSearch" not in payload:
        raise RuntimeError(f"ECOS 오류: {payload.get('RESULT') or payload}")
    rows = payload["StatisticSearch"]["row"]
    if not rows:
        raise RuntimeError("ECOS 응답에 금리 값이 없습니다.")
    wrong = {row["ITEM_NAME1"] for row in rows if ITEM_NAME_PART not in row["ITEM_NAME1"]}
    if wrong:
        raise RuntimeError(f"ECOS 항목 코드 {ITEM_CODE}가 주택담보대출이 아닙니다: {sorted(wrong)}")
    return {f"{row['TIME'][:4]}.{row['TIME'][4:6]}": float(row["DATA_VALUE"]) for row in rows}


def fetch(start: str, end: str, key: str | None = None) -> dict[str, float]:
    """{'2026.09': 4.88, ...}  (start/end: YYYYMM)"""
    key = key or require_env("ECOS_API_KEY")
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{urllib.parse.quote(key, safe='')}/json/kr/1/200/"
           f"{STAT_CODE}/M/{start}/{end}/{ITEM_CODE}")
    with urllib.request.urlopen(url, timeout=60) as response:
        return parse(json.loads(response.read()))


def collect(start: str, end: str, output=REFRESH) -> dict[str, float]:
    rates = fetch(start, end)
    write_atomic(output / "mortgage_rate.json", json.dumps(rates, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"금리 {len(rates)}개월: {min(rates)} ~ {max(rates)}", flush=True)
    return rates
