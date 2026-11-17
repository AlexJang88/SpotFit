"""Shared paths and helpers for the real-estate (apartment sale / jeonse / wolse) refresh pipeline."""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "부동산"
REFRESH = RAW / "refresh"                    # files downloaded from the APIs (one file per month)
BACKUP = RAW / "_backup"
REPORT = ROOT / "artifacts" / "property_refresh"

SALE_RAW = RAW / "아파트_매매_2010_2026.csv"
RENT_RAW = RAW / "아파트_전월세_2010_2026.csv"

# 서울 25개 자치구 법정동 앞 5자리 (국토교통부 API의 LAWD_CD)
SEOUL_SGG = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구",
    "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구",
    "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구",
    "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구",
}

# 신고 기한이 계약 후 30일이라 M월 거래는 M+1월 말에야 대부분 모입니다. 그래서 오늘 기준 두 달 전까지만 확정 월로 봅니다.
LAG_MONTHS = 2
REFETCH_MONTHS = 3          # 늦게 신고된 건·취소 건을 반영하려고 확정 월 중 최근 3개월은 매번 다시 받습니다.


def month_index(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[4:6]) - 1


def month_str(index: int) -> str:
    return f"{index // 12:04d}{index % 12 + 1:02d}"


def last_complete_month(today: date | None = None, lag: int = LAG_MONTHS) -> str:
    today = today or date.today()
    return month_str(today.year * 12 + today.month - 1 - lag)


def month_range(start: str, end: str) -> list[str]:
    return [month_str(i) for i in range(month_index(start), month_index(end) + 1)]


def month_end_date(ym: str) -> str:
    nxt = month_index(ym) + 1
    last = date(nxt // 12, nxt % 12 + 1, 1).toordinal() - 1
    return date.fromordinal(last).isoformat()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"환경변수 {name}가 비어 있습니다. .env.server(또는 .env.dev)에 인증키를 넣어 주세요.")
    return value


def raw_end_months() -> dict[str, str]:
    """Last month (YYYYMM) of the sale file, the jeonse/wolse file and the 주택담보대출 금리 file. Standard library only."""
    import csv
    import glob

    out = {}
    for name, path in (("sale", SALE_RAW), ("rent", RENT_RAW)):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            column = next(reader).index("계약일")
            out[name] = max(row[column][:7] for row in reader).replace("-", "")
    files = sorted(glob.glob(str(RAW / "예금은행_대출금리_신규취급액_기준__*.csv")))
    with open(files[0], encoding="cp949", newline="") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    row = next(r for r in rows[1:] if r[0].strip() == "주택담보대출 (연리%)")
    valid = [month for month, value in zip(header[1:], row[1:]) if value.strip() not in ("", "-")]
    out["rate"] = max(valid).replace(".", "")
    return out


def write_report(name: str, payload: dict) -> None:
    import json
    from datetime import datetime, timezone
    payload = {**payload, "written_at": datetime.now(timezone.utc).isoformat()}
    write_atomic(REPORT / name, json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def write_atomic(path: Path, data: bytes) -> None:
    """Write next to the target and rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
