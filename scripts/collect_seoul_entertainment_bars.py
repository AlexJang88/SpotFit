"""Collect Seoul entertainment-bar licences from the MOIS public API."""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://apis.data.go.kr/1741000/entertainment_bars/"
OUT = ROOT / "data" / "raw" / "아동범죄" / "전국 유흥주점영업 인허가 정보.csv"
META = ROOT / "data" / "raw" / "아동범죄" / "전국 유흥주점영업 인허가 정보.meta.json"


def first(row: dict, *names: str) -> str:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None:
            return str(value).strip()
    return ""


def fetch(key: str, page: int, size: int) -> tuple[list[dict], int]:
    params = {
        "serviceKey": key,
        "pageNo": page,
        "numOfRows": size,
        "type": "json",
        "_type": "json",
    }
    req = Request(
        f"{API_URL}?{urlencode(params, safe='%')}",
        headers={"User-Agent": "SpotFit/1.0", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=120) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API request failed on page {page}: {exc}; body={detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"API request failed on page {page}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        preview = raw[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"Unexpected API response: {preview}") from exc

    header = payload.get("response", {}).get("header", {})
    code = str(header.get("resultCode", "00"))
    if code not in {"00", "0", "NORMAL_SERVICE"}:
        raise RuntimeError(f"API error: {header}")

    body = payload.get("response", {}).get("body", payload)
    items = body.get("items", []) if isinstance(body, dict) else []
    if isinstance(items, dict):
        items = items.get("item", items.get("items", []))
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise RuntimeError(f"Unknown items structure: {type(items).__name__}")
    total = int(body.get("totalCount", len(items))) if isinstance(body, dict) else len(items)
    return items, total


def is_seoul(row: dict) -> bool:
    text = " ".join(str(value or "") for value in row.values())
    return "서울" in text


def permit_year(row: dict) -> int | None:
    value = first(
        row,
        "apvPermYmd", "APVPERMYMD", "permissionDate", "permitDate",
        "인허가일자", "인허가일", "허가일자",
    ).replace("-", "").replace(".", "")
    return int(value[:4]) if len(value) >= 4 and value[:4].isdigit() else None


def main() -> int:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.dev", override=True)
    key = os.getenv("DATA_GO_KR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DATA_GO_KR_API_KEY is missing")

    page_size = 1000
    rows: list[dict] = []
    page = 1
    total = None
    while total is None or len(rows) < total:
        batch, total = fetch(key, page, page_size)
        rows.extend(batch)
        print(f"page={page}, batch={len(batch):,}, accumulated={len(rows):,}, total={total:,}")
        if not batch:
            break
        page += 1
        time.sleep(0.1)

    selected = rows

    # Keep the provider's complete schema and add explicit provenance fields.
    fields = sorted({str(key) for row in selected for key in row})
    prefix = ["수집기준_시작연도", "수집기준_종료연도", "원천_API", "수집일시"]
    collected_at = datetime.now().astimezone().isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*prefix, *fields], extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "수집기준_시작연도": 2010,
                "수집기준_종료연도": 2025,
                "원천_API": API_URL,
                "수집일시": collected_at,
                **row,
            })

    metadata = {
        "source": API_URL,
        "collected_at": collected_at,
        "filter": "none; complete nationwide API response",
        "nationwide_rows": len(rows),
        "output_rows": len(selected),
        "columns": [*prefix, *fields],
        "output": str(OUT),
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
