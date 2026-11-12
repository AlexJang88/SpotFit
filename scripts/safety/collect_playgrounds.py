"""공공데이터포털 전국 어린이놀이시설 중 서울 자료를 수집한다.

필수 환경변수: DATA_GO_KR_API_KEY
공식 명세: https://www.data.go.kr/data/15124519/openapi.do
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://apis.data.go.kr/1741000/pfc3/getPfctInfo3"
SOURCE_URL = "https://www.data.go.kr/data/15124519/openapi.do"
DEFAULT_OUT = ROOT / "data" / "raw" / "safety" / "playgrounds"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_key() -> str:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.dev", override=True)
    key = os.getenv("DATA_GO_KR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DATA_GO_KR_API_KEY가 .env 또는 .env.dev에 없습니다.")
    return key


def fetch_page(key: str, page: int, page_size: int, timeout: int) -> dict:
    query = urlencode(
        {
            "serviceKey": key,
            "pageIndex": page,
            "recordCountPerPage": page_size,
        },
        safe="%",
    )
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": "SpotFit/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8-sig"))
    except HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "HTTP 403: API 키는 존재하지만 이 서비스 이용 권한이 없습니다. "
                "공공데이터포털 15124519에서 활용신청 후 승인 상태를 확인하세요."
            ) from exc
        raise RuntimeError(f"HTTP {exc.code}: 어린이놀이시설 API 호출 실패") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"어린이놀이시설 API 연결 실패: {exc.reason}") from exc


def is_seoul(row: dict) -> bool:
    address = " ".join(
        str(row.get(field) or "")
        for field in ("rgnCdNm", "ronaAddr", "ronaDaddr", "lotnoAddr", "lotnoDaddr")
    )
    return "서울" in address


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(output_root: Path, page_size: int, timeout: int, delay: float) -> Path:
    key = load_key()
    collected_at = datetime.now().astimezone()
    output_dir = output_root / collected_at.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        payload = fetch_page(key, page, page_size, timeout)
        header = payload.get("header") or {}
        if str(header.get("resultCode", "0")) not in {"0", "00"}:
            raise RuntimeError(
                f"API 오류 {header.get('resultCode')}: {header.get('resultMsg', '메시지 없음')}"
            )
        body = payload.get("body") or {}
        items = body.get("items") or []
        if isinstance(items, dict):
            items = items.get("item") or []
        all_rows.extend(items)
        total_pages = int(body.get("totalPageCnt") or 1)
        print(f"page {page}/{total_pages}: {len(items)} rows")
        page += 1
        if page <= total_pages and delay:
            time.sleep(delay)

    seoul_rows = [row for row in all_rows if is_seoul(row)]
    raw_path = output_dir / "playgrounds_nationwide.json"
    seoul_json_path = output_dir / "playgrounds_seoul.json"
    seoul_csv_path = output_dir / "playgrounds_seoul.csv"
    raw_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    seoul_json_path.write_text(
        json.dumps(seoul_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(seoul_csv_path, seoul_rows)

    manifest = {
        "dataset": "행정안전부_전국어린이놀이시설정보서비스",
        "source": SOURCE_URL,
        "api_endpoint": API_URL,
        "collected_at": collected_at.isoformat(),
        "nationwide_rows": len(all_rows),
        "seoul_rows": len(seoul_rows),
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (raw_path, seoul_json_path, seoul_csv_path)
        },
        "notes": "좌표 필드명은 원천 명세와 실제 값의 위도/경도 방향을 별도 검증할 것",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output_dir} (서울 {len(seoul_rows):,}건)")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()
    try:
        collect(args.output, args.page_size, args.timeout, args.delay)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
