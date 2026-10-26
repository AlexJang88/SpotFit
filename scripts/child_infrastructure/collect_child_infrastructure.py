"""서울 육아 인프라 원자료를 재현 가능하게 수집하는 명령행 도구.

API 응답은 수정하지 않고 날짜별 raw 폴더에 저장한다. 같은 날짜의 파일이 이미 있으면
기본적으로 덮어쓰지 않아, 팀원이 다시 실행해도 원본 증거가 보존된다.
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
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "raw" / "child_infrastructure_collected"

# 서울 열린데이터광장 Sheet API 중 서비스명이 공식 명세에서 확인된 데이터입니다.
# 새 데이터는 추측한 서비스명을 코드에 넣지 말고, 공식 명세 확인 후 이 목록에 추가합니다.
SEOUL_DATASETS = {
    "daycare": {
        "service": "ChildCareInfo",
        "title": "서울시 어린이집 정보(표준 데이터)",
        "source": "https://data.seoul.go.kr/dataList/OA-20300/S/1/datasetView.do",
        "domain": "childcare",
        "snapshot_only": True,
    },
    "kindergarten": {
        "service": "childSchoolInfo",
        "title": "서울시 유치원 일반현황",
        "source": "https://data.seoul.go.kr/dataList/OA-20566/S/1/datasetView.do",
        "domain": "education",
        "snapshot_only": True,
    },
    "public_library": {
        "service": "SeoulPublicLibraryInfo",
        "title": "서울시 공공도서관 현황정보",
        "source": "https://data.seoul.go.kr/dataList/OA-15480/A/1/datasetView.do",
        "domain": "culture",
        "snapshot_only": True,
    },
    "sports_program": {
        "service": "program",
        "title": "서울시 생활체육포털 우리동네 프로그램",
        "source": "https://data.seoul.go.kr/dataList/OA-21780/A/1/datasetView.do",
        "domain": "sports",
        "snapshot_only": True,
    },
    "air_station": {
        "service": "airPolutionMeasuringPlace",
        "title": "서울시 대기오염 측정소 정보",
        "source": "https://data.seoul.go.kr/dataList/OA-15516/A/1/datasetView.do",
        "domain": "environment",
        "snapshot_only": True,
    },
    "apartment": {
        "service": "OpenAptInfo",
        "title": "서울시 공동주택 아파트 정보",
        "source": "https://data.seoul.go.kr/dataList/OA-15818/A/1/datasetView.do",
        "domain": "housing",
        "snapshot_only": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, dataset_name: str, retries: int = 3) -> dict:
    """일시적인 오류를 재시도하며, 예외에 인증키가 든 URL을 절대 포함하지 않는다."""
    request = Request(url, headers={"User-Agent": "SpotFit-child-infra-collector/1.0"})
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8-sig"))
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"{dataset_name} API 요청 실패({type(exc).__name__}). "
                    "인증키 보호를 위해 요청 URL은 표시하지 않습니다."
                ) from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def collect_seoul(name: str, spec: dict, api_key: str, out_dir: Path, page_size: int, force: bool) -> dict:
    """서울 API의 최대 1,000행 페이징을 처리하고 하나의 JSON으로 보존한다."""
    target = out_dir / f"{name}.json"
    if target.exists() and not force:
        existing = json.loads(target.read_text(encoding="utf-8-sig"))
        return {"dataset": name, "status": "existing", "rows": len(existing.get("rows", [])), "path": str(target), "sha256": sha256(target)}
    service = spec["service"]
    rows: list[dict] = []
    start, total = 1, None
    while total is None or start <= total:
        end = start + page_size - 1
        # 서울 열린데이터광장 일반 Open API의 공식 예시는 8088 포트의 HTTP 주소를 사용합니다.
        url = f"http://openapi.seoul.go.kr:8088/{quote(api_key)}/json/{service}/{start}/{end}/"
        payload = fetch_json(url, name)
        body = payload.get(service)
        if not body:
            message = payload.get("RESULT", payload)
            raise RuntimeError(f"{name} API 응답에 {service}가 없습니다: {message}")
        result = body.get("RESULT", {})
        if result.get("CODE") not in (None, "INFO-000"):
            raise RuntimeError(f"{name} API 오류: {result}")
        total = int(body.get("list_total_count", 0))
        batch = body.get("row", [])
        rows.extend(batch)
        if not batch:
            break
        start = end + 1

    document = {"metadata": spec, "collected_at": datetime.now().astimezone().isoformat(), "rows": rows}
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": name, "status": "collected", "rows": len(rows), "path": str(target), "sha256": sha256(target)}


def write_manifest(out_dir: Path, records: list[dict]) -> Path:
    target = out_dir / "collection_manifest.csv"
    columns = ["dataset", "status", "rows", "path", "sha256", "message"]
    previous: dict[str, dict] = {}
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as stream:
            previous = {row["dataset"]: row for row in csv.DictReader(stream)}
    previous.update({row["dataset"]: row for row in records})
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(previous.values())
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="SpotFit 서울 육아 인프라 수집기")
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *SEOUL_DATASETS])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="같은 날짜에 저장된 원본을 명시적으로 다시 수집")
    args = parser.parse_args()

    # 팀원이 PowerShell 세션마다 키를 입력하지 않아도 되도록 프로젝트 루트의 .env를 읽습니다.
    # override=False이므로 서버나 CI가 주입한 환경변수가 있으면 그 값을 우선 사용합니다.
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("SEOUL_OPEN_DATA_API_KEY", "").strip()
    if not api_key:
        print("SEOUL_OPEN_DATA_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        return 2
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size는 1~1000이어야 합니다.")

    stamp = datetime.now().astimezone().strftime("%Y%m%d")
    out_dir = (args.output if args.output.is_absolute() else ROOT / args.output) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = SEOUL_DATASETS if "all" in args.datasets else {k: SEOUL_DATASETS[k] for k in args.datasets}
    records = []
    for name, spec in selected.items():
        try:
            records.append(collect_seoul(name, spec, api_key, out_dir, args.page_size, args.force))
            print(f"[완료] {name}: {records[-1]['rows']:,}행")
        except Exception as exc:  # 다른 데이터 수집은 계속 진행하고 manifest에 실패를 남긴다.
            records.append({"dataset": name, "status": "failed", "rows": 0, "message": str(exc)})
            print(f"[실패] {name}: {exc}", file=sys.stderr)
    manifest = write_manifest(out_dir, records)
    print(f"수집 기록: {manifest}")
    return 1 if any(row["status"] == "failed" for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
