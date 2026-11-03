"""서울 25개 자치구의 학원·교습소 원본을 하나의 JSON으로 수집한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
SERVICES = {
    "종로구": "neisAcademyInfo_jn", "중구": "neisAcademyInfo_jg",
    "용산구": "neisAcademyInfo_ys", "성동구": "neisAcademyInfo_sd",
    "광진구": "neisAcademyInfo_gj", "동대문구": "neisAcademyInfo_dd",
    "중랑구": "neisAcademyInfo_jr", "성북구": "neisAcademyInfo_sb",
    "강북구": "neisAcademyInfo_gb", "도봉구": "neisAcademyInfo_db",
    "노원구": "neisAcademyInfo_nw", "은평구": "neisAcademyInfo_ep",
    "서대문구": "neisAcademyInfo_sm", "마포구": "neisAcademyInfo_mp",
    "양천구": "neisAcademyInfo_yc", "강서구": "neisAcademyInfo_gs",
    "구로구": "neisAcademyInfo_gr", "금천구": "neisAcademyInfo_gc",
    "영등포구": "neisAcademyInfo_yd", "동작구": "neisAcademyInfo_dj",
    "관악구": "neisAcademyInfo_ga", "서초구": "neisAcademyInfo_sc",
    "강남구": "neisAcademyInfo_gn", "송파구": "neisAcademyInfo_sp",
    "강동구": "neisAcademyInfo_gd",
}


def request_json(api_key: str, service: str, start: int, end: int) -> dict:
    url = f"http://openapi.seoul.go.kr:8088/{quote(api_key)}/json/{service}/{start}/{end}/"
    request = Request(url, headers={"User-Agent": "SpotFit-academy-collector/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/아동/전연령_교육인프라_공식통계")
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("SEOUL_OPEN_DATA_API_KEY", "").strip()
    if not api_key:
        print("SEOUL_OPEN_DATA_API_KEY가 필요합니다.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    counts: dict[str, int] = {}
    for district, service in SERVICES.items():
        start, total = 1, None
        district_rows: list[dict] = []
        while total is None or start <= total:
            end = start + args.page_size - 1
            payload = request_json(api_key, service, start, end)
            body = payload.get(service)
            if body is None:
                raise RuntimeError(f"{district}({service}) 응답 오류: {payload.get('RESULT', payload)}")
            total = int(body.get("list_total_count", 0))
            batch = body.get("row", [])
            district_rows.extend(batch)
            if not batch:
                break
            start = end + 1
        for row in district_rows:
            row["DISTRICT"] = district
            row["SOURCE_SERVICE"] = service
        rows.extend(district_rows)
        counts[district] = len(district_rows)
        print(f"[완료] {district}: {len(district_rows):,}행")

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d")
    target = args.output / f"서울시_25개자치구_학원교습소_{stamp}.json"
    document = {
        "metadata": {
            "title": "서울시 25개 자치구 학원 교습소정보",
            "source": "https://data.seoul.go.kr/",
            "collected_at": datetime.now().astimezone().isoformat(),
            "snapshot_only": True,
            "history_warning": "개설·등록일은 있으나 현재 제공 목록에 없는 폐원시설의 완전한 이력을 보장하지 않음",
            "district_counts": counts,
        },
        "rows": rows,
    }
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {target} ({len(rows):,}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
