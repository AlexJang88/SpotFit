"""현재 공개 원자료에서 2010~2026 연도별 안전 보조지표를 복원한다.

주의:
- CCTV 패널은 2026년 현재 목록에 남아 있는 장비의 설치연월을 누적한 값이다.
  철거된 장비가 빠져 있으므로 당시의 완전한 재고가 아니다.
- 주점 패널은 인허가일과 폐업일로 각 연도 말 영업 중 업소를 복원한다.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "안전"
YEARS = range(2010, 2027)
DISTRICT_RE = re.compile(r"서울(?:특별시|시)?\s+([가-힣]+구)(?:\s|청|$)")
DISTRICTS = (
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
)


def parse_date(value: object) -> date | None:
    text = str(value or "").strip().replace(".", "-").replace("/", "-")
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m"):
        try:
            parsed = datetime.strptime(text[:10], fmt).date()
            return parsed
        except ValueError:
            continue
    return None


def district_from(*values: object) -> str | None:
    for value in values:
        text = str(value or "")
        match = DISTRICT_RE.search(text)
        if match:
            return match.group(1)
        # 일부 관리기관명은 '서울' 접두어 없이 '금천구청'처럼 제공된다.
        for district in DISTRICTS:
            if district in text:
                return district
    return None


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cctv_panel() -> tuple[Path, dict]:
    source = RAW / "교통사고" / "서울시_CCTV정보.csv"
    with source.open(encoding="cp949", newline="") as stream:
        records = list(csv.DictReader(stream))

    installed: dict[tuple[str, int, str], int] = defaultdict(int)
    source_by_district: dict[str, int] = defaultdict(int)
    dated_by_district: dict[str, int] = defaultdict(int)
    skipped = 0
    for row in records:
        district = district_from(
            row.get("소재지도로명주소"),
            row.get("소재지지번주소"),
            row.get("관리기관명"),
        )
        if district:
            source_by_district[district] += 1
        installed_on = parse_date(row.get("설치연월"))
        if not district or not installed_on:
            skipped += 1
            continue
        dated_by_district[district] += 1
        purpose = (row.get("설치목적구분") or "미분류").strip()
        try:
            cameras = int(float((row.get("카메라대수") or "1").replace(",", "")))
        except ValueError:
            cameras = 1
        installed[(district, installed_on.year, purpose)] += cameras

    districts = list(DISTRICTS)
    purposes = sorted({key[2] for key in installed})
    rows: list[dict] = []
    cumulative: dict[tuple[str, str], int] = defaultdict(int)
    for year in YEARS:
        for district in districts:
            has_history = dated_by_district[district] > 0
            for purpose in purposes:
                cumulative[(district, purpose)] += installed[(district, year, purpose)]
            rows.append(
                {
                    "연도": year,
                    "자치구": district,
                    "교통단속_CCTV_대수": cumulative[(district, "교통단속")] if has_history else "",
                    "생활방범_CCTV_대수": cumulative[(district, "생활방범")] if has_history else "",
                    "어린이보호_CCTV_대수": cumulative[(district, "어린이보호")] if has_history else "",
                    "전체_CCTV_대수": (
                        sum(cumulative[(district, purpose)] for purpose in purposes)
                        if has_history else ""
                    ),
                    "설치연월_보유행수": dated_by_district[district],
                    "현재목록_행수": source_by_district[district],
                    "복원방식": (
                        "2026년 현재 목록의 설치연월 누적(철거 이력 미반영)"
                        if has_history else "현재 목록에 설치연월이 없어 과거 복원 불가"
                    ),
                }
            )
    output = RAW / "교통사고" / "파생" / "서울시_CCTV_자치구_연도별_2010_2026.csv"
    write_csv(output, list(rows[0]), rows)
    return output, {
        "source_rows": len(records),
        "usable_rows": len(records) - skipped,
        "skipped_rows": skipped,
        "districts": len(districts),
        "districts_with_installation_dates": sum(
            count > 0 for count in dated_by_district.values()
        ),
        "output_rows": len(rows),
    }


def load_json_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", payload if isinstance(payload, list) else []))


def build_bar_panel() -> tuple[Path, dict]:
    sources = {
        "단란주점": RAW / "범죄" / "20261030" / "danran_bar.json",
        "유흥주점": RAW / "범죄" / "20261030" / "entertainment_bar.json",
    }
    records: list[tuple[str, dict]] = []
    for kind, path in sources.items():
        records.extend((kind, row) for row in load_json_rows(path))

    prepared: list[tuple[str, str, date, date | None]] = []
    skipped = 0
    for kind, row in records:
        district = district_from(row.get("RDNWHLADDR"), row.get("SITEWHLADDR"))
        opened = parse_date(row.get("APVPERMYMD"))
        closed = parse_date(row.get("DCBYMD"))
        if not district or not opened:
            skipped += 1
            continue
        prepared.append((kind, district, opened, closed))

    districts = sorted({record[1] for record in prepared})
    rows: list[dict] = []
    for year in YEARS:
        end = date(year, 12, 31)
        for district in districts:
            counts = {"단란주점": 0, "유흥주점": 0}
            for kind, record_district, opened, closed in prepared:
                if record_district != district:
                    continue
                if opened <= end and (closed is None or closed > end):
                    counts[kind] += 1
            rows.append(
                {
                    "연도": year,
                    "자치구": district,
                    "단란주점_연말_영업업소수": counts["단란주점"],
                    "유흥주점_연말_영업업소수": counts["유흥주점"],
                    "유흥_단란주점_합계": counts["단란주점"] + counts["유흥주점"],
                    "복원방식": "인허가일<=연말<폐업일(폐업일 공란은 영업 중으로 간주)",
                }
            )
    output = RAW / "범죄" / "파생" / "유흥_단란주점_자치구_연도별_2010_2026.csv"
    write_csv(output, list(rows[0]), rows)
    return output, {
        "source_rows": len(records),
        "usable_rows": len(prepared),
        "skipped_rows": skipped,
        "districts": len(districts),
        "output_rows": len(rows),
    }


def main() -> int:
    cctv_path, cctv_stats = build_cctv_panel()
    bar_path, bar_stats = build_bar_panel()
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "period": {"start": 2010, "end": 2026},
        "outputs": {
            str(cctv_path.relative_to(ROOT)): cctv_stats,
            str(bar_path.relative_to(ROOT)): bar_stats,
        },
        "limitations": [
            "2026년 값은 연중 스냅샷 또는 인허가 이력 기반이며 확정 연간 실적이 아니다.",
            "CCTV 과거값은 현재 목록에 남은 장비만 역산하여 철거 장비가 누락된다.",
            "주점 과거값은 폐업일 누락 또는 정정 이력의 영향을 받을 수 있다.",
        ],
    }
    manifest_path = RAW / "2010_2026_파생패널_메타.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
