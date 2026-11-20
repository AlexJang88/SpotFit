"""지역안전지수 공개 산출지표 원자료를 수집한다.

현재 공공데이터포털 표준데이터 다운로드 API에서 AED 데이터를 받아
서울 주소만 선별한다. API 키가 필요하지 않은 공개 다운로드 경로다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "안전" / "생활안전"
PORTAL = "https://www.data.go.kr"


def request_json(path: str, params: list[tuple[str, str]] | dict[str, str]) -> object:
    query = urlencode(params)
    url = f"{PORTAL}{path}?{query}"
    # 포털의 대용량 표준데이터 경로는 Python urllib 요청에 빈 응답을
    # 반환하는 경우가 있어 Windows 기본 curl을 동일한 공개 URL에 사용한다.
    if path == "/download/standard.json":
        completed = subprocess.run(
            ["curl.exe", "-k", "-L", "--fail", "--silent", "--show-error", url],
            check=True,
            capture_output=True,
        )
        return json.loads(completed.stdout.decode("utf-8-sig"))
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SpotFit/1.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{PORTAL}/data/15021103/standard.do",
        },
    )
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8-sig")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            content_type = response.headers.get("Content-Type", "unknown")
            raise RuntimeError(
                f"공공데이터포털이 JSON 대신 {content_type} 응답을 반환했습니다."
            ) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_standard_dataset(dataset_id: str) -> tuple[dict, list[dict]]:
    header = request_json(
        "/download/columList.json", {"pk": dataset_id, "ext": "CSV"}
    )
    table = header["tableVO"]
    # 화면용 컬럼 목록에는 제공기관 보조 컬럼이 포함되지만, 다운로드 API는
    # tableVO.colNmList만 허용하며 그 외 컬럼을 보내면 빈 응답을 반환한다.
    columns = list(table["colNmList"])
    total = int(header["totalCount"])
    page_size = 10_000
    rows: list[dict] = []
    for page in range(1, (total + page_size - 1) // page_size + 1):
        params = [
            ("publicDataPk", dataset_id),
            ("svcTableNm", table["svcTableNm"]),
            ("perPage", str(page_size)),
            ("page", str(page)),
            ("totalCount", str(total)),
        ]
        params.extend(("colNmList", column) for column in columns)
        page_rows = request_json("/download/standard.json", params)
        if not isinstance(page_rows, list):
            raise RuntimeError(f"{dataset_id} 응답이 목록 형식이 아닙니다.")
        rows.extend(page_rows)
        print(f"{dataset_id}: page {page}, {len(page_rows)} rows")
    return header, rows


def is_seoul(row: dict) -> bool:
    text = " ".join(str(value or "") for value in row.values())
    return "서울" in text


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_id = "15021103"
    header, nationwide = collect_standard_dataset(dataset_id)
    seoul = [row for row in nationwide if is_seoul(row)]
    columns = list(header["tableVO"]["colNmList"])
    csv_path = OUT / "전국자동심장충격기표준데이터_서울.csv"
    json_path = OUT / "전국자동심장충격기표준데이터_서울.json"
    write_csv(csv_path, seoul, columns)
    json_path.write_text(json.dumps(seoul, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset_id": dataset_id,
        "dataset": header["fileName"],
        "source": "https://www.data.go.kr/data/15021103/standard.do",
        "collected_at": datetime.now().astimezone().isoformat(),
        "nationwide_rows": len(nationwide),
        "seoul_rows": len(seoul),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (csv_path, json_path)
        },
    }
    (OUT / "자동심장충격기_수집메타.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"AED: 전국 {len(nationwide):,}건, 서울 {len(seoul):,}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
