"""Download and catalogue official candidate data for improving traffic model v94."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "94차_성능개선_추가수집"
RAW = BASE / "raw"
INTERIM = BASE / "interim"
RECORDS = BASE / "수집기록"
UA = {"User-Agent": "Mozilla/5.0"}


def request(url: str, data: dict | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def save(path: Path, content: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"파일": str(path.relative_to(ROOT)), "크기": len(content), "SHA256": hashlib.sha256(content).hexdigest()}


def standard_child_zones() -> tuple[list[dict], list[dict]]:
    header_url = "https://www.data.go.kr/download/columList.json?pk=15012891&ext=CSV"
    header = json.loads(request(header_url))
    table = header["tableVO"]
    columns = table["colNmList"]
    rows = []
    for page in range(1, (header["totalCount"] + 9999) // 10000 + 1):
        query = [
            ("publicDataPk", "15012891"),
            *[("colNmList", col) for col in columns],
            ("totalCount", str(header["totalCount"])),
            ("svcTableNm", table["svcTableNm"]),
            ("perPage", "10000"),
            ("page", str(page)),
        ]
        url = "https://www.data.go.kr/download/standard.json?" + urllib.parse.urlencode(query)
        rows.extend(json.loads(request(url)))
    name_map = {item["columCode"]: item["columNm"] for item in header["columList"]}
    frame = pd.DataFrame(rows).rename(columns=name_map)
    raw_path = RAW / "공공데이터포털" / "전국어린이보호구역표준데이터_20260922.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False, encoding="utf-8-sig")
    address = frame["소재지도로명주소"].fillna("") + " " + frame["소재지지번주소"].fillna("")
    seoul = frame[address.str.contains(r"서울특별시|서울시", regex=True)].copy()
    seoul_path = INTERIM / "서울_어린이보호구역_CCTV_도로폭_최신.csv"
    seoul_path.parent.mkdir(parents=True, exist_ok=True)
    seoul.to_csv(seoul_path, index=False, encoding="utf-8-sig")
    files = []
    for path in (raw_path, seoul_path):
        content = path.read_bytes()
        files.append({"파일": str(path.relative_to(ROOT)), "크기": len(content), "SHA256": hashlib.sha256(content).hexdigest()})
    return files, [{
        "데이터": "전국어린이보호구역표준데이터",
        "제공기관": "경찰청·교육부 / 지방자치단체",
        "출처페이지": "https://www.data.go.kr/data/15012891/standard.do",
        "직접조회주소": header_url,
        "갱신주기": "반기",
        "기준일/수정일": "2026-09-22",
        "94차개선활용": "높음",
        "활용필드": "시설 위치, CCTV 설치 여부·대수, 보호구역 도로폭",
        "한계": "최신 스냅샷이므로 연도별 시설 변화는 별도 과거자료 필요",
        "전체행": len(frame),
        "서울행": len(seoul),
    }]


def file_data(data_id: str, title: str, cycle: str, use: str, limitation: str) -> tuple[dict, dict]:
    page_url = f"https://www.data.go.kr/data/{data_id}/fileData.do"
    html = request(page_url).decode("utf-8", errors="replace")
    match = re.search(r"fn_fileDataDown\('([^']+)',\s*'([^']+)',\s*'([^']*)','([^']+)',\s*'([^']+)'\)", html)
    if not match:
        raise RuntimeError(f"다운로드 매개변수를 찾지 못했습니다: {data_id}")
    pk, detail_pk, atch, detail_sn, hist = match.groups()
    payload = {
        "publicDataDetailPk": detail_pk, "publicDataPk": pk, "atchFileId": atch,
        "fileDetailSn": detail_sn, "publicDataTyCode": "PR0051",
    }
    info = json.loads(request("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do", payload))
    atch = info["atchFileId"]
    detail_sn = info["fileDetailSn"]
    registered = info["fileDataRegistVO"]
    data_name = registered.get("dataNm") or title
    original = registered.get("orginlFileNm") or f"{data_name}.{registered.get('atchFileExtsn') or 'dat'}"
    download_url = "https://www.data.go.kr/cmm/cmm/fileDownload.do?" + urllib.parse.urlencode({
        "atchFileId": atch, "fileDetailSn": detail_sn, "dataNm": data_name,
    })
    content = request(download_url)
    file_info = save(RAW / "공공데이터포털" / original, content)
    source = {
        "데이터": title, "제공기관": info["dataSetFileDetailInfo"].get("insttNm"),
        "출처페이지": page_url, "직접다운로드주소": download_url, "갱신주기": cycle,
        "기준일/수정일": info["dataSetFileDetailInfo"].get("stdrDe"), "94차개선활용": use,
        "활용필드": info["dataSetFileDetailInfo"].get("publicDataDc", ""), "한계": limitation,
    }
    return file_info, source


def main() -> None:
    for directory in (RAW, INTERIM, RECORDS):
        directory.mkdir(parents=True, exist_ok=True)
    files, sources = standard_child_zones()
    targets = [
        ("15162408", "교통사고도로위험도지수", "연간", "높음", "전국 도로링크 자료; 서울 공간필터와 자치구·연도 집계가 필요"),
        ("15118850", "최근 5년 어린이보호구역 내 어린이 교통사고 통계", "연간", "참고", "전국 합계로 자치구 구분이 없어 모델 직접 입력 불가"),
        ("15087037", "시간대별 교통사고 통계", "연간", "참고", "전국 전체사고 합계로 어린이·서울·자치구 구분이 없음"),
    ]
    for args in targets:
        file_info, source = file_data(*args)
        files.append(file_info)
        sources.append(source)
    pd.DataFrame(files).to_csv(RECORDS / "파일검증.csv", index=False, encoding="utf-8-sig")
    workbook = RECORDS / "94차_성능개선_데이터출처_갱신주기.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(sources).to_excel(writer, sheet_name="출처와갱신주기", index=False)
        pd.DataFrame(files).to_excel(writer, sheet_name="확보파일검증", index=False)
        for ws in writer.book:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(70, max(14, max(len(str(c.value or "")) for c in col[:50]) + 2))
    print(json.dumps({"files": files, "sources": sources, "workbook": str(workbook.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
