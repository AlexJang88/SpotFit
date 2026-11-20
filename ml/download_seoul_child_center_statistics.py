from pathlib import Path
import json

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://stat.eseoul.go.kr"
TABLE_URL = (
    BASE
    + "/statHtml/statHtml.do?orgId=201&tblId=DT_201004_B060003&conn_path=I2&mode=tab"
)
OUT = ROOT / "data" / "raw" / "지역아동센터" / "서울시_지역아동센터_자치구별통계_2017_2025_원본.xlsx"
YEARS = list(range(2017, 2026))


def successful_controls(form):
    controls = []
    for tag in form.find_all(["input", "select", "textarea"]):
        name = tag.get("name")
        if not name or tag.has_attr("disabled"):
            continue
        if tag.name == "input":
            input_type = (tag.get("type") or "text").lower()
            if input_type in {"submit", "button", "image", "file", "reset"}:
                continue
            if input_type in {"checkbox", "radio"} and not tag.has_attr("checked"):
                continue
            controls.append((name, tag.get("value", "")))
        elif tag.name == "select":
            selected = tag.find_all("option", selected=True) or tag.find_all("option")[:1]
            controls.extend((name, option.get("value", option.get_text(strip=True))) for option in selected)
        else:
            controls.append((name, tag.get_text()))
    return controls


def replace_single(controls, name, value):
    controls[:] = [(key, val) for key, val in controls if key != name]
    controls.append((name, value))


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    response = session.get(TABLE_URL, timeout=60)
    response.raise_for_status()
    form = BeautifulSoup(response.text, "html.parser").find("form", id="ParamInfo")
    if form is None:
        raise RuntimeError("통계표 다운로드 양식을 찾지 못했습니다.")

    data = [(key, val) for key, val in successful_controls(form) if key != "timeChkY"]
    data.extend(("timeChkY", str(year)) for year in YEARS)
    period = "Y," + ",".join(map(str, YEARS)) + ",@"
    field_list = [
        {"targetId": "PRD", "targetValue": "", "prdValue": period},
        {"targetId": "ITM_ID", "targetValue": "T001", "prdValue": ""},
    ]
    field_list.extend(
        {"targetId": "OV_L1_ID", "targetValue": value, "prdValue": ""}
        for value in ["001", *[f"001{i:03d}" for i in range(1, 26)]]
    )
    field_list.extend(
        {"targetId": "OV_L2_ID", "targetValue": value, "prdValue": ""}
        for value in [
            "001", "001001", "001002", "002", "002001", "002002",
            "002002001", "002002002", "003", "003001", "003002",
        ]
    )
    for key, value in {
        "view": "xlsx", "viewSubKind": "2_1", "viewKind": "2", "smblYn": "Y",
        "prdSort": "asc", "downSort": "asc",
        "fieldList": json.dumps(field_list, ensure_ascii=False),
        "rowAxis": "A", "colAxis": "TIME,B",
    }.items():
        replace_single(data, key, value)

    prepared = session.post(BASE + "/statHtml/downGrid.do", data=data, timeout=120)
    prepared.raise_for_status()
    file_token = prepared.json().get("file")
    if not file_token:
        raise RuntimeError(f"다운로드 파일 토큰을 받지 못했습니다: {prepared.text[:500]}")
    replace_single(data, "file", file_token)
    download = session.post(BASE + "/statHtml/downNormal.do", data=data, timeout=120)
    download.raise_for_status()
    if len(download.content) < 1000 or download.content[:2] != b"PK":
        raise RuntimeError("정상 XLSX 파일이 아닙니다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(download.content)
    print(OUT)
    print(len(download.content))


if __name__ == "__main__":
    main()
