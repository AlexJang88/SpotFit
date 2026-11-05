"""서울통계 차량통행속도(월별·구별) 2017~2025 CSV를 내려받는다."""

from __future__ import annotations

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE = "https://stat.eseoul.go.kr"
TABLE_ID = "DT_201004_I040003"
TABLE_URL = f"{BASE}/statHtml/statHtml.do?orgId=201&tblId={TABLE_ID}"
OUTPUT = Path("data/raw/안전/교통사고/서울시_차량통행속도_월별_구별_2017_2025.csv")


def selected_form_values(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.select_one("#ParamInfo")
    if form is None:
        raise RuntimeError("서울통계 ParamInfo 폼을 찾지 못했습니다.")
    values = []
    for element in form.select("input[name], select[name], textarea[name]"):
        name = element.get("name")
        if not name or element.has_attr("disabled"):
            continue
        if element.name == "input":
            kind = element.get("type", "text").lower()
            if kind in {"checkbox", "radio"} and not element.has_attr("checked"):
                continue
            values.append((name, element.get("value", "")))
        elif element.name == "select":
            options = element.select("option[selected]") or element.select("option")[:1]
            values.extend((name, option.get("value", "")) for option in options)
        else:
            values.append((name, element.get_text()))
    return values


def main() -> None:
    session = requests.Session()
    response = session.get(TABLE_URL, timeout=60)
    response.raise_for_status()
    form = selected_form_values(response.text)

    months = [f"{year}{month:02d}" for year in range(2025, 2016, -1) for month in range(12, 0, -1)]
    districts = [f"003{number:03d}" for number in range(1, 26)]
    fields = [
        {"targetId": "PRD", "targetValue": "", "prdValue": f"M,{','.join(months)},@"},
        {"targetId": "ITM_ID", "targetValue": "T001", "prdValue": ""},
        *(
            {"targetId": "OV_L1_ID", "targetValue": district, "prdValue": ""}
            for district in districts
        ),
    ]
    overrides = {
        "view": "csv", "viewKind": "2", "viewSubKind": "2_3", "isFirst": "N",
        "dataOpt": "ko", "fieldList": json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
        "rowAxis": "A", "colAxis": "TIME,ITEM", "periodStr": "M",
        "reqCellCnt": str(len(months) * len(districts)),
        "classAllArr": json.dumps([{"objVarId": "A", "ovlSn": "1"}], separators=(",", ":")),
    }
    form = [(key, overrides.get(key, value)) for key, value in form if key not in {"timeChkY", "timeChkM"}]
    result = session.post(f"{BASE}/statHtml/downGrid.do", data=form, timeout=180)
    result.raise_for_status()
    payload = result.json()
    filename = payload.get("file")
    if not filename:
        raise RuntimeError(f"서울통계 CSV 생성 실패: {payload}")
    download_form = [(key, filename if key == "file" else value) for key, value in form]
    csv_response = session.post(f"{BASE}/statHtml/downNormal.do", data=download_form, timeout=180)
    csv_response.raise_for_status()
    if b"<html" in csv_response.content[:500].lower():
        raise RuntimeError("서울통계 CSV 응답이 HTML입니다.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(csv_response.content)
    print(f"saved={OUTPUT} bytes={len(csv_response.content):,}")


if __name__ == "__main__":
    main()
