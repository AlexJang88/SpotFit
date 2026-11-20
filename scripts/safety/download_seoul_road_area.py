"""Download and normalize Seoul district road-area statistics.

Source: Seoul Statistics table DT_201004_O090023 (Road status).
The downloader submits the same public table form used by stat.eseoul.go.kr.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE = "https://stat.eseoul.go.kr"
TABLE_URL = f"{BASE}/statHtml/statHtml.do?orgId=201&tblId=DT_201004_O090023"


def selected_form_values(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.select_one("#ParamInfo")
    if form is None:
        raise RuntimeError("서울통계 ParamInfo 폼을 찾지 못했습니다.")

    values: list[tuple[str, str]] = []
    for element in form.select("input[name], select[name], textarea[name]"):
        name = element.get("name")
        if not name or element.has_attr("disabled"):
            continue
        if element.name == "input":
            input_type = element.get("type", "text").lower()
            if input_type in {"checkbox", "radio"} and not element.has_attr("checked"):
                continue
            values.append((name, element.get("value", "")))
        elif element.name == "select":
            options = element.select("option[selected]") or element.select("option")[:1]
            values.extend((name, option.get("value", option.get_text(strip=True))) for option in options)
        else:
            values.append((name, element.get_text()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/안전/교통사고/서울시_도로현황_자치구_연도별.csv"),
    )
    args = parser.parse_args()

    session = requests.Session()
    response = session.get(TABLE_URL, timeout=60)
    response.raise_for_status()
    form = selected_form_values(response.text)

    years = [str(year) for year in range(2025, 2016, -1)]
    districts = ["001"] + [f"001{number:03d}" for number in range(1, 26)]
    road_types = ["001", "001001", "001002", "001003"]
    states = ["001", "002", "003"]
    field_list = [
        {"targetId": "PRD", "targetValue": "", "prdValue": f"Y,{','.join(years)},@"},
        {"targetId": "ITM_ID", "targetValue": "T001", "prdValue": ""},
        {"targetId": "ITM_ID", "targetValue": "T002", "prdValue": ""},
    ]
    for target, values in (("OV_L1_ID", districts), ("OV_L2_ID", road_types), ("OV_L3_ID", states)):
        field_list.extend(
            {"targetId": target, "targetValue": value, "prdValue": ""} for value in values
        )

    # Ask the official table server for CSV using the table's own request format.
    overrides = {
        "view": "csv",
        "viewKind": "2",
        "viewSubKind": "2_3",
        "isFirst": "N",
        "dataOpt": "ko",
        "fieldList": json.dumps(field_list, ensure_ascii=False, separators=(",", ":")),
        "rowAxis": "A",
        "colAxis": "TIME,B,C,ITEM",
        "periodStr": "Y",
        "reqCellCnt": str(len(years) * len(districts) * len(road_types) * len(states) * 2),
        "classAllArr": json.dumps(
            [
                {"objVarId": "A", "ovlSn": "1"},
                {"objVarId": "B", "ovlSn": "2"},
                {"objVarId": "C", "ovlSn": "3"},
            ],
            separators=(",", ":"),
        ),
    }
    form = [(key, overrides.get(key, value)) for key, value in form if key not in {"timeChkY"}]
    result = session.post(f"{BASE}/statHtml/downGrid.do", data=form, timeout=120)
    result.raise_for_status()
    payload = result.json()
    filename = payload.get("file")
    if not filename:
        raise RuntimeError(f"서울통계 CSV 생성 실패: {payload}")

    download_form = [(key, filename if key == "file" else value) for key, value in form]
    csv_response = session.post(
        f"{BASE}/statHtml/downNormal.do", data=download_form, timeout=120
    )
    csv_response.raise_for_status()
    if b"<html" in csv_response.content[:500].lower():
        raise RuntimeError(f"서울통계 CSV 응답이 HTML입니다: file={filename!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(csv_response.content)
    print(f"saved={args.output} bytes={len(csv_response.content):,}")


if __name__ == "__main__":
    main()
