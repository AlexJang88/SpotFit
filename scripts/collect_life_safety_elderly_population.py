"""Collect official Seoul 65+ population for the life-safety index."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "AI수집" / "안전" / "생활안전" / "03_인구만명당_재난약자수"
OUT = ROOT / "data" / "interim" / "안전" / "생활안전" / "자치구별_65세이상인구_2017_2025.csv"
URL = "https://data.seoul.go.kr/bsp/wgs/dataset/dataTypeList.do"


def fetch(page: int) -> dict:
    body = json.dumps({
        "id": 96, "rowFilterList": [], "currentPage": page, "pagePer": 1000
    }).encode("utf-8")
    request = Request(URL, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"
    })
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    page = 1
    while True:
        payload = fetch(page)
        (RAW / f"elderly_population_page_{page}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        batch = payload.get("dataTypeList", [])
        rows.extend(batch)
        # The portal reports 1,000 as a capped count, so paginate until a short page.
        if len(batch) < 1000:
            break
        page += 1

    raw = pd.DataFrame(rows)
    data = raw[
        raw["stats_yr"].astype(int).between(2017, 2025)
        & raw["sigungu_nm"].astype(str).str.endswith("구")
        & raw["frgnr_div"].eq("계")
        & raw["sex_nm"].eq("계")
    ].copy()
    data = data.rename(columns={
        "stats_yr": "연도", "sigungu_nm": "자치구", "popl_cnt": "65세이상인구수",
        "admdong_cd": "행정구역코드",
    })
    data["연도"] = data["연도"].astype(int)
    data["출처"] = "서울 데이터 허브 고령자 현황(내외국인 성별/연도별/구별)"
    data["원자료URL"] = "https://data.seoul.go.kr/bsp/wgs/dataView/data300View/96.do"
    data = data[["연도", "자치구", "65세이상인구수", "행정구역코드", "출처", "원자료URL"]]
    data = data.sort_values(["연도", "자치구"]).reset_index(drop=True)

    if len(data) != 225 or data["자치구"].nunique() != 25:
        raise ValueError(f"Unexpected coverage: rows={len(data)}, districts={data['자치구'].nunique()}")
    if data.duplicated(["연도", "자치구"]).any() or data.isna().any().any():
        raise ValueError("Duplicate keys or missing values")
    data.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved={OUT}")
    print(f"raw_rows={len(raw)}, processed_rows={len(data)}")


if __name__ == "__main__":
    main()
