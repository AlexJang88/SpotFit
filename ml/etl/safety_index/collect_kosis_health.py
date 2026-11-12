from __future__ import annotations

import getpass
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


SEOUL_CODES = [f"001{i:02d}" for i in range(1, 26)]
TABLES = {
    "교통안전_자치구별_운전자석안전벨트착용률_2008_2025.csv": "DT_H_BELT",
    "교통안전_자치구별_연간음주운전경험률_2008_2025.csv": "DT_117075_H_DR_DRIV",
}


def main() -> None:
    key = getpass.getpass("KOSIS API key: ").strip()
    output_dir = Path("data/raw/안전/교통사고")
    for filename, table_id in TABLES.items():
        params = {
            "method": "getList", "apiKey": key, "orgId": "177", "tblId": table_id,
            "itmId": "CR", "objL1": "+".join(SEOUL_CODES), "prdSe": "Y",
            "startPrdDe": "2008", "endPrdDe": "2025", "format": "json", "jsonVD": "Y",
        }
        url = "https://kosis.kr/openapi/Param/statisticsParameterData.do?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.load(response)
        if isinstance(data, dict) and data.get("err"):
            raise RuntimeError(f"{table_id}: {data}")
        df = pd.DataFrame(data)
        keep = [c for c in ["PRD_DE", "C1_NM", "DT", "UNIT_NM"] if c in df.columns]
        df = df[keep].rename(columns={"PRD_DE": "연도", "C1_NM": "자치구", "DT": "값", "UNIT_NM": "단위"})
        df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        print(filename, len(df), sorted(df["연도"].astype(str).unique()))


if __name__ == "__main__":
    main()
