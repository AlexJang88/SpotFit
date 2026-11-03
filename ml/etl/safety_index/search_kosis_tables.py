from __future__ import annotations

import getpass
import json
import urllib.parse
import urllib.request


TERMS = [
    "도로 기능별",
    "유치원 원아수",
    "유치원생 수",
    "초등학교 학생수",
    "고령인구비율 시군구",
]


def main() -> None:
    api_key = getpass.getpass("KOSIS API key: ").strip()
    for term in TERMS:
        params = urllib.parse.urlencode(
            {
                "method": "getList",
                "apiKey": api_key,
                "searchNm": term,
                "sort": "RANK",
                "startCount": 1,
                "resultCount": 50,
                "format": "json",
                "jsonVD": "Y",
            }
        )
        url = "https://kosis.kr/openapi/statisticsSearch.do?" + params
        with urllib.request.urlopen(url, timeout=60) as response:
            rows = json.load(response)
        if isinstance(rows, dict):
            print(f"\n[{term}] {rows}")
            continue
        print(f"\n[{term}] {len(rows)}")
        for row in rows[:20]:
            print(
                row.get("ORG_ID"),
                row.get("TBL_ID"),
                row.get("TBL_NM"),
                row.get("LINK_URL", ""),
            )


if __name__ == "__main__":
    main()
