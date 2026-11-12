from __future__ import annotations

import getpass
import json
import urllib.parse
import urllib.request


TABLES = [
    ("460", "TX_315_2009_H1023"),
    ("101", "DT_1B04006"),
    ("101", "DT_1YL21241"),
    ("177", "DT_H_BELT"),
    ("177", "DT_117075_H_DR_DRIV"),
]


def main() -> None:
    key = getpass.getpass("KOSIS API key: ").strip()
    for item in ("CR", "13999001.CR"):
        for obj in ("00101", "CHS_CT.00101"):
            params = {"method":"getList","apiKey":key,"orgId":"177","tblId":"DT_H_BELT",
                      "itmId":item,"objL1":obj,"prdSe":"Y","startPrdDe":"2024",
                      "endPrdDe":"2024","format":"json","jsonVD":"Y"}
            url="https://kosis.kr/openapi/Param/statisticsParameterData.do?"+urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(url,timeout=60) as response: body=response.read().decode("utf-8")
                print(item,obj,body[:1000])
            except Exception as exc: print(item,obj,repr(exc))


if __name__ == "__main__":
    main()
