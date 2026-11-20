"""VWorld 주소 API로 좌표 없는 육아 인프라를 지오코딩하고 응답을 캐시한다."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def request_coordinate(address: str, key: str, address_type: str) -> tuple[float | None, float | None, str]:
    params = {"service": "address", "request": "getcoord", "version": "2.0", "crs": "EPSG:4326", "address": address,
              "refine": "true", "simple": "false", "format": "json", "type": address_type, "key": key}
    request = Request("https://api.vworld.kr/req/address?" + urlencode(params), headers={"User-Agent": "SpotFit-geocoder/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8-sig"))
        api_response = payload.get("response", {})
        if api_response.get("status") != "OK":
            return None, None, str(api_response.get("error", {}).get("text", api_response.get("status", "ERROR")))
        point = api_response["result"]["point"]
        return float(point["x"]), float(point["y"]), "OK"
    except Exception as exc:
        # 요청 URL에는 인증키가 있으므로 예외 문자열과 URL을 저장하거나 출력하지 않습니다.
        return None, None, f"REQUEST_ERROR:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체, 테스트 시 최대 신규 호출 수")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("VWORLD_API_KEY", "").strip()
    if not key:
        raise SystemExit("VWORLD_API_KEY 환경변수가 필요합니다.")
    collected = args.collected_dir if args.collected_dir.is_absolute() else ROOT / args.collected_dir
    cache_path = collected / "vworld_geocode_cache.csv"
    cache = pd.read_csv(cache_path, encoding="utf-8-sig") if cache_path.exists() else pd.DataFrame(columns=["address", "longitude", "latitude", "status", "address_type"])
    cached_addresses = set(cache["address"].astype(str))

    kindergarten = pd.DataFrame(json.loads((collected / "kindergarten.json").read_text(encoding="utf-8-sig"))["rows"])
    sports = pd.DataFrame(json.loads((collected / "sports_program.json").read_text(encoding="utf-8-sig"))["rows"])
    addresses = pd.concat([
        kindergarten[["ADDR"]].rename(columns={"ADDR": "address"}),
        sports[["BD_FIELD_13"]].rename(columns={"BD_FIELD_13": "address"}),
    ], ignore_index=True)["address"].dropna().astype(str).str.strip()
    addresses = [a for a in addresses.drop_duplicates() if a and a not in cached_addresses]
    if args.limit > 0:
        addresses = addresses[:args.limit]

    new_rows = []
    for index, address in enumerate(addresses, start=1):
        lon, lat, status = request_coordinate(address, key, "road")
        address_type = "road"
        if status != "OK":
            lon, lat, status = request_coordinate(address, key, "parcel")
            address_type = "parcel"
        new_rows.append({"address": address, "longitude": lon, "latitude": lat, "status": status, "address_type": address_type})
        if index % 100 == 0:
            pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"지오코딩 진행: {index}/{len(addresses)}")
        time.sleep(max(args.delay, 0))
    result = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates("address", keep="last")
    result.to_csv(cache_path, index=False, encoding="utf-8-sig")

    kindergarten.merge(result, left_on="ADDR", right_on="address", how="left").to_csv(collected / "kindergarten_geocoded.csv", index=False, encoding="utf-8-sig")
    sports.merge(result, left_on="BD_FIELD_13", right_on="address", how="left").to_csv(collected / "sports_program_geocoded.csv", index=False, encoding="utf-8-sig")
    print(result["status"].value_counts(dropna=False).to_string())
    print(f"캐시: {cache_path}")


if __name__ == "__main__":
    main()
