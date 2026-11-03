"""Collect public external features for the Seoul district traffic-accident model.

Outputs raw, reproducible records only. Aggregation is handled by
build_traffic_model_features.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "AI수집" / "안전" / "교통사고" / "모델보강"

DISTRICTS = {
    "종로구": (37.5735, 126.9790), "중구": (37.5641, 126.9979),
    "용산구": (37.5326, 126.9900), "성동구": (37.5633, 127.0371),
    "광진구": (37.5385, 127.0823), "동대문구": (37.5744, 127.0396),
    "중랑구": (37.6063, 127.0927), "성북구": (37.5894, 127.0167),
    "강북구": (37.6398, 127.0255), "도봉구": (37.6688, 127.0471),
    "노원구": (37.6542, 127.0568), "은평구": (37.6027, 126.9291),
    "서대문구": (37.5791, 126.9368), "마포구": (37.5663, 126.9019),
    "양천구": (37.5170, 126.8665), "강서구": (37.5509, 126.8495),
    "구로구": (37.4955, 126.8875), "금천구": (37.4569, 126.8955),
    "영등포구": (37.5264, 126.8963), "동작구": (37.5124, 126.9393),
    "관악구": (37.4784, 126.9516), "서초구": (37.4837, 127.0324),
    "강남구": (37.5172, 127.0473), "송파구": (37.5146, 127.1059),
    "강동구": (37.5301, 127.1238),
}


def get_json(url: str, params: dict, attempts: int = 4) -> dict:
    last = None
    for n in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (n + 1))
    raise RuntimeError(f"request failed: {url} {params}: {last}")


def collect_transit() -> None:
    out_dir = RAW / "대중교통"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://data.seoul.go.kr/bsp/wgs/theme/data/12/1"
    jobs = [(y, m, gu) for y in (2024, 2025) for m in range(1, 13) for gu in DISTRICTS]

    def one(job):
        y, m, gu = job
        payload = get_json(url, {"crtr_yr": y, "crtr_mm": m, "type": 1, "crtr_atdrc": gu})
        return y, m, gu, payload

    records = []
    raw_by_year = {2024: [], 2025: []}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            y, m, gu, payload = fut.result()
            raw_by_year[y].append({"year": y, "month": m, "district": gu, "response": payload})
            rows = (payload.get("data") or {}).get("atdrcSttusInfo") or []
            if rows:
                row = dict(rows[0])
                row.update({"year": y, "month": m, "district": gu})
                records.append(row)
            if i % 100 == 0:
                print(f"transit {i}/{len(jobs)}")
    for y, rows in raw_by_year.items():
        p = out_dir / f"서울데이터허브_자치구월별대중교통_{y}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for row in sorted(rows, key=lambda z: (z["month"], z["district"])):
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pd.DataFrame(records).sort_values(["year", "month", "district"]).to_csv(
        out_dir / "서울데이터허브_자치구월별대중교통_2024_2025.csv", index=False, encoding="utf-8-sig"
    )


def collect_weather() -> None:
    out_dir = RAW / "기상"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "district_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = "https://archive-api.open-meteo.com/v1/archive"
    frames = []
    for i, (gu, (lat, lon)) in enumerate(DISTRICTS.items(), 1):
        cache = cache_dir / f"{gu}.csv.gz"
        if cache.exists():
            daily = pd.read_csv(cache)
        else:
            payload = get_json(url, {
                "latitude": lat, "longitude": lon,
                "start_date": "2010-01-01", "end_date": "2025-12-31",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
                "timezone": "Asia/Seoul", "models": "era5_land",
            }, attempts=8)
            daily = pd.DataFrame(payload["daily"])
            daily.insert(0, "district", gu)
            daily.insert(1, "latitude", lat)
            daily.insert(2, "longitude", lon)
            daily.to_csv(cache, index=False, encoding="utf-8-sig", compression="gzip")
            time.sleep(8)
        frames.append(daily)
        print(f"weather {i}/{len(DISTRICTS)} {gu}")
    pd.concat(frames, ignore_index=True).to_csv(
        out_dir / "OpenMeteo_ERA5Land_자치구일별_2010_2025.csv.gz",
        index=False, encoding="utf-8-sig", compression="gzip"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "transit", "weather"], default="all")
    args = parser.parse_args()
    if args.only in ("all", "transit"):
        collect_transit()
    if args.only in ("all", "weather"):
        collect_weather()


if __name__ == "__main__":
    main()
