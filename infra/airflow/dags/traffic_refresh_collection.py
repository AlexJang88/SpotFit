"""Monthly ingestion for a future, separately validated integrated model release.

This DAG collects public TOPIS workbooks. It never trains or promotes a model:
other annual sources and ground-truth labels must be complete first.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from airflow.sdk import dag, task


@dag(
    dag_id="traffic_refresh_collection",
    schedule="0 23 L * *",  # 매월 말일 23:00 (Asia/Seoul)
    start_date=datetime(2026, 11, 1, tzinfo=ZoneInfo("Asia/Seoul")),
    catchup=False,
    max_active_runs=1,
    tags=["traffic", "collection"],
)
def traffic_refresh_collection():
    @task(retries=2)
    def collect_topis() -> dict:
        from scripts.traffic_data.collect_topis_refresh import RAW_ROOT, collect

        # January may be the first chance to receive the previous December.
        year = datetime.now(ZoneInfo("Asia/Seoul")).year
        results = []
        if year > 2026:
            previous = year - 1
            if not (RAW_ROOT / str(previous) / "manifest.json").is_file() or not all(
                (RAW_ROOT / str(previous) / f"{month:02d}_topis_traffic_{previous}.xlsx").is_file()
                for month in range(1, 13)
            ):
                results.append(collect(previous))
        results.append(collect(year))
        return {"years": [{"year": result["year"],
                            "months_collected": result["months_collected"],
                            "complete_year": result["complete_year"]}
                           for result in results]}

    @task
    def report_annual_readiness(collection: dict) -> dict:
        from scripts.traffic_data.refresh_readiness import readiness

        # A missing source is an expected 'not ready' state, not permission to
        # run training against an incomplete year.
        current_year = datetime.now(ZoneInfo("Asia/Seoul")).year
        target_year = max(2026, current_year - 1)
        result = readiness(target_year)
        print(result)
        return {"collection": collection, "readiness": result}

    report_annual_readiness(collect_topis())


traffic_refresh_collection()
