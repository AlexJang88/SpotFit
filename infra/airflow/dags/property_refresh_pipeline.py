"""Monthly apartment price pipeline: download -> preprocess -> retrain -> validate -> replace served models.

 1. collect_molit_and_rate : 국토교통부 실거래가(매매·전월세) + 한국은행 금리를 data/raw/부동산/refresh/ 로 내려받음
                             (표준 라이브러리만 사용. DATA_GO_KR_API_KEY, ECOS_API_KEY 필요)
 2. refresh_models         : 백엔드 API가 원본 병합 -> 입력 파일 3개 생성 -> 재학습 -> 검증 -> 모델 교체 -> 서비스 캐시 갱신
                             (pandas/xgboost는 백엔드 이미지에만 있으므로 백엔드가 실행하고, 여기서는 끝날 때까지 기다림)
New models only replace the served ones when they pass validation; otherwise the task fails and the old models stay.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from airflow.sdk import dag, task

BACKEND_URL = os.environ.get("PROPERTY_REFRESH_BACKEND_URL", "http://backend:8000")
POLL_SECONDS = 30
TIMEOUT_SECONDS = 4 * 3600


def _call(path: str, method: str = "GET", admin: bool = False) -> dict:
    headers = {"X-Model-Admin-Key": os.environ["MODEL_TRAINING_API_KEY"]} if admin else {}
    request = urllib.request.Request(BACKEND_URL + path, method=method, headers=headers,
                                     data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"백엔드 {method} {path} → {error.code}: {error.read().decode('utf-8', 'replace')}") from error


@dag(
    dag_id="property_refresh_pipeline",
    schedule="0 4 5 * *",                       # 매월 5일 04:00 (한국시간): 신고 기한이 지난 두 달 전 거래까지 확정
    start_date=datetime(2026, 12, 1, tzinfo=ZoneInfo("Asia/Seoul")),
    catchup=False,
    max_active_runs=1,
    tags=["property", "collection", "training"],
)
def property_refresh_pipeline():
    @task(retries=2)
    def collect_molit_and_rate() -> dict:
        from scripts.property_data.run_property_refresh import stage_collect
        return stage_collect()

    @task
    def refresh_models(collected: dict) -> dict:
        started = _call("/api/models/property/refresh", "POST", admin=True)
        deadline = time.monotonic() + TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            status = _call("/api/models/property/refresh")
            if status["state"] == "succeeded":
                return {"collected": collected, "report": status.get("last_report")}
            if status["state"] == "failed":
                raise RuntimeError(f"모델 갱신 실패: {status['message']} / {json.dumps(status.get('last_report'), ensure_ascii=False)[:1500]}")
        raise TimeoutError(f"모델 갱신이 {TIMEOUT_SECONDS // 3600}시간 안에 끝나지 않았습니다 (시작: {started.get('started_at')})")

    refresh_models(collect_molit_and_rate())


property_refresh_pipeline()
