"""Run the heavy half of the monthly property refresh (merge raw -> inputs -> retrain -> validate -> replace models)
in a child process and reload the service caches when it succeeds.

The Airflow DAG downloads the raw months (light, standard library only) and then calls this through
POST /api/models/property/refresh, because only the backend image has pandas / xgboost / scikit-learn.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "artifacts" / "property_refresh" / "refresh.log"
REPORT = ROOT / "artifacts" / "property_refresh" / "last_run.json"
_LOCK = Lock()
_STATE: dict = {"state": "idle", "started_at": None, "finished_at": None, "exit_code": None, "message": ""}


def refresh_status() -> dict:
    report = None
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {**_STATE, "log_path": str(LOG), "last_report": report}


def _clear_service_caches() -> None:
    from app.services.neighborhood_recommendations import clear_neighborhood_cache
    from app.services.recommender_v1 import clear_recommender_cache
    clear_recommender_cache()
    clear_neighborhood_cache()


def _run() -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} property refresh =====\n")
            log.flush()
            result = subprocess.run([sys.executable, "-m", "scripts.property_data.run_property_refresh", "--stage", "process"],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        ok = result.returncode == 0
        if ok:
            _clear_service_caches()                     # the new models are served from the next request on
        _STATE.update(state="succeeded" if ok else "failed", exit_code=result.returncode,
                      message="모델 갱신 완료" if ok else "모델 갱신 실패: 기존 모델을 그대로 사용 중입니다. 로그와 last_report를 확인하세요.")
        logger.log(logging.INFO if ok else logging.ERROR, "property refresh finished: exit_code=%s", result.returncode)
    except Exception as error:
        _STATE.update(state="failed", exit_code=-1, message=str(error))
        logger.exception("property refresh crashed")
    finally:
        _STATE["finished_at"] = datetime.now(timezone.utc).isoformat()
        _LOCK.release()


def start_refresh() -> dict:
    if not (ROOT / "scripts" / "property_data" / "run_property_refresh.py").is_file():
        raise RuntimeError("scripts/property_data/run_property_refresh.py가 없습니다.")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("부동산 모델 갱신이 이미 실행 중입니다.")
    _STATE.update(state="running", started_at=datetime.now(timezone.utc).isoformat(), finished_at=None,
                  exit_code=None, message="")
    Thread(target=_run, name="property-refresh", daemon=True).start()
    return refresh_status()
