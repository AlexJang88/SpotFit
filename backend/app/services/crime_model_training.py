from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from app.services.child_crime_exposure import clear_forecast_cache, model_status

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "child_crime" / "rebuild_service_crime_models.py"
LOG = ROOT / "artifacts" / "crime_service_training.log"
_LOCK = Lock()
_STATE = {"state": "idle", "started_at": None, "finished_at": None, "exit_code": None, "message": ""}


def training_status() -> dict:
    return {**_STATE, "models": model_status(), "log_path": str(LOG)}


def _run() -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        _STATE.update(state="succeeded" if result.returncode == 0 else "failed", exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat(), message="재학습 완료" if result.returncode == 0 else "재학습 실패: 로그를 확인하세요.")
        if result.returncode == 0:
            clear_forecast_cache()
    except Exception as exc:
        _STATE.update(state="failed", exit_code=-1, finished_at=datetime.now(timezone.utc).isoformat(), message=str(exc))
    finally:
        _LOCK.release()


def start_training(force: bool = False) -> dict:
    if not force and model_status()["ready"]:
        return {**training_status(), "message": "모델이 이미 준비되어 있어 재학습하지 않았습니다."}
    if not SCRIPT.is_file():
        raise RuntimeError(f"재학습 스크립트가 없습니다: {SCRIPT}")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("범죄 모델 재학습이 이미 실행 중입니다.")
    _STATE.update(state="running", started_at=datetime.now(timezone.utc).isoformat(), finished_at=None, exit_code=None, message="재학습을 시작했습니다.")
    Thread(target=_run, name="crime-model-training", daemon=True).start()
    return training_status()


def verify_admin_key(value: str | None) -> None:
    expected = os.getenv("MODEL_TRAINING_API_KEY", "")
    if not expected:
        raise RuntimeError("MODEL_TRAINING_API_KEY가 설정되지 않아 재학습 API가 비활성화되어 있습니다.")
    import secrets
    if value is None or not secrets.compare_digest(value, expected):
        raise PermissionError("모델 관리 키가 올바르지 않습니다.")
