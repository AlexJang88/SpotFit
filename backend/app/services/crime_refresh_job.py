"""Authenticated background refresh; detailed errors stay in administrator logs."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / 'artifacts/crime_refresh/refresh.log'
REPORT = LOG.with_name('last_run.json')
_LOCK = Lock()
_STATE = {'state': 'idle', 'started_at': None, 'finished_at': None, 'message': ''}
logger = logging.getLogger(__name__)


def refresh_status():
    try:
        report = json.loads(REPORT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        report = None
    return {**_STATE, 'last_report': report}


def _run(collection_run=None):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open('a', encoding='utf-8') as log:
            log.write(f'\n{datetime.now(timezone.utc).isoformat()} crime refresh started\n'); log.flush()
            command = [sys.executable, '-m', 'scripts.child_crime.run_refresh', '--stage', 'process']
            if collection_run:
                command.extend(['--collection-run', collection_run])
            result = subprocess.run(command,
                                    cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONIOENCODING': 'utf-8'},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=4*3600, check=False)
        if result.returncode:
            raise RuntimeError(f'Refresh exited {result.returncode}')
        from app.services.child_crime_exposure import clear_forecast_cache
        from app.services.neighborhood_recommendations import clear_neighborhood_cache
        clear_forecast_cache(); clear_neighborhood_cache()
        _STATE.update(state='succeeded', message='처리 결과를 확인하세요. 자료 부족·검증 미통과 시 기존 모델을 유지합니다.')
        logger.info('crime refresh finished')
    except Exception:
        logger.exception('crime refresh failed; active release retained')
        _STATE.update(state='failed', message='범죄모델 갱신 실패. 기존 모델을 유지합니다. 관리자 로그를 확인하세요.')
    finally:
        _STATE['finished_at'] = datetime.now(timezone.utc).isoformat()
        _LOCK.release()


def start_refresh(collection_run=None):
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('범죄 데이터 갱신이 이미 실행 중입니다.')
    _STATE.update(state='running', started_at=datetime.now(timezone.utc).isoformat(), finished_at=None, message='')
    try:
        Thread(target=_run, args=(collection_run,), name='crime-refresh', daemon=True).start()
    except Exception:
        _LOCK.release()
        raise
    return refresh_status()
