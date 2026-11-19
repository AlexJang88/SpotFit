"""Monthly public downloads and validated crime-model refresh. Annual inputs only."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from airflow.sdk import dag, task


def call(method, payload=None):
    url = os.environ.get('CRIME_REFRESH_BACKEND_URL', 'http://backend:8000') + '/api/models/crime/refresh'
    key = os.environ.get('MODEL_TRAINING_API_KEY', '')
    if not key:
        raise RuntimeError('MODEL_TRAINING_API_KEY is required on both Airflow and backend')
    request = Request(url, method=method, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={'X-Model-Admin-Key': key, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


@dag(dag_id='crime_refresh_pipeline', schedule='0 23 L * *',
     start_date=datetime(2026, 11, 1, tzinfo=ZoneInfo('Asia/Seoul')), catchup=False,
     max_active_runs=1, tags=['crime', 'collection', 'training'])
def crime_refresh_pipeline():
    @task(retries=2, retry_delay=timedelta(minutes=5))
    def collect_sources():
        from scripts.child_crime.collect_refresh import collect
        report = collect()
        # Neither an empty response nor a failed core download is a successful collection.
        if 'crime' not in report['sources'] or 'migration' not in report['sources']:
            raise RuntimeError('Core download failed; consult raw/child_crime_exposure/refresh manifest')
        return report

    @task(execution_timeout=timedelta(hours=5))
    def refresh(collected):
        started = call('POST', {'collection_run': collected['run_id']})
        deadline = time.monotonic() + 4*3600 + 120
        while time.monotonic() < deadline:
            time.sleep(30)
            state = call('GET')
            if state['started_at'] != started['started_at']:
                raise RuntimeError('Backend restarted or job identity changed')
            if state['state'] == 'failed':
                raise RuntimeError(state['message'])
            if state['state'] == 'succeeded':
                return {'collection_run': collected['run_id'], 'result': state['last_report']}
        raise TimeoutError('Crime refresh timed out; inspect backend before retrying')

    refresh(collect_sources())


crime_refresh_pipeline()
