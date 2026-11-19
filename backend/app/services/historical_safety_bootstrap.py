"""Refresh the historical reference chart from observed source data on startup."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from app.services.historical_safety_reference import PATH, ROOT, reference_frame

LOGGER = logging.getLogger(__name__)
PANEL = ROOT / 'data/processed/child_crime_exposure/borough_panel/panel.csv'
TRAFFIC = ROOT / 'data/raw/94차 모델 데이터/교통사고'


def start_historical_safety_bootstrap() -> None:
    sources = [PANEL, *TRAFFIC.glob('자치구별_어린이교통사고_*.xlsx')]
    if len(sources) < 3 or not all(path.is_file() for path in sources):
        LOGGER.info('Historical safety reference unavailable: crime panel or traffic files missing')
        return
    if PATH.is_file() and PATH.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return
    script = ROOT / 'scripts/build_historical_safety_reference.py'

    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', check=False)
    if result.returncode:
        LOGGER.error('Historical safety reference rebuild failed: %s', (result.stderr or result.stdout)[-2000:])
        return
    reference_frame.cache_clear()
    LOGGER.info('Historical safety reference rebuilt: %s', result.stdout.strip())
