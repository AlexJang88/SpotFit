"""Rebuild lightweight service chart aggregates when raw property data is newer."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/raw/부동산'
OUTPUTS = (
    ROOT / 'data/processed/property_price_history/annual_by_dong.csv',
    ROOT / 'data/processed/property_price_history/annual_monthly_by_dong.csv',
)
SOURCES = (RAW / '아파트_매매_2010_2026.csv', RAW / '아파트_전월세_2010_2026.csv')


def start_price_history_bootstrap() -> None:
    if not all(path.is_file() for path in SOURCES):
        logger.info('Price history not built: source transaction files are unavailable')
        return
    newest_source = max(path.stat().st_mtime for path in SOURCES)
    if all(path.is_file() and path.stat().st_mtime >= newest_source for path in OUTPUTS):
        return
    script = ROOT / 'scripts/property_data/build_price_history.py'
    if not script.is_file():
        logger.warning('Price history build script unavailable: %s', script)
        return

    def run() -> None:
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', check=False)
        if result.returncode:
            logger.error('Price history rebuild failed: %s', (result.stderr or result.stdout)[-2000:])
            return
        from app.services.final_model_artifacts import clear_caches
        clear_caches()
        logger.info('Price history rebuilt: %s', result.stdout.strip())

    Thread(target=run, name='price-history-bootstrap', daemon=True).start()
