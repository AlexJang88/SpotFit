"""Historical observed proxy, intentionally separate from forecast model scores."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'data/processed/neighborhood_reference/annual_safety_reference.csv'


@lru_cache(maxsize=1)
def reference_frame() -> pd.DataFrame:
    if not PATH.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(PATH, encoding='utf-8-sig')
        expected = {'district', 'year', 'crime_reference_score', 'traffic_reference_score'}
        if not expected.issubset(frame.columns):
            raise ValueError(f'missing columns: {sorted(expected - set(frame.columns))}')
        return frame
    except (OSError, ValueError, pd.errors.ParserError):
        LOGGER.exception('Historical safety reference could not be read: %s', PATH)
        return pd.DataFrame()


def reference_series_by_district(crime_weight: float, traffic_weight: float) -> dict[str, list[dict]]:
    frame = reference_frame()
    if frame.empty:
        return {}
    frame = frame.copy()
    frame['score'] = (frame['crime_reference_score'] * crime_weight +
                      frame['traffic_reference_score'] * traffic_weight).round(4)
    return {str(name): [{'year': int(row.year), 'score': float(row.score),
                         'kind': 'historical_reference'} for row in group.itertuples()]
            for name, group in frame.sort_values('year').groupby('district')}
