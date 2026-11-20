"""Build an observed, relative borough safety reference; not a model backcast."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/processed/child_crime_exposure/borough_panel/panel.csv'
TRAFFIC = ROOT / 'data/raw/94차 모델 데이터/교통사고'
OUTPUT = ROOT / 'data/processed/neighborhood_reference/annual_safety_reference.csv'


def build() -> pd.DataFrame:
    crime = pd.read_csv(PANEL, usecols=['gu', 'year', 'crime', 'child'])
    crime = crime.rename(columns={'gu': 'district'})
    crime['district'] = crime['district'].astype(str).str.strip()
    crime = crime.loc[crime['year'].between(2017, 2025) & crime['child'].gt(0)].copy()
    accident_rows = []
    for path in sorted(TRAFFIC.glob('자치구별_어린이교통사고_*.xlsx')):
        frame = pd.read_excel(path)
        district_col, measure_col = frame.columns[:2]
        frame[district_col] = frame[district_col].ffill().astype(str).str.strip()
        # These nationwide workbooks list Seoul's 25 boroughs immediately after the total.
        # Other cities repeat names (e.g. 중구), so a suffix-only filter would corrupt joins.
        all_accidents = frame.loc[frame[measure_col].eq('사고[건]')]
        accidents = all_accidents.iloc[1:26]
        if len(accidents) != 25 or set(accidents[district_col]) != set(crime.loc[crime.year.eq(2025), 'district']):
            raise ValueError(f'Seoul borough block is incomplete or changed: {path}')
        for column in frame.columns[3:]:
            # The annual total is the first column for each year; following columns are months.
            label = str(column).strip()
            if label.isdigit() and 2017 <= int(label) <= 2025:
                part = accidents[[district_col, column]].copy()
                part.columns = ['district', 'child_accidents']
                part['year'] = int(label)
                accident_rows.append(part)
    if not accident_rows:
        raise ValueError('No annual child traffic accident counts found')
    traffic = pd.concat(accident_rows, ignore_index=True)
    traffic['child_accidents'] = pd.to_numeric(traffic['child_accidents'], errors='coerce')
    merged = crime.merge(traffic, on=['district', 'year'], validate='one_to_one')
    merged['crime_per_10k_children'] = merged['crime'] / merged['child'] * 10000
    merged['traffic_per_10k_children'] = merged['child_accidents'] / merged['child'] * 10000
    merged = merged.dropna(subset=['crime_per_10k_children', 'traffic_per_10k_children'])
    # Never publish a partial year: quintiles require all 25 Seoul boroughs.
    valid_years = merged.groupby('year')['district'].nunique()
    merged = merged.loc[merged['year'].isin(valid_years.index[valid_years.eq(25)])].copy()
    if merged.empty:
        raise ValueError('No year with complete 25-borough coverage')
    for risk, score in [('crime_per_10k_children', 'crime_reference_score'),
                        ('traffic_per_10k_children', 'traffic_reference_score')]:
        merged = merged.sort_values(['year', risk, 'district'])
        merged[score] = 100 - (merged.groupby('year').cumcount() // 5) * 20
    columns = ['district', 'year', 'crime', 'child', 'child_accidents',
               'crime_per_10k_children', 'traffic_per_10k_children',
               'crime_reference_score', 'traffic_reference_score']
    result = merged[columns].sort_values(['district', 'year']).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    return result


if __name__ == '__main__':
    data = build()
    print(f'Wrote {len(data)} reference rows for {sorted(data.year.unique().tolist())} to {OUTPUT}')
