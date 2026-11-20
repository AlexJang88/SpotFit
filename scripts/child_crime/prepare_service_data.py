"""Rebuild crime-only raw inputs into training and inference datasets."""
from pathlib import Path
import os
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('CRIME_PROCESSED_ROOT', ROOT / 'data/processed/child_crime_exposure'))


def main():
    for script in ('build_borough_experiment_panel.py', 'build_priority_population_features.py'):
        subprocess.run([sys.executable, str(Path(__file__).with_name(script))], cwd=ROOT, check=True)
    import pandas as pd
    from ml.training import compare_child_crime_borough_v2 as core
    from ml.training import optimize_borough_crime_count_3y as crime
    panel = pd.read_csv(OUT / 'borough_panel/panel.csv')
    expected_years = list(range(int(panel.year.min()), int(panel.year.max()) + 1))
    for gu, rows in panel.groupby('gu'):
        if sorted(rows.year.tolist()) != expected_years:
            raise ValueError(f'{gu}: missing/duplicate years would corrupt history features')
    priority = pd.read_csv(OUT / 'population_features/priority_features.csv')
    merged = panel.merge(priority, on=['year', 'gu'], how='left', validate='one_to_one')
    merged, _, _ = crime.add_crime_history(merged)
    for folder in ('training', 'inference'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    summary = {}
    for horizon in (2, 3):
        pairs = core.make_pairs(merged, horizon)
        pairs.to_csv(OUT / f'training/{horizon}y.csv', index=False, encoding='utf-8-sig')
        summary[f'{horizon}y'] = {'rows': len(pairs), 'origin_years': [int(pairs.year.min()), int(pairs.year.max())],
            'target_years': [int(pairs.target_year.min()), int(pairs.target_year.max())]}
    year = int(panel.year.max())
    latest = merged.loc[merged.year.eq(year)].sort_values('gu')
    if len(latest) != 25 or latest.gu.nunique() != 25:
        raise ValueError('Inference dataset must contain the 25 Seoul boroughs.')
    latest.to_csv(OUT / 'inference/latest_features.csv', index=False, encoding='utf-8-sig')
    (OUT / 'dataset_manifest.json').write_text(json.dumps({
        'raw_directory': 'data/raw/child_crime_exposure', 'preprocessing_directory': str(OUT.relative_to(ROOT)),
        'target': 'log(1 + crime / child_population_0_17 * 10000)', 'training': summary,
        'inference': {'rows': len(latest), 'base_year': year, 'target_years': [year+2, year+3]},
        'model_outputs': 'models/recommender/v1/child_crime_exposure (model, metadata, delivery forecasts)',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
