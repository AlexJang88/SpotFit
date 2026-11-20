"""Crime refresh: isolated inputs -> preprocessing -> temporal backtest -> refit -> atomic release.

python -m scripts.child_crime.run_refresh --stage all
Use --stage process --force to validate/retrain unchanged local inputs.
No failed/partial run changes the active service release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/child_crime_exposure'
PROCESSED = ROOT / 'data/processed/child_crime_exposure'
SERVICE = ROOT / 'models/recommender/v1/child_crime_exposure'
REPORT = ROOT / 'artifacts/crime_refresh/last_run.json'
CRIME_FILE = '서울시_5대범죄_2014_2025_원자료.csv'
MIGRATION_FILE = '서울시 인구이동(2015년 이후) 통계.csv'


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preflight():
    """Fail before hashing/copying gigabytes if the backend image is out of date."""
    from importlib.util import find_spec
    required = ('numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'openpyxl')
    missing = [name for name in required if find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing training dependencies: {missing}. Rebuild the backend image with requirements-server.txt.')


@contextmanager
def exclusive_run():
    """OS-released cross-process lock, works on Linux and Windows, including crashes."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / 'refresh.lock').open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if handle.read(1) == b'':
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_crime(frame, approved_missing=None):
    import numpy as np
    from scripts.child_crime.build_borough_experiment_panel import GU
    keys = ['년도', '시군구', '범죄']
    if not set([*keys, '발생건수', '체포수']).issubset(frame.columns) or frame.duplicated(keys).any():
        raise ValueError('Crime schema/duplicate keys invalid')
    if set(frame['범죄']) != {'강간·강제추행', '강도', '살인', '절도', '폭력'}:
        raise ValueError('Crime categories changed; manual mapping review required')
    values = frame[['발생건수', '체포수']].to_numpy(float)
    missing = {(int(r['년도']), r['시군구'], r['범죄'], col)
               for r in frame.to_dict('records') for col in ('발생건수', '체포수') if np.isnan(float(r[col]))}
    if missing - set(approved_missing or []):
        raise ValueError(f'Unapproved missing crime counts: {sorted(missing - set(approved_missing or []))}')
    finite = values[~np.isnan(values)]
    if not np.isfinite(finite).all() or (finite < 0).any() or not (finite == np.floor(finite)).all():
        raise ValueError('Crime counts contain missing/negative/non-integer values')
    for year, rows in frame.groupby('년도'):
        if set(rows['시군구']) != set(GU) or len(rows) != 125:
            raise ValueError(f'{year}: expected 25 boroughs x 5 crimes')


def merge_download(existing, incoming, keys, values):
    """Keep old coverage; never silently rewrite conflicting historic observations."""
    import pandas as pd
    if incoming.duplicated(keys).any():
        raise ValueError(f'Download has duplicate keys: {keys}')
    left, right = existing.set_index(keys), incoming.set_index(keys)
    overlap = left.index.intersection(right.index)
    a, b = left.loc[overlap, values], right.loc[overlap, values]
    if not ((a == b) | (a.isna() & b.isna())).all().all():
        raise ValueError('Published historical values changed; review revisions before replacement')
    return pd.concat([left, right.loc[right.index.difference(left.index)]]).reset_index()


def command(script, *args, env):
    subprocess.run([sys.executable, str(ROOT / 'scripts/child_crime' / script), *map(str, args)],
                   cwd=ROOT, env=env, check=True, timeout=7200)


def input_gate(panel, config):
    """No forward-fill or median imputation can conceal missing serving-year features."""
    import numpy as np
    latest = panel.loc[panel.year.eq(panel.year.max())]
    features = set(config['two']['features'] + config['old_crime']['features'] +
                   config['base']['population_features'] + config['base']['new_crime_winner']['features'] +
                   config['three']['direct_features'])
    missing = sorted(c for c in features if c not in latest or not np.isfinite(latest[c].to_numpy(float)).all())
    if len(latest) != 25 or latest.gu.nunique() != 25 or missing:
        raise ValueError(f'Latest year is incomplete; missing features: {missing}')
    years = sorted(panel.year.unique())
    if years != list(range(int(min(years)), int(max(years)) + 1)):
        raise ValueError('Missing years would corrupt lag features')
    return int(latest.year.max())


def backtest(data: Path, env: dict, base_year: int) -> dict:
    import numpy as np
    import pandas as pd
    from ml.training.optimize_exposure_rank_3y import rank_metrics
    panel = pd.read_csv(data / 'borough_panel/panel.csv')
    predictions = {2: [], 3: []}
    target_years = list(range(base_year - 2, base_year + 1))
    for origin in range(base_year - 5, base_year - 1):
        out = data / 'backtest' / str(origin)
        command('train_locked_service_models.py', '--base-year', origin, '--forecast-only', '--output-root', out, env=env)
        for h in (2, 3):
            if origin + h not in target_years:
                continue
            forecast = pd.read_csv(out / f'{h}y/service_forecast.csv')
            actual = panel.loc[panel.year.eq(origin+h), ['gu', 'y']].rename(columns={'gu': 'district', 'y': 'actual'})
            baseline = panel.loc[panel.year.eq(origin), ['gu', 'y']].rename(columns={'gu': 'district', 'y': 'baseline'})
            rows = forecast.merge(actual, on='district', validate='one_to_one').merge(baseline, on='district', validate='one_to_one')
            if len(rows) != 25:
                raise ValueError('Backtest needs 25 boroughs per target year')
            predictions[h].append(rows)
    report = {}
    for h, parts in predictions.items():
        rows = pd.concat(parts, ignore_index=True)
        if len(rows) != 75:
            raise ValueError('Three complete temporal folds required')
        if not np.isfinite(rows[['actual', 'predicted_y', 'baseline']]).all().all():
            raise ValueError('Nonfinite backtest predictions')
        metric = lambda column: rank_metrics(rows.actual.to_numpy(), rows[column].to_numpy(), rows.prediction_year.to_numpy())
        candidate, baseline = metric('predicted_y'), metric('baseline')
        # Fixed acceptance thresholds, not a new hyperparameter search on the holdout.
        passed = (candidate['Spearman'] >= max(.8, baseline['Spearman'] - .01)
                  and candidate['rank_MAE'] <= min(3., baseline['rank_MAE'] + .5)
                  and candidate['exact_quintile_pct'] >= 50)
        report[str(h)] = {'target_years': target_years, 'candidate': candidate, 'persistence': baseline, 'passed': passed}
        rows.to_csv(data / f'backtest/{h}y_predictions.csv', index=False, encoding='utf-8-sig')
    return report


def validate_artifacts(path: Path, base_year: int):
    import numpy as np
    import pandas as pd
    for h in (2, 3):
        folder = path / f'{h}y'
        frame = pd.read_csv(folder / 'service_forecast.csv')
        formula = pd.read_csv(folder / 'feature_score_formula.csv')
        detail = pd.read_csv(folder / 'feature_score_breakdown.csv')
        if len(frame) != 25 or frame.district.nunique() != 25:
            raise ValueError('Release borough count invalid')
        if not frame.base_year.eq(base_year).all() or not frame.prediction_year.eq(base_year+h).all():
            raise ValueError('Release years invalid')
        if not np.isfinite(frame[['predicted_y', 'safety_score']]).all().all() or not frame.safety_score.between(20-1e-8,100+1e-8).all():
            raise ValueError('Invalid predictions/scores')
        if not np.isclose(formula.weight_pct.sum(), 100) or (formula.weight_pct < 0).any():
            raise ValueError('Invalid weights')
        if detail.duplicated(['district', 'feature']).any() or len(detail) != 25*len(formula):
            raise ValueError('Incomplete scoring breakdown')
        calculated = detail.groupby('district').weighted_contribution.sum().sort_index()
        if not np.allclose(calculated, frame.set_index('district').safety_score.sort_index()):
            raise ValueError('Score sum mismatch')
        if sorted(frame.safety_rank) != list(range(1, 26)) or frame.safety_grade.value_counts().to_dict() != dict.fromkeys('ABCDE', 5):
            raise ValueError('Invalid ranks/grades')
        if not (folder / f'child_crime_exposure_{h}y.joblib').is_file():
            raise ValueError('Missing fitted model')


def process(force=False, promote=True, collection_run=None) -> dict:
    preflight()
    import pandas as pd
    from importlib.metadata import version
    config_path = ROOT / 'ml/config/child_crime_exposure_locked.json'
    raw_hashes = {p.name: digest(p) for p in RAW.iterdir() if p.is_file()}
    fingerprint = {'raw': raw_hashes}
    fingerprint['runtime'] = {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'openpyxl')}
    fingerprint['locked_config'] = digest(config_path)
    if collection_run is not None and (not isinstance(collection_run, str) or not collection_run.isalnum()):
        raise ValueError('Invalid collection run identifier')
    source_report = RAW / 'refresh' / collection_run / 'manifest.json' if collection_run else RAW / 'refresh/latest.json'
    if collection_run and not source_report.is_file():
        raise ValueError('Requested collection manifest is missing')
    collection = json.loads(source_report.read_text(encoding='utf-8')) if source_report.exists() else {}
    collected = RAW / 'refresh' / collection.get('run_id', 'unavailable')
    # Content hashes, not request timestamps/counts: revisions must retrain and
    # identical monthly downloads must not create another release.
    download_hashes = {p.name: digest(p) for p in collected.glob('*.csv')}
    fingerprint['downloads'] = download_hashes
    fingerprint['pipeline_code'] = {str(p.relative_to(ROOT)): digest(p)
                                    for p in sorted((ROOT / 'scripts/child_crime').glob('*.py'))}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    pointer = SERVICE / 'current_release.json'
    active = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else {}
    if not force and active.get('input_signature') == signature:
        return {'state': 'skipped', 'reason': 'No changed inputs', 'release': active['release']}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    data = PROCESSED / 'refresh' / run_id
    inputs = data / 'inputs'
    inputs.mkdir(parents=True)
    # Large MDIS files are read-only hardlinks when supported; hashes are checked again before promotion.
    for source in RAW.iterdir():
        if source.is_file():
            target = inputs / source.name
            if '_연간자료_' in source.name:
                try:
                    os.link(source, target)
                    continue
                except OSError:
                    pass
            shutil.copy2(source, target)
    for name in ('business_density','student_movement','household_structure','births_deaths'):
        if name in collection.get('sources', {}):
            shutil.copy2(collected/f'{name}.csv',inputs/f'{name}_refresh.csv')
    if (collected / 'crime.csv').exists():
        existing, incoming = pd.read_csv(inputs / CRIME_FILE), pd.read_csv(collected / 'crime.csv')
        # Explicitly preserve the original model's known sparse-cell convention.
        # Do NOT extend it to newly missing cells/years or claim these are observed zeros.
        approved = {(int(r['년도']), r['시군구'], r['범죄'], col)
                    for r in existing.to_dict('records') for col in ('발생건수', '체포수') if int(r['년도']) <= 2025 and pd.isna(r[col])}
        validate_crime(existing, approved); validate_crime(incoming, approved)
        write_json(data / 'legacy_missing_counts.json', {'cells': sorted(approved),
                   'policy': 'Existing aggregation sums sparse cells as zero; retained for reproducibility, not confirmed true zeros. New missing cells block refresh.'})
        merge_download(existing, incoming, ['년도','시군구','범죄'], ['발생건수','체포수']).to_csv(inputs / CRIME_FILE,index=False,encoding='utf-8-sig')
    if (collected / 'migration.csv').exists():
        existing, incoming = pd.read_csv(inputs / MIGRATION_FILE), pd.read_csv(collected / 'migration.csv')
        keys = ['통계연도', '구명', '동명', '성별명', '이동유형명']
        normalized = []
        for frame in (existing, incoming):
            frame = frame.loc[pd.to_numeric(frame['통계연도'], errors='coerce').notna()].copy()
            frame['통계연도'] = pd.to_numeric(frame['통계연도']).astype(int)
            frame['인구수'] = pd.to_numeric(frame['인구수'], errors='raise')
            normalized.append(frame)
        existing, incoming = normalized
        merge_download(existing, incoming, keys, ['인구수']).to_csv(inputs / MIGRATION_FILE,index=False,encoding='utf-8-sig')
    for source in (collected.glob('*연령별인구현황_연간.csv') if 'population' in collection.get('sources', {}) else []):
        target = inputs / source.name
        downloaded = pd.read_csv(source, encoding='cp949', dtype=str)
        if target.exists():
            old = pd.read_csv(target, encoding='cp949', dtype=str)
            # Ignore other provinces; only Seoul is a model input.
            # Only total-by-single-age columns are model inputs; newer downloads
            # may additionally contain male/female columns with identical totals.
            normalize = lambda d: d.loc[d['행정구역'].str.startswith('서울특별시')].set_index('행정구역').filter(regex='_계_').sort_index().sort_index(axis=1)
            if not normalize(old).equals(normalize(downloaded)):
                raise ValueError('MOIS historical Seoul population revised; review before replacement')
        shutil.copy2(source, target)
    from scripts.child_crime.collect_licences import KINDS
    for kind, korean in KINDS.items():
        name = f'서울시 {korean}영업 인허가 정보.csv'
        if kind not in collection.get('sources', {}):
            continue
        incoming = pd.read_csv(collected / name, dtype=str).fillna('')
        existing = pd.read_csv(inputs / name, encoding='cp949', dtype=str).fillna('')
        if incoming['관리번호'].duplicated().any():
            raise ValueError('Duplicate licence IDs')
        # Keep archived licences missing from the provider's current snapshot.
        retained = existing.loc[~existing['관리번호'].isin(incoming['관리번호'])]
        pd.concat([retained, incoming], ignore_index=True).to_csv(inputs / name,index=False,encoding='utf-8-sig')
    write_json(data / 'input_manifest.json', fingerprint)
    env = {**os.environ, 'CRIME_RAW_ROOT': str(inputs), 'CRIME_PROCESSED_ROOT': str(data),
           'PYTHONPATH': str(ROOT), 'PYTHONIOENCODING': 'utf-8'}
    command('prepare_service_data.py', env=env)
    panel = pd.read_csv(data / 'inference/latest_features.csv')
    base_year = input_gate(panel, json.loads(config_path.read_text(encoding='utf-8')))
    crime_year = int(pd.read_csv(inputs / CRIME_FILE)['년도'].max())
    if crime_year != base_year:
        raise ValueError(f'Crime year {crime_year} and population year {base_year} differ; await complete annual sources')
    validation = backtest(data, env, base_year)
    write_json(data / 'validation.json', validation)
    if not all(v['passed'] for v in validation.values()):
        return {'state': 'rejected', 'reason': 'Temporal quality gate failed; active release unchanged', 'validation': validation, 'data': str(data)}
    release = SERVICE / 'releases' / run_id
    command('train_locked_service_models.py', '--output-root', release, env=env)
    validate_artifacts(release, base_year)
    write_json(release / 'validation.json', validation)
    result = {'state': 'validated', 'release': run_id, 'input_signature': signature, 'base_year': base_year,
              'target_years': [base_year+2, base_year+3], 'data': str(data), 'validation': validation,
              'collection_errors': collection.get('errors', {}),
              'manual_sources': ['MDIS annual business census'],
              'source_policy': 'Unavailable downloads retain existing inputs. New forecast years require complete lagged model features.'}
    # Detect edits to linked MDIS or any input while the long-running job was executing.
    if (any(digest(RAW / name) != value for name, value in raw_hashes.items())
            or any(digest(collected / name) != value for name, value in download_hashes.items())
            or digest(config_path) != fingerprint['locked_config']):
        raise ValueError('Inputs changed during refresh; refusing promotion')
    if promote:
        # Do not break combined service by activating crime years without matching other domains.
        sys.path.insert(0, str(ROOT / 'backend'))
        from app.services.final_model_artifacts import traffic_forecasts, property_forecasts
        for h in (2, 3):
            traffic, _ = traffic_forecasts(h)
            if not traffic.prediction_year.eq(base_year+h).all():
                result.update(state='awaiting_companion_models', reason='Traffic target year mismatch')
                return result
            properties = property_forecasts(h, target_year=base_year+h)
            if any(properties['property_models'][kind]['prediction_year'] != base_year+h for kind in ('sale','rent')):
                result.update(state='awaiting_companion_models', reason='Property target year mismatch')
                return result
        write_json(release / 'pipeline_manifest.json', result)
        # One atomic pointer switches both horizons; previous release remains available for rollback.
        write_json(pointer, {**result, 'state': 'promoted', 'previous_release': active.get('release')})
        result['state'] = 'promoted'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('collect','process','all'), default='all')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-promote', action='store_true')
    parser.add_argument('--collection-run', help='Pin processing to one completed collection snapshot.')
    args = parser.parse_args()
    report = {'started_at': datetime.now(timezone.utc).isoformat()}
    try:
        with exclusive_run():
            if args.stage in ('collect', 'all'):
                from scripts.child_crime.collect_refresh import collect
                report['collection'] = collect()
            if args.stage in ('process', 'all'):
                report['process'] = process(args.force, not args.no_promote,
                                            report.get('collection', {}).get('run_id') or args.collection_run)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if 'error' in report or report.get('process', {}).get('state') == 'rejected' else 0


if __name__ == '__main__':
    raise SystemExit(main())
