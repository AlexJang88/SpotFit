"""Create a small, explicit-allowlist Codex handoff; never export secrets or raw records.

No service mutation, training, git commit/push, source-file rewriting, or deletion.
Aggregated borough panels are included; raw datasets and joblib binaries are not.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/handoff'
OUT = ROOT / 'artifacts/handoff'
STAMP = '20261110'
TEMPLATE = Path('C:/Users/M/Desktop/SPOT_FIT_발표자료.pptx')

ALLOWLIST = [
    'docs/handoff/README.md',
    'docs/handoff/PROJECT_STATE.md',
    'docs/handoff/MODEL_RESULTS.json',
    'scripts/package_spotfit_handoff.py',
    'scripts/build_spotfit_project_deck.py',
    'scripts/build_spotfit_crime_improvements_deck.py',
    'scripts/inspect_spotfit_template.py',
    'scripts/render_spotfit_presentation.ps1',
    'scripts/audit_spotfit_presentation.ps1',
    'scripts/verify_spotfit_deck_evidence.py',
    'frontend/src/views/pages/HomePage.tsx',
    'frontend/src/views/pages/readability-refresh.css',
    'backend/app/services/child_crime_exposure.py',
    'backend/app/services/crime_model_training.py',
    'backend/app/data/admin_boundary_registry.json',
    'ml/training/train_service_crime_exposure_1y.py',
    'ml/training/optimize_exposure_2y_2027.py',
    'ml/training/optimize_exposure_rank_3y.py',
    'ml/training/optimize_borough_crime_count_3y.py',
    'ml/training/optimize_exposure_integrated_v2.py',
    'scripts/child_crime/rebuild_service_crime_models.py',
    'scripts/child_crime/export_service_crime_forecasts.py',
    'scripts/child_crime/build_borough_experiment_panel.py',
    'scripts/child_crime/inspect_borough_inputs.py',
    'scripts/child_crime/build_priority_population_features.py',
    'scripts/child_infrastructure/build_child_environment_scores.py',
    'ml/training/train_child_environment_forecasts.py',
    'ml/inference/predict_child_environment.py',
    'ml/etl/property_amount/train_and_save_final_models.py',
    'docs/crime-exposure-service.md',
    'docs/admin-boundary-automation.md',
    'docs/architecture/spotfit-erd-draft.md',
    'requirements-dev.txt',
    'requirements-server.txt',
    'artifacts/presentations/SPOT_FIT_프로젝트발표_템플릿유지_20261110.pptx',
    'artifacts/presentations/SPOT_FIT_범죄모델_개선과정_5장_20261110.pptx',
    'artifacts/presentations/template_review/service-ui.png',
    'artifacts/presentations/template_review/verification.json',
    'artifacts/presentations/crime_improvement_review/contact.jpg',
    'artifacts/presentations/crime_improvement_review/overflow-audit.json',
    'artifacts/crime_rate_3y/REPORT.md',
    'artifacts/crime_rate_3y/모델설명.json',
    'artifacts/crime_rate_3y/기준모델부터_성능개선기록.csv',
    'artifacts/child_crime_borough_v2/panel.csv',
    'artifacts/child_crime_borough_v2/feature_groups.json',
    'artifacts/child_crime_borough_v2/feature_dictionary.csv',
    'artifacts/child_crime_borough_v2/feature_coverage.csv',
    'artifacts/child_crime_borough_v2/source_audit.json',
    'artifacts/child_crime_borough_v2/experiment_report.md',
    'artifacts/child_crime_borough_v2/final_summary.csv',
    'artifacts/child_population_priority_v1/priority_features.csv',
    'artifacts/child_population_priority_v1/feature_groups.json',
    'artifacts/child_crime_exposure_rank_3y_v1/result.json',
    'artifacts/child_crime_exposure_rank_3y_v1/experiment_report.md',
    'artifacts/child_crime_exposure_integrated_v2/result.json',
    'artifacts/child_crime_exposure_2y_2027_v1/result.json',
    'artifacts/child_crime_exposure_2y_2027_v1/forecast_2027.csv',
    'artifacts/child_crime_exposure_2y_2027_v1/holdout_2025_predictions.csv',
    'artifacts/child_crime_exposure_2y_2027_v1/all_comparisons.csv',
    'models/recommender/v1/child_crime_exposure/service_manifest.json',
    'models/recommender/v1/child_crime_exposure/1y/model_metadata.json',
    'models/recommender/v1/child_crime_exposure/1y/service_forecast.csv',
    'models/recommender/v1/child_crime_exposure/1y/raw_forecast.csv',
    'models/recommender/v1/child_crime_exposure/2y/model_metadata.json',
    'models/recommender/v1/child_crime_exposure/2y/service_forecast.csv',
    'models/recommender/v1/child_environment/model_metadata.json',
    'models/recommender/v1/child_environment/REPORT.md',
    'data/processed/child_environment_scores/FORMULA.md',
    'data/processed/child_environment_scores/weights.csv',
    'data/processed/child_environment_scores/scored_training_panel_updated.csv',
    'models/recommender/v1/apart_sale/model_contract.json',
    'models/recommender/v1/apart_rent/model_contract.json',
    'models/recommender/v1/traffic_accident/모델정보.json',
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8', errors='replace').strip()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(relative, hash_file=False):
    path = ROOT / relative
    result = {'path': relative, 'exists': path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size, sha256=sha(path) if hash_file else None)
    return result


def directory_record(relative):
    path = ROOT / relative
    entries = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else []
    return {'path': relative, 'exists': path.is_dir(), 'files': len(entries),
            'bytes': sum(p.stat().st_size for p in entries),
            'note': 'Directory inventory only; NOT automatically included in ZIP'}


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    one_path = 'models/recommender/v1/child_crime_exposure/1y/model_metadata.json'
    two_path = 'models/recommender/v1/child_crime_exposure/2y/model_metadata.json'
    one, two = read(one_path), read(two_path)
    report = {
        'as_of': '2026-11-10',
        'target': {'name': '아동 지역범죄 노출지수', 'formula': 'ln(1 + five_major_crimes / resident_children_age_0_17 * 10000)',
                   'not_actual_child_victimization_rate': True, 'log_vs_raw_ablation_verified': False},
        'service_horizons': [1,2],
        'service_1y': {'source': one_path, 'base_year': 2025, 'target_year': 2026, 'metadata': one, 'separate_holdout_in_metadata': False},
        'service_2y': {'source': two_path, 'base_year': 2025, 'target_year': 2027,
                       'metadata': {key: two[key] for key in ['horizon','training_origin_years','training_target_years','n_training_pairs','cv_years','locked_holdout','baseline_cv','strict_forward_history','strict_winner_cv','baseline_holdout','strict_holdout','continuous_y_tradeoff']}},
        'historical_3y_rank_research': {'source': 'artifacts/child_crime_exposure_rank_3y_v1/result.json',
                                       'not_deployed_as_current_1y_or_2y': True,
                                       'metrics': read('artifacts/child_crime_exposure_rank_3y_v1/result.json')},
        'historical_3y_blend_research': {'source': 'artifacts/child_crime_exposure_integrated_v2/result.json',
                                        'not_deployed_as_current_1y_or_2y': True,
                                        'blend': read('artifacts/child_crime_exposure_integrated_v2/result.json')['blend']},
        'interpretation_rules': ['Spearman and R2 are not classification accuracy percentages.',
                                 'Compare improvements only within same target/horizon/validation conditions.',
                                 'Current 2y holdout quintile accuracy improved but rank MAE and Y RMSE worsened.',
                                 'Service score = predicted Y ascending rank quintile, not feature-weighted score.',
                                 'CV used for model selection is not independent future accuracy.'],
    }
    write_json(DOCS/'MODEL_RESULTS.json', report)

    safe_paths = []
    for rel in ALLOWLIST:
        p = (ROOT / rel).resolve()
        if ROOT not in p.parents:
            raise ValueError(f'Outside workspace allowlist entry: {rel}')
        if p.name.startswith('.env') or p.suffix.lower() in {'.joblib','.pkl','.pem','.key'} or 'data/raw/' in rel:
            raise ValueError(f'Forbidden export category: {rel}')
        safe_paths.append(rel)
    packaged = [file_record(p, True) for p in safe_paths]
    model_files = [file_record(p.relative_to(ROOT).as_posix(), p.stat().st_size <= 2_000_000)
                   for p in (ROOT/'models/recommender/v1').rglob('*') if p.is_file() and p.suffix in {'.joblib','.json','.csv','.md'}]
    packages = {}
    for name in ('python-pptx','pandas','numpy','scikit-learn','joblib','Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    inventory_dirs = [
        'data/raw/아동범죄','data/raw/아동범죄_행정동','data/raw/아동인프라',
        'data/interim','data/processed','models/recommender/v1','models/experiments',
        'artifacts/child_crime_borough_v2','artifacts/child_population_priority_v1',
        'artifacts/child_crime_exposure_2y_2027_v1',
        'artifacts/child_crime_exposure_rank_3y_v1','artifacts/child_crime_exposure_integrated_v2',
    ]
    manifest = {
        'as_of': '2026-11-10', 'original_workspace': str(ROOT),
        'purpose': 'Context, selected source snapshots, presentation assets, metadata and aggregated panels; not a full deploy backup',
        'git': {'branch': git('branch','--show-current'), 'head': git('rev-parse','HEAD'),
                'head_subject': git('log','-1','--format=%s'), 'status_porcelain': git('-c','core.quotepath=false','status','--short')},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
                        'fonts_required_not_bundled': ['네모고딕','네모고딕 Medium','네모고딕 Light']},
        'original_template': {'path': str(TEMPLATE), 'exists': TEMPLATE.is_file(), 'sha256': sha(TEMPLATE) if TEMPLATE.is_file() else None,
                              'archive_path': 'external/SPOT_FIT_발표자료.pptx'},
        'packaged_files': packaged,
        'excluded_categories': ['credentials/.env','raw individual records','joblib model binaries','Docker persistent volumes','node_modules','browser profiles'],
        'runtime_asset_inventory_not_all_packaged': model_files,
        'transfer_directory_inventory': [directory_record(p) for p in inventory_dirs],
        'notes': ['SHA256 for model inventory is omitted for files >2 MB; byte counts alone are not integrity verification.',
                  'A manifest entry for a runtime asset does NOT mean the asset is in this archive.',
                  'Manifest itself is included but not self-hashed.',
                  'No git commits, pushes or model retraining are performed by this script.'],
    }
    write_json(DOCS/'TRANSFER_MANIFEST.json', manifest)
    patch = subprocess.check_output(['git','diff','HEAD','--binary','--','frontend/src/views/pages/HomePage.tsx'], cwd=ROOT)
    archive = OUT/f'SpotFit_Codex_handoff_{STAMP}.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('START_HERE.txt', 'Read workspace/docs/handoff/README.md first.\nExtract to a separate folder and merge deliberately; do not overwrite existing work.\nNo credentials, raw datasets, joblib binaries or Docker volumes included.\n')
        for entry in packaged:
            if entry['exists']:
                z.write(ROOT/entry['path'], 'workspace/'+entry['path'])
        z.write(DOCS/'TRANSFER_MANIFEST.json', 'workspace/docs/handoff/TRANSFER_MANIFEST.json')
        z.writestr('git_tracked_changes.patch', patch)
        if TEMPLATE.is_file():
            z.write(TEMPLATE, 'external/SPOT_FIT_발표자료.pptx')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in packaged:
            if entry['exists']:
                data = z.read('workspace/'+entry['path'])
                assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
        for name in z.namelist():
            assert not Path(name).name.startswith('.env')
            assert not name.lower().endswith(('.joblib','.pkl','.pem','.key'))
    missing = [e['path'] for e in packaged if not e['exists']]
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
                      'packaged_files': sum(e['exists'] for e in packaged), 'missing_allowlist_files': missing,
                      'zip_crc_and_file_hashes_verified': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
