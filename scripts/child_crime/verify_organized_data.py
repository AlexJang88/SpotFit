"""Verify regenerated inputs against original data and both deployed forecasts."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from ml.training import optimize_exposure_rank_3y as rank
    from scripts.child_crime import export_service_crime_forecasts as export
    out = ROOT / 'data/processed/child_crime_exposure'
    audit = ROOT / 'artifacts/data_cleanup_20261117'
    comparisons = {}
    for old_name, new_name in [('before_panel.csv', 'borough_panel/panel.csv'),
                               ('before_priority_features.csv', 'population_features/priority_features.csv')]:
        old = pd.read_csv(audit / old_name).set_index(['gu', 'year']).sort_index()
        new = pd.read_csv(out / new_name).set_index(['gu', 'year']).sort_index()
        assert old.index.equals(new.index)
        common = old.columns.intersection(new.columns)
        np.testing.assert_allclose(old[common].to_numpy(float), new[common].to_numpy(float), rtol=1e-11, atol=1e-10, equal_nan=True)
        comparisons[new_name] = {'rows': len(new), 'matching_columns': len(common), 'removed_unused_columns': sorted(set(old)-set(new))}
    latest = pd.read_csv(out / 'inference/latest_features.csv').sort_values('gu')
    checks = {}
    for horizon in (2, 3):
        target = ROOT / f'models/recommender/v1/child_crime_exposure/{horizon}y'
        bundle = joblib.load(target / f'child_crime_exposure_{horizon}y.joblib')
        if horizon == 2:
            predicted = bundle['model'].predict(latest[['gu', *bundle['features']]])
            if bundle['mode'] == 'delta':
                predicted += latest.y.to_numpy()
            formula = export.ridge_numeric_formula(bundle)
        else:
            base = bundle['baseline_bundle']
            crime = rank.predict_selected_crime(base['crime_candidate'], base['crime_models'], latest, latest,
                                                base['new_crime_winner'], base['type_counts'])
            child = rank.fitted_population_predict(base['population_model'], latest, base['population_features'], base['population_mode'])
            baseline = rank.exposure(crime, child)
            direct = bundle['direct_model'].predict(latest[['gu', *bundle['direct_features']]])
            if bundle['direct_mode'] == 'delta':
                direct += latest.y.to_numpy()
            if bundle['direct_mode'] not in ('direct', 'delta'):
                raise ValueError('Unsupported deployed direct model mode')
            predicted = bundle['baseline_weight'] * baseline + bundle['direct_weight'] * direct
            formula = export.integrated_formula(bundle)
        expected = pd.read_csv(target / 'service_forecast.csv').sort_values('district')
        assert latest.gu.tolist() == expected.district.tolist()
        np.testing.assert_allclose(predicted, expected.predicted_y.to_numpy(), rtol=1e-10, atol=1e-10)
        scratch = audit / f'check_{horizon}y'
        scratch.mkdir(exist_ok=True)
        rescored = export.add_service_columns(expected[['district','base_year','prediction_year','predicted_y']],
                                             'predicted_y', formula, horizon, scratch).sort_values('district')
        np.testing.assert_allclose(rescored.safety_score.to_numpy(), expected.safety_score.to_numpy(), rtol=1e-11, atol=1e-10)
        assert rescored.safety_grade.tolist() == expected.safety_grade.tolist()
        assert rescored.safety_rank.tolist() == expected.safety_rank.tolist()
        checks[f'{horizon}y'] = {'boroughs':25, 'max_prediction_difference':float(np.max(np.abs(predicted-expected.predicted_y.to_numpy()))),
            'scores_ranks_grades_unchanged': True, 'weight_sum_pct': float(formula.weight_pct.sum())}
    result = {'passed': True, 'preprocessing': comparisons, 'deployed_models': checks}
    (audit / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
