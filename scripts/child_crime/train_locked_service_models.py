"""Recreate selected 2y/3y models without historical feature searches."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from ml.training import compare_child_crime_borough_v2 as old
    from ml.training import optimize_borough_crime_count_3y as crime
    from ml.training import optimize_exposure_2y_2027 as two
    from ml.training import optimize_exposure_rank_3y as rank
    from ml.training import optimize_exposure_integrated_v2 as direct
    from scripts.child_crime import export_service_crime_forecasts as export
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=export.SERVICE)
    parser.add_argument('--base-year', type=int, help='Fit only labels available by this year (also used for backtests).')
    parser.add_argument('--forecast-only', action='store_true', help='Backtest predictions without service scoring artifacts.')
    args = parser.parse_args()
    config = json.loads((ROOT / 'ml/config/child_crime_exposure_locked.json').read_text(encoding='utf-8'))
    data = Path(os.environ.get('CRIME_PROCESSED_ROOT', ROOT / 'data/processed/child_crime_exposure'))
    export.PANEL = data / 'borough_panel/panel.csv'
    panel = pd.read_csv(data / 'borough_panel/panel.csv').merge(
        pd.read_csv(data / 'population_features/priority_features.csv'), on=['gu','year'], validate='one_to_one')
    if args.base_year is not None:
        panel = panel.loc[panel.year.le(args.base_year)].copy()
    latest = panel.loc[panel.year.eq(panel.year.max())].sort_values('gu').copy()
    if len(latest) != 25:
        raise ValueError('Expected 25 Seoul boroughs')
    base_year = int(latest.year.max())
    if args.base_year is not None and base_year != args.base_year:
        raise ValueError('Requested base year is missing from the panel.')
    spec = config['two']
    y2, model2 = two.fit_predict(two.pairs(panel), latest, spec['features'], spec['spec'], spec['mode'])
    bundle2 = {'model': model2, **spec}
    c = config['base']
    history, _, types = crime.add_crime_history(panel)
    latest_new = history.loc[history.year.eq(base_year)].sort_values('gu')
    latest['target_year'] = base_year+3
    columns = list(dict.fromkeys([*old.CORE, *config['old_crime']['features']]))
    models = rank.fitted_crime_models(old.make_pairs(panel,3), latest, columns,
        config['old_crime']['features'], config['old_crime']['alpha'],
        crime.make_pairs(history,types), latest_new, types, c['new_crime_winner'])
    crime_prediction = rank.predict_selected_crime(c['crime_candidate'], models, latest, latest_new,
        c['new_crime_winner'], types)
    pairs3 = rank.population_pairs(panel)
    child_prediction, population_model = rank.fit_population(pairs3, latest,
        c['population_features'], c['population_spec'], c['population_mode'])
    b = config['three']
    direct_y, direct_model = direct.fit_locked(pairs3, latest, {
        'features': b['direct_features'], 'spec': b['direct_spec'], 'mode': b['direct_mode']})
    y3 = b['baseline_weight']*rank.exposure(crime_prediction,child_prediction) + b['direct_weight']*direct_y
    baseline = {**c, 'crime_models': models, 'population_model': population_model,
        'integrated_weights': pd.DataFrame(config['integrated_weights'])}
    bundle3 = {**b, 'baseline_bundle': baseline, 'direct_model': direct_model}
    for horizon,bundle,prediction in ((2,bundle2,y2),(3,bundle3,y3)):
        target = args.output_root / f'{horizon}y'
        target.mkdir(parents=True,exist_ok=True)
        joblib.dump(bundle,target/f'child_crime_exposure_{horizon}y.joblib')
        frame = pd.DataFrame({'district':latest.gu, 'base_year':base_year,
            'prediction_year':base_year+horizon, 'predicted_y':np.asarray(prediction)})
        if args.forecast_only:
            frame.to_csv(target/'service_forecast.csv',index=False,encoding='utf-8-sig')
            continue
        formula = export.ridge_numeric_formula(bundle) if horizon==2 else export.integrated_formula(bundle)
        export.add_service_columns(frame,'predicted_y',formula,horizon,target).to_csv(
            target/'service_forecast.csv',index=False,encoding='utf-8-sig')
        metadata = target/'model_metadata.json'
        current_metadata = dict(bundle['metadata'])
        current_metadata['refresh'] = {'base_year': base_year, 'prediction_year': base_year+horizon,
            'algorithm_policy': 'locked hyperparameters; refit on available labels',
            'historical_metrics_note': 'Original experiment metrics; see validation.json for this refresh.',
            'weight_policy': 'refitted standardized Ridge' if horizon == 2 else 'locked integrated sensitivity formula (not newly estimated)'}
        metadata.write_text(json.dumps(current_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        export.write_scoring_metadata(metadata,formula)
    print(json.dumps({'output':str(args.output_root),'horizons':[2,3],'base_year':base_year}))


if __name__=='__main__':
    main()
