"""Export real model price outputs for installations without raw CSV inputs.

Without --source-url, run local property models. With an explicit trusted API
URL, snapshot its existing model outputs and retain source/time provenance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-url')
    args = parser.parse_args()
    if args.source_url:
        with urlopen(args.source_url, timeout=120) as response:
            source = json.load(response)
        models = {"sale": source['models']['apart_sale'], "rent": source['models']['apart_rent']}
        rows = source['neighborhoods']
        provenance = {"type": "model_api_snapshot", "source_url": args.source_url}
    else:
        sys.path.insert(0, str(ROOT / 'backend'))
        from app.services.recommender_v1 import _property_predictions
        _, sale, sale_meta = _property_predictions('apart_sale')
        _, rent, rent_meta = _property_predictions('apart_rent')
        rows = sale.merge(rent, on=['district', 'neighborhood'], how='outer', validate='one_to_one').to_dict('records')
        models = {"sale": sale_meta, "rent": rent_meta}
        provenance = {"type": "local_model_snapshot"}
    for row in rows:
        row['id'] = f"{row['district']}|{row['neighborhood']}"
        for key in ['apart_sale_amount', 'apart_rent_amount', 'apart_sale_per_pyeong', 'apart_rent_per_pyeong']:
            value = row[key]
            row[key] = float(value) if value is not None and math.isfinite(float(value)) else None
    assert rows and len({(r['district'], r['neighborhood']) for r in rows}) == len(rows)
    assert all(model['is_future_forecast'] is False for model in models.values())
    provenance['exported_at_utc'] = datetime.now(timezone.utc).isoformat()
    out = ROOT / 'models/recommender/v1/property_service_forecasts.json'
    out.write_text(json.dumps({'schema_version': 1, 'neighborhoods': rows, 'property_models': models,
                               'provenance': provenance}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Exported {len(rows)} real model outputs to {out}')


if __name__ == '__main__':
    main()
