"""Read-only checks for the presentation; no training or service mutations."""
import json
import math
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

OUT = Path('artifacts/presentations/template_review')


def get(path):
    with urlopen('http://127.0.0.1:8080' + path, timeout=60) as response:
        return json.load(response)


def main():
    checks = []
    payloads = {}
    for h in (1, 2):
        data = get(f'/api/safety/crime/district-forecasts?horizon={h}')
        rows = data['forecasts']
        assert len(rows) == data['district_count'] == 25
        assert len({r['district'] for r in rows}) == 25
        assert sorted(r['predicted_rank'] for r in rows) == list(range(1, 26))
        assert data['data_year'] == 2025 and data['prediction_year'] == 2025 + h
        payloads[str(h)] = data
        checks.append({'name': f'{h}y_forecast_api', 'passed': True, 'details': '25 districts; ranks 1–25; correct base/target year'})
    one = get('/api/safety/crime/district-forecasts?district=%EA%B0%95%EB%82%A8%EA%B5%AC')
    assert one['district_count'] == 1
    assert one['forecasts'][0] == next(r for r in payloads['1']['forecasts'] if r['district'] == '강남구')
    try:
        get('/api/safety/crime/district-forecasts?district=unknown')
        raise AssertionError('Expected 404')
    except HTTPError as error:
        assert error.code == 404
    checks.append({'name': 'district_filter_and_404', 'passed': True})
    for data in payloads.values():
        for r in data['forecasts']:
            bucket = (r['predicted_rank'] - 1) // 5
            assert r['predicted_grade'] == 'ABCDE'[bucket]
            assert r['predicted_score'] == 100 - 20 * bucket
            assert r['risk_rank'] + r['predicted_rank'] == 26
            assert abs(math.expm1(r['predicted_exposure_y']) - r['predicted_exposure_rate_per_10k']) < .1
    checks.append({'name': 'score_grade_and_formula_consistency', 'passed': True, 'rows': 50})
    result = {'checked_on': '2026-11-10', 'scope': 'read-only local API smoke checks; not full regression tests', 'checks': checks, 'payloads': payloads}
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
