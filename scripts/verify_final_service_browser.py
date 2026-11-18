"""Browser smoke check against the local development service (no data writes)."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        with page.expect_response(lambda r: '/api/neighborhoods/catalog' in r.url) as response:
            page.goto('http://127.0.0.1:8080/', wait_until='domcontentloaded')
        assert response.value.status == 200
        assert response.value.json()['transaction'] == 'sale'
        expect(page.locator('.map-context')).to_contain_text('추천', timeout=30000)
        print('initial catalog OK', flush=True)

        page.get_by_role('button', name='전용면적', exact=True).click()
        page.locator('#area').fill('84')
        print('area panel:', page.locator('#discovery-panel').inner_text(), flush=True)
        page.get_by_role('button', name='평', exact=True).click()
        pyeong = float(page.locator('#area').input_value())
        assert abs(pyeong - 84 / (400 / 121)) < .0001
        page.get_by_role('button', name='㎡', exact=True).click()
        assert float(page.locator('#area').input_value()) == 84
        print('area conversion OK', flush=True)
        page.get_by_role('button', name='설정 닫기', exact=True).click()
        page.get_by_role('button', name='거래 유형', exact=True).click()
        with page.expect_response(lambda r: '/api/neighborhoods/catalog' in r.url and 'transaction=wolse' in r.url) as response:
            page.get_by_role('button', name='월세', exact=True).click()
        assert response.value.status == 200
        data = response.value.json()
        assert data['transaction'] == 'wolse'
        print('monthly catalog OK', flush=True)
        quote_count = sum(r['wolse']['available'] for r in data['neighborhoods'])
        assert quote_count > 0
        expect(page.locator('.map-context')).to_contain_text('지도에 5곳 추천', timeout=30000)
        page.get_by_role('button', name='보증금·월세 예산 설정 →', exact=True).click()
        with page.expect_response(lambda r: '/api/neighborhoods/catalog' in r.url and 'deposit_manwon=10000' in r.url) as response:
            page.locator('#deposit').fill('10000')
        assert response.value.status == 200
        assert response.value.json()['deposit_manwon'] == 10000
        expect(page.locator('.map-context')).to_contain_text('보증금 1억원')
        page.get_by_role('button', name='설정 닫기', exact=True).click()
        page.get_by_role('button', name='예측연도', exact=True).click()
        with page.expect_response(lambda r: '/api/neighborhoods/catalog' in r.url and 'horizon=3' in r.url) as response:
            page.get_by_role('button', name='2028년', exact=True).click()
        data = response.value.json()
        assert data['traffic_prediction_year'] == data['crime_prediction_year'] == 2028
        page.get_by_role('button', name='설정 닫기', exact=True).click()
        page.get_by_role('button', name='지역 검색', exact=True).click()
        page.locator('#region-search').fill('무악동')
        page.locator('.search-results button').filter(has_text='무악동').click()
        expect(page.locator('.selection-card')).to_contain_text('대표 연 전환율')
        expect(page.locator('.selection-card')).to_contain_text('예상 월세 범위')
        output = ROOT / 'artifacts/final_service_smoke'
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / 'monthly-desktop.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output / 'monthly-mobile.png'))
        assert not errors, errors
        print(json.dumps({'passed': True, 'monthly_dongs': quote_count,
                          'area_toggle': '84m2 <-> 25.41평', 'years': [2027,2028],
                          'screenshots': str(output)}, ensure_ascii=False))
        browser.close()


if __name__ == '__main__':
    main()
