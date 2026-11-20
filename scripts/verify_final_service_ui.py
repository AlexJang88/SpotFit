"""Verify compact controls, budget-range recommendations and map interaction."""
import argparse
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_service_ui_review'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until='domcontentloaded', timeout=60000)
        expect(page.locator('.recommend-marker')).to_have_count(5, timeout=120000)
        assert page.locator('.recommendations, .recommend-card, .service-sidebar, .app-header').count() == 0
        assert page.locator('#discovery-panel').count() == 0
        assert page.locator('.discovery-bar').bounding_box()['height'] < 90
        page.screenshot(path=str(OUT / '01_desktop.png'))

        page.get_by_role('button', name='예산 범위', exact=True).click()
        expect(page.locator('#discovery-panel')).to_be_visible()
        page.locator('#budget-min').fill('43000')
        page.locator('#budget').fill('60000')
        expect(page.locator('.map-context')).to_contain_text('4.3억원–6억원')
        expect(page.locator('.map-context')).not_to_contain_text('업데이트', timeout=30000)
        response = page.request.get(args.url + '/api/neighborhoods/recommendations?budget_manwon=60000&min_budget_manwon=43000')
        assert response.ok
        rows = response.json()['recommendations']
        assert len({r['district'] for r in rows}) == len(rows)
        assert all(43000 <= r['estimated_amount_manwon'] <= 60000 for r in rows)
        catalog = page.request.get(args.url + '/api/neighborhoods/catalog?horizon=2').json()
        for row in rows:
            prices = [round(n['apart_sale_amount'], 4) for n in catalog['neighborhoods']
                      if n['district'] == row['district'] and n['apart_sale_amount']
                      and 43000 <= round(n['apart_sale_amount'], 4) <= 60000]
            assert row['estimated_amount_manwon'] == min(prices)

        lower = page.get_by_role('slider', name='예산 하한 슬라이더')
        lower.focus()
        lower.press('ArrowRight')
        expect(page.locator('#budget-min')).to_have_value('44000')
        # Drag the upper thumb (native range), verifying mouse and text stay in sync.
        upper = page.get_by_role('slider', name='예산 상한 슬라이더')
        rect = upper.bounding_box()
        track_width = rect['width'] - 24
        x = rect['x'] + 12 + 60000 / 300000 * track_width
        y = rect['y'] + rect['height'] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(rect['x'] + 12 + 100000 / 300000 * track_width, y, steps=12)
        page.mouse.up()
        assert 95000 <= float(page.locator('#budget').input_value()) <= 105000
        page.screenshot(path=str(OUT / '06_budget_range.png'))
        # Reversed ranges must clear stale map recommendations.
        page.locator('#budget-min').fill('110000')
        expect(page.locator('.error')).to_be_visible()
        expect(page.locator('.recommend-marker')).to_have_count(0)
        page.locator('#budget-min').fill('0')
        page.locator('#budget').fill('1')
        expect(page.locator('.empty-state')).to_be_visible()
        page.locator('#budget').fill('50000')
        page.get_by_role('button', name='설정 닫기').click()

        page.get_by_role('button', name='전용면적', exact=True).click()
        expect(page.locator('.area-presets button')).to_have_count(3, timeout=30000)
        sale = page.request.get(args.url + '/api/neighborhoods/area-presets?transaction=sale').json()
        assert sale['basis'] == 'transactions'
        assert [r['label'] for r in sale['options']] == ['84㎡대', '59㎡대', '114㎡대']
        page.locator('.area-presets button').first.click()
        expect(page.locator('#area')).to_have_value(str(sale['options'][0]['area_m2']))
        page.screenshot(path=str(OUT / '07_area_presets.png'))
        page.keyboard.press('Escape')
        assert page.locator('#discovery-panel').count() == 0

        page.get_by_role('button', name='거래 유형', exact=True).click()
        page.get_by_role('button', name='전세', exact=True).click()
        page.get_by_role('button', name='전용면적', exact=True).click()
        expect(page.locator('.area-presets')).to_contain_text('49㎡대')
        page.locator('#area').fill('84')
        page.get_by_role('button', name='설정 닫기').click()
        expect(page.locator('.recommend-marker')).to_have_count(5, timeout=30000)
        page.locator('.recommend-marker').first.click()
        expect(page.locator('.selection-card')).to_contain_text('전용 84㎡')
        page.get_by_role('button', name='가격·점수 자세히 보기').click()
        expect(page.locator('dialog')).to_be_visible()
        page.get_by_role('button', name='상세 닫기').click()
        page.get_by_role('button', name='지역 요약 닫기').click()

        page.get_by_role('button', name='예측연도', exact=True).click()
        with page.expect_response(lambda r: '/api/neighborhoods/catalog?horizon=3' in r.url) as response:
            page.get_by_role('button', name='2028년', exact=True).click()
        assert response.value.status == 200
        page.get_by_role('button', name='지역 검색', exact=True).click()
        page.locator('#region-search').fill('역삼1동')
        page.locator('.search-results button').filter(has_text='역삼1동').click()
        expect(page.locator('.selection-card h2')).to_contain_text('역삼1동')
        page.get_by_role('button', name='지역 요약 닫기').click()
        page.get_by_role('button', name='지역·이용 안내', exact=True).click()
        expect(page.locator('.boundary-status')).to_be_visible()
        page.keyboard.press('Escape')

        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, device_scale_factor=1)
        mobile.on('pageerror', lambda e: errors.append(str(e)))
        mobile.goto(args.url, wait_until='domcontentloaded', timeout=60000)
        expect(mobile.locator('.recommend-marker')).to_have_count(5, timeout=120000)
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert mobile.locator('.discovery-bar').bounding_box()['height'] < 90
        mobile.screenshot(path=str(OUT / '05_mobile.png'), full_page=True)
        mobile.get_by_role('button', name='예산 범위', exact=True).click()
        expect(mobile.locator('#budget-min')).to_be_visible()
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.screenshot(path=str(OUT / '08_mobile_budget.png'), full_page=True)
        mobile.get_by_role('button', name='전용면적', exact=True).click()
        expect(mobile.locator('.area-presets button')).to_have_count(3)
        mobile.get_by_role('button', name='설정 닫기').click()
        mobile.locator('.recommend-marker').first.click()
        expect(mobile.locator('.selection-card')).to_be_visible()
        browser.close()
    assert not errors, errors
    print('PASS: compact toolbar, no bottom cards, budget bounds, keyboard/mouse slider, invalid/empty range, transaction-based area presets, map/search/detail/year, mobile; no JS errors')


if __name__ == '__main__':
    main()
