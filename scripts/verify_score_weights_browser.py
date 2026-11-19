"""Read-only browser smoke test against the development service at localhost:8080.

Requires Playwright and Microsoft Edge. Settings only affect this isolated context.
Run: python scripts/verify_score_weights_browser.py
"""
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def weighted_response(response, endpoint, percent):
    query = parse_qs(urlparse(response.url).query)
    return (f'/api/neighborhoods/{endpoint}?' in response.url
            and query.get('crime_weight') == [str(percent / 100)])


def main():
    output = Path(__file__).resolve().parents[1] / 'artifacts/score_weights_smoke'
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:8080/', wait_until='domcontentloaded')
        page.get_by_role('button', name='점수 반영 비율 설정', exact=True).click()
        expect(page.locator('#crime-weight')).to_have_value('50')
        expect(page.locator('#discovery-panel [role=status]')).to_contain_text('적용 완료', timeout=60000)

        for percent, label in [(70, '범죄 중심'), (30, '교통 중심')]:
            with page.expect_response(lambda r: weighted_response(r, 'catalog', percent), timeout=60000) as catalog_response:
                with page.expect_response(lambda r: weighted_response(r, 'recommendations', percent), timeout=60000) as recommendation_response:
                    page.get_by_role('button', name=label, exact=True).click()
            assert catalog_response.value.status == recommendation_response.value.status == 200
            catalog = catalog_response.value.json()
            recommendations = recommendation_response.value.json()
            assert abs(catalog['weights']['crime'] - percent / 100) < 1e-9
            districts = {row['district']: row for row in catalog['districts']}
            for row in districts.values():
                expected = row['crime_score'] * percent / 100 + row['traffic_score'] * (100 - percent) / 100
                assert abs(row['score'] - expected) < .0001
            for row in recommendations['recommendations']:
                assert row['district_score'] == districts[row['district']]
            assert [sum(row['grade'] == grade for row in districts.values()) for grade in 'ABCDE'] == [5] * 5
            expect(page.locator('#discovery-panel [role=status]')).to_contain_text('적용 완료')
        print('Weighted catalog, recommendations and grades OK', flush=True)

        # Native keyboard range interaction, 1% increment, and persistence.
        page.locator('#crime-weight').focus()
        with page.expect_response(lambda r: weighted_response(r, 'catalog', 31), timeout=60000):
            page.locator('#crime-weight').press('ArrowRight')
        expect(page.locator('#crime-weight')).to_have_value('31')
        page.reload(wait_until='domcontentloaded')
        page.get_by_role('button', name='점수 반영 비율 설정', exact=True).click()
        expect(page.locator('#crime-weight')).to_have_value('31')
        expect(page.locator('#discovery-panel [role=status]')).to_contain_text('적용 완료', timeout=60000)
        page.screenshot(path=str(output / 'desktop.png'))
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            for selector in ['.discovery-bar', '#discovery-panel']:
                box = page.locator(selector).bounding_box()
                assert box and box['x'] >= 0 and box['x'] + box['width'] <= width + 1
            page.screenshot(path=str(output / f'mobile-{width}.png'))
        page.get_by_role('button', name='기본값 50:50으로 초기화', exact=True).click()
        expect(page.locator('#crime-weight')).to_have_value('50')
        expect(page.locator('#discovery-panel [role=status]')).to_contain_text('적용 완료', timeout=60000)
        page.locator('#crime-weight').press('Escape')
        expect(page.locator('#discovery-panel')).to_have_count(0)
        expect(page.get_by_role('button', name='점수 반영 비율 설정', exact=True)).to_be_focused()
        assert not errors, errors
        print(f'PASS: slider, persistence, reset, responsive layout and keyboard navigation. Screenshots: {output}')
        browser.close()


if __name__ == '__main__':
    main()
