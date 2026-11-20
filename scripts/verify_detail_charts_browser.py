"""Smoke-check that detailed price and district score charts render in the live UI."""
from playwright.sync_api import sync_playwright, expect


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        with page.expect_response(lambda response: '/api/neighborhoods/catalog' in response.url) as catalog_response:
            page.goto('http://127.0.0.1:8080/', wait_until='domcontentloaded')
        print('catalog:', catalog_response.value.status, len(catalog_response.value.json()['districts'][0].get('historical_reference_series', [])))
        expect(page.locator('.map-context')).to_be_visible(timeout=30000)
        page.get_by_role('button', name='지역 검색').click()
        page.locator('#region-search').fill('강남구')
        page.locator('.search-results button').filter(has_text='강남구').first.click()
        page.get_by_role('button', name='가격·점수 자세히 보기').click()
        dialog = page.locator('dialog.service-dialog')
        expect(dialog).to_be_visible()
        expect(dialog.locator('.detail-trend svg')).to_have_count(3)
        assert len(catalog_response.value.json()['districts'][0]['historical_reference_series']) == 9
        assert not errors, errors
        print('District detail: price, historical reference and forecast charts rendered (3 SVGs)')
        browser.close()


if __name__ == '__main__':
    main()
