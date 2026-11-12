"""Smoke-test the running local final service with the real model outputs."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_service_ui_review'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto('http://127.0.0.1:5181', wait_until='networkidle', timeout=60000)
        button = page.get_by_role('button', name='예산에 맞는 동네 5곳 찾기 →')
        button.wait_for()
        page.wait_for_function("!document.querySelector('.primary-button').disabled")
        page.screenshot(path=str(OUT / '01_desktop.png'))
        button.click()
        page.locator('.recommend-card').first.wait_for()
        assert page.locator('.recommend-card').count() == 5
        page.locator('.recommend-card').first.click()
        assert '동네점수' in page.locator('.selection-card').inner_text()
        page.screenshot(path=str(OUT / '02_recommendation.png'))
        search = page.locator('#region-search')
        search.fill('역삼1동')
        page.locator('.search-results button').first.click()
        assert page.locator('.selection-card h2').inner_text().startswith('역삼1동')
        assert '강남구 동네점수' in page.locator('.selection-card').inner_text()
        page.screenshot(path=str(OUT / '03_dong.png'))
        page.get_by_role('button', name='가격·점수 자세히 보기').click()
        assert page.locator('dialog').is_visible()
        page.get_by_role('button', name='상세 닫기').click()
        with page.expect_response(lambda r: '/api/neighborhoods/catalog?horizon=3' in r.url) as response:
            page.get_by_role('button', name='2028년 3년 모델').click()
        assert response.value.status == 200
        assert page.locator('.recommend-card').count() == 0
        page.wait_for_function("document.querySelector('.map-context').textContent.includes('2028')")
        page.locator('#budget').fill('1')
        button.click()
        page.locator('.empty-state').wait_for()
        assert page.locator('.recommend-card').count() == 0
        page.get_by_role('button', name='전세', exact=True).click()
        page.locator('#budget').fill('50000')
        page.locator('#area').fill('84')
        button.click()
        page.locator('.recommend-card').first.wait_for()
        page.locator('.recommend-card').first.click()
        assert '전용 84㎡' in page.locator('.selection-card').inner_text()
        page.screenshot(path=str(OUT / '04_rent_area.png'))
        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, device_scale_factor=1)
        mobile.on('pageerror', lambda e: errors.append(str(e)))
        mobile.goto('http://127.0.0.1:5181', wait_until='networkidle', timeout=60000)
        mobile.wait_for_function("!document.querySelector('.primary-button').disabled")
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.screenshot(path=str(OUT / '05_mobile.png'), full_page=True)
        browser.close()
    assert not errors, errors
    print('PASS: desktop, top5, selection, dong parent score, detail, horizon, empty budget, rent/area, mobile; no JS errors')


if __name__ == '__main__':
    main()
