"""Read-only browser walkthrough for the final presentation (no backend changes)."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/presentations/final_20261116/captures'
OUT.mkdir(parents=True, exist_ok=True)
results = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 900}, device_scale_factor=1)
    page.goto('http://127.0.0.1:8080', wait_until='domcontentloaded')
    expect(page.locator('.recommend-marker')).to_have_count(5, timeout=60000)
    page.screenshot(path=str(OUT / '01_home.png'))
    results.append('기본 화면: 추천 마커 5개 표시')
    page.get_by_role('button', name='예산 범위', exact=True).click()
    page.locator('#budget-min').fill('30000')
    page.locator('#budget').fill('60000')
    expect(page.locator('.map-context')).not_to_contain_text('업데이트', timeout=30000)
    page.screenshot(path=str(OUT / '02_budget.png'))
    response = page.request.get('http://127.0.0.1:8080/api/neighborhoods/recommendations?budget_manwon=60000&min_budget_manwon=30000')
    assert response.ok
    data = response.json()['recommendations']
    assert len(data) == 5 and len({r['district'] for r in data}) == 5
    assert all(30000 <= r['estimated_amount_manwon'] <= 60000 for r in data)
    results.append('예산 3~6억원: API 추천 5개, 서로 다른 구, 전부 예산 범위 내')
    page.get_by_role('button', name='설정 닫기').click()
    page.locator('.recommend-marker').first.click()
    expect(page.locator('.selection-card')).to_be_visible()
    page.screenshot(path=str(OUT / '03_recommend.png'))
    page.get_by_role('button', name='가격·점수 자세히 보기').click()
    expect(page.locator('dialog')).to_be_visible()
    page.screenshot(path=str(OUT / '04_detail.png'))
    results.append('추천 마커 → 요약 카드 → 상세 다이얼로그 표시')
    page.get_by_role('button', name='상세 닫기').click()
    page.get_by_role('button', name='지역 요약 닫기').click()
    page.get_by_role('button', name='지역 검색', exact=True).click()
    page.locator('#region-search').fill('역삼1동')
    page.locator('.search-results button').filter(has_text='역삼1동').click()
    expect(page.locator('.selection-card h2')).to_contain_text('역삼1동')
    page.screenshot(path=str(OUT / '05_search.png'))
    results.append('역삼1동 검색: 해당 행정동 요약 카드 표시')
    page.get_by_role('button', name='지역 요약 닫기').click()
    page.get_by_role('button', name='예측연도', exact=True).click()
    page.get_by_role('button', name='2028년', exact=True).click()
    response = page.request.get('http://127.0.0.1:8080/api/neighborhoods/catalog?horizon=3')
    assert response.ok
    results.append('2028년 선택 및 horizon=3 카탈로그 API HTTP 200')
    browser.close()
(OUT.parent / 'capture_checks.json').write_text(json.dumps({'checked_at':'2026-11-16','checks':results}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False))
