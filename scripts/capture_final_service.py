"""Read-only screenshots of the LAN service for presentation use."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/presentations/final_service_capture'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1000}, device_scale_factor=1)
        page.goto('http://192.168.0.3:8080', wait_until='networkidle', timeout=60000)
        page.screenshot(path=str(OUT / '01_home.png'), full_page=True)
        print(page.locator('body').inner_text())
        print('INPUTS', page.locator('input').evaluate_all('(els)=>els.map(e=>({placeholder:e.placeholder,type:e.type,id:e.id}))'))
        (OUT / 'page_text.txt').write_text(page.locator('body').inner_text(), encoding='utf-8')
        search = page.locator('#region-search')
        if search.count():
            search.fill('역삼')
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / '02_search.png'), full_page=True)
            print('SEARCH',page.locator('body').inner_text())
            results = page.locator('.search-results button')
            if results.count():
                results.first.click()
                page.wait_for_timeout(2000)
                page.screenshot(path=str(OUT / '03_dong.png'), full_page=True)
                print('SELECTED',page.locator('body').inner_text())
        browser.close()

if __name__ == '__main__':
    main()
