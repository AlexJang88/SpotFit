"""Five insert-ready crime-model slides using the completed project deck's theme."""
from pathlib import Path
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches

from build_spotfit_project_deck import (
    ROOT, OUT, GREEN, GREY, DARK, PALE, MEDIUM,
    fresh, text, rect, panel, table, chart, note,
)

SOURCE = OUT / 'SPOT_FIT_프로젝트발표_템플릿유지_20261110.pptx'
DEST = OUT / 'SPOT_FIT_범죄모델_개선과정_5장_20261110.pptx'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def banner(s, content, y=3.25):
    rect(s, 1.25, y, 17.5, .8, PALE)
    text(s, content, 1.5, y+.15, 17, .5, 24, DARK, MEDIUM)


def footer(s, content):
    text(s, content, 1.25, 10.12, 17.5, .45, 17, GREY)


def main():
    prs = Presentation(SOURCE)
    original_count = len(prs.slides)
    old = pd.read_csv(ROOT / 'artifacts/crime_rate_3y/기준모델부터_성능개선기록.csv')
    rank = read('artifacts/child_crime_exposure_rank_3y_v1/result.json')
    integrated = read('artifacts/child_crime_exposure_integrated_v2/result.json')
    one = read('models/recommender/v1/child_crime_exposure/1y/model_metadata.json')['winner']
    two = read('models/recommender/v1/child_crime_exposure/2y/model_metadata.json')
    b, core, best = two['baseline_cv'], *two['strict_forward_history']
    bh, h = two['baseline_holdout'], two['strict_holdout']

    # 1. Earliest recorded crime-rate study: same-target comparisons only.
    s = fresh(prs, '범죄모델 개선 ① · 초기 범죄율 모델', 'Crime Model Evolution · Original Baseline')
    banner(s, '출발점: 현재 범죄율 유지 → 인구·안전자원 Feature를 학습하는 3년 예측')
    text(s, '초기 Y = 5대범죄 발생건수 ÷ 전체 인구 × 10,000  |  목표: 3년 뒤 범죄율의 오차 감소', 1.25, 4.22, 17.5, .55, 21, GREY)
    combinations = ['현재 범죄율만 유지', '현재 범죄율 + 안전자원·건강', '위 조합 + 인구·가구', '동일 11개 Feature · Ridge로 변경']
    methods = ['Persistence', 'Ridge α=1', 'ElasticNet', 'Ridge α=1']
    rows = [['단계', '모델 기법', '변경한 Feature / 로직', 'CV RMSE ↓', '직전 대비 개선']]
    for i, row in old.iterrows():
        rows.append([f'{i}  '+('기준' if i==0 else '개선'), methods[i], combinations[i], f'{row.CV_RMSE:.4f}', '—' if i==0 else f'{row["직전기록대비_RMSE개선_pct"]:.2f}%'])
    table(s, rows, 1.25, 5.0, 17.5, 3.05, [1.6, 2.7, 7, 2.6, 3.1], 21)
    rect(s, 1.25, 8.45, 8.5, 1.3, 'FAFAFA')
    text(s, '기준 대비 CV RMSE 4.60% 감소', 1.5, 8.6, 8, .5, 26, GREEN, MEDIUM)
    text(s, '2025 별도검증 RMSE: 17.40 → 12.85  /  26.18% 감소', 1.5, 9.16, 8, .38, 18, GREY)
    rect(s, 10.05, 8.45, 8.7, 1.3, 'FAFAFA')
    text(s, '추가 Feature의 구체적인 내용', 10.3, 8.57, 8.2, .45, 22, GREEN, MEDIUM)
    text(s, '인구·세대·남성비율·증감률 / 경찰관서·CCTV·스트레스', 10.3, 9.16, 8.2, .4, 19)
    footer(s, '초기 모델은 전체 인구 기준입니다. 이후 아동인구 기반 Y와 오차 수치를 직접 비교하지 않습니다.')
    note(s, '초기 범죄율 3년 연구는 64개 그룹 조합×8개 알고리즘=512개 학습모델을 비교했다. CV 목표연도 2022–2024, 별도검증 2025. 최종 11개 Feature: 범죄율_1만명당, 인구수, 세대수, 세대당인구, 남성비율, 인구수_전년변화율, 세대수_전년변화율, 경찰관서_1만명당, CCTV_1만명당, CCTV_1만명당_전년변화율, 스트레스인지율. CV MAE·Spearman은 기준보다 악화했으므로 모든 지표 개선으로 주장하지 않는다. 이 페이지의 개선율은 동일한 초기 Y 내 비교만 의미한다.', ['artifacts/crime_rate_3y/REPORT.md','artifacts/crime_rate_3y/기준모델부터_성능개선기록.csv'])

    # 2. A changed target and objective, with measured three-year research results.
    s = fresh(prs, '범죄모델 개선 ② · 아동 노출과 순위로 전환', 'Target Redesign · Component Models & Ensemble')
    banner(s, '새 Y = log(1 + 5대범죄 발생건수 ÷ 0–17세 아동인구 × 10,000)')
    panel(s, '로직 1  범죄·아동인구 분리 예측', '범죄건수와 아동인구를 각각 예측\n두 예측값으로 최종 Y와 순위 계산\n각 모델의 오차보다 최종 순위로 선택', 1.25, 4.45, 8.5, 2.5)
    panel(s, '로직 2  직접 Y 모델과 앙상블', '기존 분리예측 92% + 직접 Y 예측 8%\n직접 모델에는 학생 전출입 Feature 추가\n혼합 비율은 CV 순위점수로 선택', 10.05, 4.45, 8.7, 2.5)
    rb, rw, blend = rank['baseline_cv'], rank['winner_cv'], integrated['blend']['cv']
    table(s, [['3년 실험 / CV 2022–2024', '순위 상관 ↑', '순위오차 ↓', '5분위 일치 ↑', 'Y RMSE ↓'],
              ['현재 Y 유지', f'{rb["Spearman"]:.4f}', f'{rb["rank_MAE"]:.2f}위', f'{rb["exact_quintile_pct"]:.1f}%', f'{rb["y_RMSE"]:.4f}'],
              ['범죄·아동인구 분리 예측', f'{rw["Spearman"]:.4f}', f'{rw["rank_MAE"]:.2f}위', f'{rw["exact_quintile_pct"]:.1f}%', f'{rw["y_RMSE"]:.4f}'],
              ['분리 예측 + 직접 Y 앙상블', f'{blend["Spearman"]:.4f}', f'{blend["rank_MAE"]:.2f}위', f'{blend["exact_quintile_pct"]:.1f}%', f'{blend["y_RMSE"]:.4f}']],
          1.25, 7.25, 17.5, 1.95, [6.4,2.7,2.7,3.0,2.7], 20)
    text(s, '아동인구 모델 Feature: 인구구조 · 사업체밀도 · 가구구조 · 인구이동 · 출생사망 · 추세', 1.25, 9.45, 17.5, .45, 20, GREEN)
    footer(s, '3년 연구모델은 현재 서비스와 별개입니다. Y는 실제 아동 피해율이 아닌 지역 노출 대리지표입니다.')
    note(s, '범죄 분리모델은 기존 범죄 하위모델과 범죄유형별 합산모델의 로그예측 50:50 평균, 아동인구 모델은 Ridge alpha .01 변화량 학습. 이후 직접 Y 모델은 Ridge alpha10 잔차 방식, 학생 이동 15개 변수와 핵심8개 사용. 최종 앙상블 92:8. 2025 홀드아웃에서는 앙상블 Spearman .973077, 순위오차1.28, 5분위84%, top20 80%, Y RMSE .176441. 이는 Y 유지 RMSE .121570보다 크므로 순위 개선을 절대값 개선으로 과장하지 않는다. 개별 Feature 전체 목록은 아래 JSON에 기록됨. 선택규칙과 학습기간이 다른 2년 서비스 모델에 대한 직접 우열 비교는 금지.', ['artifacts/child_crime_exposure_rank_3y_v1/result.json','artifacts/child_crime_exposure_rank_3y_v1/experiment_report.md','artifacts/child_crime_exposure_integrated_v2/result.json'])

    # 3. Precise inputs and search scope of the currently deployed two-year model.
    s = fresh(prs, '범죄모델 개선 ③ · 현재 모델의 Feature 선택', 'Feature Selection · From Broad Candidates to 12 Inputs')
    banner(s, '현재 2년 실험: 최신 데이터 가용성을 확인하고, 그룹을 추가할 때마다 순위 성능 비교')
    table(s, [['탐색 범위', '실제로 비교한 내용'],
              ['보조 Feature 후보', '인구·가구구조, 유흥시설, 이동, 출생·사망, 연령 코호트, 학생 수, 스트레스'],
              ['정보 가용성 비교', '2025년 값이 있는 변수 / 공표 지연에 맞춰 과거값을 사용하는 변수'],
              ['예측 로직 4종', 'Y 직접 / Y 변화량 / 범죄·인구 각각 직접 / 범죄·인구 각각 변화량'],
              ['모델 기법', 'Ridge α 6종 비교 · 최종 α=10, 미래 Y 직접 예측']],
          1.25, 4.4, 17.5, 2.75, [3.2,14.3], 20)
    table(s, [['최종 수치 Feature', '핵심 8개', '추가한 4개'],
              ['현재 수준 · 3개', '현재 Y / log 범죄건수 / log 아동인구', '—'],
              ['과거 수준 · 6개', 'Y의 1년 전·2년 전 값', '범죄·아동인구 각각 1년 전·2년 전 값'],
              ['변화량 · 3개', 'Y / log 범죄 / log 아동인구의 전년차', '—']],
          1.25, 7.5, 17.5, 1.95, [3.2,7,7.3], 20)
    text(s, '공통 전처리: 결측 중앙값 대체 → 표준화 → 자치구 One-hot 추가 → Ridge 학습', 1.25, 9.65, 17.5, .4, 21, GREEN)
    footer(s, '최종 선택은 현재값·시차·변화량 중심. 다른 후보가 빠졌다고 실제 범죄와 무관하다는 뜻은 아닙니다.')
    note(s, '이 2년 실험은 개별 Feature 전 조합 전수탐색이 아니라 그룹 전진선택이다. strict/lag 정책과 로직·alpha 조합의 비교 결과는 1056행. strict에는 9개 그룹, lag 정책은 고용·사업체·산업조직·학생이동도 포함. alpha=.01,.1,1,10,100,1000. 모델 family는 Ridge만 실행되었고 트리·패널 기법은 앞선 연구에서 검토되었다. 12개 수치 Feature는 '+', '.join(one['features'])+'. log_crime/log_child는 코드상 log1p 변환값. 결측대체와 스케일링은 각 학습 구간 내부에서 적합한다. 현재서비스1년에도 동일12개수치Feature와자치구사용.', ['ml/training/optimize_exposure_2y_2027.py','models/recommender/v1/child_crime_exposure/2y/model_metadata.json','artifacts/child_crime_exposure_2y_2027_v1/all_comparisons.csv','ml/training/train_service_crime_exposure_1y.py'])

    # 4. Clear improvement trail within a single horizon/validation protocol.
    s = fresh(prs, '범죄모델 개선 ④ · 현재 2년 모델의 개선 폭', 'Measured Improvement · Like-for-like CV Comparison')
    text(s, '동일 2년 예측 · 2022–2024 시간순 CV 평균', 1.25, 3.3, 17.5, .6, 27, GREEN, MEDIUM)
    chart(s, ['현재 Y 유지', '핵심 8개', '추세 포함 12개'], [('5분위 일치율',[b['exact_quintile_pct'], core['exact_quintile_pct'], best['exact_quintile_pct']])], 1.25, 4.3, 9.4, 3.65, 100, 20, '0.0', True)
    text(s, '5분위 일치율 (%) · 높을수록 좋음', 1.25, 3.98, 9.4, .4, 18, GREY)
    panel(s, '단계별로 무엇이 달라졌나?', '기준: 현재 Y를 미래에도 그대로 사용\n1단계: Ridge로 현재값·Y 추세 학습\n2단계: 범죄·인구의 과거 수준 추가\n지역 차이는 자치구 인코딩으로 반영', 11.0, 3.45, 7.75, 3.2)
    text(s, f'기준 대비 5분위 +{best["exact_quintile_pct"]-b["exact_quintile_pct"]:.1f}%p\n순위오차 {b["rank_MAE"]:.2f} → {best["rank_MAE"]:.2f}위', 11.2, 7.0, 7.3, 1, 27, GREEN, MEDIUM)
    table(s, [['지표', '기준: 현재 Y 유지', '1단계: 핵심 8개', '2단계: 최종 12개'],
              ['5분위 일치율 ↑', f'{b["exact_quintile_pct"]:.1f}%', f'{core["exact_quintile_pct"]:.1f}%', f'{best["exact_quintile_pct"]:.1f}%'],
              ['평균 순위오차 ↓', f'{b["rank_MAE"]:.2f}위', f'{core["rank_MAE"]:.2f}위', f'{best["rank_MAE"]:.2f}위'],
              ['종합 순위점수 ↑', f'{b["ranking_score"]:.2f}', f'{core["ranking_score"]:.2f}', f'{best["ranking_score"]:.2f}']], 1.25, 8.3, 17.5, 1.6, [4.7,4.3,4.3,4.2], 20)
    footer(s, 'CV는 모델 선택에 사용한 평가입니다. 최종 단계에서 모든 지표가 함께 개선된 것은 아닙니다.')
    note(s, '2년 strict 정책의 baseline→core→trend 기록. 순위점수 = Spearman 재척도40% + 정규화 순위오차25% + top20적중20% + 5분위15%. 가중치는 모델 선택용 지표 가중치이며 사용자 지역점수의 Feature 가중치가 아니다. core→최종: 5분위+2.6667%p, 순위오차1.09333→1.06667, 종합95.55855→95.98120. Spearman .974872→.974615 및 Y RMSE .122094→.122990은 악화. 기준Y RMSE .121544. 따라서 현재결과는 순위종합기준의 개선이다.', ['models/recommender/v1/child_crime_exposure/2y/model_metadata.json','ml/training/optimize_exposure_rank_3y.py'])

    # 5. Deployment and an honest independent-year result, not blanket accuracy.
    s = fresh(prs, '범죄모델 개선 ⑤ · 최종 성능과 서비스 적용', 'Final Results · Service Horizons & Validation Limits')
    table(s, [['현재 서비스', '기법 / 예측 로직', '입력 → 목표연도', 'CV 5분위 일치'],
              ['1년 모델', 'Ridge α=1 / 현재 Y + 예측 변화량', '2025 → 2026', f'{one["exact_quintile_pct"]:.1f}%'],
              ['2년 모델', 'Ridge α=10 / 미래 Y 직접 예측', '2025 → 2027', f'{best["exact_quintile_pct"]:.1f}%']], 1.25, 3.35, 17.5, 1.9, [2.8,7.0,4.5,3.2], 21)
    text(s, '2025년 별도검증: 2년 모델은 등급 일치가 개선되었지만, 순위오차와 Y 오차는 악화', 1.25, 5.65, 17.5, .55, 24, GREEN, MEDIUM)
    table(s, [['지표', '기준: 현재 Y 유지', '최종 2년 모델', '변화 해석'],
              ['5분위 일치율 ↑', f'{bh["exact_quintile_pct"]:.0f}%', f'{h["exact_quintile_pct"]:.0f}%', '+8%p · 25개 구 중 17개 → 19개'],
              ['평균 순위오차 ↓', f'{bh["rank_MAE"]:.2f}위', f'{h["rank_MAE"]:.2f}위', '0.16위 증가 · 악화'],
              ['Y RMSE ↓', f'{bh["y_RMSE"]:.4f}', f'{h["y_RMSE"]:.4f}', '절대 노출지수 오차는 미개선'],
              ['상위 20% 적중률 ↑', f'{bh["top20_overlap_pct"]:.0f}%', f'{h["top20_overlap_pct"]:.0f}%', '범죄노출 상위 5개 중 4개 일치']], 1.25, 6.4, 17.5, 2.55, [4.0,3.9,3.9,5.7], 20)
    rect(s, 1.25, 9.3, 17.5, .65, PALE)
    text(s, '다음 개선: 새 연도 재검증 · 공표 시차 반영 · 범죄유형·선행변수 조합을 최종 Y 순위로 평가', 1.45, 9.43, 17.1, .4, 21, DARK, MEDIUM)
    footer(s, '2026년 현재: 2026년은 올해 추정, 2027년은 내년 예측. 1년 모델의 별도검증 지표는 현재 메타데이터에 없습니다.')
    note(s, '1년 CV Spearman .974615, 순위오차1.2, 5분위78.6667, top20100. 2년 CV Spearman .974615, 순위오차1.066667, 5분위84, top20100. 검증 학습창 및 horizon이 달라 수치만으로 어느 기간 모델이 확실히 우수하다고 판단하지 않는다. 2년 2025 홀드아웃 Spearman .967692(기준 .972308), 순위종합90.337(기준89.396). 현재 score는 Y 오름차순25구를5개씩100/80/60/40/20점과A–E로매핑하며 Feature가중합이아니다. API는배포된예측CSV를조회. 3년연구모델은현서비스미연결. 실제아동피해율아님. 새연도에독립평가가필요하며현재메타데이터의지표는미래성능보장이아니다.', ['models/recommender/v1/child_crime_exposure/1y/model_metadata.json','models/recommender/v1/child_crime_exposure/2y/model_metadata.json','backend/app/services/child_crime_exposure.py'])

    # Only remove the inherited slides from this new in-memory document.
    # Source presentation is never saved/modified.
    for slide_id in list(prs.slides._sldIdLst)[:original_count]:
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
    assert len(prs.slides) == 5
    prs.save(DEST)
    print(DEST)
    print('5 slides; same theme, frame, fonts and 20 × 11.25 in canvas.')


if __name__ == '__main__':
    main()
