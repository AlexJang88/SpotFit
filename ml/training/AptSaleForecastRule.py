# 아파트 매매 동네 평당 금액 1년·3년 뒤 예측: 서울 최근 12개월 변화율 규칙 (과거 시험에서 가장 잘 맞은 방식)
#   1년 뒤 = 동네 현재 가격 × (1 + g)          3년 뒤 = 동네 현재 가격 × (1 + g)^3
#   현재 가격 = 최근 12개월(2025-10~2026-09) 동네 평당 금액 중앙값 (거래 10건 이상인 동네만)
#   g         = 서울 전체 최근 12개월 중앙값 ÷ 직전 12개월 중앙값 - 1
#   과거 시험(1년): 오차율 중앙값 4.7~5.9%, 10% 이내 73~80% (추세가 이어진 해 기준, 방향이 바뀐 해는 약 15%)
#   한계: 시장 방향이 이어진다는 가정이며 모든 동네가 같은 비율로 움직임. 3년은 검증할 수 없고 1년 규칙을 3번 이어 곱한 값임
# 실행: python AptSaleForecastRule.py
import json, os
import numpy as np, pandas as pd
from apt_sale_data import OUT, PROC, Y, load

d = load()
a = pd.read_csv(PROC + '아파트_매매.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '계약일', Y])
a['동'] = a['행정동_코드(상권분석)'].astype(int)
a = a[a['동'].isin(d.index)]
dt = pd.to_datetime(a['계약일'])
a['m'] = dt.dt.year * 12 + dt.dt.month - 1
m_hi = int(a['m'].max())

recent = a[a['m'] > m_hi - 12]                                              # 최근 12개월
before = a[(a['m'] <= m_hi - 12) & (a['m'] > m_hi - 24)]                    # 직전 12개월
g = recent[Y].median() / before[Y].median() - 1
print(f'서울 최근 12개월 중앙값 {recent[Y].median():,.0f}만원 vs 직전 12개월 {before[Y].median():,.0f}만원 → 변화율 {g * 100:+.1f}%')

stat = recent.groupby('동')[Y].agg(['median', 'size']).reindex(d.index)
now = stat['median'].where(stat['size'] >= 10)
out = pd.DataFrame({'자치구': d['자치구'], '행정동': d['행정동'], '현재 최근12개월 실거래(만원/평)': now.round(0)})
out['1년 뒤 예측(만원/평)'] = (now * (1 + g)).round(0)
out['1년 뒤 예상 변화율(%)'] = round(g * 100, 1)
out['3년 뒤 예측(만원/평)'] = (now * (1 + g) ** 3).round(0)
out['3년 뒤 예상 변화율(%)'] = round(((1 + g) ** 3 - 1) * 100, 1)
out['예측 가능 여부'] = np.where(now.notna(), '최근12개월 실거래 있음', '최근12개월 거래 10건 미만')

out.reset_index().to_csv(OUT + '아파트매매_1년3년뒤예측.csv', index=False, encoding='utf-8-sig')
with open(OUT + '아파트매매_1년3년뒤예측.json', 'w', encoding='utf-8') as f:
    json.dump(out.reset_index().dropna(subset=['1년 뒤 예측(만원/평)']).to_dict('records'), f, ensure_ascii=False, indent=2)
print(f'저장: 아파트매매_1년3년뒤예측.csv | 예측 가능 {int(now.notna().sum())}곳 / {len(d)}곳')
print(out[out['행정동'].isin(['반포2동', '성수2가1동', '상계1동'])].drop(columns=['자치구', '예측 가능 여부']).to_string(index=False))
