# 월세 계약으로 보증금-월세 교환 관계(전환율) 추정 + 기본 보증금 통계 (명세서 apartment_dong_rent_codex_spec_v2.md 1~2단계 시험판)
#   전환율: 같은 건물(법정동+지번)·같은 면적대(5㎡) 안에서 보증금이 더 큰 계약의 월세가 얼마나 낮은가. 월세 = alpha + beta × 보증금, r(연율) = -12 × beta
#     건물·면적대마다 평균을 뺀 값으로 기울기를 구함(그룹 고정효과) → 서로 다른 건물·면적을 섞어 상관을 구하는 오류를 피함
#     기준 시점 e 이전 12개월 월세 계약만 사용 (미래 거래 없음). 행정동 → 자치구 → 서울시 순으로 표본이 충분한 가장 좁은 단계를 씀
#   기본 보증금: e 이전 12개월 월세 계약 중 같은 행정동·비슷한 면적(±10㎡)의 평균·중앙값·표본 수, 부족하면 자치구 → 서울시
#   한계: 단지명이 처리된 파일에 없어 법정동+지번을 단지 대신 씀. 통계적 관계이며 법정 전환율이나 개별 계약 조건이 아님
# 실행: python RentConversionEstimate.py
import numpy as np, pandas as pd
from apt_jeonse_data import DATA, OUT

MIN_GROUPS, MIN_PAIRS = 5, 30                     # 전환율을 쓰려면 비교 가능한 건물·면적대 그룹 수와 계약 수가 이 이상
a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세_2010_2026.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '자치구', '행정동', '법정동', '지번', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', '임대면적(㎡)'])
a = a[(a['전월세구분'] == '월세') & (a['월세(만원)'] > 0) & (a['보증금(만원)'] > 0)].copy()
a['dt'] = pd.to_datetime(a['계약일'], errors='coerce')
a = a[a['dt'].notna() & (a['dt'] <= '2026-09-30')]
dup = a.duplicated(['행정동_코드(상권분석)', '법정동', '지번', '계약일', '보증금(만원)', '월세(만원)', '임대면적(㎡)']).sum()
a = a.rename(columns={'행정동_코드(상권분석)': '동', '보증금(만원)': 'D', '월세(만원)': 'R', '임대면적(㎡)': 'A'})
a['m'] = a['dt'].dt.year * 12 + a['dt'].dt.month - 1
a['건물'] = a['법정동'].astype(str) + ' ' + a['지번'].astype(str) + ' ' + (a['A'] / 5).round().astype(int).astype(str)
print(f'월세 계약 {len(a):,}건 (보증금>0, 월세>0) | 완전 중복 {dup:,}건 | 건물·면적대 {a["건물"].nunique():,}개')

def slope(w, key):
    """w: 윈도우 계약. key: 구분 단위 열(동/자치구/None=서울). 건물·면적대 평균을 뺀 값으로 기울기 → 전환율"""
    g = w.groupby('건물'); n = g['D'].transform('size')
    w = w[(n >= 3)].copy()
    if len(w) == 0: return pd.DataFrame(columns=['r', 'groups', 'n'])
    w['dD'] = w['D'] - w.groupby('건물')['D'].transform('mean'); w['dR'] = w['R'] - w.groupby('건물')['R'].transform('mean')
    w = w[w.groupby('건물')['dD'].transform(lambda s: s.abs().sum()) > 0]
    gk = key if key else np.zeros(len(w))
    t = w.assign(xy=w['dD'] * w['dR'], xx=w['dD'] ** 2).groupby(gk).agg(xy=('xy', 'sum'), xx=('xx', 'sum'), n=('D', 'size'), groups=('건물', 'nunique'))
    t['r'] = -12 * t['xy'] / t['xx']
    return t[['r', 'groups', 'n']]

rows = []
for e in [y * 12 + 11 for y in range(2017, 2026)] + [2026 * 12 + 8]:
    w = a[(a['m'] > e - 12) & (a['m'] <= e)].copy()
    seoul = slope(w, None); gu = slope(w.assign(k=w['자치구']), 'k'); dong = slope(w.assign(k=w['동']), 'k')
    dong_ok = dong[(dong['groups'] >= MIN_GROUPS) & (dong['n'] >= MIN_PAIRS) & (dong['r'] > 0)]
    gu_ok = gu[(gu['groups'] >= MIN_GROUPS) & (gu['n'] >= MIN_PAIRS) & (gu['r'] > 0)]
    nd = w['동'].nunique()
    rows.append({'기준': f'{e // 12}-{e % 12 + 1:02d}', '서울 전환율(%)': round(float(seoul['r'].iloc[0]) * 100, 2), '서울 계약수': int(seoul['n'].iloc[0]),
                 '자치구 전환율 중앙(%)': round(gu_ok['r'].median() * 100, 2), '자치구 사용가능': f'{len(gu_ok)}/{gu["n"].shape[0]}',
                 '행정동 전환율 중앙(%)': round(dong_ok['r'].median() * 100, 2) if len(dong_ok) else np.nan, '행정동 사용가능': f'{len(dong_ok)}/{nd}'})
res = pd.DataFrame(rows)
print(); print(res.to_string(index=False))
res.to_csv(OUT + '월세_전환율_추정_기준시점별.csv', index=False, encoding='utf-8-sig')

# ---- 기본 보증금 (최근 12개월 월세 계약, 같은 동·비슷한 면적 → 자치구 → 서울) ----
e = 2026 * 12 + 8; w = a[(a['m'] > e - 12) & (a['m'] <= e)]
def default_deposit(dong, gu, area, min_n=10):
    for level, sub in [('dong', w[(w['동'] == dong)]), ('gu', w[w['자치구'] == gu]), ('seoul', w)]:
        s = sub[(sub['A'] - area).abs() <= 10]
        if len(s) >= min_n: return level, len(s), float(s['D'].mean()), float(s['D'].median())
    return 'seoul', len(w), float(w['D'].mean()), float(w['D'].median())
names = a.groupby('동')[['자치구', '행정동']].last()
out = []
for d, r_ in names.iterrows():
    lv, n, mean_, med_ = default_deposit(d, r_['자치구'], 84.0)
    out.append({'행정동_코드': d, '자치구': r_['자치구'], '행정동': r_['행정동'], '기준': '2026-09', '면적(㎡)': 84, '사용 단계': lv, '표본 수': n, '평균 보증금(만원)': round(mean_), '중앙값 보증금(만원)': round(med_)})
dd = pd.DataFrame(out); dd.to_csv(OUT + '월세_기본보증금_84㎡_2026-09.csv', index=False, encoding='utf-8-sig')
print(); print('[기본 보증금: 84㎡, 2026-09 기준] 사용 단계:', dd['사용 단계'].value_counts().to_dict(), '| 평균/중앙값 차이 중앙 %.1f%%' % ((dd['평균 보증금(만원)'] / dd['중앙값 보증금(만원)'] - 1) * 100).median())
print(dd[dd['행정동'].isin(['반포2동', '성수2가1동', '상계1동', '개포2동'])].drop(columns=['행정동_코드', '자치구']).to_string(index=False))
print(); print('저장: 월세_전환율_추정_기준시점별.csv, 월세_기본보증금_84㎡_2026-09.csv')
