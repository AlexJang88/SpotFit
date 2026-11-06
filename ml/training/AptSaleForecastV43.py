# 모델 버전 43: v41과 같은 x값(10개)으로 '1년 뒤·3년 뒤 동네 평당 금액'을 직접 맞히는 모델 (XGBoost, 시간 순 평가)
#   v41은 '지금(2021~2026년 전체) 동네 평당 금액'을 맞히는 모델이라 시간 정보가 없음. v43은 정답을 '기준 시점으로부터 h개월 뒤 동네 평당 금액'으로 바꾸고, 항목은 기준 시점 직전 12개월 거래로 다시 계산함
#   기준 시점 e(매월 말): 항목 = e 직전 12개월 거래의 층수·거래일(년)·완공연도·평균 전용면적 동네 평균 + 정적 항목(직장인구·유동인구·단지 평균 세대수·지역난방 비율·100세대당 경비 인원·한강공원 거리, v41과 같음)
#                         정답 = e+h개월 시점의 직전 12개월 동네 평당 금액 중앙값(거래 10건 이상). h=12(1년), h=36(3년)
#   평가(1년): 정답이 시험 기준 시점 이전에 이미 알려진 기준 시점만 학습에 사용. 기준선 = ① 그대로(e 시점 가격) ② 서울 최근 12개월 변화율 적용
#   한계: 3년은 정답을 아는 시험 시점이 없어 검증 불가. 모델은 2021~2026년 5.7년(사이클 1번)만 보았고, 트리 모델은 학습 범위 밖 시점을 외삽하지 못함
# 실행: python AptSaleForecastV43.py   (내 PC에서만, 서버 기록 없음)
import json, os
import joblib, numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from apt_sale_data import OUT, PROC, Y, load
from apt_sale_compare import xgb

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
d = load()
cx = pd.read_csv(DATA + 'processed/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
area = pd.read_csv(DATA + 'raw/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
hp = pd.read_csv(DATA + 'raw/기타/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)
static = pd.DataFrame(index=d.index)
static['직장인구 수'] = d['직장인구 수']; static['유동인구 수'] = d['유동인구 수']
static['단지 평균 세대수'] = cx['단지 평균 세대수'].reindex(d.index); static['지역난방 단지 비율(%)'] = cx['지역난방 단지 비율(%)'].reindex(d.index); static['100세대당 경비 인원'] = cx['100세대당 경비 인원'].reindex(d.index)
static['가장 가까운 한강공원까지 거리(km)'] = cKDTree(np.c_[hx, hy]).query(area.reindex(d.index)[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000
a = pd.read_csv(PROC + '아파트_매매.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '계약일', Y, '층', '건축년도', '전용면적(㎡)'])
a['동'] = a['행정동_코드(상권분석)'].astype(int); a = a[a['동'].isin(d.index)]; dt = pd.to_datetime(a['계약일'])
a['m'] = dt.dt.year * 12 + dt.dt.month - 1; a['거래일(년)'] = dt.dt.year + (dt.dt.dayofyear - 1) / 365
m_lo, m_hi = int(a['m'].min()), int(a['m'].max()); ends = list(range(m_lo + 11, m_hi + 1))
lab = lambda m: f'{m // 12}-{m % 12 + 1:02d}'
P, S, F = {}, {}, {}
for e in ends:
    w = a[(a['m'] >= e - 11) & (a['m'] <= e)]; g = w.groupby('동')[Y].agg(['median', 'size']).reindex(d.index)
    P[e] = g['median'].where(g['size'] >= 10); S[e] = w[Y].median()
    f = w.groupby('동').agg(층수=('층', 'mean'), 거래일=('거래일(년)', 'mean'), 완공연도=('건축년도', 'mean'), 면적=('전용면적(㎡)', 'mean')).reindex(d.index)
    f.columns = ['층수', '거래일(년)', '완공연도', '평균 전용면적(㎡)']; F[e] = f
FEAT = ['층수', '거래일(년)', '완공연도', '직장인구 수', '유동인구 수', '평균 전용면적(㎡)', '단지 평균 세대수', '지역난방 단지 비율(%)', '100세대당 경비 인원', '가장 가까운 한강공원까지 거리(km)']      # v41의 10개
print(f'거래 기간 {lab(m_lo)} ~ {lab(m_hi)} | 기준 시점 {lab(ends[0])} ~ {lab(ends[-1])} ({len(ends)}개) | 동네 {len(d)}곳 | 항목 {len(FEAT)}개 (v41과 같음)')
def feats(e):
    df = F[e].join(static); df['현재 가격'] = P[e]; df['e'] = e; return df
def build(h):
    rows = []
    for e in ends:
        if (e + h) in P:
            df = feats(e); df['정답'] = P[e + h]; rows.append(df)
    D = pd.concat(rows); return D.dropna(subset=['현재 가격', '정답'])
def metr(true, hat): e = (hat - true).abs() / true * 100; return round(e.median(), 1), round(e.mean(), 1), round((e <= 10).mean() * 100)

print('\n[1년 뒤 예측: 시간 순 평가]  오차율 중앙% | 평균% | 10% 이내%')
D12 = build(12); res = []
for name, lo, hi in [('시험 A', 2024 * 12 + 9, m_hi - 12), ('시험 B', 2023 * 12 + 9, 2024 * 12 + 8)]:
    tr = D12[D12['e'] + 12 <= lo]; te = D12[(D12['e'] >= lo) & (D12['e'] <= hi)]
    mdl = xgb().fit(tr[FEAT], np.log(tr['정답'])); hat = np.exp(mdl.predict(te[FEAT]))
    sea = te['현재 가격'] * (S[te['e'].iloc[0]] / S[te['e'].iloc[0] - 12]) if False else te['현재 가격'] * te['e'].map(lambda e: S[e] / S[e - 12] if (e - 12) in S else 1.0)
    for k, v in [('① 그대로', te['현재 가격']), ('② 서울 최근12개월 변화율 적용', sea), ('③ v43 (v41 항목 → 1년 뒤 가격)', pd.Series(hat, index=te.index))]:
        mdn, mn, w10 = metr(te['정답'], v); res.append({'시험': f'{name}: 기준 {lab(lo)}~{lab(hi)} → 정답 {lab(lo + 12)}~{lab(hi + 12)}', '학습 기준시점 수': tr['e'].nunique(), '방법': k, '오차율 중앙': mdn, '평균': mn, '10% 이내': w10})
pd.set_option('display.width', 230); print(pd.DataFrame(res).to_string(index=False))

# 최종: 정답이 있는 모든 기준 시점으로 학습 → 가장 최근 시점에서 1년·3년 뒤 예측
now = feats(m_hi); out = pd.DataFrame({'자치구': d['자치구'], '행정동': d['행정동'], '현재 최근12개월 실거래(만원/평)': now['현재 가격'].round(0)})
for h, nm in [(12, '1년'), (36, '3년')]:
    D = build(h); mdl = xgb().fit(D[FEAT], np.log(D['정답']))
    ok = now[FEAT].notna().all(axis=1) | True; pr = pd.Series(np.exp(mdl.predict(now[FEAT])), index=d.index)
    out[f'{nm} 뒤 예측(v43, 만원/평)'] = pr.round(0).where(now['현재 가격'].notna()); out[f'{nm} 뒤 예상 변화율(%)'] = ((pr / now['현재 가격'] - 1) * 100).round(1)
    ch = out[f'{nm} 뒤 예상 변화율(%)'].dropna(); imp = pd.Series(mdl.feature_importances_, index=FEAT).sort_values(ascending=False)
    print(f'\n[{nm} 모델] 학습 기준시점 {D["e"].nunique()}개·{len(D)}행 | 예상 변화율 중앙 {ch.median():+.1f}% (최소 {ch.min():+.0f}% ~ 최대 {ch.max():+.0f}%), ±30% 넘는 동네 {int((ch.abs() > 30).sum())}곳 / {len(ch)}곳 | 중요 항목:', {k: round(float(v), 2) for k, v in imp.head(4).items()})
    joblib.dump(mdl, OUT + f'아파트매매_v43_{nm}뒤모델.joblib')
out['예측 가능 여부'] = np.where(now['현재 가격'].notna(), '최근12개월 실거래 있음', '최근12개월 거래 10건 미만')
out.reset_index().to_csv(OUT + '아파트매매_v43_1년3년뒤예측.csv', index=False, encoding='utf-8-sig')
with open(OUT + '아파트매매_v43_1년3년뒤예측.json', 'w', encoding='utf-8') as f: json.dump(out.reset_index().dropna(subset=['1년 뒤 예측(v43, 만원/평)']).to_dict('records'), f, ensure_ascii=False, indent=2)
print('저장:', os.path.abspath(OUT + '아파트매매_v43_1년3년뒤예측.csv'))
print(out[out['행정동'].isin(['반포2동', '개포2동', '성수2가1동', '상계1동', '사직동'])].drop(columns='예측 가능 여부').to_string())
