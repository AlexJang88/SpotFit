# v15 아파트 전세 1년·3년 뒤 평당 보증금 예측 (v15 항목 14개 → h개월 뒤 동네 평당 보증금, XGBoost, 시간 순 평가) + 동네별 JSON
#   v15 항목 = v13의 13개 + 가장 가까운 한강공원까지 거리 (현재 수준 모델 AptJeonseXGB_v15.py 와 같음)
#   기준 시점 e(매월 말): 항목 = e 직전 12개월 전세 거래로 계산한 층수·거래일(년)·완공연도·평균 임대면적 + 정적 항목(직장인구·CCTV·1인가구·연령대 비율 5개·경비 인원·한강공원 거리)
#                         정답 = e+h개월 시점의 직전 12개월 동네 평당 보증금 중앙값(거래 10건 이상) (h=12: 1년, h=36: 3년)
#   평가(1년): 학습에는 정답이 시험 시작 전에 이미 알려진 기준 시점만 사용. 3년은 정답을 아는 시험 시점이 없어 검증 불가
#   JSON: 동네별로 현재(최근 12개월 전세 실거래 중앙값), 1년·3년 뒤 예측
# 실행: python AptJeonseXGB_v15.py  →  python AptJeonseForecastV15.py   (내 PC에서만, 서버 기록 없음)
import json, os
import joblib, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from apt_jeonse_data import AREA, DATA, OUT, X_COLS, Y, load
from apt_sale_compare import AGE_COLS, add_age, xgb
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------- 1. 데이터 준비 ----------
HG = '가장 가까운 한강공원까지 거리(km)'
GUARD = '100세대당 경비 인원'
d = add_age(load())
cx = pd.read_csv(DATA + '행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
area = pd.read_csv(DATA + '../raw/부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드').reindex(d.index)
hp = pd.read_csv(DATA + '../raw/기타/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)

static = pd.DataFrame(index=d.index)                                        # 시간이 지나도 변하지 않는 동네 특성
for c in ['직장인구 수', '생활방범 CCTV 수', '1인가구 비율(%)'] + [c for c in AGE_COLS if c != '50대 주민 비율(%)']: static[c] = d[c]
static[GUARD] = cx[GUARD].reindex(d.index)
static[HG] = cKDTree(np.c_[hx, hy]).query(area[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000

a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '계약일', '전월세구분', Y, '층', '건축년도', '임대면적(㎡)'])
a = a[a['전월세구분'] == '전세']
a['동'] = a['행정동_코드(상권분석)'].astype(int)
a = a[a['동'].isin(d.index)]
dt = pd.to_datetime(a['계약일'], errors='coerce')
a = a[dt.notna()]; dt = dt[dt.notna()]
a['m'] = dt.dt.year * 12 + dt.dt.month - 1
a['거래일(년)'] = dt.dt.year + (dt.dt.dayofyear - 1) / 365
m_lo, m_hi = int(a['m'].min()), int(a['m'].max())
ends = list(range(m_lo + 11, m_hi + 1))                                     # 기준 시점: 직전 12개월이 모두 데이터 안에 있는 달
lab = lambda m: f'{m // 12}-{m % 12 + 1:02d}'

P, S, F = {}, {}, {}                                                        # 시점별 동네 가격 / 서울 가격 / 동네 항목
for e in ends:
    w = a[(a['m'] >= e - 11) & (a['m'] <= e)]
    g = w.groupby('동')[Y].agg(['median', 'size']).reindex(d.index)
    P[e], S[e] = g['median'].where(g['size'] >= 10), w[Y].median()
    f = w.groupby('동').agg(층수=('층', 'mean'), 거래일=('거래일(년)', 'mean'), 완공연도=('건축년도', 'mean'), 면적=('임대면적(㎡)', 'mean')).reindex(d.index)
    f.columns = ['층수', '거래일(년)', '완공연도', AREA]
    F[e] = f

FEAT = ['층수', '거래일(년)', '완공연도', AREA] + list(static.columns)         # v15의 14개
print(f'전세 거래 기간 {lab(m_lo)} ~ {lab(m_hi)} | 기준 시점 {len(ends)}개 | 동네 {len(d)}곳 | 항목 {len(FEAT)}개 (v13 13개 + 한강공원 거리)')

def feats(e):
    df = F[e].join(static)
    df['현재 가격'], df['e'] = P[e], e
    return df

def build(h):
    rows = []
    for e in ends:
        if (e + h) in P:
            df = feats(e)
            df['정답'] = P[e + h]
            rows.append(df)
    return pd.concat(rows).dropna(subset=['현재 가격', '정답'])

def score(true, hat):
    err = (hat - true).abs() / true * 100
    r2 = 1 - ((true - hat) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    return err.median(), (err <= 10).mean() * 100, (hat - true).abs().mean(), r2

# ---------- 2. 1년 뒤 예측 시험 (시간 순) ----------
D12 = build(12)
TESTS = {'시험 A': (2024 * 12 + 9, m_hi - 12), '시험 B': (2023 * 12 + 9, 2024 * 12 + 8)}       # (첫 기준 시점, 마지막 기준 시점)
rows = []
for name, (lo, hi) in TESTS.items():
    tr = D12[D12['e'] + 12 <= lo]
    te = D12[(D12['e'] >= lo) & (D12['e'] <= hi)]
    model = xgb().fit(tr[FEAT], np.log(tr['정답']))
    seoul_growth = te['e'].map(lambda e: S[e] / S[e - 12] - 1 if (e - 12) in S else 0.0)
    preds = {'그대로': te['현재 가격'],
             '서울 변화율 규칙': te['현재 가격'] * (1 + seoul_growth),
             'v15 예측 모델': pd.Series(np.exp(model.predict(te[FEAT])), index=te.index)}
    for method, hat in preds.items():
        mid, within10, mae, r2 = score(te['정답'], hat)
        rows.append({'방법': method, '시험': f'{name} ({lab(lo)}~{lab(hi)} 기준)', '오차율 중앙(%)': round(mid, 1), '10% 이내(%)': round(within10),
                     '평균 오차(만원/평)': round(mae), '결정계수 R²': round(r2, 3)})
res = pd.DataFrame(rows)
order = ['그대로', '서울 변화율 규칙', 'v15 예측 모델']
print('\n===== 1년 뒤 예측 시험 결과 (오차율·평균 오차는 낮을수록, 10% 이내·결정계수는 높을수록 좋음) =====')
for name in res['시험'].unique():
    print(f'\n[{name}]')
    print(res[res['시험'] == name].drop(columns='시험').set_index('방법').reindex(order).to_string())

# ---------- 3. 최종 예측과 JSON ----------
perf = pd.read_csv(OUT + '아파트전세_XGB_v15_성능.csv')
imp15 = pd.read_csv(OUT + '아파트전세_XGB_v15_중요도.csv', index_col=0).iloc[:, 0]
print('\n===== v15 현재 수준 모델 (동네 평당 보증금을 직접 맞힘, 5분할 x 20번 반복) =====')
print(perf.rename(columns={'R² 평균': '결정계수 R²'}).iloc[:2].to_string(index=False))
print('\n항목 중요도 (섞으면 R²가 떨어지는 정도)')
print(imp15.round(3).to_string())

now = feats(m_hi)
out = pd.DataFrame({'자치구': d['자치구'], '행정동': d['행정동'], '현재 평당 보증금(최근12개월 실거래, 만원)': now['현재 가격'].round(0)})
print('\n===== v15 1년·3년 뒤 예측 =====')
for h, nm in [(12, '1년'), (36, '3년')]:
    D = build(h)
    model = xgb().fit(D[FEAT], np.log(D['정답']))
    ok = now[FEAT].notna().sum(axis=1) >= len(FEAT) - 2
    pred = pd.Series(np.exp(model.predict(now[FEAT])), index=d.index).where(ok)
    out[f'{nm} 뒤 예측(만원/평)'] = pred.round(0)
    out[f'{nm} 뒤 예상 변화율(%)'] = ((pred / now['현재 가격'] - 1) * 100).round(1)
    ch = out[f'{nm} 뒤 예상 변화율(%)'].dropna()
    pi = permutation_importance(model, D[FEAT], np.log(D['정답']), n_repeats=5, random_state=0)       # 현재 수준 모델과 같은 방식
    imp = pd.Series(pi.importances_mean, index=FEAT).sort_values(ascending=False)
    print(f'\n[{nm} 뒤 모델] 학습 시점 {D["e"].nunique()}개 | 변화율 중앙 {ch.median():+.1f}% (범위 {ch.min():+.0f}%~{ch.max():+.0f}%, ±30% 넘는 동네 {int((ch.abs() > 30).sum())}곳)')
    print('항목 중요도')
    print(imp.round(3).to_string())
    joblib.dump(model, OUT + f'아파트전세_v15_{nm}뒤모델.joblib')
out['비고'] = np.where(now['현재 가격'].notna(), '', '최근12개월 전세 거래 10건 미만')
out.reset_index().to_csv(OUT + '아파트전세_v15_동네별예측_1년3년뒤.csv', index=False, encoding='utf-8-sig')
short = out.reset_index()[['행정동_코드', '자치구', '행정동', '현재 평당 보증금(최근12개월 실거래, 만원)', '1년 뒤 예측(만원/평)', '3년 뒤 예측(만원/평)']].dropna(subset=['1년 뒤 예측(만원/평)'])
with open(OUT + '아파트전세_v15_동네별예측_1년3년뒤.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(short.to_json(orient='records', force_ascii=False)), f, ensure_ascii=False, indent=2)       # NaN은 null, 상세는 CSV
print(f'\n저장: 아파트전세_v15_동네별예측_1년3년뒤.json ({len(short)}곳)')
print(short[short['행정동'].isin(['반포2동', '성수2가1동', '상계1동', '개포2동'])].drop(columns=['행정동_코드', '자치구']).to_string(index=False))
