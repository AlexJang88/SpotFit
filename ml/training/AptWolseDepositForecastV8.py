# 월세 v8 아파트 월세 1년·2년·3년 뒤 평당 보증금 예측 (2011~2026 합본 월세 계약, XGBoost, 시간 순 평가) + 동네별 JSON
#   y = 평당 보증금(만원) = 월세 계약의 보증금(만원) ÷ 임대면적(평). 월세 계약만 사용. 짝 모델: AptWolseForecastV8.py
#   v8 = v4와 같은 항목 3개 (6·12개월 상승률, 상대 가격). 2011년 이후 전체 학습 + 가격을 '구성 고정 가격'으로 계산 (순월세/반전세 비중 변화로 가격이 흔들리는 것을 제거). 제곱 오차 학습
#   항목(3개): 6개월 상승률, 12개월 상승률, 상대 가격(서울 대비). 모두 구성 고정 가격 기준으로 계산
#   구성 고정 가격: 계약마다 배수 = 보증금 ÷ (월세×12). 전체 월세 계약 배수의 중앙값을 기준으로 A그룹(이하)·B그룹(초과)으로 나누고, 동네·기간마다 두 그룹의 중앙값을 따로 구해 반반 평균 = 동네 가격. 서울 가격·6개월·12개월 상승률·상대 가격·정답 모두 같은 방식
#   데이터: data/raw/부동산/아파트_전월세_2010_2026.csv 의 월세 계약 (월세 계약은 2011-01부터, 2026-09까지만 사용)
#   기준 시점 e(매월 말): 정답 = e+h개월 시점의 직전 12개월 동네 구성 고정 평당 보증금(월세 거래 10건 이상, 두 그룹 모두 1건 이상) (h=12: 1년, h=24: 2년, h=36: 3년)
#   학습 방식: 정답을 현재 가격 대비 비율(로그)로 바꿔 맞히고, 예측 후 현재 가격을 곱해 평당 보증금으로 되돌림. 학습은 2011년 이후 전체 기준 시점
#   평가: 시간 순. 시험 구간 첫 기준 시점 전에 정답이 이미 알려진 기준 시점만 학습에 사용(2011년 이후). 기준선: 그대로, 서울 변화율 규칙. 학습 시점이 24개 미만인 해는 시험을 건너뜀(1년 뒤는 2018년 기준부터, 3년 뒤는 2020년 기준부터)
#   한계: 월세는 보증금과 월세가 서로 대체 관계라 두 모델(월세, 월세 보증금)을 같이 봐야 함. 정답이 계산된 가격이라 오차는 같은 방식의 '그대로'와만 비교해야 함. 2011~2014년은 B그룹 거래가 적음. 두 그룹이 모두 있는 동네만 예측
# 실행: python AptWolseDepositForecastV8.py   (내 PC에서만, 서버 기록 없음)
import json
import joblib, numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.inspection import permutation_importance
from apt_jeonse_data import AREA, DATA, OUT, Y
from apt_sale_compare import xgb

RAW = DATA + '../raw/'
HG = '가장 가까운 한강공원까지 거리(km)'
M6, M12, REL = '6개월 상승률(%)', '12개월 상승률(%)', '상대 가격(서울 대비)'

# ---------- 1. 데이터 준비 ----------
a = pd.read_csv(DATA + '../raw/부동산/아파트_전월세_2010_2026.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', Y, '층', '건축년도', '임대면적(㎡)'])
a = a[(a['전월세구분'] == '월세') & (a['월세(만원)'] > 0)]
a['dt'] = pd.to_datetime(a['계약일'], errors='coerce')
a = a[a['dt'].notna() & (a['dt'] <= pd.Timestamp('2026-09-30'))].copy()                  # 2026-09까지 (매매와 같은 기준 시점)
a['동'] = a['행정동_코드(상권분석)'].astype(int)
a['m'] = a['dt'].dt.year * 12 + a['dt'].dt.month - 1
a['거래일(년)'] = a['dt'].dt.year + (a['dt'].dt.dayofyear - 1) / 365
a['배수'] = a['보증금(만원)'] / (a['월세(만원)'] * 12)                                          # 월세 몇 년 치가 보증금인가
CUT = a['배수'].quantile(0.5)                                                               # 전체 월세 계약의 중앙값 (기준값)
a['bin'] = (a['배수'] > CUT).astype(int)                                                    # 0 = A그룹(보증금 작은 쪽), 1 = B그룹
d = a.groupby('동')[['자치구', '행정동']].last().rename_axis('행정동_코드')                    # 동네 목록과 이름도 합본에서

area = pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드').reindex(d.index)
hp = pd.read_csv(RAW + '부동산/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)

static = pd.DataFrame(index=d.index)                                                      # 시간이 지나도 변하지 않는 값으로 취급하는 동네 특성
static[HG] = cKDTree(np.c_[hx, hy]).query(area[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000


m_lo, m_hi = int(a['m'].min()), int(a['m'].max())
ends = list(range(m_lo + 11, m_hi + 1))                                                   # 기준 시점: 직전 12개월이 모두 데이터 안에 있는 달
lab = lambda m: f'{m // 12}-{m % 12 + 1:02d}'

def fixed(w):                                                                             # 구성 고정 가격: 두 그룹 중앙값의 반반 평균
    gg = w.groupby(['동', 'bin'])[Y].agg(['median', 'size']).unstack('bin')
    sz = gg['size'].reindex(columns=[0, 1])
    ok = (sz >= 1).all(axis=1) & (sz.sum(axis=1) >= 10)                                    # 두 그룹 모두 있고 합 10건 이상
    return gg['median'].reindex(columns=[0, 1]).mean(axis=1).where(ok).reindex(d.index)

P6 = {}                                                                                   # 시점별 동네 최근 6개월 가격 (거래 10건 이상)
for e in range(m_lo + 5, m_hi + 1):
    w6 = a[(a['m'] >= e - 5) & (a['m'] <= e)]
    P6[e] = fixed(w6)

P, S, F = {}, {}, {}                                                                      # 시점별 동네 가격 / 서울 가격 / 동네 항목
for e in ends:
    w = a[(a['m'] >= e - 11) & (a['m'] <= e)]
    P[e], S[e] = fixed(w), w.groupby('bin')[Y].median().mean()
    f = w.groupby('동').agg(층수=('층', 'mean'), 거래일=('거래일(년)', 'mean'), 완공연도=('건축년도', 'mean'), 면적=('임대면적(㎡)', 'mean')).reindex(d.index)
    f.columns = ['층수', '거래일(년)', '완공연도', AREA]
    f[M6] = (P6[e] / P6[e - 6] - 1) * 100 if (e - 6) in P6 else np.nan                    # 최근 6개월 가격 / 그 직전 6개월 가격
    f[M12] = (P[e] / P[e - 12] - 1) * 100 if (e - 12) in P else np.nan                    # 최근 12개월 가격 / 그 직전 12개월 가격
    f[REL] = P[e] / S[e]                                                                  # 같은 시기 서울 전체 가격 대비 이 동네 가격
    F[e] = f

FEAT = [M6, M12, REL]
print(f'월세 거래 {len(a):,}건 | 기간 {lab(m_lo)} ~ {lab(m_hi)} | 기준 시점 {len(ends)}개 | 동네 {len(d)}곳 | 항목 {len(FEAT)}개')

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

# ---------- 2. 시간 순 예측 시험 (1년·2년·3년) ----------
TRAIN_FROM = 2011 * 12                                                                   # 학습에 쓰는 첫 기준 시점 (2011-01 = 자료 시작, 전체 사용)
RATIO = lambda df: np.log(df['정답'] / df['현재 가격'])                                       # 학습 정답: 현재 가격 대비 비율(로그)
SETS = {12: ('1년', range(2015, 2026)), 24: ('2년', range(2015, 2025)), 36: ('3년', range(2016, 2024))}

def run_tests(h):
    D = build(h); rows = []; last = None
    for ym in SETS[h][1]:
        lo = ym * 12; hi = lo + 11                                                        # 시험 기준 시점: ym년 1월 ~ 12월
        tr = D[(D['e'] + h <= lo) & (D['e'] >= TRAIN_FROM)]                                                          # 학습: 정답이 시험 시작 전에 이미 알려진 기준 시점만
        te = D[(D['e'] >= lo) & (D['e'] <= hi)]
        if len(te) == 0 or tr['e'].nunique() < 24: continue
        model = xgb().fit(tr[FEAT], RATIO(tr))
        last = (ym, model, te)
        seoul_growth = te['e'].map(lambda e: S[e] / S[e - 12] - 1 if (e - 12) in S else 0.0)
        preds = {'그대로': te['현재 가격'], '서울 변화율 규칙': te['현재 가격'] * (1 + seoul_growth) ** (h // 12), f'v8 예측 모델': te['현재 가격'] * np.exp(model.predict(te[FEAT]))}
        for method, hat in preds.items():
            mid, within10, mae, r2 = score(te['정답'], hat)
            rows.append({'기간': SETS[h][0], '시험': f'{ym}년 기준', '방법': method, '오차율 중앙(%)': mid, '10% 이내(%)': within10, '평균 오차(만원/평)': mae, '결정계수 R²': r2})
    return pd.DataFrame(rows), last

order = ['그대로', '서울 변화율 규칙', f'v8 예측 모델']
col = ['오차율 중앙(%)', '10% 이내(%)', '평균 오차(만원/평)', '결정계수 R²']
allres = []
for h in SETS:
    nm = SETS[h][0]
    res, (ym, hold_model, hold_te) = run_tests(h)
    allres.append(res)
    print()
    print(f'===== {nm} 뒤 예측 시험 (시간 순, 시험별 표는 CSV로 저장) =====')
    avg = res.groupby('방법')[col].mean().round(3).reindex(order)
    print(f'[{nm} 뒤: 시험 {res["시험"].nunique()}번 평균]'); print(avg.to_string())
    pi = permutation_importance(hold_model, hold_te[FEAT], RATIO(hold_te), n_repeats=5, random_state=0)       # 모델이 처음 보는 가장 최근 시험 구간에서 측정
    imp = pd.Series(pi.importances_mean, index=FEAT).sort_values(ascending=False)
    print(f'[{nm} 뒤 모델 항목 중요도: {ym}년 시험 구간(학습에 안 쓴 데이터)에서 섞으면 R²가 떨어지는 정도, 상위 8개]'); print(imp.head(8).round(3).to_string())
pd.concat(allres).to_csv(OUT + f'아파트월세보증금_v8_시험별결과.csv', index=False, encoding='utf-8-sig')          # 시험별 표는 파일로만 저장

# ---------- 3. 최종 예측과 JSON ----------
now = feats(m_hi)
out = pd.DataFrame({'자치구': d['자치구'], '행정동': d['행정동'], '현재 평당 보증금(최근12개월 구성 고정 가격, 만원)': now['현재 가격'].round(0)})
print()
print(f'===== 월세 평당 보증금 v8 1년·2년·3년 뒤 예측 (2026-09 기준) =====')
for h, nm in [(12, '1년'), (24, '2년'), (36, '3년')]:
    D = build(h); D = D[D['e'] >= TRAIN_FROM]
    model = xgb().fit(D[FEAT], RATIO(D))
    pred = (now['현재 가격'] * np.exp(model.predict(now[FEAT]))).where(now['현재 가격'].notna())
    out[f'{nm} 뒤 예측(만원/평)'] = pred.round(0)
    out[f'{nm} 뒤 예상 변화율(%)'] = ((pred / now['현재 가격'] - 1) * 100).round(1)
    ch = out[f'{nm} 뒤 예상 변화율(%)'].dropna()
    fit_r2 = score(D['정답'], D['현재 가격'] * np.exp(model.predict(D[FEAT])))[3]                  # 학습 데이터에서의 결정계수 (참고용, 과적합 확인)
    print(f'[{nm} 뒤 모델] 학습 시점 {D["e"].nunique()}개 | 변화율 중앙 {ch.median():+.1f}% (범위 {ch.min():+.0f}%~{ch.max():+.0f}%, ±30% 넘는 동네 {int((ch.abs() > 30).sum())}곳)')
    print(f'학습 데이터 결정계수 {fit_r2:.3f} (참고용. 위 시간 순 시험 R²와 크게 차이 나면 과적합)')
    joblib.dump(model, OUT + f'아파트월세보증금_v8_{nm}뒤모델.joblib')
out['비고'] = np.where(now['현재 가격'].notna(), '', '최근12개월 월세 거래 10건 미만')
out.reset_index().to_csv(OUT + f'아파트월세보증금_v8_동네별예측_1년2년3년뒤.csv', index=False, encoding='utf-8-sig')
short = out.reset_index()[['행정동_코드', '자치구', '행정동', '현재 평당 보증금(최근12개월 구성 고정 가격, 만원)', '1년 뒤 예측(만원/평)', '2년 뒤 예측(만원/평)', '3년 뒤 예측(만원/평)']].dropna(subset=['1년 뒤 예측(만원/평)'])
with open(OUT + f'아파트월세보증금_v8_동네별예측_1년2년3년뒤.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(short.to_json(orient='records', force_ascii=False)), f, ensure_ascii=False, indent=2)       # NaN은 null, 상세는 CSV
print()
print(f'저장: 아파트월세보증금_v8_동네별예측_1년2년3년뒤.json ({len(short)}곳)')
print(short[short['행정동'].isin(['반포2동', '성수2가1동', '상계1동', '개포2동'])].drop(columns=['행정동_코드', '자치구']).to_string(index=False))