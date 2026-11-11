# v19 아파트 전세 1년·3년 뒤 평당 보증금 예측 (2011~2026 합본 전세 거래, XGBoost, 시간 순 평가) + 동네별 JSON
#   v19 = v18 + 업무지구(강남·광화문·여의도) 거리 3개
#   시작 항목: 전세 v15 예측 모델(AptJeonseForecastV15.py)의 14개에서 동네 정보를 2개만 남김. 층수·거래일(년)·완공연도·평균 임대면적 + 직장인구·40대 주민 비율 + 한강공원 거리 = 7개
#     남긴 2개는 전세 v15 현재 수준 모델의 항목 중요도에서 동네 정보 중 높은 두 개(직장인구 0.085, 40대 비율 0.068). 나머지(CCTV·1인가구·나머지 연령대·경비 인원)는 2021년 이후 값이라 뺌
#   이 버전의 항목 10개: 층수, 거래일(년), 완공연도, 평균 임대면적, 직장인구, 40대 주민 비율, 한강공원 거리, 6개월 상승률, 12개월 상승률, 업무지구(강남·광화문·여의도) 거리 3개
#   데이터: data/processed/부동산(전처리)/아파트_전월세_2010_2026.csv 의 전세 거래 (전세 계약은 2011-01부터, 2026-09까지만 사용). y = 평당 보증금
#   기준 시점 e(매월 말): 정답 = e+h개월 시점의 직전 12개월 동네 평당 보증금 중앙값(거래 10건 이상) (h=12: 1년, h=36: 3년)
#   학습 방식: 정답을 현재 가격 대비 비율(로그)로 바꿔 맞히고, 예측 후 현재 가격을 곱해 평당 보증금으로 되돌림 (매매 v55와 같음). 현재 가격은 항목에 넣지 않음
#   평가: 시간 순. 시험 구간 첫 기준 시점 전에 정답이 이미 알려진 기준 시점만 학습에 사용. 기준선: 그대로, 서울 변화율 규칙. 3년 뒤는 학습 시점이 부족한 2016년 기준 시험은 건너뜀
#   한계: 직장인구·40대 비율은 2021년 이후 값을 전 기간에 적용
# 실행: python AptJeonseForecastV19.py   (내 PC에서만, 서버 기록 없음)
import contextlib, io, json
import joblib, numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.inspection import permutation_importance
from apt_jeonse_data import AREA, DATA, OUT, Y, load
from apt_sale_compare import AGE_COLS, add_age, xgb

RAW = DATA + '../raw/'
BIZ = {'강남(GBD)': (127.0276, 37.4979), '광화문(CBD)': (126.9770, 37.5657), '여의도(YBD)': (126.9242, 37.5216)}      # 업무지구 기준점(경도, 위도), 매매 v44~v55와 같음
HG = '가장 가까운 한강공원까지 거리(km)'
M6, M12, RT, REL = '6개월 상승률(%)', '12개월 상승률(%)', '주택담보대출 금리(연%)', '상대 가격(서울 대비)'

# ---------- 1. 데이터 준비 ----------
a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세_2010_2026.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', Y, '층', '건축년도', '임대면적(㎡)'])
a = a[a['전월세구분'] == '전세']
a['dt'] = pd.to_datetime(a['계약일'], errors='coerce')
a = a[a['dt'].notna() & (a['dt'] <= pd.Timestamp('2026-09-30'))].copy()                  # 2026-09까지 (매매와 같은 기준 시점)
a['동'] = a['행정동_코드(상권분석)'].astype(int)
a['m'] = a['dt'].dt.year * 12 + a['dt'].dt.month - 1
a['거래일(년)'] = a['dt'].dt.year + (a['dt'].dt.dayofyear - 1) / 365
d = a.groupby('동')[['자치구', '행정동']].last().rename_axis('행정동_코드')                    # 동네 목록과 이름도 합본에서

with contextlib.redirect_stdout(io.StringIO()):                                           # 전세 v15의 동네 특성 (직장인구·연령대): 출력 문구는 숨김
    old = add_age(load())
area = pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드').reindex(d.index)
hp = pd.read_csv(RAW + '기타/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)

def haversine_km(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))

static = pd.DataFrame(index=d.index)                                                      # 시간이 지나도 변하지 않는 값으로 취급하는 동네 특성
static['직장인구 수'] = old['직장인구 수'].reindex(d.index)
static['40대 주민 비율(%)'] = old['40대 주민 비율(%)'].reindex(d.index)
static[HG] = cKDTree(np.c_[hx, hy]).query(area[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000
lon, lat = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True).transform(area['엑스좌표_값'].values, area['와이좌표_값'].values)
biz = pd.DataFrame({f'{k}까지 거리(km)': haversine_km(lo, la, lon, lat) for k, (lo, la) in BIZ.items()}, index=d.index)

m_lo, m_hi = int(a['m'].min()), int(a['m'].max())
ends = list(range(m_lo + 11, m_hi + 1))                                                   # 기준 시점: 직전 12개월이 모두 데이터 안에 있는 달
lab = lambda m: f'{m // 12}-{m % 12 + 1:02d}'

P6 = {}                                                                                   # 시점별 동네 최근 6개월 가격 (거래 10건 이상)
for e in range(m_lo + 5, m_hi + 1):
    w6 = a[(a['m'] >= e - 5) & (a['m'] <= e)]
    g6 = w6.groupby('동')[Y].agg(['median', 'size']).reindex(d.index)
    P6[e] = g6['median'].where(g6['size'] >= 10)

P, S, F = {}, {}, {}                                                                      # 시점별 동네 가격 / 서울 가격 / 동네 항목
for e in ends:
    w = a[(a['m'] >= e - 11) & (a['m'] <= e)]
    g = w.groupby('동')[Y].agg(['median', 'size']).reindex(d.index)
    P[e], S[e] = g['median'].where(g['size'] >= 10), w[Y].median()
    f = w.groupby('동').agg(층수=('층', 'mean'), 거래일=('거래일(년)', 'mean'), 완공연도=('건축년도', 'mean'), 면적=('임대면적(㎡)', 'mean')).reindex(d.index)
    f.columns = ['층수', '거래일(년)', '완공연도', AREA]
    f[M6] = (P6[e] / P6[e - 6] - 1) * 100 if (e - 6) in P6 else np.nan                    # 최근 6개월 가격 / 그 직전 6개월 가격
    f[M12] = (P[e] / P[e - 12] - 1) * 100 if (e - 12) in P else np.nan                    # 최근 12개월 가격 / 그 직전 12개월 가격
    F[e] = f

FEAT = ['층수', '거래일(년)', '완공연도', AREA, '직장인구 수', '40대 주민 비율(%)', HG, M6, M12, *list(biz.columns)]
print(f'전세 거래 {len(a):,}건 | 기간 {lab(m_lo)} ~ {lab(m_hi)} | 기준 시점 {len(ends)}개 | 동네 {len(d)}곳 | 항목 {len(FEAT)}개')

def feats(e):
    df = F[e].join(static).join(biz)
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

# ---------- 2. 시간 순 예측 시험 (1년·3년) ----------
RATIO = lambda df: np.log(df['정답'] / df['현재 가격'])                                       # 학습 정답: 현재 가격 대비 비율(로그)
SETS = {12: ('1년', range(2015, 2026)), 36: ('3년', range(2016, 2024))}

def run_tests(h):
    D = build(h); rows = []; last = None
    for ym in SETS[h][1]:
        lo = ym * 12; hi = lo + 11                                                        # 시험 기준 시점: ym년 1월 ~ 12월
        tr = D[D['e'] + h <= lo]                                                          # 학습: 정답이 시험 시작 전에 이미 알려진 기준 시점만
        te = D[(D['e'] >= lo) & (D['e'] <= hi)]
        if len(te) == 0 or tr['e'].nunique() < 24: continue
        model = xgb().fit(tr[FEAT], RATIO(tr))
        last = (ym, model, te)
        seoul_growth = te['e'].map(lambda e: S[e] / S[e - 12] - 1 if (e - 12) in S else 0.0)
        preds = {'그대로': te['현재 가격'], '서울 변화율 규칙': te['현재 가격'] * (1 + seoul_growth) ** (h // 12), f'v19 예측 모델': te['현재 가격'] * np.exp(model.predict(te[FEAT]))}
        for method, hat in preds.items():
            mid, within10, mae, r2 = score(te['정답'], hat)
            rows.append({'기간': SETS[h][0], '시험': f'{ym}년 기준', '방법': method, '오차율 중앙(%)': mid, '10% 이내(%)': within10, '평균 오차(만원/평)': mae, '결정계수 R²': r2})
    return pd.DataFrame(rows), last

order = ['그대로', '서울 변화율 규칙', f'v19 예측 모델']
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
pd.concat(allres).to_csv(OUT + f'아파트전세_v19_시험별결과.csv', index=False, encoding='utf-8-sig')          # 시험별 표는 파일로만 저장

# ---------- 3. 최종 예측과 JSON ----------
now = feats(m_hi)
out = pd.DataFrame({'자치구': d['자치구'], '행정동': d['행정동'], '현재 평당 보증금(최근12개월 실거래, 만원)': now['현재 가격'].round(0)})
print()
print(f'===== v19 1년·3년 뒤 예측 (2026-09 기준) =====')
for h, nm in [(12, '1년'), (36, '3년')]:
    D = build(h)
    model = xgb().fit(D[FEAT], RATIO(D))
    pred = (now['현재 가격'] * np.exp(model.predict(now[FEAT]))).where(now['현재 가격'].notna())
    out[f'{nm} 뒤 예측(만원/평)'] = pred.round(0)
    out[f'{nm} 뒤 예상 변화율(%)'] = ((pred / now['현재 가격'] - 1) * 100).round(1)
    ch = out[f'{nm} 뒤 예상 변화율(%)'].dropna()
    fit_r2 = score(D['정답'], D['현재 가격'] * np.exp(model.predict(D[FEAT])))[3]                  # 학습 데이터에서의 결정계수 (참고용, 과적합 확인)
    print(f'[{nm} 뒤 모델] 학습 시점 {D["e"].nunique()}개 | 변화율 중앙 {ch.median():+.1f}% (범위 {ch.min():+.0f}%~{ch.max():+.0f}%, ±30% 넘는 동네 {int((ch.abs() > 30).sum())}곳)')
    print(f'학습 데이터 결정계수 {fit_r2:.3f} (참고용. 위 시간 순 시험 R²와 크게 차이 나면 과적합)')
    joblib.dump(model, OUT + f'아파트전세_v19_{nm}뒤모델.joblib')
out['비고'] = np.where(now['현재 가격'].notna(), '', '최근12개월 전세 거래 10건 미만')
out.reset_index().to_csv(OUT + f'아파트전세_v19_동네별예측_1년3년뒤.csv', index=False, encoding='utf-8-sig')
short = out.reset_index()[['행정동_코드', '자치구', '행정동', '현재 평당 보증금(최근12개월 실거래, 만원)', '1년 뒤 예측(만원/평)', '3년 뒤 예측(만원/평)']].dropna(subset=['1년 뒤 예측(만원/평)'])
with open(OUT + f'아파트전세_v19_동네별예측_1년3년뒤.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(short.to_json(orient='records', force_ascii=False)), f, ensure_ascii=False, indent=2)       # NaN은 null, 상세는 CSV
print()
print(f'저장: 아파트전세_v19_동네별예측_1년3년뒤.json ({len(short)}곳)')
print(short[short['행정동'].isin(['반포2동', '성수2가1동', '상계1동', '개포2동'])].drop(columns=['행정동_코드', '자치구']).to_string(index=False))