# 서울 아파트 월세 계약 단위 모델 (명세서 apartment_dong_rent_codex_spec_v2.md 3단계): 동네 시계열 모델(월세 v8·보증금 v10)과 별개로 같은 시간순 시험에서 비교
#   A 기준: X = 행정동, 면적, 층, 건물나이, 직전 분기까지의 동네 월세 수준 → Y = 월세/㎡
#   B 보증금 조건부: A의 X + 보증금/㎡ → Y = 월세/㎡
#   C 환산 월세: Y = 환산월세/㎡, 환산월세 = 월세 + 보증금 × r / 12 (r = 그 분기 이전 12개월 월세 계약으로 건물·면적대 안에서 추정한 연 전환율, 행정동 → 자치구 → 서울)
#      월세 = 예측 환산월세 − 보증금 × r / 12, 보증금 = (예측 환산월세 − 월세) × 12 / r
#   B2 보증금 조건부 + 건물 이력: B의 X + 같은 건물(법정동+지번)·같은 면적대(5㎡)의 직전 분기 말까지 24개월 월세·보증금 평균/중앙값/계약 수 + 보증금 상대 위치(동 대비, 건물 대비, 로그, 총액)
#      건물 이력이 없으면 값 없음으로 두고 B처럼 동작. 서비스에서는 사용자가 단지(건물)를 고를 때만 건물 이력을 쓸 수 있음
#   기준선: 같은 행정동 직전 12개월 월세/㎡ 중앙값 × 면적, 자치구×보증금 구간 중앙값 × 면적
#   누수 방지: 계약이 속한 분기의 지역 통계·전환율은 모두 직전 분기 말까지의 계약으로만 계산. 학습 2012~2023, 검증 2024(조기 종료·예측구간), 시험 2025-01~2026-09
#   포함: 월세>0, 보증금>0 월세 계약만(전세·무보증 월세 제외). 완전 중복 제거, 월세/㎡·보증금/㎡ 양끝 0.1% 제외. 평가 지표는 월세 금액(만원)의 MAE
#   기본 보증금 = 사용자에게 처음 보여 줄 초기값: 직전 12개월 같은 행정동·같은 면적대(10㎡) 월세 계약 10건 이상일 때만 평균·중앙값 제공. 부족하면 '예측 불가'(자치구·서울로 넓히지 않음), 평가도 예측 가능한 계약만
#   시나리오: ② 사용자가 보증금 입력 → 월세 (핵심 평가)  ③ 사용자가 월세 입력 → 보증금
# 실행: python RentContractModels.py   (내 PC에서만, 서버 기록 없음)
import json, time
import joblib, numpy as np, pandas as pd
from xgboost import XGBRegressor
from apt_jeonse_data import DATA, OUT, RAW

T0 = time.time()
MIN_GROUPS, MIN_PAIRS = 5, 30
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}      # 주소 DB의 옛 행정동 코드 → 상권분석 코드 (MapComplex.py)
use = ['행정동_코드(상권분석)', '자치구', '행정동', '법정동', '지번', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', '임대면적(㎡)', '층', '건축년도']
a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세_2010_2026.csv', low_memory=False, usecols=use)
a = a[a['전월세구분'] == '월세'].rename(columns={'행정동_코드(상권분석)': '동', '보증금(만원)': 'D', '월세(만원)': 'R', '임대면적(㎡)': 'A', '층': 'F', '건축년도': 'Y0'})
rep = {'월세 계약(원자료)': len(a)}
a['dt'] = pd.to_datetime(a['계약일'], errors='coerce')
a = a[a['dt'].notna() & (a['dt'] <= '2026-09-30')]; rep['2026-09 이전·날짜 정상'] = len(a)
a = a[(a['R'] > 0) & (a['D'] > 0) & (a['A'] > 0)]; rep['월세>0·보증금>0'] = len(a)
a = a[~a.duplicated(['동', '법정동', '지번', '계약일', 'D', 'R', 'A', 'F'])]; rep['완전 중복 제거 후'] = len(a)
a['rs'], a['ds'] = a['R'] / a['A'], a['D'] / a['A']
for c in ['rs', 'ds']:
    lo, hi = a[c].quantile([0.001, 0.999]); a = a[(a[c] >= lo) & (a[c] <= hi)]
rep['월세/㎡·보증금/㎡ 양끝 0.1% 제외 후'] = len(a)
a['m'] = a['dt'].dt.year * 12 + a['dt'].dt.month - 1
a['q'] = a['m'] // 3
a['age'] = a['dt'].dt.year - a['Y0']
a['gu'] = a['자치구'].astype('category').cat.codes
a['b'] = pd.factorize(a['법정동'].astype(str) + ' ' + a['지번'].astype(str) + ' ' + (a['A'] / 5).round().astype(int).astype(str))[0]
a['abin'] = (a['A'] // 10).clip(upper=12).astype(int)
a = a.reset_index(drop=True)
print(f'계약 {len(a):,}건 | ' + ' → '.join(f'{k} {v:,}' for k, v in rep.items()))

# ---------- 행정동 매핑 검증 ----------
area_codes = set(pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949')['행정동_코드'].unique())
codes = set(a['동'].unique())
mp = pd.DataFrame({'항목': ['계약에 부여된 행정동 코드 수', '영역 표에 없는 코드 수', '옛 코드가 남은 계약 수', '매핑 실패 계약 수'],
                   '값': [len(codes), len(codes - area_codes), int(a['동'].isin(OLD.keys()).sum()), int(a['동'].isna().sum())]})
by_year = a.groupby(a['dt'].dt.year)['동'].nunique().rename('행정동 코드 수').reset_index().rename(columns={'dt': '연도'})
mp.to_csv(OUT + '월세계약모델_행정동매핑검증.csv', index=False, encoding='utf-8-sig'); by_year.to_csv(OUT + '월세계약모델_연도별행정동수.csv', index=False, encoding='utf-8-sig')
print(); print(mp.to_string(index=False)); print('연도별 행정동 코드 수:', dict(zip(by_year['연도'], by_year['행정동 코드 수'])))

# ---------- 분기별 지역 통계·전환율 (직전 분기 말까지의 12개월) ----------
a = a.sort_values('m').reset_index(drop=True); mm = a['m'].values
def window(e): return a.iloc[np.searchsorted(mm, e - 11, 'left'):np.searchsorted(mm, e, 'right')]
def slopes(w):
    n = w.groupby('b')['D'].transform('size'); w = w[n >= 3]
    dD = w['D'] - w.groupby('b')['D'].transform('mean'); dR = w['R'] - w.groupby('b')['R'].transform('mean')
    w = w.assign(xy=dD * dR, xx=dD ** 2); w = w[w.groupby('b')['xx'].transform('sum') > 0]
    out = {}
    for lvl, key in [('dong', '동'), ('gu', 'gu'), ('seoul', None)]:
        t = w.assign(k=0).groupby(key if key else 'k').agg(xy=('xy', 'sum'), xx=('xx', 'sum'), n=('D', 'size'), g=('b', 'nunique'))
        t['r'] = -12 * t['xy'] / t['xx']; out[lvl] = t[['r', 'g', 'n']]
    return out
dong_gu = a.groupby('동')['gu'].last()
q_lo, q_hi = (2012 * 12 - 1 + 1) // 3, a['q'].max()
STAT, RATE, BASEB, DEF = [], [], [], []
test_q0 = (2025 * 12) // 3
edges = a.loc[a['m'] < 2024 * 12, 'ds'].quantile(np.linspace(0, 1, 11)[1:-1]).values                 # 보증금/㎡ 구간 경계(학습 기간으로 정함)
a['dbin'] = np.searchsorted(edges, a['ds'].values)
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1; w = window(e)
    if len(w) == 0: continue
    s = w.groupby('동').agg(dong_rs=('rs', 'median'), dong_ds=('ds', 'median'), dong_cnt=('rs', 'size')); s['q'] = q; STAT.append(s.reset_index())
    sg = w.groupby('gu').agg(gu_rs=('rs', 'median'), gu_ds=('ds', 'median')); sg['q'] = q
    sl = slopes(w); rd = sl['dong']; rg = sl['gu']; rs_ = sl['seoul']['r'].iloc[0]
    d_ok = rd[(rd['g'] >= MIN_GROUPS) & (rd['n'] >= MIN_PAIRS) & (rd['r'] > 0)]
    g_ok = rg[(rg['g'] >= MIN_GROUPS) & (rg['n'] >= MIN_PAIRS) & (rg['r'] > 0)]
    r = pd.DataFrame({'동': dong_gu.index}); r['gu'] = dong_gu.values
    r['r'] = r['동'].map(d_ok['r']); r['r_level'] = np.where(r['r'].notna(), 'dong', '')
    rg_map = r['gu'].map(g_ok['r']); fill = r['r'].isna() & rg_map.notna()
    r.loc[fill, 'r'] = rg_map[fill]; r.loc[fill, 'r_level'] = 'gu'
    fill = r['r'].isna(); r.loc[fill, 'r'] = rs_; r.loc[fill, 'r_level'] = 'seoul'
    r['q'] = q; RATE.append(r[['q', '동', 'r', 'r_level']]); STAT[-1]['seoul_rs'] = w['rs'].median()
    STAT[-1] = STAT[-1].merge(dong_gu.rename('gu').reset_index(), on='동').merge(sg.reset_index().drop(columns='q'), on='gu', how='left')
    bb = w.groupby(['gu', 'dbin'])['rs'].median().rename('gub_rs').reset_index(); bb['q'] = q; BASEB.append(bb)
    if q >= test_q0:                                                                                       # 기본 보증금(초기값): 시험 기간 분기만, 같은 행정동·면적대만
        t = w.groupby(['동', 'abin'])['D'].agg(['mean', 'median', 'size']).reset_index(); t['q'] = q; DEF.append(t)
BLD = []
for q in range(q_lo, q_hi + 1):                                                                            # 건물 이력: 직전 분기 말까지 24개월
    wb = a.iloc[np.searchsorted(mm, 3 * q - 1 - 23, 'left'):np.searchsorted(mm, 3 * q - 1, 'right')]
    if len(wb) == 0: continue
    t = wb.groupby('b').agg(b_rs=('rs', 'median'), b_ds=('ds', 'median'), b_cnt=('rs', 'size'), b_R=('R', 'mean'), b_D=('D', 'mean')).reset_index(); t['q'] = q; BLD.append(t)
BLD = pd.concat(BLD, ignore_index=True)
STAT = pd.concat(STAT, ignore_index=True); RATE = pd.concat(RATE, ignore_index=True); BASEB = pd.concat(BASEB, ignore_index=True)
print(f'분기별 통계 {STAT["q"].nunique()}개 분기 | {time.time() - T0:.0f}초')

d = a.merge(STAT.drop(columns='gu'), on=['q', '동'], how='left').merge(RATE, on=['q', '동'], how='left').merge(BASEB, on=['q', 'gu', 'dbin'], how='left').merge(BLD, on=['q', 'b'], how='left')
d = d[d['dong_rs'].notna() & d['r'].notna()].copy()
d['dD_b'], d['ds_rel_b'], d['ds_rel_dong'], d['logD'], d['D_abs'] = d['D'] - d['b_D'], d['ds'] / d['b_ds'], d['ds'] / d['dong_ds'], np.log(d['D']), d['D']
d['E'] = d['R'] + d['D'] * d['r'] / 12                                                                     # 환산 월세
d['es'] = d['E'] / d['A']
d['동c'] = pd.Categorical(d['동'], categories=sorted(a['동'].unique()))
tr, va, te = d[d['m'] < 2024 * 12], d[(d['m'] >= 2024 * 12) & (d['m'] < 2025 * 12)], d[d['m'] >= 2025 * 12]
print(f'학습 {len(tr):,} | 검증 {len(va):,} | 시험 {len(te):,}')

# ---------- 기본 보증금 = 사용자에게 처음 보여 줄 초기값 (시험 계약마다 직전 12개월, 같은 행정동·같은 면적대 계약 10건 이상일 때만, 자치구·서울로 넓히지 않음) ----------
DEF = pd.concat(DEF, ignore_index=True).rename(columns={'mean': 'def_D', 'median': 'def_med', 'size': 'def_n'})
te = te.merge(DEF[DEF['def_n'] >= 10], on=['q', '동', 'abin'], how='left')
te['avail'] = te['def_D'].notna()
print(f'예측 가능(같은 동·면적대 직전 12개월 월세 계약 10건 이상): 계약 {te["avail"].mean() * 100:.1f}% ({int(te["avail"].sum()):,}/{len(te):,}건), 시험 기간 마지막 분기 기준 동네 {te[te["q"] == te["q"].max()].groupby("동")["avail"].any().sum()}/{te["동"].nunique()}곳')

BASE_X = ['동c', 'A', 'F', 'age', 'dong_rs', 'dong_ds', 'dong_cnt', 'gu_rs', 'seoul_rs']
def fit(X, y):
    m = XGBRegressor(n_estimators=4000, learning_rate=0.1, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method='hist', enable_categorical=True, n_jobs=8, early_stopping_rounds=60, random_state=0)
    m.fit(tr[X], tr[y], eval_set=[(va[X], va[y])], verbose=False); return m
mods = {}
BLD_X = ['b_rs', 'b_ds', 'b_cnt', 'b_R', 'b_D', 'dD_b', 'ds_rel_b', 'ds_rel_dong', 'logD', 'D_abs']
MODELS = ['A', 'B', 'B2', 'C']
for nm, X, y in [('A', BASE_X, 'rs'), ('B', BASE_X + ['ds'], 'rs'), ('B2', BASE_X + ['ds'] + BLD_X, 'rs'), ('C', BASE_X, 'es')]:
    mods[nm] = (fit(X, y), X, y); print(f'모델 {nm} 학습 완료 (트리 {mods[nm][0].best_iteration + 1}개) {time.time() - T0:.0f}초')

def pred_rent(nm, df, dep=None):
    m, X, y = mods[nm]; x = df[X].copy()
    if nm in ('B', 'B2'): x['ds'] = (df['D'] if dep is None else dep) / df['A']
    p = m.predict(x) * df['A']
    if nm == 'C': p = p - (df['D'] if dep is None else dep) * df['r'] / 12
    return p
def pred_dep(df):                                                                                       # 모델 C 역산: 실제 월세 → 보증금
    m, X, y = mods['C']; return (m.predict(df[X]) * df['A'] - df['R']) * 12 / df['r']
def metrics(true, hat, area):
    e = hat - true; return {'월세 MAE(만원)': e.abs().mean(), 'RMSE': np.sqrt((e ** 2).mean()), 'R²': 1 - (e ** 2).sum() / ((true - true.mean()) ** 2).sum(), '월세/㎡ MAE': (e / area).abs().mean(), '오차율 중앙(%)': (e.abs() / true * 100).median()}
allte = te; te = te[te['avail']].copy()                                                                      # 이후 평가는 예측 가능한 계약만
rows = []
base_dong = te['dong_rs'] * te['A']; base_bin = te['gub_rs'].fillna(te['gu_rs']) * te['A']
rows += [{'시나리오': '② 사용자가 보증금을 입력 → 월세', '모델': '기준선: 동 월세/㎡ 중앙값', **metrics(te['R'], base_dong, te['A'])},
         {'시나리오': '② 사용자가 보증금을 입력 → 월세', '모델': '기준선: 자치구×보증금 구간 중앙값', **metrics(te['R'], base_bin, te['A'])}]
for nm in MODELS:
    p = pred_rent(nm, te); neg = int((p <= 0).sum()); rows.append({'시나리오': '② 사용자가 보증금을 입력 → 월세', '모델': f'{nm}' + {'A': ' 기준(보증금 안 씀)', 'B': ' 보증금 조건부', 'B2': ' 보증금 조건부 + 건물 이력', 'C': ' 환산 월세'}[nm] + (f' (음수 {neg}건)' if neg else ''), **metrics(te['R'], p.clip(lower=1), te['A'])})
pB_all = pred_rent('B', allte); rows.append({'시나리오': '참고: 예측 불가 포함 전체 시험 계약', '모델': 'B 보증금 조건부 (전체)', **metrics(allte['R'], pB_all.clip(lower=1), allte['A'])})
res = pd.DataFrame(rows); print(); print(res.round(3).to_string(index=False))
dgap = pd.DataFrame([{'초기값': '평균', 'MAE(만원)': (te['def_D'] - te['D']).abs().mean(), '오차율 중앙(%)': ((te['def_D'] - te['D']).abs() / te['D'] * 100).median(), '실제 보증금 대비 평균 편향(%)': (te['def_D'].mean() / te['D'].mean() - 1) * 100},
                     {'초기값': '중앙값', 'MAE(만원)': (te['def_med'] - te['D']).abs().mean(), '오차율 중앙(%)': ((te['def_med'] - te['D']).abs() / te['D'] * 100).median(), '실제 보증금 대비 평균 편향(%)': (te['def_med'].mean() / te['D'].mean() - 1) * 100}])
print(); print('[초기 보증금이 실제 계약 보증금과 얼마나 다른가]'); print(dgap.round(1).to_string(index=False))
ph = pred_dep(te).clip(lower=0); e3 = (ph - te['D']).abs(); e3b = (te['def_D'] - te['D']).abs()
s3 = pd.DataFrame([{'시나리오': '③ 사용자가 월세를 입력 → 보증금', '모델': 'C 역산', '보증금 MAE(만원)': e3.mean(), '오차율 중앙(%)': (e3 / te['D'] * 100).median()},
                   {'시나리오': '③ 사용자가 월세를 입력 → 보증금', '모델': '기준선: 초기 보증금(평균)', '보증금 MAE(만원)': e3b.mean(), '오차율 중앙(%)': (e3b / te['D'] * 100).median()}])
print(); print(s3.round(2).to_string(index=False))
res.to_csv(OUT + '월세계약모델_성능비교.csv', index=False, encoding='utf-8-sig'); s3.to_csv(OUT + '월세계약모델_보증금역산.csv', index=False, encoding='utf-8-sig'); dgap.to_csv(OUT + '월세계약모델_초기보증금.csv', index=False, encoding='utf-8-sig')

hb = te['b_cnt'].notna(); pB2 = pred_rent('B2', te).clip(lower=1); pBb = pred_rent('B', te).clip(lower=1)
hist = pd.DataFrame([{'구분': '건물 이력 있음', '계약수': int(hb.sum()), 'B MAE': (pBb[hb] - te['R'][hb]).abs().mean(), 'B2 MAE': (pB2[hb] - te['R'][hb]).abs().mean()},
                     {'구분': '건물 이력 없음', '계약수': int((~hb).sum()), 'B MAE': (pBb[~hb] - te['R'][~hb]).abs().mean(), 'B2 MAE': (pB2[~hb] - te['R'][~hb]).abs().mean()}])
print(); print('[건물 이력 유무별 월세 MAE]'); print(hist.round(2).to_string(index=False)); hist.to_csv(OUT + '월세계약모델_건물이력유무별.csv', index=False, encoding='utf-8-sig')
# ---------- 보증금 구간별·면적대별·행정동별 오차, 예측구간 ----------
best = min(MODELS, key=lambda n: (pred_rent(n, te).clip(lower=1) - te['R']).abs().mean()); pb = pred_rent(best, te).clip(lower=1)
print(); print(f'월세 MAE가 가장 낮은 모델: {best}')
te = te.assign(err=(pb - te['R']).abs(), errA=(pred_rent('A', te).clip(lower=1) - te['R']).abs(), errD=(base_dong - te['R']).abs())
te['보증금 구간'] = pd.cut(te['D'], [0, 3000, 10000, 30000, 1e9], labels=['<3천', '3천~1억', '1억~3억', '3억+']); te['면적대'] = pd.cut(te['A'], [0, 40, 60, 85, 1e4], labels=['<40㎡', '40~60', '60~85', '85㎡+'])
by = []
for col in ['보증금 구간', '면적대']:
    t = te.groupby(col, observed=True).agg(계약수=('R', 'size'), **{f'{best} MAE': ('err', 'mean'), 'A MAE': ('errA', 'mean'), '동 중앙값 기준선 MAE': ('errD', 'mean')}).reset_index().rename(columns={col: '구분'}); t.insert(0, '분류', col); by.append(t)
dm = te.groupby('동').agg(계약수=('R', 'size'), mae=('err', 'mean'), base=('errD', 'mean'))
dm['구간'] = pd.cut(dm['계약수'], [0, 30, 100, 300, 1e9], labels=['<30건', '30~100', '100~300', '300건+'])
t = dm.groupby('구간', observed=True).agg(행정동수=('mae', 'size'), **{f'{best} MAE': ('mae', 'mean'), '동 중앙값 기준선 MAE': ('base', 'mean')}).reset_index().rename(columns={'구간': '구분'}); t.insert(0, '분류', '시험 기간 동네 거래량'); by.append(t)
by = pd.concat(by, ignore_index=True); print(by.round(2).to_string(index=False)); by.to_csv(OUT + '월세계약모델_구간별성능.csv', index=False, encoding='utf-8-sig')
vr = va['R'].values / pred_rent(best, va).clip(lower=1).values                                           # 검증 기간 실제/예측 비율
q05, q95 = np.quantile(vr, [0.05, 0.95]); cov = ((te['R'] >= pb * q05) & (te['R'] <= pb * q95)).mean()
print(f'예측구간(검증 기간 실제/예측 비율의 5~95%): 시험 기간 커버리지 {cov * 100:.1f}% (목표 90%)')
dm.reset_index().rename(columns={'mae': f'{best} MAE', 'base': '동 중앙값 기준선 MAE'}).to_csv(OUT + '월세계약모델_행정동별성능.csv', index=False, encoding='utf-8-sig')
for nm in MODELS: joblib.dump(mods[nm][0], OUT + f'월세계약모델_{nm}.joblib')
json.dump({'rep': rep, 'best': best, 'coverage': float(cov), 'q05': float(q05), 'q95': float(q95), 'features': BASE_X}, open(OUT + '월세계약모델_요약.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(); print(f'저장: 월세계약모델_*.csv, 월세계약모델_A/B/C.joblib | {time.time() - T0:.0f}초')
