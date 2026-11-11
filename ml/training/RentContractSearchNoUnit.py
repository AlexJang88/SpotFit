# 서울 아파트 월세 계약 단위 모델: 같은 호실(층·면적) 직전 계약을 쓰지 않는 항목 조합 + 모델 기법 비교 (RentContractSearch.py에서 호실 항목을 뺀 버전)
#   서비스에서는 사용자가 층까지 정하지 않으므로 같은 호실 직전 계약(u_*) 항목을 제외. 건물(법정동+지번+면적대) 이력은 사용
#   항목 단계: G1 = B2(동 코드·계약·동 통계·건물 이력 24개월) → G2 K-apt 단지 정보 → G3 계절·금리·동네 추세·한강/업무지구 거리 → G4 둘 다
#   기법: XGBoost(제곱/절대/Huber), 히스토그램 부스팅(제곱/절대), RandomForest, 릿지. 선택 기준은 검증 2024 월세 MAE, 시험 2025-01~2026-09는 보고용
#   예측 가능 기준: 같은 행정동·면적대 직전 12개월 월세 계약 10건 이상(자치구·서울로 넓히지 않음)
# 실행: python RentContractSearchNoUnit.py   (내 PC에서만, 서버 기록 없음)
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
a['u'] = pd.factorize(a['법정동'].astype(str) + ' ' + a['지번'].astype(str) + ' ' + a['F'].astype(str) + ' ' + a['A'].round(1).astype(str))[0]
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
# ======================================================================================================
# 항목 조합 + 모델 기법 탐색 (선택 기준은 검증 2024의 월세 MAE, 시험 2025-01~2026-09는 보고용)
# ======================================================================================================
import itertools
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from pyproj import Transformer
from scipy.spatial import cKDTree

def P(*x): print(*x, flush=True)
# ---------- 추가 항목 재료 ----------
KA = pd.read_csv(DATA + '../interim/기타/단지_행정동.csv', encoding='utf-8-sig', low_memory=False)
tok = KA['법정동주소'].astype(str).str.split(',').str[0].str.split()
KA['key'] = tok.str[1] + '|' + tok.str[2] + '|' + tok.str[3].str.extract(r'(\d+)')[0].fillna('')
for c in ['세대수', '동수', '총주차대수', '최고층수', '임대세대수', '승강기(승객용)']: KA[c] = pd.to_numeric(KA[c], errors='coerce')
KA['ka_park'] = KA['총주차대수'] / KA['세대수']; KA['ka_lease'] = KA['임대세대수'] / KA['세대수']; KA['ka_elev'] = KA['승강기(승객용)'] / KA['동수']
for c, n in [('난방방식', 'ka_heat'), ('복도유형', 'ka_corr'), ('분양형태', 'ka_sale')]:
    KA[n] = pd.factorize(KA[c])[0].astype(float); KA.loc[KA[c].isna(), n] = np.nan
KA = KA.rename(columns={'세대수': 'ka_hh', '동수': 'ka_dongs', '최고층수': 'ka_top'}).sort_values('ka_hh', ascending=False).drop_duplicates('key')
KA_COLS = ['ka_hh', 'ka_dongs', 'ka_top', 'ka_park', 'ka_lease', 'ka_elev', 'ka_heat', 'ka_corr', 'ka_sale']
KA = KA[['key'] + KA_COLS]
rt = pd.read_csv(RAW + '기타/예금은행_대출금리_신규취급액_기준__20261106130754.csv', encoding='cp949')
rt = pd.to_numeric(rt[rt['계정항목별'].str.strip() == '주택담보대출 (연리%)'].iloc[0, 1:], errors='coerce')
RATE = {int(c[:4]) * 12 + int(c[5:7]) - 1: v for c, v in rt.items()}
area = pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
hp = pd.read_csv(RAW + '기타/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)
STAT_D = pd.DataFrame(index=area.index)
STAT_D['hg'] = cKDTree(np.c_[hx, hy]).query(area[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000
lon, lat = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True).transform(area['엑스좌표_값'].values, area['와이좌표_값'].values)
def hav(lo0, la0, lo, la):
    p1, p2 = np.radians(la0), np.radians(la)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo - lo0) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))
for k, (lo, la) in {'gbd': (127.0276, 37.4979), 'cbd': (126.9770, 37.5657), 'ybd': (126.9242, 37.5216)}.items(): STAT_D[k] = hav(lo, la, lon, lat)
STAT_D = STAT_D.reset_index().rename(columns={'행정동_코드': '동'})
TR = []                                                                                                   # 동네 최근 6개월·그 이전 12개월 월세 수준 (분기마다)
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1
    w6 = a.iloc[np.searchsorted(mm, e - 5, 'left'):np.searchsorted(mm, e, 'right')]
    wp = a.iloc[np.searchsorted(mm, e - 23, 'left'):np.searchsorted(mm, e - 12, 'right')]
    if len(w6) == 0: continue
    t = w6.groupby('동')['rs'].median().rename('dong_rs6').to_frame().join(wp.groupby('동')['rs'].median().rename('dong_rs_prev')); t['q'] = q; TR.append(t.reset_index())
TR = pd.concat(TR, ignore_index=True)
Rt = a[['u', 'm', 'R', 'D', 'A']].rename(columns={'m': 'um', 'R': 'uR', 'D': 'uD', 'A': 'uA'}).sort_values('um')
def addx(df):
    df = df.reset_index(drop=True)
    L = df[['u', 'q']].copy(); L['e'] = 3 * L['q'] - 1; L['i'] = np.arange(len(L))
    x = pd.merge_asof(L.sort_values('e'), Rt, left_on='e', right_on='um', by='u', direction='backward').sort_values('i')
    df['u_rs'] = (x['uR'] / x['uA']).values; df['u_ds'] = (x['uD'] / x['uA']).values; df['u_gap'] = (x['e'] - x['um']).values; df['u_dD'] = df['D'].values - x['uD'].values
    df['key'] = df['자치구'] + '|' + df['법정동'].astype(str) + '|' + df['지번'].astype(str).str.extract(r'(\d+)')[0].fillna('')
    df = df.merge(KA, on='key', how='left').merge(TR, on=['q', '동'], how='left').merge(STAT_D, on='동', how='left')
    df['mo'] = df['m'] % 12 + 1; df['since'] = df['m'] - (3 * df['q'] - 1); df['rate'] = df['m'].map(RATE); df['rate12'] = df['rate'] - (df['m'] - 12).map(RATE)
    df['dong_trend'] = df['dong_rs'] / df['dong_rs_prev'] - 1; df['dong_rs6r'] = df['dong_rs6'] / df['dong_rs']
    return df
tr, va, te = addx(tr), addx(va), addx(te); te = te[te['avail']].copy()
P(f'추가 항목 준비 | 건물 이력 있음 {te["b_cnt"].notna().mean() * 100:.0f}% | 같은 호실 직전 계약 있음 {te["u_rs"].notna().mean() * 100:.0f}% | K-apt 단지 정보 있음 {te["ka_hh"].notna().mean() * 100:.0f}% | 학습 {len(tr):,} 검증 {len(va):,} 시험(예측 가능) {len(te):,} | {time.time() - T0:.0f}초')

BLD_X = ['b_rs', 'b_ds', 'b_cnt', 'b_R', 'b_D', 'dD_b', 'ds_rel_b', 'ds_rel_dong', 'logD', 'D_abs']
F1 = BASE_X + ['ds'] + BLD_X                                                         # B2
G2 = F1 + KA_COLS
G3 = F1 + ['mo', 'since', 'rate', 'rate12', 'dong_trend', 'dong_rs6r', 'hg', 'gbd', 'cbd', 'ybd']
G4 = G2 + G3[len(F1):]
FS = {'G1 (B2: 동·계약·건물 이력)': F1, 'G2 (+K-apt 단지 정보)': G2, 'G3 (+계절·금리·동네 추세·위치)': G3, 'G4 (K-apt + 계절·금리·추세·위치 모두)': G4}
OBJ = {'XGBoost 제곱 오차': ('reg:squarederror', False), 'XGBoost 절대 오차(월세 금액 기준)': ('reg:absoluteerror', True), 'XGBoost Huber(월세 금액 기준)': ('reg:pseudohubererror', True)}
def mae(df, p): return float((p - df['R']).abs().mean())
def xgb_fit(X, obj='reg:squarederror', w=False, depth=7, lr=0.1, mcw=5, lam=1.0, col=0.8, n=4000):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8, colsample_bytree=col, min_child_weight=mcw, reg_lambda=lam, tree_method='hist', max_bin=128, enable_categorical=True, n_jobs=8, early_stopping_rounds=60, random_state=0, objective=obj)
    m.fit(tr[X], tr['rs'], sample_weight=tr['A'] if w else None, eval_set=[(va[X], va['rs'])], sample_weight_eval_set=[va['A']] if w else None, verbose=False)
    return m
def row(stage, name, X, pv, pt, trees=np.nan):
    return {'단계': stage, '구분': name, '항목 수': len(X), '검증 MAE': mae(va, pv), '시험 MAE': mae(te, pt), '시험 오차율 중앙(%)': float(((pt - te['R']).abs() / te['R'] * 100).median()), '트리': trees}
res = []
# ---------- 1단계: 항목 조합 (XGBoost 제곱 오차) ----------
for name, X in FS.items():
    m = xgb_fit(X); res.append(row('항목', name, X, m.predict(va[X]) * va['A'], m.predict(te[X]) * te['A'], m.best_iteration + 1))
    P(f'[항목] {name:36s} {len(X):2d}개 | 검증 {res[-1]["검증 MAE"]:.2f} 시험 {res[-1]["시험 MAE"]:.2f} | {time.time() - T0:.0f}초')
R1 = pd.DataFrame(res); bestF = R1.loc[R1['검증 MAE'].idxmin(), '구분']; XB = FS[bestF]; P(f'→ 검증 MAE가 가장 낮은 항목 조합: {bestF}')
# ---------- 2단계: 모델 기법 (같은 항목) ----------
def run_model(nm):
    if nm in OBJ:
        m = xgb_fit(XB, *OBJ[nm]); return m, lambda df: m.predict(df[XB]) * df['A']
    if nm.startswith('히스토그램'):
        l1 = '절대' in nm; Xh = [c for c in XB if c != '동c']
        m = HistGradientBoostingRegressor(loss='absolute_error' if l1 else 'squared_error', learning_rate=0.1, max_iter=800, max_leaf_nodes=127, min_samples_leaf=20, early_stopping=False, random_state=0)
        m.fit(tr[Xh], tr['rs'], sample_weight=tr['A'] if l1 else None)
        best = (1e18, 0)
        for k, p in enumerate(m.staged_predict(va[Xh])):                                                   # 검증 MAE가 가장 낮은 단계 선택
            if k % 20 == 19:
                v = mae(va, p * va['A'])
                if v < best[0]: best = (v, k)
        kb = best[1]; m.best_iteration = kb + 1
        return m, lambda df: list(itertools.islice(m.staged_predict(df[Xh]), kb, kb + 1))[0] * df['A']
    Xn = [c for c in XB if c != '동c']; med = tr[Xn].median()
    if nm == 'RandomForest':
        m = RandomForestRegressor(n_estimators=100, max_depth=22, min_samples_leaf=5, max_features=0.5, n_jobs=8, random_state=0).fit(tr[Xn].fillna(-999), tr['rs'])
        return m, lambda df: m.predict(df[Xn].fillna(-999)) * df['A']
    mu, sd = tr[Xn].fillna(med).mean(), tr[Xn].fillna(med).std() + 1e-9
    z = lambda df: ((df[Xn].fillna(med) - mu) / sd).clip(-6, 6)
    m = Ridge(alpha=10).fit(z(tr), tr['rs']); return m, lambda df: m.predict(z(df)) * df['A']
TECH = list(OBJ) + ['히스토그램 부스팅 제곱 오차', '히스토그램 부스팅 절대 오차(월세 금액 기준)', 'RandomForest', '릿지(직선)']
fns = {}; models = {}
for nm in TECH:
    t0 = time.time(); m, f = run_model(nm); fns[nm] = f; models[nm] = m
    res.append(row('모델 기법', nm, XB, f(va), f(te), getattr(m, 'best_iteration_', getattr(m, 'best_iteration', np.nan))))
    P(f'[기법] {nm:36s} | 검증 {res[-1]["검증 MAE"]:.2f} 시험 {res[-1]["시험 MAE"]:.2f} | {time.time() - t0:.0f}초 (누적 {time.time() - T0:.0f}초)')
R2 = pd.DataFrame([r for r in res if r['단계'] == '모델 기법']); bestT = R2.loc[R2['검증 MAE'].idxmin(), '구분']; P(f'→ 검증 MAE가 가장 낮은 기법: {bestT}')
# ---------- 3단계: 가장 좋은 기법의 설정 조정 (XGBoost일 때) ----------
final_nm, final_model, final_fn = bestT, models[bestT], fns[bestT]; best_val = R2['검증 MAE'].min()
if bestT in OBJ:
    for cfg in [dict(depth=6, mcw=5, lam=1, col=0.8), dict(depth=9, mcw=10, lam=5, col=0.8), dict(depth=11, mcw=20, lam=10, col=0.7), dict(depth=8, mcw=3, lam=1, col=0.6)]:
        t0 = time.time(); m = xgb_fit(XB, *OBJ[bestT], lr=0.08, **cfg)
        res.append(row('설정 조정', f'{bestT} {cfg}', XB, m.predict(va[XB]) * va['A'], m.predict(te[XB]) * te['A'], m.best_iteration + 1))
        P(f'[설정] {cfg} | 검증 {res[-1]["검증 MAE"]:.2f} 시험 {res[-1]["시험 MAE"]:.2f} | {time.time() - t0:.0f}초 (누적 {time.time() - T0:.0f}초)')
        if res[-1]['검증 MAE'] < best_val: best_val = res[-1]['검증 MAE']; final_nm, final_model, final_fn = res[-1]['구분'], m, (lambda mm_: (lambda df: mm_.predict(df[XB]) * df['A']))(m)
if bestT.startswith('히스토그램') and '절대' not in bestT:
    Xh = [c for c in XB if c != '동c']
    for lr_, leaf_, msl_, it_ in [(0.03, 255, 50, 2666), (0.05, 127, 50, 1600)]:
        t0 = time.time(); m = HistGradientBoostingRegressor(loss='squared_error', learning_rate=lr_, max_iter=it_, max_leaf_nodes=leaf_, min_samples_leaf=msl_, early_stopping=False, random_state=0).fit(tr[Xh], tr['rs'])
        bv = (1e18, 0)
        for k, p in enumerate(m.staged_predict(va[Xh])):
            if k % 20 == 19:
                v = mae(va, p * va['A'])
                if v < bv[0]: bv = (v, k)
        kb = bv[1]; fn = (lambda m_, kb_: (lambda df: list(itertools.islice(m_.staged_predict(df[Xh]), kb_, kb_ + 1))[0] * df['A']))(m, kb)
        res.append(row('설정 조정', f'히스토그램 부스팅 제곱 오차 lr={lr_} 잎={leaf_} 최소표본={msl_}', XB, fn(va), fn(te), kb + 1))
        P(f'[설정] lr={lr_} 잎={leaf_} 최소표본={msl_} | 검증 {res[-1]["검증 MAE"]:.2f} 시험 {res[-1]["시험 MAE"]:.2f} 트리 {kb + 1} | {time.time() - t0:.0f}초 (누적 {time.time() - T0:.0f}초)')
        if res[-1]['검증 MAE'] < best_val: best_val = res[-1]['검증 MAE']; final_nm, final_model, final_fn = res[-1]['구분'], m, fn
RES = pd.DataFrame(res); RES.to_csv(OUT + '월세계약모델_호실제외_탐색결과.csv', index=False, encoding='utf-8-sig')
joblib.dump(final_model, OUT + '월세계약모델_호실제외_최고성능.joblib')
# ---------- 최종 후보 상세 (시험, 예측 가능한 계약) ----------
P(); P(f'검증 MAE 최저: {final_nm} | 검증 {best_val:.2f}')
raw = final_fn(te); pt = raw.clip(lower=1); e = (pt - te['R']).abs()
P('시험 월세 MAE %.2f | RMSE %.2f | R² %.3f | 오차율 중앙 %.1f%% | 음수 예측 %d건' % (e.mean(), np.sqrt(((pt - te['R']) ** 2).mean()), 1 - ((pt - te['R']) ** 2).sum() / ((te['R'] - te['R'].mean()) ** 2).sum(), (e / te['R'] * 100).median(), int((raw <= 0).sum())))
for lbl, mask in [('건물 이력 있음', te['b_cnt'].notna()), ('건물 이력 없음', te['b_cnt'].isna())]:
    P(f'  {lbl}: {int(mask.sum()):,}건, MAE {e[mask].mean():.2f}')
gb = te.assign(err=e, bin=pd.cut(te['D'], [0, 3000, 10000, 30000, 1e9], labels=['<3천', '3천~1억', '1억~3억', '3억+']))
P('보증금 구간별 MAE:', gb.groupby('bin', observed=True)['err'].mean().round(2).to_dict())
if hasattr(final_model, 'feature_importances_') and len(final_model.feature_importances_) == len(XB):
    P('항목 중요도 상위 12:', pd.Series(final_model.feature_importances_, index=XB).sort_values(ascending=False).head(12).round(3).to_dict())
json.dump({'features': XB, 'best_feature_set': bestF, 'best_technique': bestT, 'selected': final_nm, 'test_mae': float(e.mean())}, open(OUT + '월세계약모델_호실제외_최고성능요약.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
P(); P(f'저장: 월세계약모델_호실제외_탐색결과.csv, 월세계약모델_호실제외_최고성능.joblib | {time.time() - T0:.0f}초')
