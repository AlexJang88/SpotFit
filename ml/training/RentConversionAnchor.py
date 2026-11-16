# 월세 계약 단위: 전환율 target을 '직전 12개월 시장 전환율과의 차이'로 바꾼 연도별 반복 시험 (같은 건물 전세만, 빠른 설정)
#   기준 수준(앵커): 계약 분기 시작 전 12개월의 유효 계약 전환율 중앙값 — 서울 / 동(30건 이상, 없으면 자치구 30건 이상, 없으면 서울)
#   A: y = 전환율 − 서울 앵커,  B: y = 전환율 − 동 앵커.  예측 전환율 = 앵커 + 예측값.  앵커 2개는 항목으로도 넣음
#   비교: 기존 전환율 target (앵커 없음). 직접 예측 결과는 RentConversionBuildingOnly.py (같은 계약·같은 설정)
# 실행: python RentConversionAnchor.py   (내 PC에서만, 서버 기록 없음)
import json, time
import joblib, numpy as np, pandas as pd
from xgboost import XGBRegressor
from apt_jeonse_data import DATA, OUT, RAW

T0 = time.time()
MIN_GROUPS, MIN_PAIRS = 5, 30
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}      # 주소 DB의 옛 행정동 코드 → 상권분석 코드 (MapComplex.py)
use = ['행정동_코드(상권분석)', '자치구', '행정동', '법정동', '지번', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', '임대면적(㎡)', '층', '건축년도']
a = pd.read_csv(DATA + '../raw/부동산/아파트_전월세_2010_2026.csv', low_memory=False, usecols=use)
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
q_lo, q_hi = (2012 * 12 - 1 + 1) // 3, a['q'].max() + 1                                                  # 마지막 분기 다음 분기 = 2026-09 말 기준 통계(서비스용)
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
RATE_TBL = RATE.copy()                                                                                       # 아래에서 RATE가 금리 사전으로 바뀌므로 전환율 표를 따로 보관
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
KA = pd.read_csv(DATA + '../raw/부동산/단지_행정동.csv', encoding='utf-8-sig', low_memory=False)
tok = KA['법정동주소'].astype(str).str.split(',').str[0].str.split()
KA['key'] = tok.str[1] + '|' + tok.str[2] + '|' + tok.str[3].str.extract(r'(\d+)')[0].fillna('')
for c in ['세대수', '동수', '총주차대수', '최고층수', '임대세대수', '승강기(승객용)']: KA[c] = pd.to_numeric(KA[c], errors='coerce')
KA['ka_park'] = KA['총주차대수'] / KA['세대수']; KA['ka_lease'] = KA['임대세대수'] / KA['세대수']; KA['ka_elev'] = KA['승강기(승객용)'] / KA['동수']
for c, n in [('난방방식', 'ka_heat'), ('복도유형', 'ka_corr'), ('분양형태', 'ka_sale')]:
    KA[n] = pd.factorize(KA[c])[0].astype(float); KA.loc[KA[c].isna(), n] = np.nan
KA = KA.rename(columns={'세대수': 'ka_hh', '동수': 'ka_dongs', '최고층수': 'ka_top'}).sort_values('ka_hh', ascending=False).drop_duplicates('key')
KA_COLS = ['ka_hh', 'ka_dongs', 'ka_top', 'ka_park', 'ka_lease', 'ka_elev', 'ka_heat', 'ka_corr', 'ka_sale']
KA = KA[['key'] + KA_COLS]
rt = pd.read_csv(RAW + '부동산/예금은행_대출금리_신규취급액_기준__20261106130754.csv', encoding='cp949')
rt = pd.to_numeric(rt[rt['계정항목별'].str.strip() == '주택담보대출 (연리%)'].iloc[0, 1:], errors='coerce')
RATE = {int(c[:4]) * 12 + int(c[5:7]) - 1: v for c, v in rt.items()}
area = pd.read_csv(RAW + '부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
hp = pd.read_csv(RAW + '부동산/한강공원 위치현황.csv', encoding='cp949').iloc[1:]
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
tr, va, te = addx(tr), addx(va), addx(te); te_all = te.copy(); te = te[te['avail']].copy()
P(f'추가 항목 준비 | 건물 이력 있음 {te["b_cnt"].notna().mean() * 100:.0f}% | 같은 호실 직전 계약 있음 {te["u_rs"].notna().mean() * 100:.0f}% | K-apt 단지 정보 있음 {te["ka_hh"].notna().mean() * 100:.0f}% | 학습 {len(tr):,} 검증 {len(va):,} 시험(예측 가능) {len(te):,} | {time.time() - T0:.0f}초')

# ======================================================================================================
BASE_X = ['동c', 'A', 'F', 'age', 'dong_rs', 'dong_ds', 'dong_cnt', 'gu_rs', 'seoul_rs']
BLD_X = ['b_rs', 'b_ds', 'b_cnt', 'b_R', 'b_D', 'dD_b', 'ds_rel_b', 'ds_rel_dong', 'logD', 'D_abs']
F1 = BASE_X + ['ds'] + BLD_X
Xh = [c for c in (F1 + KA_COLS) if c != '동c']
# ---------- 전세 시세 (분기마다 직전 12개월) ----------
J0 = pd.read_csv(DATA + '../raw/부동산/아파트_전월세_2010_2026.csv', low_memory=False, usecols=use)
J0 = J0[J0['전월세구분'] == '전세'].rename(columns={'행정동_코드(상권분석)': '동', '보증금(만원)': 'D', '임대면적(㎡)': 'A'})
J0['dt'] = pd.to_datetime(J0['계약일'], errors='coerce'); J0 = J0[J0['dt'].notna() & (J0['dt'] <= '2026-09-30') & (J0['D'] > 0) & (J0['A'] > 0)].copy()
J0['js'] = J0['D'] / J0['A']; lo, hi = J0['js'].quantile([0.001, 0.999]); J0 = J0[(J0['js'] >= lo) & (J0['js'] <= hi)]
J0['m'] = J0['dt'].dt.year * 12 + J0['dt'].dt.month - 1
J0['bk'] = J0['법정동'].astype(str) + ' ' + J0['지번'].astype(str) + ' ' + (J0['A'] / 5).round().astype(int).astype(str)
J0['abin'] = (J0['A'] // 10).clip(upper=12).astype(int); J0 = J0.sort_values('m').reset_index(drop=True); jm = J0['m'].values
JB, JD = [], []
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1; w = J0.iloc[np.searchsorted(jm, e - 11, 'left'):np.searchsorted(jm, e, 'right')]
    if len(w) == 0: continue
    t = w.groupby('bk')['js'].agg(['median', 'size']).reset_index().rename(columns={'median': 'j_b', 'size': 'j_bn'}); t['q'] = q; JB.append(t)
    t = w.groupby(['동', 'abin'])['js'].agg(['median', 'size']).reset_index().rename(columns={'median': 'j_d', 'size': 'j_dn'}); t['q'] = q; JD.append(t)
JB = pd.concat(JB, ignore_index=True); JD = pd.concat(JD, ignore_index=True)
def add_j(df):
    df = df.copy(); df['bk'] = df['법정동'].astype(str) + ' ' + df['지번'].astype(str) + ' ' + (df['A'] / 5).round().astype(int).astype(str)
    df = df.merge(JB, on=['q', 'bk'], how='left').merge(JD, on=['q', '동', 'abin'], how='left')
    use_b = df['j_bn'] >= 2; use_d = ~use_b & (df['j_dn'] >= 10)
    df['js_ref'] = np.where(use_b, df['j_b'], np.where(use_d, df['j_d'], np.nan)); df['j_src'] = np.where(use_b, 2, np.where(use_d, 1, 0)); df['j_n'] = np.where(use_b, df['j_bn'], df['j_dn'])
    df['J'] = df['js_ref'] * df['A']; df['gapJ'] = df['J'] - df['D']; df['D_J'] = df['D'] / df['J']; df['logJ'] = np.log(df['J'])
    df['rc'] = 12 * df['R'] / df['gapJ']
    df['ok'] = df['J'].notna() & (df['D_J'] <= 0.8) & (df['rc'] >= 0.005) & (df['rc'] <= 0.15)
    return df
ALL = add_j(pd.concat([tr, va, te_all], ignore_index=True))
XS = [c for c in Xh if c not in ('ds', 'dD_b', 'ds_rel_b', 'ds_rel_dong', 'logD', 'D_abs')] + ['J', 'logJ', 'j_src', 'j_n']
XD = [c for c in Xh if c not in ('ds', 'dD_b', 'ds_rel_b', 'ds_rel_dong', 'logD', 'D_abs')] + ['R', 'rs', 'J', 'logJ', 'j_src', 'j_n']
LO, HI = 0.005, 0.15
def fit(X, y, df): return HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=300, max_leaf_nodes=127, min_samples_leaf=50, early_stopping=False, random_state=0).fit(df[X], y)
def R2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
ALL['yr'] = ALL['dt'].dt.year
ALL['okb'] = ALL['ok'] & (ALL['j_src'] == 2)
P('같은 건물 전세가 있는(유효) 계약 비율: ' + ', '.join(f'{y}:{v * 100:.0f}%' for y, v in ALL.groupby('yr')['okb'].mean().items()))
P(f'전체 {len(ALL):,}건 | 연도별 유효(전환율 정의 가능) 비율: ' + ', '.join(f'{y}:{v * 100:.0f}%' for y, v in ALL.groupby('yr')['ok'].mean().items()))
# ---------- 앵커: 직전 12개월 시장 전환율 ----------
K = ALL[ALL['okb']].sort_values('m'); km = K['m'].values; AN = []
for q in range(ALL['q'].min(), ALL['q'].max() + 1):
    e = 3 * q - 1; w = K.iloc[np.searchsorted(km, e - 11, 'left'):np.searchsorted(km, e, 'right')]
    if len(w) < 100: continue
    gd = w.groupby('동')['rc'].agg(['median', 'size']); gg = w.groupby('gu')['rc'].agg(['median', 'size'])
    t = pd.DataFrame({'동': ALL['동'].unique()}); t['gu'] = t['동'].map(ALL.groupby('동')['gu'].last())
    t['an_s'] = w['rc'].median()
    t['an_d'] = t['동'].map(gd.loc[gd['size'] >= 30, 'median']).fillna(t['gu'].map(gg.loc[gg['size'] >= 30, 'median'])).fillna(t['an_s'])
    t['q'] = q; AN.append(t.drop(columns='gu'))
ALL = ALL.merge(pd.concat(AN, ignore_index=True), on=['동', 'q'], how='left')
P('앵커 있는 계약 %.1f%%' % (ALL['an_s'].notna().mean() * 100))
XA = XS + ['an_s', 'an_d']
rows = []
for Y in range(2014, 2027):
    trn, tst = ALL[ALL['yr'] < Y], ALL[(ALL['yr'] == Y) & ALL['okb'] & ALL['an_s'].notna()]
    tk = trn[trn['okb'] & trn['an_s'].notna()]; t0 = time.time()
    preds = {'기존(앵커 없음)': fit(XS, tk['rc'], tk).predict(tst[XS]),
             'A 서울 앵커 차이': tst['an_s'].values + fit(XA, tk['rc'] - tk['an_s'], tk).predict(tst[XA]),
             'B 동 앵커 차이': tst['an_d'].values + fit(XA, tk['rc'] - tk['an_d'], tk).predict(tst[XA]),
             '앵커만(동, 모델 없음)': tst['an_d'].values}
    row = {'시험 연도': Y, '시험 계약': len(tst), '정답 전환율 중앙': tst['rc'].median() * 100}
    msg = []
    for lab, r in preds.items():
        r = np.clip(r, LO, HI); pr = np.clip(r * tst['gapJ'] / 12, 1, None); pdp = np.clip(tst['J'] - 12 * tst['R'] / r, 100, None)
        e1 = (pr - tst['R']).abs() / tst['R'] * 100; e2 = (pdp - tst['D']).abs() / tst['D'] * 100
        row[f'월세 오차율중앙 {lab}'] = e1.median(); row[f'보증금 오차율중앙 {lab}'] = e2.median(); row[f'편향 {lab}'] = (r - tst['rc']).mean() * 100
        row[f'월세 MAE {lab}'] = (pr - tst['R']).abs().mean(); row[f'보증금 MAE {lab}'] = (pdp - tst['D']).abs().mean()
        msg.append(f'{lab} 월세 {e1.median():.1f}% 보증금 {e2.median():.1f}% 편향 {row[f"편향 {lab}"]:+.2f}%p')
    rows.append(row); P(f'{Y} | 시험 {len(tst):,} | 정답 {row["정답 전환율 중앙"]:.2f}% | ' + ' | '.join(msg) + f' | {time.time() - t0:.0f}초')
Rt = pd.DataFrame(rows); Rt.to_csv(OUT + '월세계약_연도별반복시험_전환율앵커.csv', index=False, encoding='utf-8-sig')
P('\n요약 (연도별 오차율 중앙의 평균 / 표준편차 / 최대, 2015년까지 vs 2016년부터)')
for lab in preds:
    for k in ['월세', '보증금']:
        v = Rt[f'{k} 오차율중앙 {lab}']
        P(f'  {lab:16s} {k:4s} 평균 {v.mean():.2f} | 표준편차 {v.std():.2f} | 최대 {v.max():.2f} | {Rt.loc[Rt["시험 연도"] <= 2015, v.name].mean():.2f} vs {Rt.loc[Rt["시험 연도"] >= 2016, v.name].mean():.2f}')
P(f'\n저장: 월세계약_연도별반복시험_전환율앵커.csv | 끝 {time.time() - T0:.0f}초')
