# 월세 계약 단위 최종 모델: 전환율 B (보증금·월세 중 하나를 입력하면 나머지를 예측)
#   전환율 r = 12×월세÷(기준 전세가−보증금). 기준 전세가 J = 계약 직전 12개월 같은 건물·면적대(5㎡) 전세 실거래 2건 이상의 ㎡당 중앙값 × 면적 (대체값 없음)
#   동 앵커 an_d = 계약 분기 시작 전 12개월 유효 계약 전환율 중앙값 (동 30건 이상 → 자치구 30건 이상 → 서울), an_s = 서울 중앙값
#   target = r − an_d, 예측 r̂ = an_d + 모델 출력 (0.5%~15%로 자름)
#   보증금 입력 → 월세 = r̂×(J−보증금)÷12 | 월세 입력 → 보증금 = J − 12×월세÷r̂
#   항목 28개: 면적·층·연식, 동·자치구·서울 월세 수준 5개, 건물 이력 5개, K-apt 9개, 기준 전세가 4개(J, logJ, j_src, j_n), 앵커 2개 (입력 금액 관련 항목 없음)
#   설정: 히스토그램 부스팅 학습률 0.03, 잎 255, 최소 잎 표본 50, 트리 1640개 (RentConversionFinalB.py에서 검증 2024로 선택)
#   성능 근거 (RentConversionFinalB.py, 학습 ≤2024 → 시험 2025-01~2026-09, 104,355건):
#     월세 MAE 27.99만원, 오차율 중앙 12.9%, R² 0.856 | 보증금 MAE 7,062만원, 오차율 중앙 31.0%, R² 0.715 | 전환율 오차 중앙 0.65%p, 편향 +0.02%p
#   행정동 단위 시험 (이 파일): 학습 ≤2024 → 2025-01~2026-09, (동·면적대·분기) 묶음별로 동 안 건물 예측의 중앙값 ↔ 실제 중앙값 (결과: 월세계약_전환율모델_행정동시험.csv)
#   이 파일: 2012~2026-09 유효 계약 전체로 다시 학습 → 2026-09 말 기준 서비스 표(건물 정보 포함)와 함께 저장 (이후 시험 불가, 위 결과가 성능 근거)
#   서비스 함수: predict() = 집 한 채, dong_service() = 행정동·면적대 (docs/wolse-conversion-dong-service.md)
#   예측 가능 조건: 같은 건물·면적대 직전 12개월 전세 2건 이상, 보증금 입력은 보증금 ≤ 전세가×0.8
# 실행: python RentConversionFinalInput.py   (출력: 월세계약_전환율모델_최종_입력.joblib, 월세계약_전환율모델_최종_입력요약.json, 월세계약_전환율모델_입력_행정동시험.csv)
#   ※ 이 파일은 RentConversionFinal.py와 같은 모델·항목·학습인데, 데이터를 data/processed/월세_전환율.csv 한 파일에서만 읽음 (ml/etl/BuildWolseInput.py가 만듦)
#     달라진 점: 읽는 파일 1개 (거래마다 K-apt 단지 정보가 이미 붙어 있음) · 날짜 기준은 아래 END 한 곳 · 결과 이름에 _입력 · 서비스 단지 표(kapt)는 거래에 나온 단지만 담음
import json, os, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
DATA_FILE, OUT = BASE + 'processed/월세_전환율.csv', BASE + 'processed/output/'
END = '2026-09-30'          # 학습·기준 전세가에 쓰는 마지막 계약일. 파일에는 2026-10-20까지 있음. 원본 모델과 같은 결과를 내려고 9월 말로 둠 (10월까지 쓰려면 이 날짜를 바꾸고 다시 학습)

T0 = time.time()
def P(*x): print(*x, flush=True)
MIN_GROUPS, MIN_PAIRS = 5, 30
TREES = 1640
LO, HI, DJ_MAX = 0.005, 0.15, 0.8
MIN_BLD = 3                                                                                           # 행정동 결과를 보여 줄 최소 건물 수 (행정동 시험의 건물 수별 오차로 확인)
TEST = {'trained_until': '2024-12-31', 'test_period': '2025-01 ~ 2026-09', 'test_n': 104355,
        'rent': {'mae': 27.99, 'median_err_pct': 12.9, 'r2': 0.856}, 'deposit': {'mae': 7062, 'median_err_pct': 31.0, 'r2': 0.715},
        'rate': {'mae_pp': 0.93, 'median_pp': 0.65, 'bias_pp': 0.02}}
KAMAP = {'K-apt 세대수': 'ka_hh', 'K-apt 동수': 'ka_dongs', 'K-apt 최고층수': 'ka_top', 'K-apt 세대당 주차대수': 'ka_park', 'K-apt 임대세대 비율': 'ka_lease',
         'K-apt 동당 승강기 수': 'ka_elev', 'K-apt 난방방식 코드': 'ka_heat', 'K-apt 복도유형 코드': 'ka_corr', 'K-apt 분양형태 코드': 'ka_sale'}

# ---------- 월세 계약 (RentConversionFinalB.py와 같은 정리) ----------
X = pd.read_csv(DATA_FILE, low_memory=False, encoding='utf-8-sig', float_precision='round_trip').rename(columns={'구분': '전월세구분', **KAMAP})                       # 월세·전세 거래 전체 (K-apt 9개 붙어 있음)
a = X
a = a[a['전월세구분'] == '월세'].rename(columns={'행정동_코드(상권분석)': '동', '보증금(만원)': 'D', '월세(만원)': 'R', '임대면적(㎡)': 'A', '층': 'F', '건축년도': 'Y0'})
a['dt'] = pd.to_datetime(a['계약일'], errors='coerce')
a = a[a['dt'].notna() & (a['dt'] <= END)]
a = a[(a['R'] > 0) & (a['D'] > 0) & (a['A'] > 0)]
a = a[~a.duplicated(['동', '법정동', '지번', '계약일', 'D', 'R', 'A', 'F'])]
a['rs'], a['ds'] = a['R'] / a['A'], a['D'] / a['A']
for c in ['rs', 'ds']:
    lo, hi = a[c].quantile([0.001, 0.999]); a = a[(a[c] >= lo) & (a[c] <= hi)]
a['m'] = a['dt'].dt.year * 12 + a['dt'].dt.month - 1
a['q'] = a['m'] // 3
a['age'] = a['dt'].dt.year - a['Y0']
a['gu'] = a['자치구'].astype('category').cat.codes
a['bk'] = a['법정동'].astype(str) + ' ' + a['지번'].astype(str) + ' ' + (a['A'] / 5).round().astype(int).astype(str)
a['b'] = pd.factorize(a['bk'])[0]
a['abin'] = (a['A'] // 10).clip(upper=12).astype(int)
a = a.sort_values('m').reset_index(drop=True); mm = a['m'].values
P(f'월세 계약 {len(a):,}건 | {time.time() - T0:.0f}초')

# ---------- 분기별 동네 통계 (직전 분기 말까지 12개월), 건물 이력 (24개월), 기본 보증금 ----------
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
gu_name = a.groupby('gu')['자치구'].first()
q_lo, q_hi = (2012 * 12) // 3, a['q'].max() + 1                                                          # q_hi = 2026-09 말 기준 (서비스용)
STAT, RATE, DEF, BLD = [], [], [], []
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1; w = window(e)
    if len(w) == 0: continue
    s = w.groupby('동').agg(dong_rs=('rs', 'median'), dong_ds=('ds', 'median'), dong_cnt=('rs', 'size')); s['q'] = q
    sg = w.groupby('gu').agg(gu_rs=('rs', 'median'), gu_ds=('ds', 'median'))
    sl = slopes(w); rd, rg, rs_ = sl['dong'], sl['gu'], sl['seoul']['r'].iloc[0]                         # 동 전환율 기울기: 기존 모델과 같은 학습 대상(r 있는 분기)을 쓰기 위해 유지
    d_ok = rd[(rd['g'] >= MIN_GROUPS) & (rd['n'] >= MIN_PAIRS) & (rd['r'] > 0)]
    g_ok = rg[(rg['g'] >= MIN_GROUPS) & (rg['n'] >= MIN_PAIRS) & (rg['r'] > 0)]
    r = pd.DataFrame({'동': dong_gu.index, 'gu': dong_gu.values})
    r['r'] = r['동'].map(d_ok['r']).fillna(r['gu'].map(g_ok['r'])).fillna(rs_); r['q'] = q; RATE.append(r[['q', '동', 'r']])
    s = s.reset_index(); s['seoul_rs'] = w['rs'].median()
    STAT.append(s.merge(dong_gu.rename('gu').reset_index(), on='동').merge(sg.reset_index(), on='gu', how='left'))
    t = w.groupby(['동', 'abin'])['D'].agg(['mean', 'median', 'size']).reset_index(); t['q'] = q; DEF.append(t)
    wb = a.iloc[np.searchsorted(mm, e - 23, 'left'):np.searchsorted(mm, e, 'right')]
    t = wb.groupby('bk').agg(b_rs=('rs', 'median'), b_ds=('ds', 'median'), b_cnt=('rs', 'size'), b_R=('R', 'mean'), b_D=('D', 'mean')).reset_index(); t['q'] = q; BLD.append(t)
STAT, RATE, BLD = pd.concat(STAT, ignore_index=True), pd.concat(RATE, ignore_index=True), pd.concat(BLD, ignore_index=True)
DEF = pd.concat(DEF, ignore_index=True).rename(columns={'mean': 'def_D', 'median': 'def_med', 'size': 'def_n'})
P(f'분기별 통계 {STAT["q"].nunique()}개 분기 | {time.time() - T0:.0f}초')

d = a.merge(STAT.drop(columns='gu'), on=['q', '동'], how='left').merge(RATE, on=['q', '동'], how='left').merge(BLD, on=['q', 'bk'], how='left')
d = d[d['dong_rs'].notna() & d['r'].notna()].copy()

# ---------- K-apt 단지 정보: 파일에 거래마다 이미 붙어 있음 (ka_* 9개). 서비스용 단지 표만 거래에서 모음 ----------
KA_COLS = ['ka_hh', 'ka_dongs', 'ka_top', 'ka_park', 'ka_lease', 'ka_elev', 'ka_heat', 'ka_corr', 'ka_sale']
def kkey(df): return df['자치구'].astype(str) + '|' + df['법정동'].astype(str) + '|' + df['지번'].astype(str).str.extract(r'(\d+)')[0].fillna('')
KA = X.assign(key=kkey(X))[['key'] + KA_COLS].dropna(subset=KA_COLS, how='all').drop_duplicates('key')
d['key'] = kkey(d)

# ---------- 기준 전세가: 같은 건물·면적대 직전 12개월 전세 ㎡당 중앙값 (2건 이상) ----------
J0 = X
J0 = J0[J0['전월세구분'] == '전세'].rename(columns={'보증금(만원)': 'D', '임대면적(㎡)': 'A'})
J0['dt'] = pd.to_datetime(J0['계약일'], errors='coerce'); J0 = J0[J0['dt'].notna() & (J0['dt'] <= END) & (J0['D'] > 0) & (J0['A'] > 0)].copy()
J0['js'] = J0['D'] / J0['A']; lo, hi = J0['js'].quantile([0.001, 0.999]); J0 = J0[(J0['js'] >= lo) & (J0['js'] <= hi)]
J0['m'] = J0['dt'].dt.year * 12 + J0['dt'].dt.month - 1
J0['bk'] = J0['법정동'].astype(str) + ' ' + J0['지번'].astype(str) + ' ' + (J0['A'] / 5).round().astype(int).astype(str)
J0 = J0.sort_values('m').reset_index(drop=True); jm = J0['m'].values
JB = []
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1; w = J0.iloc[np.searchsorted(jm, e - 11, 'left'):np.searchsorted(jm, e, 'right')]
    if len(w) == 0: continue
    t = w.groupby('bk')['js'].agg(['median', 'size']).reset_index().rename(columns={'median': 'j_b', 'size': 'j_n'}); t['q'] = q; JB.append(t)
JB = pd.concat(JB, ignore_index=True)
def add_j(df):
    df = df.merge(JB, on=['q', 'bk'], how='left')
    df.loc[df['j_n'] < 2, ['j_b', 'j_n']] = np.nan
    df['j_src'] = np.where(df['j_b'].notna(), 2, 0); df['J'] = df['j_b'] * df['A']; df['logJ'] = np.log(df['J'])
    return df
d = add_j(d)
d['gapJ'] = d['J'] - d['D']; d['rc'] = 12 * d['R'] / d['gapJ']
d['ok'] = d['J'].notna() & (d['D'] / d['J'] <= DJ_MAX) & (d['rc'] >= LO) & (d['rc'] <= HI)
P(f'기준 전세가 있는 유효 계약 {int(d["ok"].sum()):,}/{len(d):,}건 | {time.time() - T0:.0f}초')

# ---------- 앵커: 직전 12개월 유효 계약 전환율 중앙값 ----------
K = d[d['ok']].sort_values('m'); km = K['m'].values; AN = []
dongs = pd.DataFrame({'동': dong_gu.index, 'gu': dong_gu.values})
for q in range(q_lo, q_hi + 1):
    e = 3 * q - 1; w = K.iloc[np.searchsorted(km, e - 11, 'left'):np.searchsorted(km, e, 'right')]
    if len(w) < 100: continue
    gd = w.groupby('동')['rc'].agg(['median', 'size']); gg = w.groupby('gu')['rc'].agg(['median', 'size'])
    t = dongs.copy(); t['an_s'] = w['rc'].median()
    t['an_lvl'] = np.where(t['동'].isin(gd.index[gd['size'] >= 30]), 'dong', np.where(t['gu'].isin(gg.index[gg['size'] >= 30]), 'gu', 'seoul'))
    t['an_d'] = t['동'].map(gd.loc[gd['size'] >= 30, 'median']).fillna(t['gu'].map(gg.loc[gg['size'] >= 30, 'median'])).fillna(t['an_s'])
    t['q'] = q; AN.append(t.drop(columns='gu'))
AN = pd.concat(AN, ignore_index=True)
d = d.merge(AN.drop(columns='an_lvl'), on=['동', 'q'], how='left')

XA = ['A', 'F', 'age', 'dong_rs', 'dong_ds', 'dong_cnt', 'gu_rs', 'seoul_rs', 'b_rs', 'b_ds', 'b_cnt', 'b_R', 'b_D'] + KA_COLS + ['J', 'logJ', 'j_src', 'j_n', 'an_s', 'an_d']
CFG = dict(loss='squared_error', learning_rate=0.03, max_iter=TREES, max_leaf_nodes=255, min_samples_leaf=50, early_stopping=False, random_state=0)

# ---------- 건물 정보 (bk별 행정동·주소·대표 층·건축년도·면적, 월세+전세 실거래에서) ----------
ICOLS = ['bk', '동', '자치구', '법정동', '지번', 'F', 'Y0', 'A', 'm']
SRC = pd.concat([a[ICOLS], J0.rename(columns={'행정동_코드(상권분석)': '동', '층': 'F', '건축년도': 'Y0'})[ICOLS]], ignore_index=True).sort_values('m')
SRC = SRC[SRC['동'].notna()]; SRC['동'] = SRC['동'].astype('int64')
BAND = {1: '20평 미만', 2: '20평대', 3: '30평대', 4: '40평 이상'}
def band(A): return np.clip((np.asarray(A, float) / 3.3058 // 10).astype(int), 1, 4)
def binfo(src):
    g = src.groupby('bk'); out = g[['동', '자치구', '법정동', '지번']].last()                                 # 위치는 가장 최근 거래 기준
    out['F'], out['Y0'], out['A'], out['info_n'] = g['F'].median(), g['Y0'].median(), g['A'].median(), g.size()
    out = out.reset_index(); out['band'] = band(out['A']); return out

# ---------- 행정동 단위 시험: 학습 ≤2024 → 2025-01~2026-09, 분기마다 그 시점 표만 사용 ----------
#   묶음 = (동, 면적대, 분기), 실제 계약 5건 이상. 입력 = 묶음 실제 보증금(또는 월세) 중앙값
#   서비스 방식: 그 동·면적대에서 기준 전세가 있는 건물마다 예측 → 건물 예측의 중앙값 ↔ 묶음 실제 월세(또는 보증금) 중앙값
#   비교: 동 앵커만(전환율 = an_d), 계약 묶음(실제 계약 각각 예측 후 중앙값, 참고)
trT = d[d['ok'] & d['an_s'].notna() & (d['m'] < 2025 * 12)]
P(f'\n[행정동 시험] 학습 ≤2024 {len(trT):,}건, 트리 {TREES}')
t0 = time.time(); mT = HistGradientBoostingRegressor(**CFG).fit(trT[XA], trT['rc'] - trT['an_d']); P(f'  학습 {time.time() - t0:.0f}초')
te = d[d['ok'] & d['an_s'].notna() & (d['m'] >= 2025 * 12)].copy(); te['band'] = band(te['A'])
te['rhat'] = np.clip(te['an_d'] + mT.predict(te[XA]), LO, HI)
te['pR'], te['pD'] = te['rhat'] * te['gapJ'] / 12, te['J'] - 12 * te['R'] / te['rhat']
G = te.groupby(['동', 'band', 'q']).agg(n=('R', 'size'), D_g=('D', 'median'), R_g=('R', 'median'), cR=('pR', 'median'), cD=('pD', 'median')).reset_index()
G = G[G['n'] >= 5]
def svc_rows(q, info, model):
    x = JB[(JB['q'] == q) & (JB['j_n'] >= 2)].drop(columns='q').merge(info, on='bk')
    x['q'], x['year'] = q, q // 4; x['age'] = x['year'] - x['Y0']; x['key'] = kkey(x)
    x = (x.merge(STAT[STAT['q'] == q].drop(columns=['q', 'gu']), on='동', how='left').merge(BLD[BLD['q'] == q].drop(columns='q'), on='bk', how='left')
          .merge(KA, on='key', how='left').merge(AN[AN['q'] == q].drop(columns='q'), on='동', how='left'))
    x = x[x['dong_rs'].notna() & x['an_d'].notna()].copy()
    x['j_src'], x['J'] = 2, x['j_b'] * x['A']; x['logJ'] = np.log(x['J'])
    x['r_m'] = np.clip(x['an_d'] + model.predict(x[XA]), LO, HI) if len(x) else []; x['r_a'] = x['an_d']
    return x
S = pd.concat([svc_rows(q, binfo(SRC[SRC['m'] <= 3 * q - 1]), mT) for q in sorted(G['q'].unique())], ignore_index=True)
S = S.merge(G[['동', 'band', 'q', 'D_g', 'R_g']], on=['동', 'band', 'q'])
for k in ['m', 'a']:
    S[f'R_{k}'] = np.where(S['D_g'] <= DJ_MAX * S['J'], S[f'r_{k}'] * (S['J'] - S['D_g']) / 12, np.nan)
    dp = S['J'] - 12 * S['R_g'] / S[f'r_{k}']; S[f'D_{k}'] = np.where(dp > 0, dp, np.nan)
agg = S.groupby(['동', 'band', 'q']).agg(R_m=('R_m', 'median'), R_a=('R_a', 'median'), nR=('R_m', 'count'), D_m=('D_m', 'median'), D_a=('D_a', 'median'), nD=('D_m', 'count')).reset_index()
S['w'] = S['b_cnt'].fillna(0)                                                                       # 가중: 건물의 직전 24개월 월세 계약 수 (그 시점에 알 수 있는 값)
def wmed(df, c):
    t = df[df[c].notna() & (df['w'] > 0)].sort_values(c); cw = t.groupby(['동', 'band', 'q'])['w'].cumsum(); tot = t.groupby(['동', 'band', 'q'])['w'].transform('sum')
    return t[cw >= tot / 2].groupby(['동', 'band', 'q'])[c].first().rename(c + 'w')
agg = agg.merge(pd.concat([wmed(S, 'R_m'), wmed(S, 'D_m')], axis=1).reset_index(), on=['동', 'band', 'q'], how='left')
G = G.merge(agg, on=['동', 'band', 'q'], how='left'); G[['nR', 'nD']] = G[['nR', 'nD']].fillna(0)
def gscore(pred, act, keep):
    p, y = pred[keep], act[keep]; e = (p - y).abs()
    return {'groups': int(keep.sum()), 'mae': float(e.mean()), 'median_err_pct': float((e / y * 100).median()), 'r2': float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}
DT = {'groups_total': len(G), 'contracts_in_groups': int(G['n'].sum())}
for lab, nb, act, cols in [('rent', 'nR', 'R_g', [('model', 'R_m'), ('model_weighted', 'R_mw'), ('anchor_only', 'R_a'), ('contract_median', 'cR')]),
                           ('deposit', 'nD', 'D_g', [('model', 'D_m'), ('model_weighted', 'D_mw'), ('anchor_only', 'D_a'), ('contract_median', 'cD')])]:
    DT[lab] = {}
    for mn in [1, 3, 5]:
        keep = G[nb] >= mn
        DT[lab][f'min_buildings_{mn}'] = {name: gscore(G[c], G[act], keep & G[c].notna()) for name, c in cols}
    DT[lab]['by_buildings'] = {f'{lo_}~{hi_}': gscore(G['R_m' if lab == 'rent' else 'D_m'], G[act], (G[nb] >= lo_) & (G[nb] <= hi_)) for lo_, hi_ in [(1, 2), (3, 5), (6, 10), (11, 10 ** 6)] if ((G[nb] >= lo_) & (G[nb] <= hi_)).any()}
    DT[lab]['by_band'] = {BAND[b_]: gscore(G['R_m' if lab == 'rent' else 'D_m'], G[act], (G['band'] == b_) & (G[nb] >= 3)) for b_ in BAND if ((G['band'] == b_) & (G[nb] >= 3)).any()}
for lab, unit in [('rent', '만원'), ('deposit', '만원')]:
    P(f'  [{"보증금 입력 → 월세" if lab == "rent" else "월세 입력 → 보증금"}] 묶음 {DT["groups_total"]:,}개 (계약 {DT["contracts_in_groups"]:,}건)')
    for mn in [1, 3, 5]:
        for name, v in DT[lab][f'min_buildings_{mn}'].items():
            P(f'    건물 {mn}곳 이상 | {name:16s} 묶음 {v["groups"]:5,} | MAE {v["mae"]:,.1f}{unit} | 오차율 중앙 {v["median_err_pct"]:.1f}% | R² {v["r2"]:.3f}')
    P('    건물 수별(모델): ' + ' | '.join(f'{k}곳 {v["groups"]:,}개 {v["median_err_pct"]:.1f}%' for k, v in DT[lab]['by_buildings'].items()))
    P('    면적대별(모델, 3곳 이상): ' + ' | '.join(f'{k} {v["groups"]:,}개 {v["median_err_pct"]:.1f}%' for k, v in DT[lab]['by_band'].items()))
G.assign(면적대=G['band'].map(BAND)).to_csv(OUT + '월세계약_전환율모델_입력_행정동시험.csv', index=False, encoding='utf-8-sig')
P(f'  저장: 월세계약_전환율모델_행정동시험.csv | {time.time() - T0:.0f}초')

tr = d[d['ok'] & d['an_s'].notna()]
P(f'학습 {len(tr):,}건 ({tr["dt"].min():%Y-%m} ~ {tr["dt"].max():%Y-%m}), 항목 {len(XA)}개, 트리 {TREES}')
t0 = time.time(); model = HistGradientBoostingRegressor(**CFG).fit(tr[XA], tr['rc'] - tr['an_d'])
P(f'학습 끝 {time.time() - t0:.0f}초')

# ---------- 서비스 표 (2026-09 말 기준 = q_hi) ----------
tables = {
    'dong': STAT[STAT['q'] == q_hi].drop(columns='q'),                                       # 동, dong_rs, dong_ds, dong_cnt, seoul_rs, gu, gu_rs, gu_ds
    'building': BLD[BLD['q'] == q_hi].drop(columns='q'),                                     # bk(법정동 지번 면적/5), b_rs, b_ds, b_cnt, b_R, b_D
    'jeonse': JB[(JB['q'] == q_hi) & (JB['j_n'] >= 2)].drop(columns='q'),                    # bk, j_b(전세 ㎡당 중앙값), j_n
    'anchor': AN[AN['q'] == q_hi].drop(columns='q'),                                         # 동, an_s, an_d, an_lvl
    'default_deposit': DEF[(DEF['q'] == q_hi) & (DEF['def_n'] >= 10)].drop(columns='q'),     # 동, abin, def_D, def_med, def_n (화면 초기값)
    'kapt': KA,                                                                              # key(자치구|법정동|지번 본번), ka_* 9개
    'gu_names': gu_name.to_dict(),
    'building_info': binfo(SRC),                                                             # bk, 동, 자치구, 법정동, 지번, F(층 중앙값), Y0, A(면적 중앙값), info_n, band(면적대 1~4)
    'bands': BAND,
}
def features(bundle, q):
    """q: DataFrame[자치구, 법정동, 지번, 동, A(㎡), F(층), Y0(건축년도), year(계약 연도)] → 항목 표 + 예측 가능 여부."""
    T = bundle['tables']; x = q.copy().reset_index(drop=True)
    x['bk'] = x['법정동'].astype(str) + ' ' + x['지번'].astype(str) + ' ' + (x['A'] / 5).round().astype(int).astype(str)
    x['key'] = kkey(x); x['age'] = x['year'] - x['Y0']
    x = x.merge(T['dong'], on='동', how='left').merge(T['building'], on='bk', how='left').merge(T['kapt'], on='key', how='left')
    x = x.merge(T['jeonse'], on='bk', how='left').merge(T['anchor'], on='동', how='left')
    x['j_src'] = np.where(x['j_b'].notna(), 2, 0); x['J'] = x['j_b'] * x['A']; x['logJ'] = np.log(x['J'])
    x['ok'] = x['J'].notna() & x['an_d'].notna() & x['dong_rs'].notna()
    return x
def predict(bundle, q, deposit=None, rent=None):
    """보증금(만원) 또는 월세(만원) 하나를 주면 나머지를 계산. 예측 불가 행은 NaN."""
    x = features(bundle, q); r = np.full(len(x), np.nan)
    if x['ok'].any(): r[x['ok'].values] = np.clip(x.loc[x['ok'], 'an_d'] + bundle['model'].predict(x.loc[x['ok'], bundle['features']]), LO, HI)
    x['rate'] = r
    if deposit is not None:
        D = np.asarray(deposit, float); good = x['ok'] & (D <= DJ_MAX * x['J'])
        x['deposit'], x['rent'] = D, np.where(good, r * (x['J'] - D) / 12, np.nan)
    else:
        R = np.asarray(rent, float); D = x['J'] - 12 * R / r
        x['rent'], x['deposit'] = R, np.where(x['ok'] & (D > 0), D, np.nan)
    return x
def wquant(v, w, p):
    """가중 분위수: 값 순으로 정렬해 누적 가중이 전체의 p 이상이 되는 첫 값 (행정동 시험의 가중 중앙값과 같은 방식)."""
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]; return float(v[np.searchsorted(np.cumsum(w), p * w.sum())])
def dong_service(bundle, dong, band_, year, deposit=None, rent=None, min_buildings=MIN_BLD):
    """행정동 코드 + 면적대(1~4) + 보증금 또는 월세 → 그 동·면적대 건물별 예측을 모은 중앙값·25%·75%.
       보증금 입력 → 월세: 건물의 직전 24개월 월세 계약 수(b_cnt)로 가중 (행정동 시험 오차율 중앙 17.8% → 14.4%). 가중할 건물이 없으면 가중 없이.
       월세 입력 → 보증금: 가중 없음 (가중해도 나아지지 않음). 20평 미만(band 1)은 결과 없음 (행정동 시험 오차율 중앙 81.5%).
       예측 가능한 건물이 min_buildings 미만이면 결과 없음."""
    T = bundle['tables']; info = T['building_info']; an = T['anchor'].set_index('동')
    out = {'dong': int(dong), 'band': T['bands'][band_], 'output': 'rent' if deposit is not None else 'deposit', 'unit': '만원',
           'dong_rate': float(an.at[dong, 'an_d']) if dong in an.index else None, 'dong_rate_level': an.at[dong, 'an_lvl'] if dong in an.index else None}
    if deposit is None and band_ == 1:
        return out | {'buildings_used': 0, 'weighted': False, 'result': None, 'reason': '20평 미만은 월세로 보증금을 예측하지 않습니다 (오차가 커서)'}
    rows = info[(info['동'] == dong) & (info['band'] == band_) & info['bk'].isin(T['jeonse']['bk'])].copy(); rows['year'] = year
    x = predict(bundle, rows[['자치구', '법정동', '지번', '동', 'A', 'F', 'Y0', 'year']], deposit=None if deposit is None else np.full(len(rows), deposit), rent=None if rent is None else np.full(len(rows), rent))
    x = x[x[out['output']].notna()]; v = x[out['output']].values
    out['buildings_used'] = int(len(v))
    if len(v) < min_buildings: return out | {'weighted': False, 'result': None, 'reason': f'예측 가능한 건물 {len(v)}곳 (기준 {min_buildings}곳 미만)'}
    w = x['b_cnt'].fillna(0).values if deposit is not None else np.ones(len(v))
    out['weighted'] = bool(deposit is not None and w.sum() > 0)
    if w.sum() == 0: w = np.ones(len(v))
    return out | {'result': {'median': wquant(v, w, 0.5), 'p25': wquant(v, w, 0.25), 'p75': wquant(v, w, 0.75)}}

bundle = {'model': model, 'features': XA, 'config': CFG, 'target': '전환율 − 동 앵커 (전환율 = 12×월세÷(기준 전세가−보증금))',
          'rate_clip': [LO, HI], 'deposit_to_jeonse_max': DJ_MAX, 'trained_until': f'{tr["dt"].max():%Y-%m-%d}', 'train_n': len(tr),
          'stats_q_index': int(q_hi), 'stats_basis': '2026-09 말까지 (동네 통계·앵커·전세가 12개월, 건물 이력 24개월)', 'tables': tables,
          'test_before_refit': TEST, 'dong_test': DT, 'min_buildings': MIN_BLD,
          'usage': '보증금 입력 → 월세 = 전환율×(J−보증금)÷12 (보증금 ≤ J×0.8) | 월세 입력 → 보증금 = J − 12×월세÷전환율 | J = 같은 건물·면적대 직전 12개월 전세 ㎡당 중앙값(2건 이상)×면적'}

# ---------- 확인: 저장한 표만으로 2026-07~09 계약을 양방향 예측 (학습에 들어간 계약이라 참고용, 표 연결 확인 목적) ----------
chk = d[d['ok'] & (d['m'] >= 2026 * 12 + 6)].copy(); chk['year'] = chk['dt'].dt.year
qq = chk[['자치구', '법정동', '지번', '동', 'A', 'F', 'Y0', 'year']]
pr, pd_ = predict(bundle, qq, deposit=chk['D'].values), predict(bundle, qq, rent=chk['R'].values)
ok_r, ok_d = pr['rent'].notna().values, pd_['deposit'].notna().values
er, ed = np.abs(pr['rent'].values - chk['R'].values)[ok_r], np.abs(pd_['deposit'].values - chk['D'].values)[ok_d]
P(f'확인 (2026-07~09 계약 {len(chk):,}건, 표만 사용): 월세 예측 가능 {ok_r.mean() * 100:.1f}% MAE {er.mean():.2f}만원 오차율 중앙 {np.median(er / chk["R"].values[ok_r] * 100):.1f}% '
  f'| 보증금 예측 가능 {ok_d.mean() * 100:.1f}% MAE {ed.mean():,.0f}만원 오차율 중앙 {np.median(ed / chk["D"].values[ok_d] * 100):.1f}%')
ex = tables['default_deposit'].sort_values('def_n', ascending=False).iloc[0]                          # 예시: 계약이 가장 많은 동·면적 구간
P('예시 dong_service:', dong_service(bundle, int(ex['동']), int(band([ex['abin'] * 10 + 5])[0]), 2026, deposit=float(ex['def_med'])))
bundle['check_2026Q3_in_sample'] ={'n': len(chk), 'rent_ok_pct': float(ok_r.mean() * 100), 'rent_mae': float(er.mean()), 'deposit_ok_pct': float(ok_d.mean() * 100), 'deposit_mae': float(ed.mean())}

joblib.dump(bundle, OUT + '월세계약_전환율모델_최종_입력.joblib')
json.dump({k: v for k, v in bundle.items() if k not in ('model', 'tables')} | {'tables': {k: len(v) for k, v in tables.items()}},
          open(OUT + '월세계약_전환율모델_최종_입력요약.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
P({k: len(v) for k, v in tables.items()})
P(f'저장: 월세계약_전환율모델_최종_입력.joblib, 월세계약_전환율모델_최종_입력요약.json | 끝 {time.time() - T0:.0f}초')
