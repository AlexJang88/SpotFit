# 부동산 평당 금액 분석 (전체 기간, 아파트+빌라 매매): 행정동별로 묶은 7개 항목(층수·거래일·완공일·직장인구·CCTV·1인가구·유동인구)이 평당 금액과
# 얼마나 관련 있는지(연관도)를 여러 분석 방법으로 비교하고, 행정동 기준으로 한 표에 합친다. (평당 금액을 예측하지는 않음)
# 먼저 ConvertSale.py 로 *_juso.csv 를 만들어야 함. 결과는 한 페이지(HTML)로 모아서 열림
import glob, os, unicodedata, platform, io, base64, webbrowser
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
OUT = os.path.join(RAW, '..', 'processed', 'output') + os.sep
K = '행정동_코드'
Y = '평당 금액(만원)'
GU = {11110: '종로구', 11140: '중구', 11170: '용산구', 11200: '성동구', 11215: '광진구', 11230: '동대문구', 11260: '중랑구',
      11290: '성북구', 11305: '강북구', 11320: '도봉구', 11350: '노원구', 11380: '은평구', 11410: '서대문구', 11440: '마포구',
      11470: '양천구', 11500: '강서구', 11530: '구로구', 11545: '금천구', 11560: '영등포구', 11590: '동작구', 11620: '관악구',
      11650: '서초구', 11680: '강남구', 11710: '송파구', 11740: '강동구'}      # 구 코드(앞 5자리) → 구 이름
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 → 상권분석의 옛 행정동 (KIKmix 변경 이력: 신설동·용두동←용신동, 상일제1·2동←상일동, 개포3동←일원2동)
DEAL = ['종류(빌라=1)', '완공연도', '거래일(년)', '층수']                         # 부동산 거래 항목 (행정동은 아래에서 따로 봄)

num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')     # '12,000' 같은 글자를 숫자로
def when(d):            # 계약 연/월/일 → 날짜 하나 + 2024.5 같은 숫자 하나 (연도 + 연중 경과 비율)
    t = pd.to_datetime(d[['dealYear', 'dealMonth', 'dealDay']].set_axis(['year', 'month', 'day'], axis=1), errors='coerce')
    return t, t.dt.year + (t.dt.dayofyear - 1) / 365

def load(kw):           # 상권분석 csv를 이름으로 찾아 불러옴 (모든 분기)
    files = [f for f in glob.glob(RAW + '*.csv') if kw in unicodedata.normalize('NFC', os.path.basename(f))]
    return pd.concat([pd.read_csv(f, encoding='cp949') for f in files])

# ---------- 1) 동네 특징 X 10개: 전체 기간 평균 ----------
def one_person():       # 1인가구 비율(2022~2025 월별 전체). 코드 체계가 달라 이름으로 연결 (신사동은 강남·관악 2곳이라 따로 처리)
    one = pd.read_csv(RAW + '../processed/1인가구/1인가구_전체.csv', encoding='cp949').groupby(['행정동코드', '행정동명'])[['1인가구수', '총인구']].sum().reset_index()
    one['비율'] = one['1인가구수'] / one['총인구'] * 100
    one['이름'] = one['행정동명'].str.replace('·', '').mask(one['행정동코드'] == 1121068, '관악신사동')
    nm = load('상주인구-행정동').drop_duplicates(K)[[K, '행정동_코드_명']]
    nm['이름'] = nm['행정동_코드_명'].str.replace('?', '').mask(nm[K] == 11620685, '관악신사동')
    return one.merge(nm, on='이름').set_index(K)['비율']

def cctv_count(area):   # CCTV는 위경도뿐이라, 가장 가까운 행정동 중심점에 배정 (카메라 대수 합)
    c = pd.read_csv(RAW + '기타/CCTV정보_서울특별시.csv', encoding='cp949', usecols=['카메라대수', 'WGS84위도', 'WGS84경도'])
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(c['WGS84경도'].values, c['WGS84위도'].values)   # 위경도 → 중심 좌표와 같은 좌표계(m)
    dist, i = cKDTree(area[['엑스좌표_값', '와이좌표_값']].values).query(np.c_[x, y])
    ok = dist < 5000    # 서울 밖 좌표(오류)는 제외
    return pd.Series(c['카메라대수'].values[ok]).groupby(area.index.values[i[ok]]).sum()

def dong_features():
    area = load('영역-행정동').drop_duplicates(K).set_index(K)
    X = pd.DataFrame(index=area.index)
    X['CCTV 수'] = cctv_count(area)
    X['유동인구 수'] = load('길단위인구-행정동').groupby(K)['총_유동인구_수'].mean()
    X['직장인구 수'] = load('직장인구-행정동').groupby(K)['총_직장_인구_수'].mean()
    X['거주인구 수'] = load('상주인구-행정동').groupby(K)['총_상주인구_수'].mean()
    X['집객시설 수'] = load('집객시설-행정동').groupby(K)['집객시설_수'].mean()
    st = load('점포-행정동').drop_duplicates(['기준_년분기_코드', K, '서비스_업종_코드'])          # 점포 데이터는 2025년 1분기부터만 있음
    q = st.groupby(['기준_년분기_코드', K])[['전체_점포_수', '개업_점포_수', '폐업_점포_수']].sum()          # 분기별 합계
    X['점포 수'] = q['전체_점포_수'].groupby(K).mean()
    X['창업률(%)'] = (q['개업_점포_수'] / q['전체_점포_수'] * 100).groupby(K).mean()     # 개업 점포 / 전체 점포
    X['폐업률(%)'] = (q['폐업_점포_수'] / q['전체_점포_수'] * 100).groupby(K).mean()
    X['업종 수'] = st[st['전체_점포_수'] > 0].groupby(['기준_년분기_코드', K])['서비스_업종_코드'].nunique().groupby(K).mean()   # 점포가 있는 업종 종류
    X['1인가구 비율(%)'] = one_person()
    return X

# ---------- 2) 부동산 거래 항목 + 평당 금액 y: 거래 1건이 1행 ----------
def deals():
    cols = ['_source_service', K, '연결방식', 'dealAmount', 'excluUseAr', 'dealYear', 'dealMonth', 'dealDay', 'buildYear', 'floor', 'cdealType']
    d = pd.concat([pd.read_csv(RAW + f'../processed/부동산(도로명주소)/{n}_juso.csv', low_memory=False, usecols=cols) for n in ['apt_sale', 'villa_sale']], ignore_index=True)
    link = (d['연결방식'].fillna('연결안됨').value_counts(normalize=True) * 100).round(1)       # 전체 거래의 행정동 연결 방식 비율 (리포트에 표시)
    d = d[d['cdealType'].isna() & (d['excluUseAr'] >= 20) & (d['연결방식'] == '지번 일치')]       # 해제(취소) 거래·초소형·지번이 주소 DB와 안 맞는 거래 제외
    date, year = when(d)
    return pd.DataFrame({K: d[K].astype(int).replace(OLD), '종류': d['_source_service'].map({'apt_sale': '아파트', 'villa_sale': '빌라'}),
        Y: num(d['dealAmount']) / (d['excluUseAr'] / 3.3058),                              # 거래금액(만원) / 전용면적(평)
        '종류(빌라=1)': (d['_source_service'] == 'villa_sale').astype(int),                # 종류는 파일(_source_service)로 구분
        '완공연도': d['buildYear'], '거래일': date, '거래일(년)': year, '층수': d['floor']}), link      # 완공일 = 건축년도

# ---------- 3) 행정동 기준으로 하나의 표 + 연관도 + 여러 분석 방법 ----------
all_deals, link = deals()
XD = dong_features()
df = all_deals.join(XD, on=K, how='inner').dropna(subset=[Y]).reset_index(drop=True)
n_drop = len(all_deals) - len(df)        # 상권분석에 없는 행정동의 거래는 제외
DONG = [c for c in df.columns if c not in [K, '종류', '거래일', Y] + DEAL]       # 동네 특징 10개
# 거래를 행정동별로 묶어서 한 표로: 평당 금액은 동네 중앙값, 층수·거래일·완공연도는 동네 평균, 나머지는 동네 특징
USE = ['층수', '거래일(년)', '완공연도', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']          # 분석에 쓰는 7개 항목
g = df.groupby(K).agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean'),
                         '빌라 비율(%)': ('종류(빌라=1)', lambda v: v.mean() * 100)}).join(XD)                 # 행정동 1곳 = 1행
dg = g[[Y] + USE]
corr_all = dg.corr(method='spearman')                                      # 연관도 표 (항목끼리도 포함)
corr = corr_all.loc[USE, Y]                                                # 평당 금액과의 연관도 (+: 클수록 비쌈, -: 클수록 저렴)

def eta(cat, y):        # 범주형(행정동)과 평당 금액의 연관도 = 상관비 η (0~1, 부호 없음): 동네별 평균 차이가 가격 차이를 얼마나 설명하나
    g = y.groupby(cat)
    return np.sqrt((g.size() * (g.mean() - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())
level = lambda v: '강함' if abs(v) >= .5 else '중간' if abs(v) >= .3 else '약함' if abs(v) >= .1 else '없음'
corr_tab = pd.DataFrame({'항목': USE, '연관도(스피어만)': corr.values}).assign(강도=lambda t: t['연관도(스피어만)'].map(level))
corr_tab['연관도(스피어만)'] = corr_tab['연관도(스피어만)'].map('{:+.2f}'.format)

# 같은 질문("평당 금액과 어떤 항목이 관련 있나")을 8가지 방법으로 비교. 방법마다 보는 관점이 달라서 여러 개를 같이 봄 (행정동 단위 표 기준)
dd = dg.dropna()                                                # 직장인구·1인가구가 비어 있는 동네를 뺀 표 (회귀·정보량·중요도는 빈 값을 못 다룸)
def scores(t, fast=False):                                      # t: 행정동 표(평당 금액 + 7개 항목) → 항목 × 분석 방법 점수표. fast=True는 반복 평가용(가볍게)
    z = (t[USE] - t[USE].mean()) / t[USE].std()                  # 표준화: 단위가 다른 항목을 같은 눈금(표준편차 1)으로
    S = pd.DataFrame(index=USE)                                  # 행 = 항목 7개, 열 = 분석 방법
    S['스피어만'] = t[USE].corrwith(t[Y], method='spearman')       # ① 순위가 같이 오르내리나 (이상치에 강하고 곡선 관계도 잡음)
    S['피어슨'] = t[USE].corrwith(t[Y])                            # ② 값이 직선으로 같이 움직이나
    S['켄달'] = t[USE].corrwith(t[Y], method='kendall')            # ③ 동네 두 곳씩 비교해 같은 방향인 비율 (이상치에 가장 둔감)
    P = np.linalg.inv(t[[Y] + USE].corr().values)                # ④ 편상관: 나머지 6개 항목을 모두 같게 맞췄을 때의 상관 (상관행렬의 역행렬로 계산)
    S['편상관(나머지 고정)'] = -P[0, 1:] / np.sqrt(P[0, 0] * np.diag(P)[1:])
    S['회귀 표준화 계수'] = LinearRegression().fit(z, t[Y]).coef_      # ⑤ 다른 항목이 같을 때 한 눈금 오르면 평당 금액(만원)이 얼마나 변하나
    S['라쏘 계수'] = LassoCV(cv=5, random_state=0).fit(z, t[Y]).coef_   # ⑥ 회귀와 같지만 쓸모없는 항목의 계수를 0으로 깎음 (0이면 없어도 되는 항목)
    S['상호정보량'] = mutual_info_regression(t[USE], t[Y], random_state=0)      # ⑦ 직선·순서가 아닌 관계까지 포함한 관련 정도 (부호 없음)
    imp, k = np.zeros(len(USE)), (2 if fast else 5)
    for tr, te in KFold(k, shuffle=True, random_state=0).split(t):   # ⑧ 순열 중요도: 항목 하나를 섞으면 설명력(R²)이 얼마나 떨어지나. 모델에 쓰지 않은 동네에서 측정
        m = RandomForestRegressor(100 if fast else 300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(t.iloc[tr][USE], t.iloc[tr][Y])
        imp += permutation_importance(m, t.iloc[te][USE], t.iloc[te][Y], n_repeats=(5 if fast else 10), random_state=0).importances_mean / k
    S['순열 중요도'] = imp
    return S
M = scores(dd)
z = (dd[USE] - dd[USE].mean()) / dd[USE].std()                  # 표준화 (아래 VIF 계산용)
METH = list(M.columns)                                           # 순위를 매기는 8가지 방법
M['거래 단위 스피어만(참고)'] = df[USE].corrwith(df[Y], method='spearman')    # 참고: 동네로 묶지 않고 거래 1건씩 비교한 값
M['VIF(겹침)'] = [1 / (1 - LinearRegression().fit(z.drop(columns=c), z[c]).score(z.drop(columns=c), z[c])) for c in USE]   # 다른 항목과 얼마나 겹치나 (10 넘으면 겹침이 큼)
M['평균 순위'] = M[METH].abs().rank(ascending=False).mean(axis=1)    # 방법마다 "관련 큰 순서"를 매겨 평균 (1등이 가장 관련 큼)
M = M.sort_values('평균 순위')

# 어떤 방법이 가장 좋은가: 두 가지 기준으로 평가
# ① 안정성: 동네를 무작위로 반으로 나눠 각각 분석했을 때 두 결과의 항목 순서가 얼마나 같나 (1에 가까울수록 어느 동네로 해도 같은 답, 30번 반복)
# ② 항목 고르는 능력: 학습용 동네에서 이 방법이 뽑은 상위 3개 항목만으로 평당 금액을 설명하는 직선 모델을 만들고, 모델에 쓰지 않은 동네에서 설명력(R²)을 잼 (5분할 × 2번)
rng = np.random.default_rng(0)
stab, pick = {c: [] for c in METH}, {c: [] for c in METH + ['7개 항목 모두', '무작위 3개']}
for _ in range(30):
    idx = rng.permutation(len(dd)); h = len(dd) // 2
    a, b = scores(dd.iloc[idx[:h]], True), scores(dd.iloc[idx[h:]], True)
    for c in METH: stab[c].append(a[c].abs().corr(b[c].abs(), method='spearman'))
for rep in range(2):
    for tr, te in KFold(5, shuffle=True, random_state=rep).split(dd):
        A, B = dd.iloc[tr], dd.iloc[te]; S = scores(A, True)
        fit = lambda cols: LinearRegression().fit(A[cols], A[Y]).score(B[cols], B[Y])
        for c in METH: pick[c].append(fit(list(S[c].abs().nlargest(3).index)))
        pick['7개 항목 모두'].append(fit(USE)); pick['무작위 3개'].append(fit(list(rng.choice(USE, 3, replace=False))))
judge = pd.DataFrame({'안정성': {c: np.mean(v) for c, v in stab.items()}, '상위 3개 선택 설명력(R²)': {c: np.mean(v) for c, v in pick.items()}})
jm = judge.loc[METH]
judge['종합 순위'] = (jm.iloc[:, 0].rank(ascending=False) + jm.iloc[:, 1].rank(ascending=False)).rank(method='min')     # 두 기준의 순위를 더해 다시 순위
best = judge['종합 순위'].idxmin()
judge_tab = judge.assign(**{c: judge[c].map(lambda v: '-' if pd.isna(v) else f'{v:.2f}') for c in judge.columns[:2]}, **{'종합 순위': judge['종합 순위'].map(lambda v: '-' if pd.isna(v) else f'{v:.0f}')}).rename_axis('방법').reset_index()

nm = load('상주인구-행정동').drop_duplicates(K).set_index(K)['행정동_코드_명'].str.replace('?', '·')
dong_table = g.rename(columns={Y: '평당 금액 중앙값(만원)', '층수': '평균 층수', '거래일(년)': '평균 거래일(년)', '완공연도': '평균 완공연도'}).assign(자치구=lambda t: t.index.map(lambda c: GU[c // 1000]), 행정동=nm)
dong_table = dong_table[['자치구', '행정동', '거래 수', '평당 금액 중앙값(만원)', '빌라 비율(%)', '평균 층수', '평균 거래일(년)', '평균 완공연도'] + list(XD.columns)].sort_values('평당 금액 중앙값(만원)', ascending=False).reset_index()
os.makedirs(OUT, exist_ok=True)
dong_table.round(1).to_csv(OUT + '평당금액_행정동표.csv', index=False, encoding='utf-8-sig')

# ---------- 행정동별 연관도: 동네 하나하나에서 "무엇이 평당 금액과 관련 있나" ----------
# 동네 안에서 거래마다 다른 항목(층수·거래일·완공연도·종류)은 그 동네 거래끼리 연관도를 구함 (거래 100건 이상인 동네만)
# 직장인구·CCTV 같은 동네 특징은 한 동네 안에서 값이 하나뿐이라 그 동네만으로는 연관도를 못 구함 → 서울 424곳 중 순위(백분위)로 표시
WITHIN = ['층수', '거래일(년)', '완공연도', '종류(빌라=1)']
within = lambda t: pd.Series({c: t[c].corr(t[Y], method='spearman') if t[c].nunique() > 1 else np.nan for c in WITHIN})
dc = df.groupby(K).filter(lambda t: len(t) >= 100).groupby(K)[WITHIN + [Y]].apply(within)
pct = XD[['직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']].rank(pct=True).mul(100).add_suffix(' 백분위')       # 100에 가까울수록 서울에서 많은 동네
X4 = ['직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
gname = dg.index.map(lambda c: GU[c // 1000])
gu_corr = dg.assign(자치구=gname).groupby('자치구')[USE + [Y]].apply(lambda t: t[USE].corrwith(t[Y], method='spearman'))       # 같은 자치구 안의 동네끼리 비교
gu_corr.insert(0, '동네 수', dg.groupby(gname).size())
dong_corr = dong_table.set_index(K)[['자치구', '행정동', '거래 수', '평당 금액 중앙값(만원)']].join(dc.add_prefix('동네 안 연관도: ')).join(gu_corr[X4].add_prefix('구 안 연관도: '), on='자치구').join(pct).reset_index(drop=True)
dong_corr.round(2).to_csv(OUT + '평당금액_행정동별_연관도.csv', index=False, encoding='utf-8-sig')
gu_corr.round(2).to_csv(OUT + '평당금액_자치구별_연관도.csv', encoding='utf-8-sig')
ff = lambda fm: (lambda v: '' if pd.isna(v) else fm.format(v))
fmt2 = {**{c: ff('{:+.2f}') for c in dong_corr.columns if '연관도' in c}, **{c: ff('{:.0f}') for c in dong_corr.columns if c.endswith('백분위')}}

# ---------- 4) 결과를 한 페이지(HTML)로 모으기 ----------
def img(fig):           # 그래프를 이미지로 바꿔 HTML에 넣음
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
# 집객시설 종류별 연관도: 시설 수(전체 기간 평균, 빈칸=0)와 평당 금액의 스피어만 상관계수
fac = load('집객시설-행정동').drop(columns=['기준_년분기_코드', '행정동_코드_명', '집객시설_수']).fillna(0).groupby(K).mean()
fac = fac.loc[:, fac.std() > 0].rename(columns=lambda c: c.replace('_수', '').replace('_', ' '))              # 모든 동네가 0인 시설(공항 등)은 제외
fdf = df[[K, Y, '종류']].join(fac, on=K)
dong_y = fdf.groupby(K)[Y].median()
sp = lambda m: fdf[m][fac.columns].corrwith(fdf.loc[m, Y], method='spearman')            # m: 거래 선택(True/False) → 시설별 스피어만 상관계수
fac_corr = pd.DataFrame({'거래 1건 기준': sp(fdf[Y].notna()), '동네 기준(동네별 중앙값)': fac.reindex(dong_y.index).corrwith(dong_y, method='spearman'),
                         '아파트만': sp(fdf['종류'] == '아파트'), '빌라만': sp(fdf['종류'] == '빌라'),
                         '시설이 있는 동네(%)': (fac > 0).mean() * 100}).sort_values('거래 1건 기준', ascending=False)
fac_corr['강도'] = fac_corr['거래 1건 기준'].map(level)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
sns.heatmap(fac_corr.iloc[:, :4], cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f', ax=ax)
ax.set_title('집객시설 종류별 평당 금액 연관도\n(빨강: 시설이 많을수록 비쌈, 파랑: 많을수록 저렴)'); plt.xticks(rotation=20, ha='right')
img_fac = img(fig)
fac_html = fac_corr.assign(**{c: fac_corr[c].map('{:+.2f}'.format) for c in fac_corr.columns[:4]}, **{'시설이 있는 동네(%)': fac_corr['시설이 있는 동네(%)'].map('{:.0f}'.format)}).rename_axis('집객시설').reset_index()

# 전월세(아파트·연립다세대(빌라)·단독다가구·오피스텔): 서울시 전월세가 6개 파일을 합친 자료. 종류마다 건물 성격과 금액 수준이 달라서 종류별로 따로 분석
# 행정동이 정확한 거래만: 오피스텔·빌라는 지번이 주소 DB와 일치한 거래, 단독다가구는 지번이 없어서 법정동이 행정동 하나뿐인 거래
RENT = pd.read_csv(RAW + '../processed/부동산(도로명주소)/서울시_전월세가_juso.csv', low_memory=False)
n_rent = len(RENT)
ok = ((RENT['건물용도'] == '단독다가구') & (RENT['연결방식'] == '법정동 하나뿐')) | ((RENT['건물용도'] != '단독다가구') & (RENT['연결방식'] == '지번 일치'))
RENT = RENT[ok & (RENT['임대면적'] >= 20) & (RENT['계약일'] >= 20210101)].reset_index(drop=True)         # 20㎡ 이상, 매매 분석과 같은 기간(2021년 이후 계약)
ymd = pd.to_datetime(RENT['계약일'].astype(str), format='%Y%m%d', errors='coerce')
RENT = pd.DataFrame({K: RENT[K].astype(int).replace(OLD), '종류': RENT['건물용도'].replace({'연립다세대': '빌라(연립다세대)'}), '월세 거래': num(RENT['임대료(만원)']) > 0,
                     '평당 보증금': num(RENT['보증금(만원)']) / (RENT['임대면적'] / 3.3058), '평당 월세': num(RENT['임대료(만원)']) / (RENT['임대면적'] / 3.3058),      # 보증금·월세(만원) / 면적(평)
                     '완공연도': num(RENT['건축년도']), '거래일(년)': ymd.dt.year + (ymd.dt.dayofyear - 1) / 365, '층수': num(RENT['층']), '면적(㎡)': RENT['임대면적']})
def rent_corr(yn, rent, kd):    # kd 종류의 전월세 거래 → 평당 보증금(또는 월세)과의 연관도 표. rent=True: 월세 거래만, False: 전세(월세 0)만
    t = RENT[(RENT['월세 거래'] == rent) & (RENT['종류'] == kd)].join(XD, on=K, how='inner')
    t = t.rename(columns={yn: 'y'}).drop(columns=[c for c in ['평당 보증금', '평당 월세'] if c != yn]).dropna(subset=['y'])
    c = t[['y'] + [x for x in t.columns if x not in [K, '종류', '월세 거래', 'y']]].corr(method='spearman')['y'].drop('y').dropna()      # 숫자 항목: 스피어만 (값이 아예 없는 항목은 뺌)
    tab = pd.DataFrame({'항목': [('[동네] ' if x in DONG else '[부동산] ') + x for x in c.index] + ['[부동산] 행정동 (상관비 η)'], '연관도': list(c.values) + [eta(t[K], t['y'])]})     # 행정동은 이름이라 상관비 η
    tab = tab.sort_values('연관도', key=abs, ascending=False).assign(강도=lambda x: x['연관도'].map(level))
    tab['연관도'] = tab['연관도'].map('{:+.2f}'.format)
    return t, tab
RENT_Y = [('① 전세(월세 0)의 평당 보증금', '평당 보증금', False), ('② 월세 거래의 평당 보증금', '평당 보증금', True), ('③ 월세 거래의 평당 월세', '평당 월세', True)]
RENT_K = ['아파트', '빌라(연립다세대)', '오피스텔', '단독다가구']
rent_res = {(kd, lb): rent_corr(yn, rw, kd) for kd in RENT_K for lb, yn, rw in RENT_Y}      # 종류 4 × y 3 = 12개 표
fig, ax = plt.subplots(figsize=(7, 4.5))
corr.sort_values().plot.barh(ax=ax, color=['tomato' if v > 0 else 'steelblue' for v in corr.sort_values()])
ax.set(xlabel='연관도 (스피어만 상관계수)', title='평당 금액과의 연관도 (행정동 단위)  빨강: 클수록 비쌈, 파랑: 클수록 저렴')
img_bar = img(fig)
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr_all, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', ax=ax)
ax.set_title('전체 연관도 표 (평당 금액 + 7개 항목, 행정동 단위)'); plt.xticks(rotation=45, ha='right')
img_heat = img(fig)
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.heatmap(M[METH].abs().div(M[METH].abs().max()), cmap='Reds', vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax)
ax.set_title('분석 방법별 관련 정도 (각 방법의 최댓값 = 1, 진할수록 평당 금액과 관련 큼)'); plt.xticks(rotation=30, ha='right')
img_meth = img(fig)
fig, ax = plt.subplots(figsize=(6.5, 4))
df.groupby([df['거래일'].dt.year, '종류'])[Y].median().unstack().plot(marker='o', ax=ax)
ax.set(xlabel='거래 연도', ylabel='평당 금액 중앙값 (만원/평)', title='연도별 평당 금액 변화')
img_year = img(fig)

OTHER = ['순열 중요도', 'VIF(겹침)', '평균 순위']
meth_tab = M.assign(**{c: M[c].map(lambda v: '-' if pd.isna(v) else f'{v:+.2f}' if abs(v) < 100 else f'{v:+,.0f}') for c in M.columns if c not in OTHER},
                     **{'순열 중요도': M['순열 중요도'].map('{:.3f}'.format), 'VIF(겹침)': M['VIF(겹침)'].map('{:.1f}'.format), '평균 순위': M['평균 순위'].map('{:.1f}'.format)}).rename_axis('항목').reset_index()
how = pd.DataFrame([
    ['스피어만', '동네들을 평당 금액 순서로 세웠을 때, 이 항목의 순서도 비슷하게 서나?', '기본 방법. 가격은 일부 동네가 아주 비싸서 값이 아닌 순서로 비교하는 게 안전함', '오르다 내리는 관계(U자 등)는 못 잡음'],
    ['피어슨', '이 항목이 커질 때 평당 금액이 직선으로 같이 커지나?', '가장 흔한 방법이라 스피어만과 비교용', '아주 비싼 동네 몇 곳에 크게 흔들림'],
    ['켄달', '동네 두 곳을 짝지었을 때, 이 항목이 큰 쪽이 평당 금액도 큰 비율은?', '이상치에 가장 둔감해서 스피어만 결과를 한 번 더 확인', '값이 스피어만보다 작게 나옴 (크기가 아니라 순위로 비교)'],
    ['편상관(나머지 고정)', '다른 6개 항목이 모두 같다고 쳤을 때, 이 항목과 평당 금액의 관계는?', '항목끼리 서로 영향을 주고받아 생기는 가짜 관계를 걷어냄 (예: CCTV와 직장인구가 같이 움직이는 경우)', '항목이 서로 많이 겹치면 불안정'],
    ['회귀 표준화 계수', '다른 항목이 같을 때, 이 항목이 평균보다 한 칸 크면 평당 금액이 몇 만원 오르나?', '모든 항목을 동시에 놓고 "각자의 몫"을 금액(만원)으로 보려고', '서로 겹치는 항목(VIF가 큰 것)은 계수가 불안정'],
    ['라쏘 계수', '회귀와 같지만, 도움이 안 되는 항목의 몫을 0으로 깎아서 "꼭 필요한 항목"만 남김', '여러 항목 중 진짜 필요한 것만 고르려고', '0으로 깎인 항목이 정말 무관한 건지, 다른 항목과 겹쳐서 밀려난 건지는 구분 못 함'],
    ['상호정보량', '이 항목을 알면 평당 금액에 대해 얼마나 더 알게 되나? (직선이든 곡선이든 상관없이)', '직선·순서가 아닌 복잡한 관계까지 포함하려고. 부호는 없음', '값의 크기를 해석하기 어렵고, 동네가 적어서(약 410곳) 흔들림'],
    ['순열 중요도', '이 항목의 값을 뒤죽박죽 섞으면 설명력이 얼마나 떨어지나? (많이 떨어질수록 중요한 항목)', '항목들의 복잡한 조합 효과까지 반영. 모델이 학습에 쓰지 않은 동네에서 측정해 "외운 효과"를 뺌', '랜덤포레스트 모델에 의존, 서로 겹치는 항목은 중요도가 나뉨']], columns=['방법', '쉽게 말하면', '왜 골랐나', '주의할 점'])
none = ', '.join(c for c in USE if abs(corr[c]) < 0.1) or '없음'
top_sp = corr.abs().idxmax()
table = lambda d, cls='t': d.to_html(index=False, border=0, classes=cls, float_format='{:,.0f}'.format)
rent_html = ''.join(f'<h5>{kd}</h5><div class="row">' + ''.join(f'<div><b>{lb}</b><br>분석 거래 {len(rent_res[kd, lb][0]):,}건(행정동 {rent_res[kd, lb][0][K].nunique()}곳){table(rent_res[kd, lb][1], "t l")}</div>' for lb, _, _ in RENT_Y) + '</div>' for kd in RENT_K)
top3 = ', '.join(M.index[:3])
desc = pd.DataFrame([
    ['평당 금액(y)', '거래금액(만원) ÷ (전용면적 ÷ 3.3058). 동네별로는 중앙값', '부동산 거래'],
    ['층수', '거래한 집의 층 (지하는 음수), 동네 평균', '부동산 거래'], ['거래일(년)', '계약 연/월/일을 하나로 합쳐 숫자로 (예: 2024.5), 동네 평균', '부동산 거래'],
    ['완공연도', '건축년도(완공일), 동네 평균', '부동산 거래'], ['직장인구 수', '그 동네에서 일하는 사람 수 (전체 기간 평균)', '직장인구'],
    ['CCTV 수', '동네의 CCTV 카메라 대수 합 (위경도로 가장 가까운 동네에 배정)', 'CCTV 정보'], ['1인가구 비율(%)', '1인가구 ÷ 총인구', '1인가구'],
    ['유동인구 수', '길을 지나다니는 사람 수 (전체 기간 평균)', '길단위인구']], columns=['항목', '뜻', '출처'])
n_apt, n_villa = (df['종류'] == '아파트').sum(), (df['종류'] == '빌라').sum()

html = f"""<meta charset="utf-8"><title>평당 금액 분석</title>
<style>body{{font-family:"Malgun Gothic",sans-serif;margin:24px auto;max-width:1150px;line-height:1.6}} h2{{margin-top:36px;border-bottom:2px solid #ddd}}
.t{{border-collapse:collapse}} .t td,.t th{{border:1px solid #ddd;padding:4px 10px;text-align:right}} .t td:first-child,.t th{{text-align:left}} .l td{{text-align:left!important}} .box{{background:#f6f8fa;padding:12px 18px;border-radius:8px}}
.row{{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}} img{{max-width:100%}} #wrap,#wrap2{{max-height:420px;overflow:auto;width:fit-content}}</style>
<h1>행정동별 데이터로 보는 평당 금액 (전체 기간, 아파트+빌라 매매)</h1>
<div class="box"><b>한눈에 보기</b><ul>
<li>거래 <b>{len(df):,}건</b> (아파트 {n_apt:,}건, 빌라 {n_villa:,}건 · {int(df['거래일(년)'].min())}~{int(df['거래일(년)'].max())}년) · 동네 <b>{df[K].nunique()}곳</b> · 평당 금액(전용면적 기준) 평균 <b>{df[Y].mean():,.0f}만원</b>, 중앙값 {df[Y].median():,.0f}만원</li>
<li>행정동 단위(동네 {len(dg)}곳)로 묶은 7개 항목 중 평당 금액과 연관도(스피어만)가 가장 큰 것: <b>{top_sp}</b>({corr[top_sp]:+.2f}) · 관계가 거의 없는 항목(0.1 미만): {none}</li>
<li>8가지 분석 방법을 종합한 관련 큰 순서(상위 3): <b>{top3}</b> · 8가지 방법 중 가장 좋았던 방법(안정성·항목 고르는 능력 기준): <b>{best}</b></li></ul></div>

<h2>1. 어떻게 분석했나</h2>
<ol><li><b>지번 → 행정동 변환</b>: 실거래 데이터의 법정동(umdNm)+지번(jibun)을 도로명주소 DB(juso.go.kr)의 지번별 행정동과 연결했어요 (<code>apt_sale_juso.csv</code>, <code>villa_sale_juso.csv</code>). 법정동 하나가 행정동 여러 개에 걸쳐도 지번마다 정확한 행정동을 알 수 있어요. 주소 DB에서 지번이 정확히 일치한 거래만 썼어요(주소 DB에 없는 지번은 분석에서 뺐어요). 지번이 깨진 건(엑셀 날짜 변환)은 고친 지번(<code>*_with_hdong.csv</code>)을 썼어요.</li>
<li><b>평당 금액(y)</b>: <code>거래금액(만원) ÷ (전용면적 ÷ 3.3058)</code>. 해제(취소) 거래와 20㎡ 미만은 뺐어요. (㎡당 금액은 이 값 ÷ 3.3058이에요)</li>
<li><b>거래일</b>: 계약 연·월·일을 하나의 날짜로 합치고, 연도에 연중 경과 비율을 더한 숫자(예: 2024.5)로 바꿨어요. 전체 기간을 쓰기 때문에 시간에 따라 오른 가격이 이 항목에 반영돼요.</li>
<li><b>행정동 단위로 묶기</b>: 거래를 <b>행정동 코드(8자리)</b>별로 묶어 동네 1곳을 1행으로 만들었어요. 평당 금액은 동네 중앙값, 층수·거래일·완공연도는 동네 평균이에요. 여기에 동네 특징(직장인구·CCTV·1인가구·유동인구)을 붙였어요. 상권분석 자료(유동·직장·거주인구, 집객시설, 점포)는 코드가 같고, CCTV는 좌표로 가까운 행정동에, 1인가구는 코드 체계가 달라 행정동 이름으로 연결했어요. 최근 새로 나뉘거나 이름이 바뀐 동(신설동·용두동→용신동, 상일제1·2동→상일동, 개포3동→일원2동)은 KIKmix 변경 이력대로 상권분석의 옛 행정동에 합쳤어요.</li></ol>
{table(desc, 't l')}

<h2>2. 연관도: 무엇이 평당 금액과 관련 있나</h2>
<p>이 분석은 <b>행정동 단위</b>예요(동네 1곳 = 1행). 평당 금액은 그 동네 거래의 중앙값이고, 항목은 위 표의 7개예요. <b>읽는 법:</b> 연관도는 -1~+1이에요. <b>+</b>는 그 항목이 클수록 평당 금액이 비싸고, <b>-</b>는 클수록 저렴해요. 절댓값 0.5 이상 강함, 0.3 이상 중간, 0.1 이상 약함, 0.1 미만 없음으로 봤어요. (원인이 아니라 같이 움직이는 정도예요)</p>
<h4>2-1. 항목별 연관도 (행정동 단위, 스피어만)</h4>{table(corr_tab, 't l')}
<h4>2-2. 한눈에 보기 (항목끼리의 연관도 포함)</h4>
<div class="row">{img_bar}{img_heat}</div>

<h4>2-3. 집객시설 종류별 연관도 (거래 1건 단위)</h4>
<p>집객시설({len(fac.columns)}종류)을 하나씩 따로 평당 금액과 비교했어요. 시설 수는 동네별 전체 기간 평균이고 빈칸은 0으로 봤어요. <b>거래 1건 기준</b>은 거래마다 그 동네의 시설 수를 붙여 비교한 값, <b>동네 기준</b>은 동네별 평당 금액 중앙값과 비교한 값, <b>아파트만·빌라만</b>은 거래 1건 기준을 종류별로 나눠 본 값이에요.</p>
<div class="row"><div>{table(fac_html, 't')}</div>{img_fac}</div>

<h4>2-4. 전월세 거래: 종류별 (거래 1건 단위)</h4>
<p>서울시 전월세가(2021~2026년 6개 파일을 합친 자료, 전체 {n_rent:,}건)를 법정동+지번 → 도로명주소 DB로 행정동에 연결했어요. 아파트·오피스텔·빌라(연립다세대)는 <b>지번이 일치한 거래</b>(97~99%)를, 단독다가구는 지번이 없어서 <b>법정동이 행정동 하나뿐인 거래</b>(약 15%)만 썼어요(행정동이 여러 개에 걸친 큰 법정동은 빠짐). 2021년 이후 계약, 임대면적 20㎡ 이상만 썼고 면적은 임대면적이에요(단독다가구는 전용면적이 아닐 수 있어요).
종류마다 건물 성격과 금액 수준이 달라서 <b>종류별로 나눠서 각각</b> 분석했어요. 보증금과 월세는 서로 바꿔 낼 수 있어서 한 값으로 합치지 않고 ① 전세의 평당 보증금, ② 월세 거래의 평당 보증금, ③ 월세 거래의 평당 월세로 나눴어요. 행정동은 이름이라 상관비 η로 봤고, 값이 아예 없는 항목(예: 단독다가구의 층수)은 표에서 뺐어요. 매매 평당 금액과 뜻이 달라서 따로 봤어요.</p>
{rent_html}

<h2>3. 여러 분석 방법으로 비교</h2>
<p>"어떤 항목이 평당 금액과 관련 있나"를 <b>8가지 방법</b>으로 따로 구해서 비교했어요(행정동 단위, 동네 {len(dd)}곳 — 직장인구·1인가구가 비어 있는 동네는 회귀·정보량·중요도에서 제외). 방법마다 보는 관점(직선, 순서, 이상치, 항목끼리 겹침 등)이 달라서, 여러 방법에서 일관되게 위에 오르는 항목이 믿을 만해요. 평당 금액을 예측하는 결과는 만들지 않았어요.</p>
{table(how, 't l')}
<h4>3-1. 방법별 결과 (숫자는 방법마다 단위가 달라서 서로 비교하지 않고, 순위로 비교해요)</h4>
<p>'거래 단위 스피어만(참고)' 열은 동네로 묶지 않고 거래 1건씩 비교한 값이라 순위 계산에는 쓰지 않았어요. 항목은 <b>평균 순위</b>(8가지 방법에서 관련 큰 순서의 평균, 1에 가까울수록 관련 큼) 순으로 정렬했어요. <b>VIF</b>는 그 항목이 다른 항목과 얼마나 겹치는지(10이 넘으면 많이 겹침)예요.</p>
<div style="overflow:auto">{table(meth_tab, 't')}</div>
<h4>3-2. 한눈에 보기</h4>
<div class="row">{img_meth}</div>
<h4>3-3. 어떤 방법이 가장 좋은가</h4>
<p>"좋은 방법"을 두 가지로 쟀어요. ① <b>안정성</b>: 동네를 무작위로 반씩 나눠 각각 분석했을 때 두 결과의 항목 순서가 얼마나 같은지(1에 가까울수록 어떤 동네로 해도 같은 답, 30번 반복). ② <b>항목 고르는 능력</b>: 이 방법이 뽑은 상위 3개 항목만으로 평당 금액을 설명하는 직선 모델을 만들고, 모델에 쓰지 않은 동네에서 설명력(R²)을 쟀어요(5분할 × 2번). 맨 아래 '7개 항목 모두'와 '무작위 3개'는 비교 기준이에요. 이 평가는 어떤 방법이 중요한 항목을 잘 고르는지 보려는 것이고, 평당 금액을 예측하려는 게 아니에요.</p>
{table(judge_tab, 't l')}
<p><b>결론: {best}가 가장 좋았어요.</b> 안정성 {judge.loc[best].iloc[0]:.2f}(8개 방법 평균 {jm.iloc[:, 0].mean():.2f}), 상위 3개 선택 설명력 R² {judge.loc[best].iloc[1]:.2f}(7개 항목 모두 쓴 경우 {judge.loc['7개 항목 모두'].iloc[1]:.2f}, 무작위 3개 {judge.loc['무작위 3개'].iloc[1]:.2f}). 방법 사이의 설명력 차이는 최대 {jm.iloc[:, 1].max() - jm.iloc[:, 1].min():.2f}로 {'작아서' if jm.iloc[:, 1].max() - jm.iloc[:, 1].min() < 0.05 else '커서'} {'한 방법만 믿기보다 3-1처럼 여러 방법에서 같은 항목이 위에 오르는지를 같이 보는 게 안전해요.' if jm.iloc[:, 1].max() - jm.iloc[:, 1].min() < 0.05 else '방법 선택이 결과에 영향을 줘요.'} 동네가 약 410곳뿐이라 이 평가도 어느 정도 흔들려요.</p>

<h2>4. 연도별 변화와 동네별 평당 금액</h2>
<div class="row">{img_year}</div>
<p>행정동 1곳을 1행으로 합친 표예요: 평당 금액(중앙값), 부동산 항목 요약(빌라 비율·평균 층수·평균 거래일·평균 완공연도), 동네 특징 10개. (같은 내용이 <code>평당금액_행정동표.csv</code>에도 저장돼요)</p>
<input id="q" placeholder="동네·자치구 검색 (예: 강남, 반포)" oninput="f()" size="30">
<div id="wrap">{dong_table.drop(columns=K).to_html(table_id='all', index=False, border=0, classes='t', float_format='{:,.0f}'.format, formatters={c: '{:,.1f}'.format for c in ['빌라 비율(%)', '평균 완공연도', '평균 층수', '평균 거래일(년)', '창업률(%)', '폐업률(%)', '업종 수', '1인가구 비율(%)']})}</div>
<script>function f(){{var q=document.getElementById('q').value;document.querySelectorAll('#all tbody tr').forEach(r=>r.style.display=r.innerText.includes(q)?'':'none')}}function f2(){{var q=document.getElementById('q2').value;document.querySelectorAll('#all2 tbody tr').forEach(r=>r.style.display=r.innerText.includes(q)?'':'none')}}</script>

<h2>5. 행정동별 연관도 (동네 하나하나)</h2>
<p>서울의 행정동마다 7개 항목(층수·거래일·완공연도·직장인구·CCTV·1인가구·유동인구)이 평당 금액과 어떤 관계인지 동네별로 보여줘요 (예: 연남동을 검색해 보세요). 항목마다 구하는 방법이 달라요.</p>
<ul><li><b>동네 안 연관도</b> (층수·거래일·완공연도, 종류): 그 동네 <b>안에서</b> 거래끼리 비교한 스피어만 상관계수예요(거래 100건 이상인 동네만, 값이 하나뿐이면 빈칸). 층수가 높은 거래가 평당 금액도 높은지, 최근 거래일수록 비싼지, 새 건물일수록 비싼지, 빌라(=1)가 아파트(=0)보다 비싼지를 봐요.</li>
<li><b>구 안 연관도</b> (직장인구·CCTV·1인가구·유동인구): 이 항목들은 한 동네 안에서는 값이 하나뿐이라 그 동네만으로는 연관도를 구할 수 없어요. 대신 <b>같은 자치구 안의 동네들끼리</b> 비교해서(예: 마포구 16개 동네) 직장인구가 많은 동네일수록 평당 금액이 높은지를 구하고, 그 구에 속한 동네에 같은 값을 보여줘요. 구마다 동네가 15개 안팎이라 값이 흔들릴 수 있어요.</li>
<li><b>백분위</b>: 서울 424곳 중 그 값의 순위예요(100에 가까울수록 많은 동네).</li></ul>
<p>서울 전체에서의 연관도는 위 2·3번 섹션이고, 아래 5-2에 자치구별 표가 있어요. (같은 내용이 <code>평당금액_행정동별_연관도.csv</code>, <code>평당금액_자치구별_연관도.csv</code>에도 저장돼요)</p>
<input id="q2" placeholder="동네·자치구 검색 (예: 연남, 마포)" oninput="f2()" size="30">
<div id="wrap2">{dong_corr.to_html(table_id='all2', index=False, border=0, classes='t', na_rep='', float_format='{:,.0f}'.format, formatters=fmt2)}</div>

<h4>5-2. 자치구별 연관도 (구 안의 동네끼리 비교, 스피어만)</h4>
<p>자치구 하나 안의 동네들(동네 수 열)끼리 비교한 값이에요. 서울 전체(2·3번)와 다른 구도 있어요.</p>
<div style="overflow:auto">{gu_corr.round(2).rename_axis('자치구').reset_index().to_html(index=False, border=0, classes='t', formatters={c: ff('{:+.2f}') for c in USE})}</div>

<h2>6. 알아둘 점</h2>
<ul><li>행정동 연결 방식(전체 거래 기준, <b>지번 일치만 분석에 사용</b>): {', '.join(f'{k} {v}%' for k, v in link.items())}. 상권분석에 행정동이 없어 제외한 거래: <b>{n_drop:,}건</b> (취소·20㎡ 미만·행정동 미연결 제외 후 {len(all_deals):,}건 중 {n_drop / len(all_deals) * 100:.1f}%)이에요.</li>
<li>행정동 단위로 묶으면 동네 수(약 420곳)만큼의 점으로 계산해서, 거래가 적은 동네의 값은 덜 믿을 만하고 아파트·빌라가 섞인 중앙값이에요(종류는 이번 분석 항목에서 뺐어요). 그래서 층수·거래일처럼 동네 평균을 낸 항목은 "아파트가 많은 동네"(층이 높고 평당 금액도 높음)나 "거래 시점이 늦은 동네"의 영향을 같이 받을 수 있어요.</li>
<li>동네 특징은 전체 기간의 평균이에요. 점포 관련 항목(점포 수·창업률·폐업률·업종 수)은 2025년 이후 데이터만 있어요.</li>
<li>아파트와 빌라는 평당 금액 수준이 달라요(종류 항목으로 구분). 전월세(아파트·빌라·단독다가구·오피스텔)는 매매와 금액의 뜻이 달라서, 매매 평당 금액 분석에 섞지 않고 평당 보증금·평당 월세로 2-4에서 따로 봤어요.</li></ul>"""
path = os.path.abspath(OUT + 'AptHypothesis.html')
open(path, 'w', encoding='utf-8').write(html)
webbrowser.open('file:///' + path.replace(os.sep, '/'))
print('결과 모음 저장:', path, '\n행정동 합친 표:', os.path.abspath(OUT + '평당금액_행정동표.csv'))
