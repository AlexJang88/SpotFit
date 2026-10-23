# 가설: 건물이 노후한 동네일수록 연령대(특히 육아 세대) 인구 분포가 다르다 (주거 건물이 육아 환경에 영향을 줄 것)
# 건물의 종류(아파트/빌라/오피스텔)와 건축년도를 기준으로 동네별 건축년도 분포(평균 건축년도, 연대별 건물 비율, 노후 20·30년 이상 비율)를 만들어 연령대와 비교함
# 노후 = 건축 후 20년 이상: 연령대 인구 자료가 바뀌는 기준연도(아래 3개 시점)마다 노후 여부를 다시 계산해서 갱신함
# 건축물대장(사용승인일)이 없어서, 매매·전월세 거래에 나온 건물의 건축년도를 사용함 (거래된 건물만 포함). 먼저 ConvertSale.py, AptHypothesis.py 를 실행해야 함 (집값은 평당금액_행정동표.csv 에서 가져옴)
import glob, os, unicodedata, platform, io, base64, webbrowser
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
DATA = os.path.join(RAW, '..', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 코드 → 상권분석의 옛 코드 (AptHypothesis.py 와 같음)
YEARS = range(2021, 2027)
MIN_BLD, MIN_T = 30, 10        # 한 동네에 건물이 이만큼은 있어야 비율을 믿고 사용 (전체 / 종류별)
KINDS = ['아파트', '빌라', '오피스텔']
AGES = {'10': '10대', '20': '20대', '30': '30대', '40': '40대', '50': '50대', '60_이상': '60대 이상'}
COHORT = {'~1989년 건물(%)': (1900, 1989), '1990년대 건물(%)': (1990, 1999), '2000년대 건물(%)': (2000, 2009), '2010년대 건물(%)': (2010, 2019), '2020년 이후 건물(%)': (2020, 2026)}      # 건축년도 구간
NEW = ['평균 건축년도'] + list(COHORT) + ['노후비율20(%)', '노후비율30(%)']        # 건축년도 지표: 평균 건축년도, 연대별 건물 비율, 노후(20년·30년 이상) 건물 비율
KEY = ['평균 건축년도', '1990년대 건물(%)', '2020년 이후 건물(%)', '노후비율20(%)']  # 종류별로 볼 대표 지표
MIX = [k + ' 건물(%)' for k in KINDS]                                              # 동네의 종류 구성: 아파트·빌라·오피스텔 건물 비율

def load(kw):           # 상권분석 csv를 이름으로 찾아 불러옴 (모든 분기)
    return pd.concat([pd.read_csv(f, encoding='cp949') for f in glob.glob(RAW + '*.csv') if kw in unicodedata.normalize('NFC', os.path.basename(f))])

# ---------- 1) 건물 목록: (지번, 종류) 1개 = 건물(단지) 1개, 종류와 건축년도는 거래 데이터에서 가져옴 ----------
J, c = DATA + '부동산(도로명주소)/', ['sggCd', 'umdNm', 'jibun', '행정동_코드', '연결방식']
read = lambda n, kind: pd.read_csv(J + f'{n}_juso.csv', low_memory=False, usecols=c + ['buildYear']).rename(columns={'buildYear': '건축년도'}).assign(종류=kind)
sale = pd.concat([read('apt_sale', '아파트'), read('villa_sale', '빌라')])                               # 매매: 파일로 종류 구분
rent = pd.read_csv(J + '서울시_전월세가_juso.csv', low_memory=False, usecols=c + ['건축년도', '건물용도'])
rent = rent[rent['건물용도'].isin(['아파트', '연립다세대', '오피스텔'])].assign(종류=lambda t: t['건물용도'].map({'아파트': '아파트', '연립다세대': '빌라', '오피스텔': '오피스텔'}))[sale.columns]   # 전월세: 건물용도로 종류 구분 (단독다가구는 지번이 없어 건물을 구분할 수 없어 뺌)
b = pd.concat([sale, rent])
b = b[b['연결방식'] == '지번 일치'].assign(건축년도=lambda t: pd.to_numeric(t['건축년도'], errors='coerce'))
b = b[b['건축년도'].between(1900, 2026)]                                           # 말이 안 되는 건축년도(998, 8404 등)는 뺌
b = b.assign(동=b['행정동_코드'].astype(int).replace(OLD))
bld = b.groupby(['sggCd', 'umdNm', 'jibun', '종류']).agg(동=('동', 'first'), 건축년도=('건축년도', 'median')).reset_index()     # 같은 지번·종류의 거래들은 한 건물로 묶음
print(f'건물(지번) {len(bld):,}곳 | 종류별:', bld['종류'].value_counts().to_dict(), f'| 거래 데이터 {len(b):,}건에서 추출')

# ---------- 2) 동네별 건축년도 분포. 기준연도마다 노후 여부(기준연도 - 건축년도 >= 20, 30)를 다시 계산해서 갱신 ----------
def dong_stats(t, y):       # t: 건물 목록(전체 또는 한 종류), y: 기준연도 → 동네별 건축년도 지표
    t = t[t['건축년도'] <= y]                              # 그 해에 이미 지어진 건물만 (이후에 지어진 건물은 제외)
    age = y - t['건축년도']                                # 기준연도 - 건축년도 = 건물 나이 (기준연도가 바뀌면 다시 계산)
    g = t.assign(건물수=1, 노후20=age >= 20, 노후30=age >= 30, **{k: t['건축년도'].between(a, z) for k, (a, z) in COHORT.items()}).groupby('동')
    d = pd.concat([g['건물수'].sum(), g['건축년도'].mean().rename('평균 건축년도'), g[list(COHORT) + ['노후20', '노후30']].mean().mul(100)], axis=1)
    return d.rename(columns={'노후20': NEW[-2], '노후30': NEW[-1]}).assign(연도=y)
panel = pd.concat([dong_stats(bld, y) for y in YEARS])                                              # 전체 종류
ptype = {kd: pd.concat([dong_stats(bld[bld['종류'] == kd], y) for y in YEARS]) for kd in KINDS}    # 종류별
mix = pd.concat([bld[bld['건축년도'] <= y].groupby('동')['종류'].value_counts(normalize=True).unstack(fill_value=0).mul(100).add_suffix(' 건물(%)').assign(연도=y) for y in YEARS])   # 동네의 종류 구성

# ---------- 3) 연령대 인구 비율 (상주인구) ----------
p = load('상주인구-행정동')
cols = {f'연령대_{k}_상주인구_수': v + '(%)' for k, v in AGES.items()}
# 상주인구 연령대 값은 분기별로 있지만 실제로는 몇 번만 바뀜(같은 값이 여러 분기 반복) → 값이 바뀐 분기를 새 자료의 기준연도로 봄
tot = p.groupby('기준_년분기_코드')[list(cols)].sum()
start = pd.Series(tot.index, index=tot.index).groupby((tot.diff().abs().sum(axis=1) > 0).cumsum()).transform('min')
p['연도'] = p['기준_년분기_코드'].map(start // 10)
qlab = lambda q: f'{q // 10}년 {q % 10}분기'
end = pd.Series(tot.index, index=tot.index).groupby(start).max()                                   # 같은 값이 이어진 마지막 분기
LABEL = {int(a // 10): f'{qlab(a)}~{qlab(z)}' for a, z in end.items()}                              # 기준연도 → '2023년 4분기~2026년 2분기'
print('연령대 자료의 기준연도(값이 바뀐 시점):', sorted(p['연도'].unique()), '→ 분기:', sorted(start.unique()))
p['합'] = p[list(cols)].sum(axis=1)
for k, v in cols.items(): p[v] = p[k] / p['합'] * 100
SHARE = list(cols.values())                                              # 연령대별 비율 (특정 연령대를 미리 고르거나 합치지 않고 모두 같은 방식으로 봄)
BAD = {2022}        # 값이 이상한 자료(2022년 4분기~2023년 3분기: 서울 전체 20대 1,460천 → 1,076천 → 1,350천 명으로 급감 후 회복)는 분석에서 뺌
print('연령대 자료별 서울 전체 인구(천 명):'); print(tot.groupby(start).first().div(1000).round(0).rename(columns=cols).to_string())
pop = p[~p['연도'].isin(BAD)].groupby(['행정동_코드', '연도'])[SHARE].mean().reset_index()
def join(d, minb=0):        # 동네 지표 + 연령대 비율을 (행정동, 기준연도)로 합치고, 건물이 minb곳 미만인 동네는 뺌
    d = d.reset_index().rename(columns={'동': '행정동_코드'}).merge(pop, on=['행정동_코드', '연도'])
    return d[d['건물수'] >= minb] if '건물수' in d else d
df = join(panel, MIN_BLD)                                                  # 동네 × 연도 (전체 종류)
dft = {kd: join(ptype[kd], MIN_T) for kd in KINDS}                         # 종류별
dmix = join(mix).merge(df[['행정동_코드', '연도']], on=['행정동_코드', '연도'])        # 종류 구성 (전체 분석에 들어간 동네만)
SNAP = sorted(df['연도'].unique()); LAST = SNAP[-1]                         # 연령대 자료의 기준연도, 가장 최근 시점
LL = f'최근 연령대 자료({LABEL[LAST]})'
print(f'동네 × 기준연도 {len(df)}행 (동네 {df["행정동_코드"].nunique()}곳) | 기준연도별 동네 수:', df.groupby('연도').size().to_dict(), '| 종류별 동네 수(최근):', {kd: int((d['연도'] == LAST).sum()) for kd, d in dft.items()})

# ---------- 4) 건축년도 지표와 연령대 비율의 연관도 (스피어만) ----------
cor = lambda t, rows=NEW: t[rows + SHARE].corr(method='spearman').loc[rows, SHARE]
last = cor(df[df['연도'] == LAST])                                          # 전체 종류, 가장 최근 자료
by_year = pd.DataFrame({LABEL[y]: cor(df[df['연도'] == y]).loc['평균 건축년도'] for y in SNAP}).T      # 평균 건축년도와 연령대의 연관도가 자료 시점마다 같은지
bytype = pd.concat({kd: cor(d[d['연도'] == LAST], KEY) for kd, d in dft.items()})                # 종류별: 그 종류 건물만으로 만든 동네 지표
bytype.index = [f'{a} · {z}' for a, z in bytype.index]
first = cor(df[df['연도'] == SNAP[0]])                                          # 첫 번째 연령대 자료로 같은 분석 (결과가 같은지 확인)
cmix = dmix[dmix['연도'] == LAST][MIX + SHARE].corr(method='spearman').loc[MIX, SHARE]            # 동네의 종류 구성(아파트·빌라·오피스텔 비율) × 연령대
print(f'\n[{LL}, 전체 종류, 건축년도 지표 × 연령대 비율 연관도]'); print(last.round(2).to_string())
print(f'\n[{LL}, 종류별 건축년도 지표 × 연령대 비율 연관도]'); print(bytype.round(2).to_string())
print(f'\n[{LL}, 동네의 종류 구성 × 연령대 비율 연관도]'); print(cmix.round(2).to_string())

# 집값(동네 평당 금액 중앙값)이 같은 동네끼리 비교했을 때도 연관이 남는지: 순위로 바꾼 뒤 집값의 영향을 뺀 나머지끼리 상관 (편 스피어만 상관)
price = pd.read_csv(OUT + '평당금액_행정동표.csv').set_index('행정동_코드')['평당 금액 중앙값(만원)']
dl = df[df['연도'] == LAST].assign(집값=lambda t: t['행정동_코드'].map(price)).dropna(subset=['집값'])
rk = dl[KEY + SHARE + ['집값']].rank()
res = lambda c: rk[c] - np.polyval(np.polyfit(rk['집값'], rk[c], 1), rk['집값'])
ctl = pd.DataFrame({sh: [res(a).corr(res(sh)) for a in KEY] for sh in SHARE}, index=KEY)
# 노후(20년 이상) 건물 비율이 낮은/높은 동네의 평균 연령대 비율
q = dl['노후비율20(%)']; grp = pd.cut(q, [-1, q.quantile(.25), q.quantile(.75), 101], labels=['노후 낮은 25%', '중간 50%', '노후 높은 25%'])
cmp = dl.groupby(grp, observed=True)[SHARE].mean().T
cmp['차이(높은-낮은)'] = cmp['노후 높은 25%'] - cmp['노후 낮은 25%']
cmp = cmp.rename_axis('연령대').reset_index()
rc = lambda n: 1.96 / np.sqrt(n - 1)                                        # 이 크기(절댓값)보다 작은 연관도는 우연과 구분되지 않음 (95% 수준의 근사)
n_all, n_off = int((df['연도'] == LAST).sum()), int((dft['오피스텔']['연도'] == LAST).sum())
print(f'\n[집값을 같게 맞춘 뒤의 연관도, 동네 {len(dl)}곳]'); print(ctl.round(2).to_string())
print('\n[노후 높은/낮은 동네의 평균 연령대 비율(%)]'); print(cmp.round(1).to_string(index=False))
print(f'\n우연과 구분되는 최소 연관도(절댓값): 전체 동네 {n_all}곳 {rc(n_all):.2f}, 오피스텔 {n_off}곳 {rc(n_off):.2f}')

# ---------- 5) 결과 한 페이지(HTML) ----------
def img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
def heat(d, title, size):
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(d.astype(float), cmap='RdBu_r', center=0, vmin=-.7, vmax=.7, annot=True, fmt='.2f', ax=ax)
    ax.set_title(title); plt.xticks(rotation=30, ha='right')
    return img(fig)
tt = bld[bld['건축년도'] <= LAST]                                            # 서울 전체의 종류별 건축년도 분포 (최근 기준연도까지 지어진 건물)
seoul = tt.groupby('종류')['건축년도'].agg(건물수='size', 평균건축년도='mean').join(pd.DataFrame({k: tt.groupby('종류')['건축년도'].apply(lambda v: v.between(a, z).mean() * 100) for k, (a, z) in COHORT.items()}))
seoul['노후(20년 이상)(%)'] = tt.groupby('종류')['건축년도'].apply(lambda v: (v <= LAST - 20).mean() * 100)
seoul = seoul.reindex(KINDS).reset_index()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, kd in zip(axs, KINDS):                                              # 종류별 산점도: 평균 건축년도 vs 10대 비율
    d = dft[kd][dft[kd]['연도'] == LAST]; k = np.polyfit(d['평균 건축년도'], d['10대(%)'], 1); xs = np.array([d['평균 건축년도'].min(), d['평균 건축년도'].max()])
    ax.scatter(d['평균 건축년도'], d['10대(%)'], s=14, alpha=.6); ax.plot(xs, k[0] * xs + k[1], 'r--'); ax.set(xlabel='동네 건물의 평균 건축년도', title=f'{kd} (동네 {len(d)}곳)')
axs[0].set_ylabel('10대 인구 비율 (%)')
imgs = [heat(last, f'{LL}: 전체 종류, 건축년도 지표 × 연령대 비율', (9, 4.2)), heat(bytype, f'{LL}: 종류별 건축년도 지표 × 연령대 비율', (9, 5.2)),
        heat(cmix, f'{LL}: 동네의 종류 구성 × 연령대 비율', (9, 2.6)), img(fig), heat(by_year, '평균 건축년도 × 연령대 비율, 자료 시점별 (전체 종류)', (9, 2.8)), heat(ctl, f'{LL}: 집값을 같게 맞춘 뒤의 연관도 (전체 종류)', (9, 2.6)), heat(first, f'첫 번째 연령대 자료({LABEL[SNAP[0]]}): 전체 종류, 건축년도 지표 × 연령대 비율', (9, 4.2))]
tab = lambda t, f='{:,.1f}': t.to_html(index=False, border=0, classes='t', float_format=f.format)
html = f"""<meta charset="utf-8"><title>건물 노후도와 연령대</title>
<style>body{{font-family:"Malgun Gothic",sans-serif;margin:24px auto;max-width:1000px;line-height:1.6}} h2{{margin-top:32px;border-bottom:2px solid #ddd}} .t{{border-collapse:collapse}} .t td,.t th{{border:1px solid #ddd;padding:4px 10px;text-align:right}} .box{{background:#f6f8fa;padding:12px 18px;border-radius:8px}}</style>
<h1>건물 종류·건축년도와 연령대 인구 분포</h1>
<div class="box"><b>가설</b>: 건물이 노후한 동네일수록 연령대 분포가 다르다. 주거 건물이 육아 환경에 영향을 줄 것이다.<br>
<b>한눈에 보기 ({LL})</b>: 노후 건물 비율(20년 이상)과 연령대별 인구 비율의 연관도 — {', '.join(f'{k.replace('(%)', '')} {v:+.2f}' for k, v in last.loc['노후비율20(%)'].items())}. 우연과 구분되는 최소 크기는 약 {rc(n_all):.2f}라서 이보다 작은 값은 관계가 없다고 봐요.</div>
<h2>1. 어떻게 분석했나</h2>
<ul><li><b>건물 종류</b>: 아파트, 빌라(연립·다세대), 오피스텔로 나눴어요. 매매는 파일로, 전월세는 건물용도로 구분했어요. 단독·다가구는 지번이 없어 건물을 구분할 수 없어 뺐어요.</li>
<li><b>건물 목록</b>: 건축물대장(사용승인일)이 없어서 매매·전월세 거래에 나온 건물의 건축년도를 썼어요(지번+종류 1개 = 건물 1개, 건물 {len(bld):,}곳: {', '.join(f'{k} {v:,}' for k, v in bld['종류'].value_counts().items())}).</li>
<li><b>건축년도 지표</b>(동네별): 건물의 <b>평균 건축년도</b>, 건축 연대별 건물 비율(~1989 / 1990년대 / 2000년대 / 2010년대 / 2020년 이후), 노후(20년·30년 이상) 건물 비율. 전체는 건물 {MIN_BLD}곳 이상, 종류별은 {MIN_T}곳 이상인 동네만 썼어요. 달력 연도(2021, 2026)를 비교하는 대신 건물의 건축년도 자체를 기준으로 봤어요.</li>
<li><b>노후 판단</b>: 연령대 자료가 바뀐 시점의 연도({', '.join(map(str, SNAP))})를 기준연도로 두고, 기준연도마다 <code>기준연도 - 건축년도</code>로 건물 나이를 다시 계산해서 노후 여부를 갱신했어요. 그 해에 아직 지어지지 않은 건물은 뺐어요.</li>
<li><b>연령대 자료</b>: 상주인구 연령대는 분기별로 있지만 값이 같은 분기가 이어져서 <b>실제로는 3번만 갱신된 자료</b>예요(2021년 1분기, 2022년 4분기, 2023년 4분기). 이 중 두 번째(2022년 4분기~2023년 3분기)는 서울 전체 20대가 1,460천 명→1,076천 명→1,350천 명으로 급감했다 회복해서 <b>값이 이상하다고 보고 분석에서 뺐어요</b>. 남은 두 자료(첫 번째: {LABEL[SNAP[0]]}, 최근: {LABEL[SNAP[-1]]})로 분석하고, 값이 처음 나온 분기의 연도를 기준연도로 썼어요(정확한 조사 기준일은 자료에 없어서 가정이에요).</li>
<li><b>연령대 비율</b>: 상주인구의 연령대별 인구 ÷ 전체. <b>특정 연령대를 미리 고르지 않고 6개 연령대를 모두 같은 방식으로</b> 봤어요. 육아 환경과 가장 직접 관련된 영유아(0~9세) 자료가 없어서, 자녀 세대에 가장 가까운 <b>10대 비율</b>을 대표로 봤어요(10~19세 자녀가 있는 가구의 부모는 주로 40·50대예요). 연관도는 스피어만 상관계수(-1~+1)예요. 평균 건축년도는 클수록 새 건물이 많다는 뜻이라, 노후 비율과는 부호가 반대로 나와요.</li></ul>
<h2>2. 결과</h2>
<h4>2-1. 서울의 건물 종류별 건축년도 분포 ({LAST}년 기준, %)</h4>{tab(seoul)}
<h4>2-2. 전체 종류: 건축년도 지표 × 연령대 비율</h4>{imgs[0]}
<p>이 크기(절댓값)보다 작은 연관도는 우연과 구분되지 않아요: 전체 동네 {n_all}곳 기준 약 {rc(n_all):.2f}, 오피스텔(동네 {n_off}곳) 기준 약 {rc(n_off):.2f}. 표에 값이 많아서 우연히 큰 값이 하나쯤 나올 수도 있으니, 한 칸보다 여러 칸이 같은 방향인지를 봐야 해요.</p>
<h4>2-3. 노후 건물 비율이 낮은 동네 vs 높은 동네의 연령대 비율 (평균, %)</h4><p>노후(20년 이상) 건물 비율을 기준으로 동네를 아래 25%, 중간 50%, 위 25%로 나눠 비교했어요. 연관도(상관계수)보다 직접 읽기 쉬워요.</p>{tab(cmp)}
<h4>2-4. 종류별: 건축년도 지표 × 연령대 비율</h4><p>아파트·빌라·오피스텔 건물만으로 동네 지표를 따로 만들어 비교했어요. 종류마다 입주하는 연령대가 달라서(예: 오피스텔은 1인가구·젊은 층) 결과가 다를 수 있어요.</p>{imgs[1]}
<h4>2-5. 동네의 종류 구성(아파트·빌라·오피스텔 건물 비율) × 연령대 비율</h4>{imgs[2]}
<h4>2-6. 종류별 산점도: 평균 건축년도 vs 10대 비율</h4><p>10대 비율은 '자녀 세대'에 가장 가까운 자료예요(영유아 0~9세는 자료에 없어요). 육아 환경이 영향을 준다면 노후 동네에서 10대 비율이 낮을 것이라는 가설의 직접적인 예측이라서 골랐어요.</p>{imgs[3]}
<h4>2-7. 집값이 같은 동네끼리 비교해도 연관이 남는가</h4><p>동네의 평당 금액(매매 중앙값)이 같다고 보고 그 영향을 뺀 뒤의 연관도예요(순위로 바꾼 편상관). 노후도와 연령대의 관계가 집값 때문에 생긴 것인지 확인하려는 거예요. 동네 {len(dl)}곳.</p>{imgs[5]}
<h4>2-8. 첫 번째 연령대 자료로 같은 분석을 하면 결과가 같은가</h4>{imgs[6]}<p>두 자료에서 평균 건축년도와 연령대의 연관도를 나란히 보면:</p>{imgs[4]}
<h2>3. 알아둘 점</h2>
<ul><li>연관도는 같이 움직이는 정도이고 원인이 아니에요. 동네의 소득·집값·주택 종류도 연령대에 영향을 주고, 건물 종류 구성이 건축년도와 섞여 있어요.</li>
<li>건물 목록이 <b>거래된 건물만</b> 포함해서, 거래가 적은 건물(예: 오래 거주하는 노후 주택)이 덜 잡혀요. 동네의 실제 건물 구성과 다를 수 있어요.</li>
<li><b>상주인구 연령대는 그 동네에 사는 사람 전체</b>의 연령대예요. 그래서 이 분석은 "건물에 사는 사람의 나이"가 아니라 <b>"동네의 건물 연식과 동네 주민 연령대의 관계"</b>예요(건물별 입주자 연령 자료는 없어요). 건물 목록에는 거래된 건물만 있어서 단독주택 거주자 등은 반영되지 않고, 동네 평균의 관계가 개인에게 그대로 성립하는 것도 아니에요.</li>
<li>'종류 구성'(2-5)은 <b>건물(지번) 수</b> 기준이라, 세대가 수백 개인 아파트 단지도 건물 1개로 세어요. 그래서 빌라가 건물 수로는 대부분(약 86%)을 차지하고, 실제 거주 가구 기준의 종류 구성과는 달라요. 가구 수 기준으로 바꾸려고 했지만, 상주인구 자료의 아파트 가구 수는 전부 0이고 상권분석 아파트 세대 수는 분기마다 크게 달라져서(서울 합계 245만→104만) 쓰지 않았어요.</li>
<li>연령대 자료는 사실상 두 시점(첫 번째와 최근)뿐이라 "시점마다 같은지"는 참고용이에요. 이상한 값이 있는 중간 자료는 뺐어요.</li>
<li>'최근 연령대 자료'는 2023년 4분기부터 2026년 2분기까지 같은 값이 이어진 자료예요. 조사 기준일이 자료에 없어서 값이 처음 나온 분기의 연도(2023년)를 기준연도로 가정했고, 노후 판단(20년 이상)도 이 연도 기준으로 계산했어요.</li>
<li>'육아 환경'을 직접 재는 변수(어린이집, 학교, 놀이터 등)는 넣지 않았고 연령대 인구로 간접 확인했어요. 종류별 동네 수가 달라서(건물이 적은 종류는 동네가 줄어요) 종류끼리 값의 크기를 곧바로 비교하기는 어려워요.</li></ul>"""
os.makedirs(OUT, exist_ok=True)
df.round(2).to_csv(OUT + '노후건물_연령대_동네연도.csv', index=False, encoding='utf-8-sig')
bld.assign(sggCd=bld['sggCd'].astype(int)).rename(columns={'sggCd': '구코드', 'umdNm': '법정동', 'jibun': '지번', '동': '행정동_코드'}).to_csv(OUT + '노후건물_건물목록.csv', index=False, encoding='utf-8-sig')      # 건물(지번+종류)별 행정동·건축년도
path = os.path.abspath(OUT + 'AgingHypothesis.html')
with open(path, 'w', encoding='utf-8') as f: f.write(html)
webbrowser.open('file:///' + path.replace(os.sep, '/'))
print('\n결과 모음 저장:', path)
