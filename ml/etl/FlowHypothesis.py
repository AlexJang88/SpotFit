# H1. 업종 비율에 따라 유동인구가 다를 것이다
import glob, os, unicodedata, platform
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import stats
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
YQ = [20251, 20252, 20253, 20254, 20261, 20262]     # 점포 데이터가 있는 분기

def load(kw):
    files = [f for f in glob.glob(DATA_DIR + '*.csv') if kw in unicodedata.normalize('NFC', os.path.basename(f))]
    df = pd.concat([pd.read_csv(f, encoding='cp949') for f in files])
    return df[df['기준_년분기_코드'].isin(YQ)] if '기준_년분기_코드' in df else df

def h1(level):
    key = f'{level}_코드'

    # 1) 데이터 불러오기
    flow = load(f'길단위인구-{level}').groupby(key)['총_유동인구_수'].mean()       # 지역별 평균 유동인구
    store = load(f'점포-{level}').drop_duplicates(['기준_년분기_코드', key, '서비스_업종_코드'])

    # 2) 업종 비율 = 그 지역 점포 중 해당 업종이 차지하는 비율
    cnt = store.groupby([key, '서비스_업종_코드_명'])['전체_점포_수'].mean().unstack(fill_value=0)
    cnt = cnt[cnt.sum(axis=1) >= 5]                                                # 점포 5개 미만 지역 제외
    ratio = cnt.div(cnt.sum(axis=1), axis=0)
    ratio = ratio[ratio.index.isin(flow.index)]
    flow = flow[ratio.index]

    # 3) 업종별 비율과 유동인구의 상관계수 (점포 많은 업종 30개)
    top = cnt.sum().nlargest(30).index
    corr = ratio[top].corrwith(flow, method='spearman').sort_values()
    print(f'\n[{level}] 유동인구와 상관 높은 업종\n', corr.tail(5).round(2))
    print(f'[{level}] 유동인구와 상관 낮은 업종\n', corr.head(5).round(2))

    plt.figure(figsize=(7, 9))
    corr.plot.barh(color=['tomato' if c > 0 else 'steelblue' for c in corr])
    plt.title(f'[{level}] 업종 비율 vs 유동인구 상관계수'); plt.tight_layout(); plt.show()

    # 4) 주력 업종(외식/서비스/소매)별 유동인구 비교
    major = store.drop_duplicates('서비스_업종_코드_명').set_index('서비스_업종_코드_명')['서비스_업종_코드']
    major = major.str[:3].map({'CS1': '외식업', 'CS2': '서비스업', 'CS3': '소매업'})
    main = cnt.T.groupby(major).sum().T.idxmax(axis=1)                            # 점포가 가장 많은 대분류
    df = pd.DataFrame({'유동인구': flow, '주력업종': main})

    print(f'\n[{level}] 주력업종별 유동인구 중앙값\n', df.groupby('주력업종')['유동인구'].agg(['count', 'median']).round(0))
    p = stats.kruskal(*[g['유동인구'] for _, g in df.groupby('주력업종')]).pvalue
    print(f'[{level}] 주력업종별로 유동인구가 다른가? p = {p:.4f} →', '다르다' if p < 0.05 else '차이 없다')

    sns.boxplot(data=df, x='주력업종', y='유동인구', log_scale=True)
    plt.title(f'[{level}] 주력업종별 유동인구 (p={p:.4f})'); plt.tight_layout(); plt.show()

# H2. 점포 수가 많으면 유동인구도 많아질 것이다
def h2(level):
    key = f'{level}_코드'

    # 1) 지역별 평균 유동인구, 평균 점포 수, 면적
    flow = load(f'길단위인구-{level}').groupby(key)['총_유동인구_수'].mean()
    store = load(f'점포-{level}').drop_duplicates(['기준_년분기_코드', key, '서비스_업종_코드'])
    store = store.groupby(['기준_년분기_코드', key])['전체_점포_수'].sum().groupby(key).mean()
    area = load(f'영역-{level}').drop_duplicates(key).set_index(key)['영역_면적']
    df = pd.DataFrame({'유동인구': flow, '점포수': store, '면적': area}).dropna()
    df = df[(df['유동인구'] > 0) & (df['점포수'] > 0)]

    # 2) 상관계수: 그냥 비교 / 면적당으로 비교 (큰 지역이라 둘 다 많은 효과 제거)
    r1, p1 = stats.spearmanr(df['점포수'], df['유동인구'])
    r2, p2 = stats.spearmanr(df['점포수'] / df['면적'], df['유동인구'] / df['면적'])
    print(f'\n[{level}] 점포수 vs 유동인구       상관계수 {r1:.2f} (p={p1:.4f})')
    print(f'[{level}] 면적당 점포수 vs 면적당 유동인구 상관계수 {r2:.2f} (p={p2:.4f})')

    # 3) 점포 수로 5개 그룹을 나눠서 그룹별 유동인구 중앙값 비교
    df['점포수 그룹'] = pd.qcut(df['점포수'], 5, labels=['1(적음)', '2', '3', '4', '5(많음)'])
    print(f'[{level}] 점포수 그룹별 유동인구 중앙값\n', df.groupby('점포수 그룹', observed=True)['유동인구'].median().round(0))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(data=df, x='점포수', y='유동인구', s=10, alpha=.4, ax=ax[0])
    ax[0].set(xscale='log', yscale='log', title=f'[{level}] 점포수 vs 유동인구 (상관 {r1:.2f})')
    sns.barplot(data=df, x='점포수 그룹', y='유동인구', estimator='median', errorbar=None, ax=ax[1])
    ax[1].set(title=f'[{level}] 점포수 그룹별 유동인구 중앙값')
    plt.tight_layout(); plt.show()

# H3. 집객시설 수가 많으면 유동인구도 많아질 것이다
def h3(level):
    key = f'{level}_코드'

    # 1) 지역별 평균 유동인구, 집객시설 수, 면적 (집객시설 파일에 없는 지역은 시설 0개)
    flow = load(f'길단위인구-{level}').groupby(key)['총_유동인구_수'].mean()
    fac = load(f'집객시설-{level}').groupby(key).mean(numeric_only=True).drop(columns='기준_년분기_코드')
    area = load(f'영역-{level}').drop_duplicates(key).set_index(key)['영역_면적']
    df = pd.DataFrame({'유동인구': flow, '면적': area}).dropna()
    df = df[df['유동인구'] > 0].join(fac).fillna(0)
    df = df.rename(columns={'집객시설_수': '집객시설수'})

    # 2) 상관계수: 그냥 비교 / 면적당으로 비교
    r1, p1 = stats.spearmanr(df['집객시설수'], df['유동인구'])
    r2, p2 = stats.spearmanr(df['집객시설수'] / df['면적'], df['유동인구'] / df['면적'])
    print(f'\n[{level}] 집객시설수 vs 유동인구       상관계수 {r1:.2f} (p={p1:.4f})')
    print(f'[{level}] 면적당 집객시설수 vs 면적당 유동인구 상관계수 {r2:.2f} (p={p2:.4f})')

    # 3) 집객시설 수로 5개 그룹 (시설 0개인 곳이 많아 같은 값이 많으므로 순위로 나눔)
    df['시설수 그룹'] = pd.qcut(df['집객시설수'].rank(method='first'), 5, labels=['1(적음)', '2', '3', '4', '5(많음)'])
    print(f'[{level}] 시설수 그룹별 유동인구 중앙값\n', df.groupby('시설수 그룹', observed=True)['유동인구'].median().round(0))

    # 4) 어떤 종류의 시설이 유동인구와 관련 있나 (지하철역, 버스정거장, 학교 ...)
    kinds = [c for c in fac.columns if c.endswith('_수') and c != '집객시설_수']
    corr = df[kinds].corrwith(df['유동인구'], method='spearman').dropna().sort_values()
    corr.index = corr.index.str.replace('_수', '')
    print(f'[{level}] 유동인구와 상관 높은 시설\n', corr.tail(5).round(2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.scatterplot(data=df, x=df['집객시설수'] + 1, y='유동인구', s=10, alpha=.4, ax=ax[0])
    ax[0].set(xscale='log', yscale='log', xlabel='집객시설수 + 1', title=f'[{level}] 집객시설수 vs 유동인구 (상관 {r1:.2f})')
    sns.barplot(data=df, x='시설수 그룹', y='유동인구', estimator='median', errorbar=None, ax=ax[1])
    ax[1].set(title=f'[{level}] 시설수 그룹별 유동인구 중앙값')
    corr.plot.barh(ax=ax[2], color=['tomato' if c > 0 else 'steelblue' for c in corr])
    ax[2].set(title=f'[{level}] 시설 종류별 유동인구 상관계수')
    plt.tight_layout(); plt.show()

# H4. 상주인구가 많아지면 유동인구도 많아질 것이다
def h4(level):
    key = f'{level}_코드'

    # 1) 지역별 평균 유동인구, 상주인구, 면적 (상주인구 파일에 없는 지역은 0명)
    flow = load(f'길단위인구-{level}').groupby(key)['총_유동인구_수'].mean()
    pop = load(f'상주인구-{level}').groupby(key).mean(numeric_only=True).drop(columns='기준_년분기_코드')
    area = load(f'영역-{level}').drop_duplicates(key).set_index(key)['영역_면적']
    df = pd.DataFrame({'유동인구': flow, '면적': area}).dropna()
    df = df[df['유동인구'] > 0].join(pop).fillna(0)
    df = df.rename(columns={'총_상주인구_수': '상주인구'})

    # 2) 상관계수: 그냥 비교 / 면적당으로 비교
    r1, p1 = stats.spearmanr(df['상주인구'], df['유동인구'])
    r2, p2 = stats.spearmanr(df['상주인구'] / df['면적'], df['유동인구'] / df['면적'])
    print(f'\n[{level}] 상주인구 vs 유동인구       상관계수 {r1:.2f} (p={p1:.4f})')
    print(f'[{level}] 면적당 상주인구 vs 면적당 유동인구 상관계수 {r2:.2f} (p={p2:.4f})')

    # 3) 상주인구로 5개 그룹 (0명인 곳이 있어 순위로 나눔)
    df['상주인구 그룹'] = pd.qcut(df['상주인구'].rank(method='first'), 5, labels=['1(적음)', '2', '3', '4', '5(많음)'])
    print(f'[{level}] 상주인구 그룹별 유동인구 중앙값\n', df.groupby('상주인구 그룹', observed=True)['유동인구'].median().round(0))

    # 4) 어떤 연령대 상주인구가 유동인구와 관련 있나 (연령대별 비율 기준)
    ages = [c for c in pop.columns if c.startswith('연령대_')]
    age_ratio = df[ages].div(df['상주인구'], axis=0)
    corr = age_ratio.corrwith(df['유동인구'], method='spearman').dropna().sort_values()
    corr.index = [c.split('_')[1] + ('대 이상' if '이상' in c else '대') for c in corr.index]   # 연령대_20_상주인구_수 → 20대
    print(f'[{level}] 연령대 비율 vs 유동인구 상관계수\n', corr.round(2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.scatterplot(data=df, x=df['상주인구'] + 1, y='유동인구', s=10, alpha=.4, ax=ax[0])
    ax[0].set(xscale='log', yscale='log', xlabel='상주인구 + 1', title=f'[{level}] 상주인구 vs 유동인구 (상관 {r1:.2f})')
    sns.barplot(data=df, x='상주인구 그룹', y='유동인구', estimator='median', errorbar=None, ax=ax[1])
    ax[1].set(title=f'[{level}] 상주인구 그룹별 유동인구 중앙값')
    corr.plot.barh(ax=ax[2], color=['tomato' if c > 0 else 'steelblue' for c in corr])
    ax[2].set(title=f'[{level}] 상주인구 연령대 비율 vs 유동인구 상관계수')
    plt.tight_layout(); plt.show()

# H5. 폐업률이 낮아지면 유동인구는 낮아질 것이다 (= 폐업률과 유동인구가 같은 방향으로 움직인다)
def h5(level):
    key = f'{level}_코드'
    yq = '기준_년분기_코드'

    # 1) 분기별·지역별 폐업률 = 폐업 점포 수 / 전체 점포 수 (점포 5개 미만은 비율이 튀어서 제외)
    flow = load(f'길단위인구-{level}').groupby([yq, key])['총_유동인구_수'].sum()
    store = load(f'점포-{level}').drop_duplicates([yq, key, '서비스_업종_코드'])
    store = store.groupby([yq, key])[['전체_점포_수', '폐업_점포_수']].sum()
    df = store.join(flow.rename('유동인구'), how='inner').reset_index()
    df = df[(df['전체_점포_수'] >= 5) & (df['유동인구'] > 0)]
    df['폐업률'] = df['폐업_점포_수'] / df['전체_점포_수'] * 100

    # 2) 지역끼리 비교: 평균 폐업률이 높은 곳이 유동인구도 많은가
    avg = df.groupby(key)[['폐업률', '유동인구']].mean()
    r1, p1 = stats.spearmanr(avg['폐업률'], avg['유동인구'])
    print(f'\n[{level}] 평균 폐업률 vs 평균 유동인구 상관계수 {r1:.2f} (p={p1:.4f})')

    # 3) 폐업률로 5개 그룹 (폐업 0%인 곳이 많아 순위로 나눔)
    avg['폐업률 그룹'] = pd.qcut(avg['폐업률'].rank(method='first'), 5, labels=['1(낮음)', '2', '3', '4', '5(높음)'])
    print(f'[{level}] 폐업률 그룹별 유동인구 중앙값\n', avg.groupby('폐업률 그룹', observed=True)['유동인구'].median().round(0))

    # 4) 시간에 따른 변화: 작년 같은 분기보다 폐업률이 낮아진 곳은 유동인구도 줄었나
    wide = df.pivot(index=key, columns=yq, values=['폐업률', '유동인구'])
    chg = pd.concat([pd.DataFrame({
        '폐업률 변화': wide['폐업률'][b] - wide['폐업률'][a],                         # %p
        '유동인구 변화율': (wide['유동인구'][b] / wide['유동인구'][a] - 1) * 100,      # %
    }) for a, b in [(20251, 20261), (20252, 20262)]]).dropna()
    r2, p2 = stats.spearmanr(chg['폐업률 변화'], chg['유동인구 변화율'])
    chg['폐업률'] = pd.cut(chg['폐업률 변화'], [-1000, -0.001, 0.001, 1000], labels=['낮아짐', '그대로', '높아짐'])
    print(f'[{level}] 폐업률 변화 vs 유동인구 변화 상관계수 {r2:.2f} (p={p2:.4f})')
    print(f'[{level}] 폐업률 변화별 유동인구 변화율(%) 중앙값\n',
          chg.groupby('폐업률', observed=True)['유동인구 변화율'].agg(['count', 'median']).round(2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.scatterplot(data=avg, x='폐업률', y='유동인구', s=10, alpha=.4, ax=ax[0])
    ax[0].set(yscale='log', xlabel='평균 폐업률 (%)', title=f'[{level}] 폐업률 vs 유동인구 (상관 {r1:.2f})')
    sns.barplot(data=avg, x='폐업률 그룹', y='유동인구', estimator='median', errorbar=None, ax=ax[1])
    ax[1].set(title=f'[{level}] 폐업률 그룹별 유동인구 중앙값')
    sns.barplot(data=chg, x='폐업률', y='유동인구 변화율', estimator='median', errorbar=None, ax=ax[2])
    ax[2].set(ylabel='유동인구 변화율 중앙값 (%)', title=f'[{level}] 작년 대비 폐업률 변화별 유동인구 변화 (상관 {r2:.2f})')
    plt.tight_layout(); plt.show()

# H6. 창업률이 높아지면 유동인구도 많아질 것이다 (창업률 = 개업 점포 수 / 전체 점포 수)
def h6(level):
    key = f'{level}_코드'
    yq = '기준_년분기_코드'

    # 1) 분기별·지역별 창업률 (점포 5개 미만은 비율이 튀어서 제외)
    flow = load(f'길단위인구-{level}').groupby([yq, key])['총_유동인구_수'].sum()
    store = load(f'점포-{level}').drop_duplicates([yq, key, '서비스_업종_코드'])
    store = store.groupby([yq, key])[['전체_점포_수', '개업_점포_수']].sum()
    df = store.join(flow.rename('유동인구'), how='inner').reset_index()
    df = df[(df['전체_점포_수'] >= 5) & (df['유동인구'] > 0)]
    df['창업률'] = df['개업_점포_수'] / df['전체_점포_수'] * 100

    # 2) 지역끼리 비교: 평균 창업률이 높은 곳이 유동인구도 많은가
    avg = df.groupby(key)[['창업률', '유동인구']].mean()
    r1, p1 = stats.spearmanr(avg['창업률'], avg['유동인구'])
    print(f'\n[{level}] 평균 창업률 vs 평균 유동인구 상관계수 {r1:.2f} (p={p1:.4f})')

    # 3) 창업률로 5개 그룹 (창업 0%인 곳이 많아 순위로 나눔)
    avg['창업률 그룹'] = pd.qcut(avg['창업률'].rank(method='first'), 5, labels=['1(낮음)', '2', '3', '4', '5(높음)'])
    print(f'[{level}] 창업률 그룹별 유동인구 중앙값\n', avg.groupby('창업률 그룹', observed=True)['유동인구'].median().round(0))

    # 4) 시간에 따른 변화: 작년 같은 분기보다 창업률이 높아진 곳은 유동인구도 늘었나
    wide = df.pivot(index=key, columns=yq, values=['창업률', '유동인구'])
    chg = pd.concat([pd.DataFrame({
        '창업률 변화': wide['창업률'][b] - wide['창업률'][a],                         # %p
        '유동인구 변화율': (wide['유동인구'][b] / wide['유동인구'][a] - 1) * 100,      # %
    }) for a, b in [(20251, 20261), (20252, 20262)]]).dropna()
    r2, p2 = stats.spearmanr(chg['창업률 변화'], chg['유동인구 변화율'])
    chg['창업률'] = pd.cut(chg['창업률 변화'], [-1000, -0.001, 0.001, 1000], labels=['낮아짐', '그대로', '높아짐'])
    print(f'[{level}] 창업률 변화 vs 유동인구 변화 상관계수 {r2:.2f} (p={p2:.4f})')
    print(f'[{level}] 창업률 변화별 유동인구 변화율(%) 중앙값\n',
          chg.groupby('창업률', observed=True)['유동인구 변화율'].agg(['count', 'median']).round(2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.scatterplot(data=avg, x='창업률', y='유동인구', s=10, alpha=.4, ax=ax[0])
    ax[0].set(yscale='log', xlabel='평균 창업률 (%)', title=f'[{level}] 창업률 vs 유동인구 (상관 {r1:.2f})')
    sns.barplot(data=avg, x='창업률 그룹', y='유동인구', estimator='median', errorbar=None, ax=ax[1])
    ax[1].set(title=f'[{level}] 창업률 그룹별 유동인구 중앙값')
    sns.barplot(data=chg, x='창업률', y='유동인구 변화율', estimator='median', errorbar=None, ax=ax[2])
    ax[2].set(ylabel='유동인구 변화율 중앙값 (%)', title=f'[{level}] 작년 대비 창업률 변화별 유동인구 변화 (상관 {r2:.2f})')
    plt.tight_layout(); plt.show()

for lv in ['상권', '행정동']:
    h1(lv)
    h2(lv)
    h3(lv)
    h4(lv)
    h5(lv)
    h6(lv)