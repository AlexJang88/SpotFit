# 연령대 가설 (상권 매출의 연령대 구성이 무엇에 따라 달라지는가)
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

# 매출 건수 연령대 (10대는 카드 매출 비중이 1% 정도로 작아서 해석 주의)
SALES = {'연령대_10_매출_건수': '10대', '연령대_20_매출_건수': '20대', '연령대_30_매출_건수': '30대',
         '연령대_40_매출_건수': '40대', '연령대_50_매출_건수': '50대', '연령대_60_이상_매출_건수': '60대+'}
FACS = ['관공서', '은행', '종합병원', '일반_병원', '약국', '유치원', '초등학교', '중학교', '고등학교', '대학교',
        '백화점', '슈퍼마켓', '극장', '숙박_시설', '공항', '철도_역', '버스_터미널', '지하철_역', '버스_정거장']

# 지역별 매출 연령대 비율 (H1, H2 공통) (상권은 2025년이 두 파일에 겹쳐서 중복 제거)
def sales_age(level):
    key = f'{level}_코드'
    sales = load(f'추정매출-{level}').drop_duplicates(['기준_년분기_코드', key, '서비스_업종_코드'])
    sales = sales.groupby(key)[list(SALES)].sum()
    sales = sales.rename(columns=SALES)
    sales = sales[sales.sum(axis=1) >= 100]               # 매출 100건 미만 지역 제외
    return sales.div(sales.sum(axis=1), axis=0)

# H1. 집객시설 비율에 따라 매출 연령대가 나뉠 것이다
def h1(level):
    key = f'{level}_코드'

    # 1) 지역별 매출 연령대 비율
    age = sales_age(level)

    # 2) 지역별 집객시설 비율 (빈칸 = 시설 없음 → 0)
    fac = load(f'집객시설-{level}')
    fac = fac.groupby(key)[[f + '_수' for f in FACS]].mean().fillna(0)
    fac.columns = FACS
    fac = fac[fac.sum(axis=1) >= 5]                       # 시설 5개 미만 지역 제외
    fac = fac.div(fac.sum(axis=1), axis=0)

    # 3) 두 데이터에 모두 있는 지역만, 30개 지역 미만에만 있는 시설(공항 등)은 제외
    both = fac.index.intersection(age.index)
    fac = fac.loc[both]
    age = age.loc[both]
    for f in FACS:
        if (fac[f] > 0).sum() < 30:
            fac = fac.drop(columns=f)
    print(f'\n[{level}] 분석 지역 수: {len(both)}')

    # 4) 시설 비율 x 매출 연령대 비율 상관계수 (행: 시설, 열: 연령대)
    corr = pd.DataFrame(index=fac.columns, columns=age.columns, dtype=float)
    for f in fac.columns:
        for a in age.columns:
            corr.loc[f, a] = stats.spearmanr(fac[f], age[a])[0]

    for a in age.columns:
        print(f'[{level}] {a} 비율과 가장 관련 높은 시설: {corr[a].idxmax()} ({corr[a].max():.2f})')

    # 5) 히트맵 (빨강: 그 시설이 많을수록 그 연령대 손님 비율 ↑)
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f')
    plt.title(f'[{level}] 시설 비율 vs 매출 연령대 비율 상관계수')
    plt.tight_layout(); plt.show()

# H2. 업종 비율에 따라 매출 연령대가 나뉠 것이다
def h2(level):
    key = f'{level}_코드'

    # 1) 지역별 매출 연령대 비율
    age = sales_age(level)

    # 2) 지역별 업종 비율 = 업종 점포 수 / 전체 점포 수 (행: 지역, 열: 업종)
    store = load(f'점포-{level}').drop_duplicates(['기준_년분기_코드', key, '서비스_업종_코드'])
    cnt = store.pivot_table(index=key, columns='서비스_업종_코드_명', values='전체_점포_수', aggfunc='mean', fill_value=0)
    cnt = cnt[cnt.sum(axis=1) >= 5]                       # 점포 5개 미만 지역 제외
    ratio = cnt.div(cnt.sum(axis=1), axis=0)
    ratio = ratio[cnt.sum().nlargest(20).index]           # 업종이 너무 많아 점포 많은 20개만

    # 3) 두 데이터에 모두 있는 지역만
    both = ratio.index.intersection(age.index)
    ratio = ratio.loc[both]
    age = age.loc[both]
    print(f'\n[{level}] 분석 지역 수: {len(both)}')

    # 4) 업종 비율 x 매출 연령대 비율 상관계수 (행: 업종, 열: 연령대)
    corr = pd.DataFrame(index=ratio.columns, columns=age.columns, dtype=float)
    for s in ratio.columns:
        for a in age.columns:
            corr.loc[s, a] = stats.spearmanr(ratio[s], age[a])[0]

    for a in age.columns:
        print(f'[{level}] {a} 비율과 가장 관련 높은 업종: {corr[a].idxmax()} ({corr[a].max():.2f})')

    # 5) 히트맵 (빨강: 그 업종이 많을수록 그 연령대 손님 비율 ↑)
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f')
    plt.title(f'[{level}] 업종 비율 vs 매출 연령대 비율 상관계수')
    plt.ylabel('')
    plt.tight_layout(); plt.show()

# H3. 거주(상주)인구 연령대와 그 지역 매출 연령대는 비슷할 것이다
POP = {'연령대_10_상주인구_수': '10대', '연령대_20_상주인구_수': '20대', '연령대_30_상주인구_수': '30대',
       '연령대_40_상주인구_수': '40대', '연령대_50_상주인구_수': '50대', '연령대_60_이상_상주인구_수': '60대+'}

def h3(level):
    key = f'{level}_코드'

    # 1) 지역별 매출 연령대 비율
    age = sales_age(level)

    # 2) 지역별 상주인구 연령대 비율
    pop = load(f'상주인구-{level}')
    pop = pop.groupby(key)[list(POP)].mean()
    pop = pop.rename(columns=POP)
    pop = pop[pop.sum(axis=1) >= 100]                     # 상주인구 100명 미만 지역 제외
    pop = pop.div(pop.sum(axis=1), axis=0)

    # 3) 두 데이터에 모두 있는 지역만
    both = pop.index.intersection(age.index)
    pop = pop.loc[both]
    age = age.loc[both]
    print(f'\n[{level}] 분석 지역 수: {len(both)}')

    # 4) 상주 연령대 x 매출 연령대 상관계수 (대각선 = 같은 연령대끼리)
    corr = pd.DataFrame(index=pop.columns, columns=age.columns, dtype=float)
    for a in pop.columns:
        for b in age.columns:
            corr.loc[a, b] = stats.spearmanr(pop[a], age[b])[0]

    for a in pop.columns:
        print(f'[{level}] 상주 {a} 비율 vs 매출 {a} 비율 상관계수 {corr.loc[a, a]:.2f}')

    # 5) 평균 연령대 구성 비교
    avg = pd.DataFrame({'상주인구': pop.mean() * 100, '매출 건수': age.mean() * 100})
    print(f'[{level}] 평균 연령대 구성(%)\n', avg.round(1))

    # 6) 시각화: 왼쪽 상관계수 히트맵, 오른쪽 평균 연령대 구성
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.6, vmax=.6, annot=True, fmt='.2f', ax=ax[0])
    ax[0].set(xlabel='매출 연령대', ylabel='상주인구 연령대', title=f'[{level}] 상주 vs 매출 연령대 비율 상관계수')
    avg.plot.bar(rot=0, ax=ax[1])
    ax[1].set(ylabel='평균 비율 (%)', title=f'[{level}] 평균 연령대 구성')
    plt.tight_layout(); plt.show()

# H4. 유동인구 수가 많으면 매출 연령대가 낮을(어릴) 것이다
def h4(level):
    key = f'{level}_코드'

    # 1) 지역별 매출 연령대 비율(%) + 유동인구 + 손님 평균연령 (10대 → 15세 ... 60대+ → 65세)
    df = sales_age(level) * 100
    df['평균연령'] = (df['10대'] * 15 + df['20대'] * 25 + df['30대'] * 35
                     + df['40대'] * 45 + df['50대'] * 55 + df['60대+'] * 65) / 100
    df['유동인구'] = load(f'길단위인구-{level}').groupby(key)['총_유동인구_수'].mean()
    df = df[df['유동인구'] > 0]

    # 2) 유동인구와의 상관계수 (음수면 유동인구가 많을수록 그 값이 작아짐)
    print(f'\n[{level}] 분석 지역 수: {len(df)}')
    corr = {}
    for c in df.columns[:-1]:
        r, p = stats.spearmanr(df['유동인구'], df[c])
        corr[c] = r
        print(f'[{level}] 유동인구 vs {c} 상관계수 {r:.2f} (p={p:.4f})')
    corr = pd.DataFrame({'유동인구': corr})

    # 3) 유동인구로 5개 그룹 → 그룹별 손님 평균연령
    df['그룹'] = pd.qcut(df['유동인구'], 5, labels=['1(적음)', '2', '3', '4', '5(많음)'])
    group = df.groupby('그룹', observed=True).mean()
    print(f'[{level}] 그룹별 손님 평균연령\n', group['평균연령'].round(1))

    # 4) 시각화: 왼쪽 그룹별 손님 평균연령, 오른쪽 상관계수 히트맵
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), width_ratios=[2, 1])
    group['평균연령'].plot(marker='o', color='#2a78d6', lw=2, ax=ax[0])
    for i, y in enumerate(group['평균연령']):
        ax[0].annotate(f'{y:.1f}세', (i, y), textcoords='offset points', xytext=(0, 8), ha='center')
    ax[0].set(xlabel='유동인구 그룹', ylabel='손님 평균연령 (세)', title=f'[{level}] 유동인구 그룹별 손님 평균연령')
    ax[0].margins(y=.15)

    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f', ax=ax[1])
    ax[1].set(title=f'[{level}] 유동인구와의 상관계수')
    ax[1].tick_params(axis='y', rotation=0)
    plt.tight_layout(); plt.show()

for lv in ['상권', '행정동']:
    h1(lv)
    h2(lv)
    h3(lv)
    h4(lv)
