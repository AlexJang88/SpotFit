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

for lv in ['상권', '행정동']:
    h1(lv)
    h2(lv)
