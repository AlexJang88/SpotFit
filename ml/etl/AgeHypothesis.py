# 연령대 가설 (유동인구의 연령대 구성이 무엇에 따라 달라지는가)
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

AGES = {'연령대_10_유동인구_수': '10대', '연령대_20_유동인구_수': '20대', '연령대_30_유동인구_수': '30대',
        '연령대_40_유동인구_수': '40대', '연령대_50_유동인구_수': '50대', '연령대_60_이상_유동인구_수': '60대+'}
FACS = ['관공서', '은행', '종합병원', '일반_병원', '약국', '유치원', '초등학교', '중학교', '고등학교', '대학교',
        '백화점', '슈퍼마켓', '극장', '숙박_시설', '공항', '철도_역', '버스_터미널', '지하철_역', '버스_정거장']

# H1. 집객시설 비율에 따라 유동인구 연령대가 나뉠 것이다
def h1(level):
    key = f'{level}_코드'

    # 1) 지역별 유동인구 연령대 비율 = 연령대 유동인구 / 전체 연령 유동인구
    flow = load(f'길단위인구-{level}')
    flow = flow.groupby(key)[list(AGES)].mean()          # 지역별 평균 (6개 분기)
    flow = flow.rename(columns=AGES)                      # 연령대_20_유동인구_수 → 20대
    age = flow.div(flow.sum(axis=1), axis=0)

    # 2) 지역별 집객시설 비율 = 시설 수 / 전체 시설 수
    fac = load(f'집객시설-{level}')
    fac = fac.groupby(key)[[f + '_수' for f in FACS]].mean().fillna(0)
    fac.columns = FACS                                    # 은행_수 → 은행
    fac = fac[fac.sum(axis=1) >= 5]                       # 시설 5개 미만 지역 제외
    fac = fac.div(fac.sum(axis=1), axis=0)

    # 3) 두 데이터에 모두 있는 지역만 남기기
    both = fac.index.intersection(age.index)
    fac = fac.loc[both]
    age = age.loc[both]
    print(f'\n[{level}] 분석 지역 수: {len(both)}')

    # 4) 너무 드문 시설 제외 (30개 지역 미만에만 있는 공항 등은 상관계수가 이상하게 나옴)
    for f in FACS:
        if (fac[f] > 0).sum() < 30:
            fac = fac.drop(columns=f)

    # 5) 시설 비율 x 연령대 비율 상관계수 표 (행: 시설, 열: 연령대)
    corr = pd.DataFrame(index=fac.columns, columns=age.columns, dtype=float)
    for f in fac.columns:
        for a in age.columns:
            corr.loc[f, a] = stats.spearmanr(fac[f], age[a])[0]

    for a in age.columns:
        print(f'[{level}] {a} 비율과 가장 관련 높은 시설: {corr[a].idxmax()} ({corr[a].max():.2f})')

    # 6) 주력 시설(그 지역에서 가장 많은 시설)별 연령대 구성 비교
    df = age.copy()
    df['주력시설'] = fac.idxmax(axis=1)
    count = df['주력시설'].value_counts()
    df = df[df['주력시설'].isin(count[count >= 20].index)]   # 지역 20개 이상인 주력시설만
    comp = df.groupby('주력시설').mean() * 100
    print(f'\n[{level}] 주력시설별 연령대 비율(%)\n', comp.round(1))

    for a in age.columns:
        groups = [g[a] for _, g in df.groupby('주력시설')]
        p = stats.kruskal(*groups).pvalue
        print(f'[{level}] 주력시설별 {a} 비율이 다른가? p = {p:.4f} →', '다르다' if p < 0.05 else '차이 없다')

    # 7) 시각화: 왼쪽 상관계수 히트맵, 오른쪽 주력시설별 연령대 구성
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f', ax=ax[0])
    ax[0].set(title=f'[{level}] 시설 비율 vs 유동인구 연령대 비율 상관계수')
    comp.plot.barh(stacked=True, colormap='viridis', ax=ax[1])
    ax[1].set(xlabel='연령대 비율 (%)', xlim=(0, 100), title=f'[{level}] 주력시설별 유동인구 연령대 구성')
    ax[1].legend(loc='lower right', ncol=6, fontsize=9)
    plt.tight_layout(); plt.show()

for lv in ['상권', '행정동']:
    h1(lv)
