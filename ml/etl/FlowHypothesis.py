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

def run(level):
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

for lv in ['상권', '행정동']:
    run(lv)
