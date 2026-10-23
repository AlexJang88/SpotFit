# 연령대 가설: 지역(상권·행정동)의 매출 연령대 구성은 무엇과 관련 있는가 (그래프는 한 페이지 HTML로 모아서 열림)
import glob, os, unicodedata, platform, io, base64, webbrowser
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import stats
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
YQ = [20251, 20252, 20253, 20254, 20261, 20262]     # 점포 데이터가 있는 분기
LEVELS = ['상권', '행정동']
AGES = ['10대', '20대', '30대', '40대', '50대', '60대+']
AGE_CODES = ['10', '20', '30', '40', '50', '60_이상']
SALES = dict(zip([f'연령대_{a}_매출_건수' for a in AGE_CODES], AGES))      # 10대는 카드 매출 비중이 1% 정도라 해석 주의
FACS = ['관공서', '은행', '종합병원', '일반_병원', '약국', '유치원', '초등학교', '중학교', '고등학교', '대학교',
        '백화점', '슈퍼마켓', '극장', '숙박_시설', '공항', '철도_역', '버스_터미널', '지하철_역', '버스_정거장']
PEOPLE = {'유동인구': ('길단위인구', '유동인구_수'), '상주인구': ('상주인구', '상주인구_수'), '직장인구': ('직장인구', '직장_인구_수')}

def load(kw):
    files = [f for f in glob.glob(DATA_DIR + '*.csv') if kw in unicodedata.normalize('NFC', os.path.basename(f))]
    df = pd.concat([pd.read_csv(f, encoding='cp949') for f in files])
    return df[df['기준_년분기_코드'].isin(YQ)] if '기준_년분기_코드' in df else df

def code(level): return f'{level}_코드'

def sales_df(level):    # 상권은 2025년이 두 파일에 겹쳐서 중복 제거
    return load(f'추정매출-{level}').drop_duplicates(['기준_년분기_코드', code(level), '서비스_업종_코드'])

def sales_age(level):   # 지역별 매출 연령대 비율 (모든 가설 공통, 매출 100건 미만 지역 제외)
    s = sales_df(level).groupby(code(level))[list(SALES)].sum().rename(columns=SALES)
    s = s[s.sum(axis=1) >= 100]
    return s.div(s.sum(axis=1), axis=0)

def area_of(level):     # 지역별 면적(㎢)
    return load(f'영역-{level}').drop_duplicates(code(level)).set_index(code(level))['영역_면적'] / 1e6

def store_cnt(level):   # 지역 x 업종 점포 수 (분기 평균, 점포 5개 미만 지역 제외)
    st = load(f'점포-{level}').drop_duplicates(['기준_년분기_코드', code(level), '서비스_업종_코드'])
    cnt = st.pivot_table(index=code(level), columns='서비스_업종_코드_명', values='전체_점포_수', aggfunc='mean', fill_value=0)
    return cnt[cnt.sum(axis=1) >= 5]

def top_ratio(cnt):     # 점포 많은 20개 업종의 비율
    return cnt.div(cnt.sum(axis=1), axis=0)[cnt.sum().nlargest(20).index]

def corr_of(X, Y):      # X의 각 열과 Y의 각 열의 스피어만 상관 (행: X, 열: Y)
    return pd.DataFrame({c: X.corrwith(Y[c], method='spearman') for c in Y})

# ---- 그래프: 그린 그래프는 이미지로 모아두고 마지막에 한 페이지로 보여줌 ----
GRAPHS = {}             # 그래프 이름 → [(라벨, 이미지)] (상권·행정동이 같은 이름이면 한 줄에 나란히)

def collect():
    plt.tight_layout()
    fig = plt.gcf()
    title = fig.axes[0].get_title()                       # '[상권] 1. 집객시설 ...' → 라벨 '상권', 이름 '1. 집객시설 ...'
    label, name = title[1:].split('] ', 1) if title.startswith('[') else ('상권 + 행정동', title)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    GRAPHS.setdefault(name, []).append((label, base64.b64encode(buf.getvalue()).decode()))

def report():
    html = '<meta charset="utf-8"><title>연령대 가설 분석</title><style>body{font-family:"Malgun Gothic",sans-serif;margin:20px} ' \
           '.row{display:flex;gap:8px;align-items:flex-start} .row div{max-width:49%} .row b{font-size:20px;color:#c0392b} img{max-width:100%}</style>'
    for name, imgs in GRAPHS.items():
        html += f'<h2>{name}</h2><div class="row">' + ''.join(f'<div><b>{l}</b><br><img src="data:image/png;base64,{i}"></div>' for l, i in imgs) + '</div>'
    out = os.path.abspath(os.path.join(DATA_DIR, '..', 'processed', 'output', 'AgeHypothesis.html'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write(html)
    webbrowser.open('file:///' + out.replace(os.sep, '/'))
    print('그래프 모음 저장:', out)

def corr_fig(corr, title, height, n):   # 상관계수 히트맵 (빨강: x가 클수록 그 연령대 손님 비율 ↑, 파랑: ↓)
    print(f'\n{title} (지역 {n}곳)\n', corr.round(2))
    plt.figure(figsize=(8, height))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.5, vmax=.5, annot=True, fmt='.2f')
    plt.title(title, fontsize=11); plt.yticks(rotation=0); plt.ylabel('')
    collect()

def age_vs(df, xs, level, name):        # H4~H10 공통: x들과 매출 연령대 비율의 상관계수 + 히트맵
    corr_fig(corr_of(df[xs], df[AGES]), f'[{level}] {name} vs 매출 연령대 비율 상관계수', 1.8 + 0.5 * len(xs), len(df))

# ---- H1~H3 ----
def h1(level):          # 집객시설 비율에 따라 매출 연령대가 나뉠 것이다
    age = sales_age(level)
    fac = load(f'집객시설-{level}').groupby(code(level))[[f + '_수' for f in FACS]].mean().fillna(0)   # 빈칸 = 시설 없음 → 0
    fac.columns = FACS
    fac = fac[fac.sum(axis=1) >= 5]                       # 시설 5개 미만 지역 제외
    fac = fac.div(fac.sum(axis=1), axis=0)
    both = fac.index.intersection(age.index)
    fac, age = fac.loc[both], age.loc[both]
    fac = fac.loc[:, (fac > 0).sum() >= 30]               # 30개 지역 미만에만 있는 시설(공항 등)은 제외
    corr_fig(corr_of(fac, age), f'[{level}] 1. 집객시설 비율 vs 매출 연령대 비율 상관계수', 7, len(both))

def h2(level):          # 업종 비율에 따라 매출 연령대가 나뉠 것이다
    age, ratio = sales_age(level), top_ratio(store_cnt(level))
    both = ratio.index.intersection(age.index)
    corr_fig(corr_of(ratio.loc[both], age.loc[both]), f'[{level}] 2. 업종 비율 vs 매출 연령대 비율 상관계수', 8, len(both))

def h3(level):          # 거주(상주)인구 연령대와 그 지역 매출 연령대는 비슷할 것이다
    age = sales_age(level)
    pop = load(f'상주인구-{level}').groupby(code(level))[[f'연령대_{a}_상주인구_수' for a in AGE_CODES]].mean()
    pop.columns = AGES
    pop = pop[pop.sum(axis=1) >= 100]                     # 상주인구 100명 미만 지역 제외
    pop = pop.div(pop.sum(axis=1), axis=0)
    both = pop.index.intersection(age.index)
    pop, age = pop.loc[both], age.loc[both]
    corr = corr_of(pop, age)                              # 대각선 = 같은 연령대끼리
    print(f'\n[{level}] 3. 상주 vs 매출 연령대 비율 (지역 {len(both)}곳)\n', corr.round(2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[1.2, 1])
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-.6, vmax=.6, annot=True, fmt='.2f', ax=ax[0])
    ax[0].set(xlabel='매출 연령대', ylabel='상주인구 연령대', title=f'[{level}] 3. 거주인구 vs 매출 연령대 비율 상관계수')
    pd.DataFrame({'상주인구': pop.mean() * 100, '매출 건수': age.mean() * 100}).plot.bar(rot=0, ax=ax[1])
    ax[1].set(ylabel='평균 비율 (%)', title=f'[{level}] 평균 연령대 구성')
    collect()

# ---- 면적 고려 (밀집도) ----
def h_density(level):   # 면적당 점포 수, 면적당 집객시설 수 (관공서, 은행 등 시설 종류별 포함)
    c, area, df = code(level), area_of(level), sales_age(level)
    st = load(f'점포-{level}').drop_duplicates(['기준_년분기_코드', c, '서비스_업종_코드'])
    df['점포 밀도'] = st.groupby(['기준_년분기_코드', c])['전체_점포_수'].sum().groupby(c).mean() / area
    fac = load(f'집객시설-{level}').groupby(c)[['집객시설_수'] + [f + '_수' for f in FACS]].mean().reindex(df.index).fillna(0)
    df['집객시설 밀도'] = fac['집객시설_수'] / area
    xs = ['점포 밀도', '집객시설 밀도']
    for f in FACS:
        if (fac[f + '_수'] > 0).sum() >= 30:              # 30개 지역 미만에만 있는 시설(공항 등)은 제외
            df[f + ' 밀도'] = fac[f + '_수'] / area
            xs.append(f + ' 밀도')
    age_vs(df.dropna(), xs, level, '1·2 보충. 면적당 점포·집객시설 수')

# ---- H4~H10 ----
def h4(level):          # 유동인구 수가 많으면 매출 연령대가 낮을(어릴) 것이다
    df = sales_age(level).assign(유동인구=load(f'길단위인구-{level}').groupby(code(level))['총_유동인구_수'].mean())
    age_vs(df[df['유동인구'] > 0], ['유동인구'], level, '4. 유동인구')

def one_person():       # 행정동별 1인가구 비율(2025년). 코드 체계가 달라 이름으로 연결 ('·'은 상권 데이터엔 '?'로 저장, 신사동은 강남·관악 2곳)
    one = pd.read_csv(DATA_DIR + '1인가구/1인가구_2025년.csv', encoding='cp949').groupby(['행정동코드', '행정동명'])[['1인가구수', '총인구']].sum().reset_index()
    one['비율'] = one['1인가구수'] / one['총인구'] * 100
    one['이름'] = one['행정동명'].str.replace('·', '').mask(one['행정동코드'] == 1121068, '관악신사동')
    nm = load('상주인구-행정동').drop_duplicates('행정동_코드')[['행정동_코드', '행정동_코드_명']]
    nm['이름'] = nm['행정동_코드_명'].str.replace('?', '').mask(nm['행정동_코드'] == 11620685, '관악신사동')
    return one.merge(nm, on='이름').set_index('행정동_코드')['비율']

def h5():               # 1인 가구가 많으면 매출 연령대가 낮을(어릴) 것이다 (행정동만)
    age_vs(sales_age('행정동').assign(**{'1인가구비율': one_person()}).dropna(), ['1인가구비율'], '행정동', '5. 1인가구 비율')

def h6(level):          # 아파트 가격이 낮으면 매출 연령대가 낮을(어릴) 것이다 (아파트 없는 지역은 제외)
    apt = load(f'아파트-{level}').groupby(code(level)).mean(numeric_only=True)
    house = apt[[c for c in apt.columns if '면적' in c and '세대_수' in c]].sum(axis=1)     # 면적별 세대 수 합
    df = sales_age(level).assign(아파트시가=apt['아파트_평균_시가'] / 1e8, **{'아파트 세대 밀도': house / area_of(level)}).dropna()   # 시가는 억 원
    age_vs(df, ['아파트시가', '아파트 세대 밀도'], level, '6. 아파트 시가·밀도')

def h7(level):          # 금·토 야간 매출이 많은 지역은 매출 연령대가 낮을(어릴) 것이다 (연령x요일x시간대 데이터가 없어 지역별 비중으로 비교)
    s = sales_df(level).groupby(code(level)).sum(numeric_only=True)
    day = s[[d + '_매출_건수' for d in ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']]]
    t = s[[f'시간대_건수{x}_매출_건수' for x in ['~06', '~11', '~14', '~17', '~21', '~24']]]   # 00~06시, ..., 21~24시
    df = sales_age(level).assign(**{'금토 매출비율': (day['금요일_매출_건수'] + day['토요일_매출_건수']) / day.sum(axis=1) * 100,
                                    '야간 매출비율': (t['시간대_건수~24_매출_건수'] + t['시간대_건수~06_매출_건수']) / t.sum(axis=1) * 100}).dropna()
    age_vs(df, ['금토 매출비율', '야간 매출비율'], level, '7. 금·토 / 야간 매출 비율')

def h8(level):          # 영업개월수, 폐업개월수에 따라 특정 연령대가 연관성이 있을 것이다
    m = load(f'상권변화지표-{level}').groupby(code(level))[['운영_영업_개월_평균', '폐업_영업_개월_평균']].mean()
    df = sales_age(level).assign(운영개월=m['운영_영업_개월_평균'], 폐업개월=m['폐업_영업_개월_평균']).dropna()
    age_vs(df, ['운영개월', '폐업개월'], level, '8. 운영·폐업 개월수')

def h9():               # 소비지출에 따라 특정 연령대가 연관성이 있을 것이다 (행정동만, 지출 총액 + 지출 항목 비율)
    items = ['식료품', '의류_신발', '생활용품', '의료비', '교통', '교육', '유흥', '여가_문화', '음식', '기타']
    amount = load('소비-행정동').groupby('행정동_코드').mean(numeric_only=True)[[i + '_지출_총금액' for i in items]]
    df = sales_age('행정동').assign(지출총액=amount.sum(axis=1) / 1e8)
    for i in items:
        df[i + ' 비율'] = amount[i + '_지출_총금액'] / amount.sum(axis=1) * 100
    age_vs(df.dropna(), ['지출총액'] + [i + ' 비율' for i in items], '행정동', '9. 소비지출')

def h10():              # 직장인구가 많을수록 특정 연령대가 많을 것이다 (행정동만)
    df = sales_age('행정동').assign(직장인구=load('직장인구-행정동').groupby('행정동_코드')['총_직장_인구_수'].mean()).dropna()
    age_vs(df[df['직장인구'] > 0], ['직장인구'], '행정동', '10. 직장인구')

# ---- H11 성별: 같은 요인을 "수"와 "비중" 두 방식으로 나란히 비교 ----
def gender_base(level): # 남·여 매출 건수 + 남·여 인원 수 (직장인구는 행정동만 있음)
    g = sales_df(level).groupby(code(level))[['남성_매출_건수', '여성_매출_건수']].sum()
    g = g[g.sum(axis=1) >= 100]
    g.columns = ['남성', '여성']
    for n, (file, suf) in PEOPLE.items():
        if level == '행정동' or n != '직장인구':
            num = load(f'{file}-{level}').groupby(code(level))[[f'남성_{suf}', f'여성_{suf}']].mean()
            g[f'남성 {n}'], g[f'여성 {n}'] = num[f'남성_{suf}'], num[f'여성_{suf}']
    return g

def gender_corr(level): # (1) 수: 남·여 인원 수, 점포 수 등이 많은 곳일수록 남·여 매출 건수도 많은가
    df, cnt = gender_base(level), store_cnt(level)
    df['점포 수'] = cnt.sum(axis=1)
    df['집객시설 수'] = load(f'집객시설-{level}').groupby(code(level))['집객시설_수'].mean().reindex(df.index).fillna(0)   # 빈칸 = 시설 없음 → 0
    df = df.join(cnt[cnt.sum().nlargest(20).index], how='inner').dropna()            # 업종별 점포 수 (점포 많은 20개 업종)
    rows = {f'{n} 수': (f'남성 {n}', f'여성 {n}') for n in PEOPLE if f'남성 {n}' in df}     # 인원 수는 같은 성별끼리
    rows.update({c: (c, c) for c in df.columns if c not in ['남성', '여성'] and not c.startswith(('남성 ', '여성 '))})
    corr = pd.DataFrame({r: [stats.spearmanr(df[m], df['남성'])[0], stats.spearmanr(df[f], df['여성'])[0]] for r, (m, f) in rows.items()},
                        index=[f'{level} 남성', f'{level} 여성']).T
    return corr, len(df)

def male_corr(level):   # (2) 비중: 남성 인구 비중 등이 높은 곳일수록 남성 매출 비중도 높은가
    g, cnt = gender_base(level), store_cnt(level)
    df = pd.DataFrame({'남성 매출 비중': g['남성'] / (g['남성'] + g['여성']) * 100})
    for n in PEOPLE:
        if f'남성 {n}' in g:
            df[f'{n} 중 남성 비중'] = g[f'남성 {n}'] / (g[f'남성 {n}'] + g[f'여성 {n}']) * 100
    df['점포 밀도'] = cnt.sum(axis=1) / area_of(level)
    df = df.join(top_ratio(cnt), how='inner').dropna()
    return df.drop(columns='남성 매출 비중').corrwith(df['남성 매출 비중'], method='spearman')

def gender_heatmap(corr, label, xlabel, vmax):    # 비중 기준 히트맵
    corr = corr.loc[corr.mean(axis=1).sort_values(ascending=False).index]      # 값이 큰 요인이 위로
    plt.figure(figsize=(7, 1.8 + 0.33 * len(corr)))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, annot=True, fmt='.2f')
    plt.title(f'[{label}] 11. 성별 분석', fontsize=11); plt.xlabel(xlabel); plt.yticks(rotation=0)
    collect()

def count_heatmap(corr, diff, sizes):             # 수 기준: 남·여 칸은 거의 같아서(둘 다 지역 규모를 따라감) 오른쪽에 "남성 - 여성" 차이를 따로 보여줌
    order = corr.mean(axis=1).sort_values(ascending=False).index
    corr, diff = corr.loc[order], diff.loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(10, 1.8 + 0.33 * len(corr)), width_ratios=[2, 1.1])
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', cbar=False, ax=ax[0])
    sns.heatmap(diff, cmap='RdBu_r', center=0, vmin=-.1, vmax=.1, annot=True, fmt='+.2f', cbar=False, yticklabels=False, ax=ax[1])
    ax[0].set_title('[수 기준] 11. 성별 분석', fontsize=11)
    ax[0].set_xlabel('남·여 매출 건수(수)와의 상관 (인원 수는 같은 성별끼리) / ' + ', '.join(sizes))
    ax[0].tick_params(axis='y', rotation=0)
    ax[1].set_title('남성 - 여성 차이 (빨강: 남성 쪽, 파랑: 여성 쪽)', fontsize=11)
    ax[1].set_xlabel('차이가 작아서 색은 ±0.1에서 최대')
    collect()

def h11():
    res = {lv: gender_corr(lv) for lv in LEVELS}
    corr = pd.concat([c for c, _ in res.values()], axis=1)                          # 상권·행정동의 남·여를 한 히트맵의 네 열로
    diff = pd.DataFrame({f'{lv} 남-여': corr[f'{lv} 남성'] - corr[f'{lv} 여성'] for lv in LEVELS})
    count_heatmap(corr, diff, [f'{lv} {n}곳' for lv, (_, n) in res.items()])
    gender_heatmap(pd.DataFrame({f'{lv} 남성 매출 비중': male_corr(lv) for lv in LEVELS}), '비중 기준',
                   '남성 매출 비중과의 상관 / 빨강: 남성 손님 많음, 파랑: 여성 손님 많음', .6)

# ---- 실행: 가설마다 상권 → 행정동 순서 (한 줄에 나란히 나옴) ----
for h in [h1, h2, h3, h_density, h4]:
    for lv in LEVELS: h(lv)
h5()                                                      # 1인가구는 행정동만
for h in [h6, h7, h8]:
    for lv in LEVELS: h(lv)
h9()                                                      # 소비는 행정동만
h10()                                                     # 직장인구는 행정동만
h11()
report()
