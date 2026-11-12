# 아파트·빌라 매매와 전월세 전처리: 분석·모델에 쓸 거래만 남기고, 금액·날짜를 숫자로 바꾸고, 평당 금액 등 자주 쓰는 값을 미리 계산해 저장
#   입력: data/interim/부동산(도로명주소)/{apt,villa}_sale_juso.csv, {apt,villa}_rent_juso.csv  (원본은 바꾸지 않음)
#   출력: data/processed/부동산(전처리)/아파트_매매.csv, 아파트_전월세.csv, 빌라_매매.csv, 빌라_전월세.csv  (엑셀에서 한글이 깨지지 않게 utf-8-sig)
#   거르는 기준: 행정동을 지번으로 정확히 찾은 거래만, 취소 거래 제외
#     면적은 거르지 않음: 평수와 상관없이 작은 집도 실제 거래라서 모두 포함 (모델 ml/training/apt_sale_data.py 도 같음)
#     (처음에는 근거 없이 20㎡ 미만을 뺐다가 없앰. 다시 빼고 싶으면 MIN_AREA 에 숫자를 넣고, 모델 코드도 같이 맞출 것)
#   같은 값이 반복되는 행은 지우지 않음: 매매는 같은 날 여러 호실을 한꺼번에 판 거래이고, 전월세는 원본의 일부 열(건물명·계약기간 등)을 빼서 같아 보이는 것
# 실행 순서: ConvertSale.py → SplitRent.py → CleanHousing.py
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
SRC, OUT = BASE + 'interim/부동산(도로명주소)/', BASE + 'processed/부동산(전처리)/'
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 → 상권분석의 옛 행정동 (AptHypothesis.py 와 같음)
GU = {11110: '종로구', 11140: '중구', 11170: '용산구', 11200: '성동구', 11215: '광진구', 11230: '동대문구', 11260: '중랑구', 11290: '성북구', 11305: '강북구', 11320: '도봉구',
      11350: '노원구', 11380: '은평구', 11410: '서대문구', 11440: '마포구', 11470: '양천구', 11500: '강서구', 11530: '구로구', 11545: '금천구', 11560: '영등포구', 11590: '동작구',
      11620: '관악구', 11650: '서초구', 11680: '강남구', 11710: '송파구', 11740: '강동구'}
KINDS = {'apt': '아파트', 'villa': '빌라'}
PY = 3.3058                                                     # 1평 = 3.3058㎡
MIN_AREA = 0                                                    # 이 값(㎡) 미만 면적의 거래는 뺌. 0이면 모두 포함
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
os.makedirs(OUT, exist_ok=True)

def step(t, mask, why, log):    # 조건에 맞는 행만 남기고, 몇 건이 빠졌는지 기록
    log.append((why, int((~mask).sum()))); return t[mask]

def dong(t):                    # 행정동 관련 열 (상권분석 자료와 붙일 때는 '행정동_코드(상권분석)'을 씀)
    k = t['행정동_코드'].astype(int)
    return {'행정동_코드': k, '행정동_코드(상권분석)': k.replace(OLD), '자치구': (k // 1000).map(GU), '행정동': t['행정동명'], '법정동': t['umdNm'], '지번': t['jibun']}

def report(name, n0, t, log, extra=()):
    print(f'\n[{name}] {n0:,}건 → {len(t):,}건 저장 ({name.replace(" ", "_")}.csv)')
    for w, k in log: print(f'   - {w}: {k:,}건 제외')
    for line in extra: print(f'   - {line}')

def sale(kind):
    s = pd.read_csv(SRC + f'{kind}_sale_juso.csv', low_memory=False); log = []; n0 = len(s)
    s = step(s, s['연결방식'] == '지번 일치', '행정동을 지번으로 정확히 못 찾음 (인접 지번·법정동 하나뿐·연결 안 됨)', log)
    s = step(s, s['cdealType'].isna(), '취소된 거래 (cdealType = O)', log)
    if MIN_AREA: s = step(s, s['excluUseAr'] >= MIN_AREA, f'전용면적 {MIN_AREA}㎡ 미만', log)
    day = pd.to_datetime(s[['dealYear', 'dealMonth', 'dealDay']].set_axis(['year', 'month', 'day'], axis=1), errors='coerce')
    amt, area = num(s['dealAmount']), s['excluUseAr']
    out = {**dong(s)}
    if kind == 'apt': out |= {'단지명': s['aptNm'], '동': s['aptDong']}
    else: out |= {'건물명': s['mhouseNm'], '주택유형': s['houseType']}                                   # 빌라: 연립 / 다세대
    out |= {'계약일': day.dt.date, '계약연도': day.dt.year, '거래금액(만원)': amt, '전용면적(㎡)': area, '전용면적(평)': (area / PY).round(2),
            '평당 금액(만원)': (amt / (area / PY)).round(1), '층': s['floor'], '건축년도': s['buildYear'], '건물 나이(년)': day.dt.year - s['buildYear']}
    extra = []
    if kind == 'villa':                                                                               # 대지권 면적: 빌라 한 채가 가진 땅의 몫. 전용면적의 5배가 넘는 값(예: 30,822㎡)은 입력 오류로 보고 빈 값
        land = num(s['landAr']); bad = land / area > 5
        out |= {'대지권 면적(㎡)': land.mask(bad), '대지지분(대지권÷전용면적)': (land / area).mask(bad).round(3)}
        extra.append(f'대지권 면적이 전용면적의 5배 넘음 → 대지권만 빈 값: {int(bad.sum()):,}건 (행은 남김)')
    else: out |= {'토지임대부': s['landLeaseholdGbn']}
    out |= {'거래유형': s['dealingGbn'], '매수자': s['buyerGbn'], '매도자': s['slerGbn'], '등기일': s['rgstDate']}
    t = pd.DataFrame(out)
    assert t[['거래금액(만원)', '계약일', '전용면적(㎡)', '층', '건축년도']].notna().all().all(), '매매에 빈 값 또는 숫자로 못 바꾼 값이 있음'
    t.to_csv(OUT + f'{KINDS[kind]}_매매.csv', index=False, encoding='utf-8-sig')
    report(f'{KINDS[kind]} 매매', n0, t, log, extra)

def rent(kind):
    r = pd.read_csv(SRC + f'{kind}_rent_juso.csv', low_memory=False); log = []; n0 = len(r)
    day = pd.to_datetime(r['계약일'].astype(str), format='%Y%m%d', errors='coerce')
    dep, mon, area = num(r['보증금(만원)']), num(r['임대료(만원)']), num(r['임대면적'])
    r = step(r, r['연결방식'] == '지번 일치', '행정동을 지번으로 정확히 못 찾음 (인접 지번·법정동 하나뿐·연결 안 됨)', log)
    r = step(r, (day >= '2021-01-01').reindex(r.index), '계약일이 2021년 이전 (원본은 2021~2026년 접수 자료라 늦게 신고된 옛 계약이 섞임)', log)
    if MIN_AREA: r = step(r, area.reindex(r.index) >= MIN_AREA, f'임대면적 {MIN_AREA}㎡ 미만 (매매와 같은 기준)', log)
    bad = ((r['전월세구분'] == '월세') & (mon.reindex(r.index) == 0)) | ((r['전월세구분'] == '전세') & (dep.reindex(r.index) == 0))
    r = step(r, ~bad, '전월세 구분과 금액이 안 맞음 (월세인데 월세 0원, 전세인데 보증금 0원)', log)
    d_, m_, a_, t_ = dep.reindex(r.index), mon.reindex(r.index), area.reindex(r.index), day.reindex(r.index)
    built = num(r['건축년도']).where(lambda v: v.between(1900, 2026))                                # 범위를 벗어난 건축년도는 빈 값
    floor = num(r['층']).where(lambda v: v <= 100)                                                    # 100층 넘는 값(예: 빌라 601층)은 입력 오류로 보고 빈 값
    t = pd.DataFrame({**dong(r), '계약일': t_.dt.date, '계약연도': t_.dt.year, '전월세구분': r['전월세구분'],
                      '보증금(만원)': d_, '월세(만원)': m_, '임대면적(㎡)': a_, '임대면적(평)': (a_ / PY).round(2),
                      '평당 보증금(만원)': (d_ / (a_ / PY)).round(1), '평당 월세(만원)': (m_ / (a_ / PY)).where(m_ > 0).round(2),    # 평당 월세는 월세 거래만
                      '층': floor, '건축년도': built, '건물 나이(년)': t_.dt.year - built,
                      '신규계약구분': r['신규계약구분'].replace({'1': '신규', '2': '갱신'})})     # 2024년 파일에만 숫자로 적힘. 원본 확인: '2'는 모두 종전보증금이 있고 일부 갱신청구권 사용 → 갱신, '1'은 종전보증금이 전혀 없음 → 신규
    assert t[['보증금(만원)', '월세(만원)', '계약일', '임대면적(㎡)']].notna().all().all(), '전월세에 빈 값 또는 숫자로 못 바꾼 값이 있음'
    t.to_csv(OUT + f'{KINDS[kind]}_전월세.csv', index=False, encoding='utf-8-sig')
    report(f'{KINDS[kind]} 전월세', n0, t, log, [f'전세 {(t["전월세구분"] == "전세").sum():,} / 월세 {(t["전월세구분"] == "월세").sum():,}',
                                                  f'건축년도 빈 값(원래 없거나 범위 밖): {t["건축년도"].isna().sum():,}건 (행은 남김)',
                                                  f'층 빈 값(100층 넘는 입력 오류): {t["층"].isna().sum():,}건 (행은 남김)'])

for kind in KINDS:
    sale(kind); rent(kind)
print('\n저장 폴더:', os.path.abspath(OUT))
