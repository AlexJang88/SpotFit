# 전세 최종 모델(AptJeonseForecastV33)이 쓰는 데이터 컬럼만 모아 한 파일로 → data/processed/부동산/전세_V33.csv
#   한 행 = 거래 1건. 전세 거래와 매매 거래를 '구분' 열로 나눠 위아래로 쌓음 (전세가율에 매매 가격이 필요)
#   입력: raw/부동산/아파트_전월세_2010_2026.csv, 아파트_매매_2010_2026.csv, 예금은행_대출금리_신규취급액_기준__*.csv
#   V33 항목 5개: 6개월 상승률, 상대 가격(서울 대비), 금리 12개월 변화, 전세가율, 신축 건물 거래 비율(건축 5년 이내) → 위치·한강공원·업무지구 파일은 필요 없음
#   전처리 원칙: 원본에 있는 행을 지우지 않음 (날짜 자르기·이상값 제거·중복 제거 없음), 행 순서도 원본 그대로. 결측은 추정해서 채우지 않고 비워 둠.
#   하는 일: 컬럼 이름 통일(평당 가격) → 계약 월의 주택담보대출 금리 붙이기 (금리 파일에 없는 달은 빈 칸)
# 실행: python BuildJeonseInput.py
import glob, os
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
RAW, OUT = BASE + 'raw/부동산/', BASE + 'processed/부동산/'
os.makedirs(OUT, exist_ok=True)
KEY = ['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '건축년도']

rent = pd.read_csv(RAW + '아파트_전월세_2010_2026.csv', low_memory=False, usecols=KEY + ['전월세구분', '평당 보증금(만원)'])
rent = rent[rent['전월세구분'] == '전세'].drop(columns='전월세구분').rename(columns={'평당 보증금(만원)': '평당 가격(만원)'}); rent.insert(0, '구분', '전세')
sale = pd.read_csv(RAW + '아파트_매매_2010_2026.csv', low_memory=False, usecols=KEY + ['평당 금액(만원)'])
sale = sale.rename(columns={'평당 금액(만원)': '평당 가격(만원)'}); sale.insert(0, '구분', '매매')
print(f'전세 {len(rent):,}건, 매매 {len(sale):,}건')
d = pd.concat([rent, sale], ignore_index=True)
d['계약일'] = pd.to_datetime(d['계약일'], errors='coerce')
assert d['계약일'].notna().all() and d['행정동_코드(상권분석)'].notna().all()
d['행정동_코드(상권분석)'] = d['행정동_코드(상권분석)'].astype(int)

rt = pd.read_csv(glob.glob(RAW + '예금은행_대출금리_신규취급액_기준__*.csv')[0], encoding='cp949')
rt = pd.to_numeric(rt[rt['계정항목별'].str.strip() == '주택담보대출 (연리%)'].iloc[0, 1:], errors='coerce')
d['주택담보대출 금리(연%)'] = (d['계약일'].dt.year.astype(str) + '.' + d['계약일'].dt.month.astype(str).str.zfill(2)).map({c: v for c, v in rt.items()})

d = d[['구분', '행정동_코드(상권분석)', '자치구', '행정동', '계약일', '평당 가격(만원)', '건축년도', '주택담보대출 금리(연%)']]
d['계약일'] = d['계약일'].dt.strftime('%Y-%m-%d')
path = OUT + '전세_V33.csv'
d.to_csv(path, index=False, encoding='utf-8-sig')
print(f'저장: {os.path.abspath(path)} | {len(d):,}행 {d.shape[1]}열 | 동 {d["행정동_코드(상권분석)"].nunique()}곳')
print(d.groupby('구분').agg(건수=('계약일', 'size'), 시작=('계약일', 'min'), 끝=('계약일', 'max')).to_string())
print('\n빈 칸 현황:\n' + d.isna().groupby(d['구분']).sum().T.loc[lambda t: t.sum(axis=1) > 0].to_string())
