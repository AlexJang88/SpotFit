# 월세 최종 모델(RentConversionFinal)이 쓰는 데이터 컬럼만 모아 한 파일로 → data/processed/월세_전환율.csv
#   한 행 = 거래 1건 (월세 거래 + 전세 거래, 원본 전월세 파일의 행 순서 그대로). 월세 거래로 학습하고, 전세 거래로 기준 전세가를 만듦
#   입력: raw/부동산/아파트_전월세_2010_2026.csv, 20261030_단지_기본정보.xlsx (K-apt 단지 기본정보 원본, 서울만 사용. 행정동 코드를 붙인 단지_행정동.csv는 쓰지 않음)
#   전처리 원칙: 원본에 있는 행을 지우지 않음 (날짜 자르기·이상값 제거·중복 제거 없음). 결측은 추정해서 채우지 않고 비워 둠.
#   하는 일: 거래마다 같은 건물의 K-apt 단지 정보 9개를 붙임 (자치구|법정동|지번 본번이 같은 단지, 같은 건물에 단지가 둘이면 세대수가 큰 쪽)
#            K-apt 항목은 모델이 쓰는 값으로 미리 계산: 세대당 주차대수, 임대세대 비율, 동당 승강기 수, 난방방식·복도유형·분양형태는 숫자 코드(단지 전체 목록에서 처음 나온 순서)
#   모델 코드에서 하는 것(이 파일에서는 안 함): 2026-09-30 이후 거래 제외, 월세·보증금·면적 0 이하 제외, 같은 계약 중복 제거, 가격 상하위 0.1% 제외
# 실행: python BuildWolseInput.py
import os
import numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
RAW, OUT = BASE + 'raw/부동산/', BASE + 'processed/'

rent = pd.read_csv(RAW + '아파트_전월세_2010_2026.csv', low_memory=False,
                   usecols=['행정동_코드(상권분석)', '자치구', '법정동', '지번', '계약일', '전월세구분', '보증금(만원)', '월세(만원)', '임대면적(㎡)', '층', '건축년도'])
rent = rent.rename(columns={'전월세구분': '구분'})
rent = rent[['구분'] + [c for c in rent.columns if c != '구분']]            # 매매·전세 입력 파일처럼 '구분'을 맨 앞에
print(f'거래 {len(rent):,}건 (월세 {(rent["구분"] == "월세").sum():,}, 전세 {(rent["구분"] == "전세").sum():,})')

# ---------- K-apt 단지 정보 → 건물 키로 거래에 붙이기 ----------
KA = pd.read_excel(RAW + '20261030_단지_기본정보.xlsx', header=1)             # K-apt 원본 (첫 줄은 안내문, 둘째 줄이 컬럼 이름)
KA = KA[KA['시도'] == '서울특별시'].copy()                                       # 서울 단지만 (3,195개)
tok = KA['법정동주소'].astype(str).str.split(',').str[0].str.split()
KA['key'] = tok.str[1] + '|' + tok.str[2] + '|' + tok.str[3].str.extract(r'(\d+)')[0].fillna('')
for c in ['세대수', '동수', '총주차대수', '최고층수', '임대세대수', '승강기(승객용)']: KA[c] = pd.to_numeric(KA[c], errors='coerce')
KA['K-apt 세대당 주차대수'] = KA['총주차대수'] / KA['세대수']
KA['K-apt 임대세대 비율'] = KA['임대세대수'] / KA['세대수']
KA['K-apt 동당 승강기 수'] = KA['승강기(승객용)'] / KA['동수']
for c, n in [('난방방식', 'K-apt 난방방식 코드'), ('복도유형', 'K-apt 복도유형 코드'), ('분양형태', 'K-apt 분양형태 코드')]:
    KA[n] = pd.factorize(KA[c])[0].astype(float); KA.loc[KA[c].isna(), n] = np.nan
KA = KA.rename(columns={'세대수': 'K-apt 세대수', '동수': 'K-apt 동수', '최고층수': 'K-apt 최고층수'}).sort_values('K-apt 세대수', ascending=False).drop_duplicates('key')
KA_COLS = ['K-apt 세대수', 'K-apt 동수', 'K-apt 최고층수', 'K-apt 세대당 주차대수', 'K-apt 임대세대 비율', 'K-apt 동당 승강기 수', 'K-apt 난방방식 코드', 'K-apt 복도유형 코드', 'K-apt 분양형태 코드']
KA = KA[['key'] + KA_COLS]
rent['key'] = rent['자치구'].astype(str) + '|' + rent['법정동'].astype(str) + '|' + rent['지번'].astype(str).str.extract(r'(\d+)')[0].fillna('')
n0 = len(rent)
rent = rent.merge(KA, on='key', how='left').drop(columns='key')
assert len(rent) == n0, '단지 붙이기에서 행 수가 달라짐'

path = OUT + '월세_전환율.csv'
rent.to_csv(path, index=False, encoding='utf-8-sig')
print(f'저장: {os.path.abspath(path)} | {len(rent):,}행 {rent.shape[1]}열 | 단지 정보가 붙은 거래 {rent["K-apt 세대수"].notna().mean() * 100:.1f}%')
d = pd.to_datetime(rent['계약일'], errors='coerce')
print(rent.groupby('구분').agg(건수=('계약일', 'size')).join(d.groupby(rent['구분']).agg(['min', 'max'])).to_string())
print('\n빈 칸 현황:\n' + rent.isna().groupby(rent['구분']).sum().T.loc[lambda t: t.sum(axis=1) > 0].to_string())
