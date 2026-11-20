# 도시계획사업(서울플랜+) 정비사업·소규모정비사업 구역 전처리 → data/processed/부동산(전처리)/도시계획사업_구역.csv
#  입력: data/interim/부동산/도시계획사업_행정동.csv (MapUrbanPlan.py 결과)
#  하는 일: 중복 구역 제거 → 행정동 못 붙인 구역 제거 → 쓰는 컬럼만 남김 → 빈 값 처리 → 추진단계를 순서 있는 숫자로 정리
#  빈 값 처리 원칙: 원본에 없는 값은 추정해서 채우지 않음. 값이 없는 이유가 "원래 그 종류에는 없는 값"이면 '있음' 표시 컬럼을 따로 둠
import os
import numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
OUT = BASE + 'processed/부동산(전처리)/'
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(BASE + 'interim/부동산/도시계획사업_행정동.csv', encoding='utf-8-sig')
n0 = len(d)
d = d[~d['중복']]
n1 = len(d)
no_dong = d['행정동_코드'].isna()
no_stage = d['추진단계 코드명'].isna()
print(f'전체 {n0:,}개 → 중복 제거 {n1:,}개 | 행정동 못 붙임 {no_dong.sum()}개 제거 | 추진단계 없음 {(no_stage & ~no_dong).sum()}개 제거')
print('제거한 구역:\n' + d[no_dong | no_stage][['지차체 코드명', '사업명칭', '사업위치']].to_string(index=False))
d = d[~no_dong & ~no_stage].copy()

# 추진단계 → 진행순서 (1 계획·후보 ~ 7 준공). 단계 이름이 사업 종류마다 다르므로 같은 위치의 단계끼리 묶음
STAGE = {'사전자문': 1, '수립범위 자문': 1, '대상지선정': 1, '기획완료': 1, '정비계획수립': 1, '열람공고': 1, '위원회심의': 1, '관리지역고시': 1,
         '구역지정': 2, '추진위구성': 2, '주민합의체 구성': 2,
         '조합설립추진중': 3, '조합설립인가 추진중(연번부여)': 3, '조합설립인가': 3,
         '건축심의': 4, '사업시행인가': 4, '사업시행계획인가': 4,
         '관리처분계획인가': 5, '착공': 6, '준공(일부)': 7, '준공': 7}
NAME = {1: '1 계획·후보', 2: '2 구역지정·추진위', 3: '3 조합설립', 4: '4 심의·사업시행인가', 5: '5 관리처분', 6: '6 착공', 7: '7 준공'}
d['진행단계'] = d['추진단계 코드명'].map(STAGE)
assert d['진행단계'].notna().all(), d.loc[d['진행단계'].isna(), '추진단계 코드명'].unique()
d['진행단계명'] = d['진행단계'].map(NAME)

d['단계진입일'] = pd.to_datetime(d['추진현황'], errors='coerce').dt.normalize()                     # 그 추진단계에 들어간 날짜
d['사업면적'] = d['사업면적'].where(d['사업면적'] > 0)                                               # 면적 0은 입력 안 된 값으로 보고 빈 값 처리
d['공급세대_있음'] = d['공급세대'].notna()                                                           # 모아타운·신속통합기획·주거환경개선은 원래 없는 값이라 채우지 않음
d['행정동_코드'] = d['행정동_코드'].astype(int)

r = d.rename(columns={'기본정보_pk': '구역ID', '지차체 코드명': '자치구', '소분류 코드명': '사업종류', '추진단계 코드명': '추진단계'})
r['자치구'] = r['자치구'].str.strip()
r = r[['구역ID', '구분', '자치구', '사업명칭', '사업위치', '행정동_코드', '연결방식', '사업종류', '추진단계', '진행단계', '진행단계명', '단계진입일', '사업면적', '공급세대', '공급세대_있음']]
r = r.sort_values(['행정동_코드', '진행단계']).reset_index(drop=True)
r.to_csv(OUT + '도시계획사업_구역.csv', index=False, encoding='utf-8-sig')

print(f'\n저장: {os.path.abspath(OUT + "도시계획사업_구역.csv")} | 구역 {len(r):,}개, 행정동 {r["행정동_코드"].nunique()}곳')
print('\n진행단계 분포:\n' + r['진행단계명'].value_counts().sort_index().to_string())
print('\n빈 값 현황 (비어 있는 구역 수):\n' + r.isna().sum().loc[lambda s: s > 0].to_string())
print(f'\n사업면적 0 → 빈 값으로 바꾼 구역: {(d["사업면적"].isna()).sum()}개')
