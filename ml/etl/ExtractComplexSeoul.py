# K-apt 단지 기본정보(전국)에서 서울 단지만 뽑기 → data/interim/기타/단지_기본정보_서울.csv (원본은 그대로)
#   원본 엑셀의 첫 줄은 안내문이라 두 번째 줄을 컬럼 이름으로 읽음
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
SRC, OUT = BASE + 'raw/기타/20261030_단지_기본정보.xlsx', BASE + 'interim/기타/'
os.makedirs(OUT, exist_ok=True)

x = pd.read_excel(SRC, header=1)
s = x[x['시도'] == '서울특별시']
print(f'전국 {len(x):,}개 → 서울 {len(s):,}개 | 단지코드 중복 {s["단지코드"].duplicated().sum()}건 | 세대수 합 {s["세대수"].sum():,.0f}')
print(s['단지분류'].value_counts().to_string())
s.to_csv(OUT + '단지_기본정보_서울.csv', index=False, encoding='utf-8-sig')
print('저장:', os.path.abspath(OUT + '단지_기본정보_서울.csv'))
