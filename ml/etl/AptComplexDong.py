# K-apt 서울 단지(MapComplex.py 결과) → 아파트 매매 모델용 행정동별 단지 항목 → data/processed/행정동별_단지정보.csv
#  아파트·주상복합 단지만 셈 (연립주택·다세대·도시형 생활주택은 제외). 단지가 하나도 없는 행정동은 모델 쪽에서 빈 값으로 둠
#  대형 건설사 = 10대 건설사 계열(삼성물산·현대건설·대우건설·GS건설·대림산업(DL)·포스코·롯데건설·SK·현대산업개발(HDC)·현대엔지니어링). 시공사 이름에 포함되면 대형으로 봄 (제 기준이며, 옛 이름 삼성건설·LG건설·럭키 포함)
import os
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
BIG = ['삼성물산', '삼성건설', '현대건설', '대우건설', 'GS건설', 'LG건설', '럭키', '대림산업', '대림건설', 'DL', '포스코', '롯데건설', 'SK건설', 'SK에코', '현대산업개발', 'HDC', '현대엔지니어링']

c = pd.read_csv(DATA + 'interim/기타/단지_행정동.csv', encoding='utf-8-sig', low_memory=False)
n0 = len(c)
c = c[c['단지분류'].isin(['아파트', '주상복합'])].dropna(subset=['행정동_코드']).copy()
c['세대수'] = c['세대수'].astype(float)
c = c[c['세대수'] > 0]                                                             # 세대수 0인 단지 제외
c['대단지'] = c['세대수'] >= 1000
c['대형건설사'] = c['시공사'].fillna('').apply(lambda s: any(b in s for b in BIG))
c['계단식'] = c['복도유형'] == '계단식'
c['최고층수'] = c['최고층수'].where(c['최고층수'] > 0)                                  # 최고층수 0은 입력 안 된 값으로 보고 빈 값 처리
c['임대세대수'] = c['임대세대수'].fillna(0).clip(upper=c['세대수'])                 # 임대가 세대수보다 큰 입력 오류는 세대수로 맞춤
print(f'단지 {n0:,}개 → 아파트·주상복합, 세대수>0 인 {len(c):,}개 | 대형 건설사 {c["대형건설사"].mean() * 100:.1f}%')
g = c.groupby('행정동_코드')
t = pd.DataFrame({
    '아파트 단지 수(K-apt)': g.size(),
    '단지 평균 세대수': g['세대수'].mean(),
    '대단지(1000세대+) 비율(%)': g['대단지'].mean() * 100,
    '세대당 주차대수': g['총주차대수'].sum() / g['세대수'].sum(),
    '계단식 비율(%)': g['계단식'].mean() * 100,
    '임대세대 비율(%)': g['임대세대수'].sum() / g['세대수'].sum() * 100,
    '대형 건설사 단지 비율(%)': g['대형건설사'].mean() * 100,
    '단지 평균 최고층수': g['최고층수'].mean(),
}).reset_index()
t.round(3).to_csv(DATA + 'processed/행정동별_단지정보.csv', index=False, encoding='utf-8-sig')
print(f'행정동 {len(t)}곳 | 저장: {os.path.abspath(DATA + "processed/행정동별_단지정보.csv")}')
print(t.drop(columns='행정동_코드').describe().round(1).T[['count', 'min', '50%', 'max']].to_string())
