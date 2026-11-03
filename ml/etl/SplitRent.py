# 서울시 전월세(행정동 변환본)에서 아파트, 빌라(연립다세대)를 따로 뽑아 저장
#   입력: data/interim/부동산(도로명주소)/서울시_전월세가_juso.csv  (ConvertSale.py 결과)
#   출력: 같은 폴더의 apt_rent_juso.csv (아파트), villa_rent_juso.csv (연립다세대)
#   행은 고르지 않고 그대로 나눔 (연결방식·취소 등 분석 기준은 모델 코드에서 적용). 엑셀에서 한글이 깨지지 않게 utf-8-sig로 저장
# 실행 순서: MergeRent.py → ConvertSale.py → SplitRent.py
import os
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim', '부동산(도로명주소)') + os.sep
r = pd.read_csv(DIR + '서울시_전월세가_juso.csv', low_memory=False)
for kind, name in [('아파트', 'apt_rent_juso.csv'), ('연립다세대', 'villa_rent_juso.csv')]:
    t = r[r['건물용도'] == kind]
    t.to_csv(DIR + name, index=False, encoding='utf-8-sig')
    print(f'{kind}: {len(t):,}행 → {name} | 전세 {(t["전월세구분"] == "전세").sum():,} / 월세 {(t["전월세구분"] == "월세").sum():,} | 지번 일치 {(t["연결방식"] == "지번 일치").mean():.1%}')
print(f'전체 {len(r):,}행 중 나머지(단독다가구·오피스텔) {(~r["건물용도"].isin(["아파트", "연립다세대"])).sum():,}행은 저장하지 않음')
