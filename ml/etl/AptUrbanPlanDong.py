# 도시계획사업 구역(CleanUrbanPlan.py 결과) → 아파트 매매 모델용 행정동별 항목 → data/processed/부동산(전처리)/도시계획사업_행정동별_아파트.csv
#  아파트에 해당하는 정비사업만 셈: 재건축·재개발·신속통합기획·주거환경개선 (가로주택·모아타운 같은 소규모정비는 빌라 중심이라 제외)
#  구역이 없는 행정동은 모델 쪽에서 0으로 채움 (이 파일에는 구역이 있는 행정동만 있음)
#  행정동_코드는 앞 8자리, 모델 동네 코드와 맞추려고 이전된 옛 코드 4개는 상권분석 코드로 바꿈 (apt_sale_data.OLD)
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 행정동_코드(주소 DB) → 행정동_코드(상권분석)
OUT = BASE + 'processed/부동산(전처리)/'

z = pd.read_csv(OUT + '도시계획사업_구역.csv', encoding='utf-8-sig')
z = z[z['구분'] == '정비사업'].copy()                                  # 소규모정비사업(가로주택정비·모아타운 등)은 아파트 모델에서 제외
z['행정동_코드'] = z['행정동_코드'].replace(OLD)
g = z.groupby('행정동_코드')
t = pd.DataFrame({
    '정비구역 수': g.size(),
    '재건축 구역 수': g.apply(lambda x: x['사업종류'].isin(['재건축(공동)', '재건축(단독)']).sum()),
    '재개발·신속통합 구역 수': g.apply(lambda x: x['사업종류'].isin(['재개발(도시정비형)', '재개발(주택정비형)', '신속통합기획']).sum()),
    '초기 단계 구역 수': g.apply(lambda x: (x['진행단계'] <= 2).sum()),         # 계획·후보, 구역지정·추진위
    '조합설립 이후~착공 전 구역 수': g.apply(lambda x: x['진행단계'].between(3, 5).sum()),
    '착공 구역 수': g.apply(lambda x: (x['진행단계'] == 6).sum()),
    '준공 구역 수': g.apply(lambda x: (x['진행단계'] == 7).sum()),             # 이미 새 아파트로 바뀐 곳
}).reset_index()
t.to_csv(OUT + '도시계획사업_행정동별_아파트.csv', index=False, encoding='utf-8-sig')
print(f'아파트용 정비구역 {len(z):,}개 → 구역이 있는 행정동 {len(t)}곳 | 저장: {os.path.abspath(OUT + "도시계획사업_행정동별_아파트.csv")}')
print(t.drop(columns='행정동_코드').sum().to_string())
