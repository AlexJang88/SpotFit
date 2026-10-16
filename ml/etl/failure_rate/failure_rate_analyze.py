import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 가설1) 유동인구 수가 적으면 폐업률이 높을것이다(업종 전체 평균으로 계산)

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 공백이 제거된 안전한 데이터 폴더 경로 설정
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# def load_file_by_keyword(data_dir, keyword):
#     pattern = os.path.join(data_dir, f"*{keyword}*.csv")
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"'{keyword}' 파일을 찾을 수 없습니다: {pattern}")
#     filepath = files[0]
#     print(f"📖 파일 로드 완료 [{keyword}]: {os.path.basename(filepath)}")
    
#     try:
#         df = pd.read_csv(filepath, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         df = pd.read_csv(filepath, encoding='cp949')
#     df.columns = df.columns.str.strip()
#     return df

# # 유동인구 파일과 점포(폐업률) 파일 로드
# df_flow = load_file_by_keyword(DATA_DIR, "길단위인구-행정동")
# df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 동기화 및 데이터 정제
# # ---------------------------------------------------------
# df_flow['기준_년분기_코드'] = df_flow['기준_년분기_코드'].astype(str)
# df_flow['행정동_코드'] = df_flow['행정동_코드'].astype(str)

# df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
# df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# # 두 파일 간에 겹치는 기간(기준_년분기_코드) 교집합 추출
# common_quarters = sorted(list(set(df_flow['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
# print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

# df_flow = df_flow[df_flow['기준_년분기_코드'].isin(common_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# # ---------------------------------------------------------
# # 3. 행정동 단위별 집계 (업종별 평균 폐업률 산출)
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 유동인구: 행정동별 총 유동인구 합계
# flow_agg = df_flow.groupby(KEYS, as_index=False)['총_유동인구_수'].sum()

# # 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (평균 폐업률 및 총 점포수 합계)
# store_agg = df_store.groupby(KEYS, as_index=False).agg({
#     '전체_점포_수': 'sum',
#     '폐업_점포_수': 'sum',
#     '폐업_률': 'mean'  # 업종별 평균 폐업률
# })

# # 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출
# store_agg['행정동_전체_폐업률'] = np.where(
#     store_agg['전체_점포_수'] > 0,
#     store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
#     0
# )

# # ---------------------------------------------------------
# # 4. 데이터 병합 (Merge)
# # ---------------------------------------------------------
# merged_df = pd.merge(flow_agg, store_agg, on=KEYS, how='inner')
# print(f"📊 최종 분석 표 크기 (전체 행정동 x 분기): {merged_df.shape[0]}건")

# # ---------------------------------------------------------
# # 5. 가설 검증 시각화 (산점도 및 회귀선)
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# sns.regplot(
#     data=merged_df, 
#     x='총_유동인구_수', 
#     y='행정동_전체_폐업률',
#     scatter_kws={'alpha': 0.3, 'color': 'royalblue'},
#     line_kws={'color': 'crimson', 'linewidth': 2.5}
# )

# plt.title('가설 검증: 행정동별 총 유동인구 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
# plt.xlabel('총 유동인구 수', fontsize=12)
# plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_유동인구_폐업률_상관시각화.png")
# plt.savefig(output_img, dpi=300)
# print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
# plt.show()

# # ---------------------------------------------------------
# # 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# # ---------------------------------------------------------
# corr_val = merged_df['총_유동인구_수'].corr(merged_df['행정동_전체_폐업률'])
# print(f"\n=== 📈 [유동인구 수] <-> [행정동 폐업률] 상관계수 결과 ===")
# print(f"피어슨 상관계수 (r): {corr_val:.5f}")

# if corr_val < 0:
#     print("💡 해석: 유동인구 수가 많을수록 폐업률이 낮아지는(혹은 적을수록 높아지는) 경향이 있어 가설과 일치합니다.")
# else:
#     print("💡 해석: 유동인구 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")



# # 가설2) 상주인구 수가 많으면 폐업률이 낮을것이다(업종 전체 평균으로 계산)

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 파일이 위치한 원본 폴더 경로 설정
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# def load_file_by_keyword(data_dir, keyword):
#     pattern = os.path.join(data_dir, f"*{keyword}*.csv")
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"'{keyword}' 파일을 찾을 수 없습니다: {pattern}")
#     filepath = files[0]
#     print(f"📖 파일 로드 완료 [{keyword}]: {os.path.basename(filepath)}")
    
#     try:
#         df = pd.read_csv(filepath, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         df = pd.read_csv(filepath, encoding='cp949')
#     df.columns = df.columns.str.strip()
#     return df

# # 상주인구 파일과 점포(폐업률) 파일 로드
# df_pop = load_file_by_keyword(DATA_DIR, "상주인구-행정동")
# df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# # ---------------------------------------------------------
# # 2. 기준2: 기간(기준_년분기_코드) 동기화
# # ---------------------------------------------------------
# df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str)
# df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str)

# df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
# df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# # 두 파일 간 공통으로 존재하는 기간(교집합) 추출
# common_quarters = sorted(list(set(df_pop['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
# print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

# df_pop = df_pop[df_pop['기준_년분기_코드'].isin(common_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# # ---------------------------------------------------------
# # 3. 기준1 & 기준3: 모든 행정동별 집계 및 점포 업종별 평균 처리
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 상주인구: 행정동별 총 상주인구 합계
# pop_agg = df_pop.groupby(KEYS, as_index=False)['총_상주인구_수'].sum()

# # 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (업종별 평균 폐업률 산출)
# store_agg = df_store.groupby(KEYS, as_index=False).agg({
#     '전체_점포_수': 'sum',
#     '폐업_점포_수': 'sum',
#     '폐업_률': 'mean'  # 업종별 평균 폐업률
# })

# # 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출 (가중 효과 반영)
# store_agg['행정동_전체_폐업률'] = np.where(
#     store_agg['전체_점포_수'] > 0,
#     store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
#     0
# )

# # ---------------------------------------------------------
# # 4. 데이터 병합 (Merge)
# # ---------------------------------------------------------
# merged_df = pd.merge(pop_agg, store_agg, on=KEYS, how='inner')
# print(f"📊 최종 분석 표 크기 (서울시 모든 행정동 x 분기): {merged_df.shape[0]}건")

# # ---------------------------------------------------------
# # 5. 가설 검증 시각화 (산점도 및 회귀선)
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# sns.regplot(
#     data=merged_df, 
#     x='총_상주인구_수', 
#     y='행정동_전체_폐업률',
#     scatter_kws={'alpha': 0.3, 'color': 'forestgreen'},
#     line_kws={'color': 'darkorange', 'linewidth': 2.5}
# )

# plt.title('가설 검증: 행정동별 총 상주인구 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
# plt.xlabel('총 상주인구 수', fontsize=12)
# plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_상주인구_폐업률_상관시각화.png")
# plt.savefig(output_img, dpi=300)
# print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
# plt.show()

# # ---------------------------------------------------------
# # 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# # ---------------------------------------------------------
# corr_val = merged_df['총_상주인구_수'].corr(merged_df['행정동_전체_폐업률'])
# print(f"\n=== 📈 [상주인구 수] <-> [행정동 폐업률] 상관계수 결과 ===")
# print(f"피어슨 상관계수 (r): {corr_val:.5f}")

# if corr_val < 0:
#     print("💡 해석: 상주인구 수가 많을수록 폐업률이 낮아지는 경향이 있어 가설과 일치합니다.")
# else:
#     print("💡 해석: 상주인구 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")



# # 가설3) 집객시설 수가 많으면 폐업률이 낮을것이다(업종 전체 평균으로 계산)

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 파일이 위치한 원본 폴더 경로 설정
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# def load_file_by_keyword(data_dir, keyword):
#     pattern = os.path.join(data_dir, f"*{keyword}*.csv")
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"'{keyword}' 파일을 찾을 수 없습니다: {pattern}")
#     filepath = files[0]
#     print(f"📖 파일 로드 완료 [{keyword}]: {os.path.basename(filepath)}")
    
#     try:
#         df = pd.read_csv(filepath, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         df = pd.read_csv(filepath, encoding='cp949')
#     df.columns = df.columns.str.strip()
#     return df

# # 집객시설 파일과 점포(폐업률) 파일 로드
# df_facility = load_file_by_keyword(DATA_DIR, "집객시설-행정동")
# df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# # ---------------------------------------------------------
# # 2. 기준2: 기간(기준_년분기_코드) 동기화
# # ---------------------------------------------------------
# df_facility['기준_년분기_코드'] = df_facility['기준_년분기_코드'].astype(str)
# df_facility['행정동_코드'] = df_facility['행정동_코드'].astype(str)

# df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
# df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# # 두 파일 간 공통으로 존재하는 기간(교집합) 추출
# common_quarters = sorted(list(set(df_facility['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
# print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

# df_facility = df_facility[df_facility['기준_년분기_코드'].isin(common_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# # ---------------------------------------------------------
# # 3. 기준1 & 기준3: 모든 행정동별 집계 및 점포 업종별 평균 처리
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 집객시설: 행정동별 총 집객시설 수 합계
# facility_agg = df_facility.groupby(KEYS, as_index=False)['집객시설_수'].sum()

# # 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (업종별 평균 폐업률 산출)
# store_agg = df_store.groupby(KEYS, as_index=False).agg({
#     '전체_점포_수': 'sum',
#     '폐업_점포_수': 'sum',
#     '폐업_률': 'mean'  # 업종별 평균 폐업률
# })

# # 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출 (가중 효과 반영)
# store_agg['행정동_전체_폐업률'] = np.where(
#     store_agg['전체_점포_수'] > 0,
#     store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
#     0
# )

# # ---------------------------------------------------------
# # 4. 데이터 병합 (Merge)
# # ---------------------------------------------------------
# merged_df = pd.merge(facility_agg, store_agg, on=KEYS, how='inner')
# print(f"📊 최종 분석 표 크기 (서울시 모든 행정동 x 분기): {merged_df.shape[0]}건")

# # ---------------------------------------------------------
# # 5. 가설 검증 시각화 (산점도 및 회귀선)
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7))

# sns.regplot(
#     data=merged_df, 
#     x='집객시설_수', 
#     y='행정동_전체_폐업률',
#     scatter_kws={'alpha': 0.3, 'color': 'purple'},
#     line_kws={'color': 'crimson', 'linewidth': 2.5}
# )

# plt.title('가설 검증: 행정동별 집객시설 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
# plt.xlabel('집객시설 수', fontsize=12)
# plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_집객시설_폐업률_상관시각화.png")
# plt.savefig(output_img, dpi=300)
# print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
# plt.show()

# # ---------------------------------------------------------
# # 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# # ---------------------------------------------------------
# corr_val = merged_df['집객시설_수'].corr(merged_df['행정동_전체_폐업률'])
# print(f"\n=== 📈 [집객시설 수] <-> [행정동 폐업률] 상관계수 결과 ===")
# print(f"피어슨 상관계수 (r): {corr_val:.5f}")

# if corr_val < 0:
#     print("💡 해석: 집객시설 수가 많을수록 폐업률이 낮아지는 경향이 있어 가설과 일치합니다.")
# else:
#     print("💡 해석: 집객시설 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")



# # 가설4) 연령대별 매출비율에 따라 폐업률이 다를 것이다(업종 전체 평균으로 계산)

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 2025년 파일 강제 지정
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# sales_filename = "서울시 상권분석서비스(추정매출-행정동)_2025년.csv"
# sales_filepath = os.path.join(DATA_DIR, sales_filename)

# if not os.path.exists(sales_filepath):
#     raise FileNotFoundError(f"지정한 2025년 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

# print(f"📖 파일 강제 로드 완료 [추정매출-2025]: {sales_filename}")
# try:
#     df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_sales = pd.read_csv(sales_filepath, encoding='cp949')
# df_sales.columns = df_sales.columns.str.strip()

# # 점포 파일 로드 함수
# def load_file_by_keyword(data_dir, keyword):
#     pattern = os.path.join(data_dir, f"*{keyword}*.csv")
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"'{keyword}' 파일을 찾을 수 없습니다: {pattern}")
#     filepath = files[0]
#     print(f"📖 파일 로드 완료 [{keyword}]: {os.path.basename(filepath)}")
    
#     try:
#         df = pd.read_csv(filepath, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         df = pd.read_csv(filepath, encoding='cp949')
#     df.columns = df.columns.str.strip()
#     return df

# df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

# df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str).str.strip()
# df_store['행정동_코드'] = df_store['행정동_코드'].astype(str).str.strip()

# sales_quarters = set(df_sales['기준_년분기_코드'].unique())
# store_quarters = set(df_store['기준_년분기_코드'].unique())

# common_quarters = sorted(list(sales_quarters.intersection(store_quarters)))

# if not common_quarters:
#     raise ValueError("⚠️ 두 파일 간에 겹치는 '기준_년분기_코드'가 없습니다.")

# print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(common_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# # ---------------------------------------------------------
# # 3. 모든 연령대별 매출비율 계산 및 행정동별 집계
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# sales_cols = [
#     '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액', 
#     '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액'
# ]

# existing_sales_cols = [col for col in sales_cols if col in df_sales.columns]

# # 매출 데이터를 행정동 단위로 합산
# sales_agg = df_sales.groupby(KEYS, as_index=False)[existing_sales_cols].sum()
# sales_agg['총_매출_금액'] = sales_agg[existing_sales_cols].sum(axis=1)

# # 각 연령대별 매출 비율(%) 계산
# age_ratio_cols = []
# for col in existing_sales_cols:
#     # 컬럼명 매핑 (예: 연령대_10_매출_금액 -> 10대_매출_비율)
#     age_prefix = col.split('_')[1] # '10', '20', ... '60_이상' 대신 '60이상' 처리
#     if '60' in age_prefix:
#         ratio_col_name = '60대이상_매출_비율'
#     else:
#         ratio_col_name = f"{age_prefix}대_매출_비율"
        
#     sales_agg[ratio_col_name] = np.where(
#         sales_agg['총_매출_금액'] > 0,
#         sales_agg[col] / sales_agg['총_매출_금액'] * 100,
#         0
#     )
#     age_ratio_cols.append(ratio_col_name)

# # 점포 및 폐업률 데이터 행정동 단위 통합 (업종별 평균 폐업률 산출)
# store_agg = df_store.groupby(KEYS, as_index=False).agg({
#     '전체_점포_수': 'sum',
#     '폐업_점포_수': 'sum',
#     '폐업_률': 'mean'
# })

# store_agg['행정동_전체_폐업률'] = np.where(
#     store_agg['전체_점포_수'] > 0,
#     store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
#     0
# )

# # ---------------------------------------------------------
# # 4. 데이터 병합 (Merge)
# # ---------------------------------------------------------
# merged_df = pd.merge(sales_agg, store_agg, on=KEYS, how='inner')
# print(f"📊 최종 분석 표 크기 (서울시 모든 행정동 x 분기): {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 5. 모든 연령대별 매출비율 vs 폐업률 피어슨 상관계수 산출 및 출력
# # ---------------------------------------------------------
# corr_results = {}
# for ratio_col in age_ratio_cols:
#     corr_val = merged_df[ratio_col].corr(merged_df['행정동_전체_폐업률'])
#     corr_results[ratio_col] = corr_val

# # 결과를 보기 좋은 DataFrame으로 변환
# corr_df = pd.DataFrame(list(corr_results.items()), columns=['연령대_매출_비율', '폐업률과의_상관계수(r)'])
# corr_df = corr_df.sort_values(by='폐업률과의_상관계수(r)', ascending=False).reset_index(drop=True)

# print("\n=== 📈 [모든 연령대별 매출 비율] <-> [행정동 폐업률] 상관계수 종합 결과 ===")
# print(corr_df.to_string(index=False))

# # ---------------------------------------------------------
# # 6. 상관계수 시각화 (Bar Plot)
# # ---------------------------------------------------------
# plt.figure(figsize=(10, 6))

# colors = ['crimson' if x > 0 else 'royalblue' for x in corr_df['폐업률과의_상관계수(r)']]
# bars = plt.bar(corr_df['연령대_매출_비율'], corr_df['폐업률과의_상관계수(r)'], color=colors, alpha=0.8, edgecolor='black')

# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.title('가설 검증: 연령대별 매출 비율 vs 행정동 전체 폐업률 상관계수 (2025년)', fontsize=14, fontweight='bold', pad=15)
# plt.xlabel('연령대별 매출 비율', fontsize=12)
# plt.ylabel('피어슨 상관계수 (r)', fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # 바 위에 수치 표시
# for bar in bars:
#     height = bar.get_height()
#     plt.annotate(f'{height:.4f}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3 if height >= 0 else -12),  # 텍스트 위치 조절
#                 textcoords="offset points",
#                 ha='center', fontsize=10, fontweight='bold')

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_전체연령대매출비율_폐업률_상관비교_2025.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 상관계수 비교 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# 아래는 업종 전체 평균으로 계산한게 아닌 각 업종별로 수치 계산
#  
# 가설1-1) 유동인구 수가 적으면 폐업률이 높을것이다(업종 전체 평균으로 계산)

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 원본 폴더 경로 설정
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

def load_file_by_keyword(data_dir, keyword):
    pattern = os.path.join(data_dir, f"*{keyword}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"'{keyword}' 파일을 찾을 수 없습니다: {pattern}")
    filepath = files[0]
    print(f"📖 파일 로드 완료 [{keyword}]: {os.path.basename(filepath)}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp949')
    df.columns = df.columns.str.strip()
    return df

# 길단위인구(유동인구) 파일과 점포 파일 로드
df_pop = load_file_by_keyword(DATA_DIR, "길단위인구-행정동")
df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# ---------------------------------------------------------
# 2. 기간(기준_년분기_코드) 데이터 정제 및 동기화
# ---------------------------------------------------------
df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str).str.strip()
df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str).str.strip()

df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str).str.strip()
df_store['행정동_코드'] = df_store['행정동_코드'].astype(str).str.strip()

pop_quarters = set(df_pop['기준_년분기_코드'].unique())
store_quarters = set(df_store['기준_년분기_코드'].unique())

common_quarters = sorted(list(pop_quarters.intersection(store_quarters)))

if not common_quarters:
    raise ValueError("⚠️ 두 파일 간에 겹치는 '기준_년분기_코드'가 없습니다. 기간을 확인해 주세요.")

print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

df_pop = df_pop[df_pop['기준_년분기_코드'].isin(common_quarters)]
df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# ---------------------------------------------------------
# 3. 데이터 집계 및 병합 (업종별 행 유지)
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 유동인구: 행정동별 총 유동인구 수 합산
pop_agg = df_pop.groupby(KEYS, as_index=False)['총_유동인구_수'].sum()

# 2) 점포 파일: 업종별 행을 유지한 채 필요한 컬럼만 추출
store_subset = df_store[KEYS + ['서비스_업종_코드_명', '폐업_률']].copy()

# 3) 병합
merged_df = pd.merge(pop_agg, store_subset, on=KEYS, how='inner')
print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

if merged_df.empty:
    raise ValueError("병합된 데이터가 없습니다.")

# ---------------------------------------------------------
# 4. 서비스 업종별 유동인구 vs 폐업률 피어슨 상관계수 산출
# ---------------------------------------------------------
industry_corr = []
for industry, group in merged_df.groupby('서비스_업종_코드_명'):
    if len(group) > 30: 
        corr = group['총_유동인구_수'].corr(group['폐업_률'])
        industry_corr.append({'서비스_업종': industry, '상관계수': corr})

corr_df = pd.DataFrame(industry_corr).sort_values(by='상관계수', ascending=False).reset_index(drop=True)

print("\n=== 📈 [업종별 유동인구] <-> [폐업률] 상관계수 결과 ===")
print(corr_df.to_string(index=False))

# ---------------------------------------------------------
# 5. 컬러 음영 히트맵(Heatmap) 시각화
# ---------------------------------------------------------
# 히트맵 표현을 위해 DataFrame을 (N x 1) 형태로 변환
heatmap_data = corr_df.set_index('서비스_업종')[['상관계수']]

plt.figure(figsize=(8, max(6, len(corr_df) * 0.4)))

# diverging 컬러맵(예: coolwarm 또는 RdBu)을 사용하여 양수/음수 음영 구분
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".3f", 
    cmap="coolwarm", 
    center=0, 
    cbar_kws={'label': '피어슨 상관계수 (r)'},
    linewidths=0.5,
    linecolor='white'
)

plt.title('가설 검증: 업종별 유동인구 수 vs 폐업률 상관계수 (컬러 음영 히트맵)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('서비스 업종', fontsize=12)
plt.xlabel('유동인구 vs 폐업률', fontsize=12)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_업종별유동인구_폐업률_히트맵.png")
plt.savefig(output_img, dpi=300)
print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
plt.show()

