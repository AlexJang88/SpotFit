import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 가설1) 업종 수에 따라 업종별 소비 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 파일 로드
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# sales_filename = "서울시 상권분석서비스(추정매출-행정동)_2025년.csv"
# sales_filepath = os.path.join(DATA_DIR, sales_filename)

# if not os.path.exists(sales_filepath):
#     raise FileNotFoundError(f"지정한 2025년 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

# print(f"📖 파일 로드 완료 [추정매출-2025]: {sales_filename}")
# try:
#     df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_sales = pd.read_csv(sales_filepath, encoding='cp949')
# df_sales.columns = df_sales.columns.str.strip()

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
#     raise ValueError("⚠️ 두 파일 간에 겹치는 '기준_년분기_코드'가 없습니다. 기간을 확인해 주세요.")

# print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(common_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# # ---------------------------------------------------------
# # 3. 변수 가공 (업종별 점포 수 및 소비 성별 비율 산출)
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 점포 파일에서 업종별 점포수 추출 (전체_점포_수 활용)
# df_store = df_store[df_store['전체_점포_수'] > 0].copy()
# store_subset = df_store[KEYS + ['서비스_업종_코드_명', '전체_점포_수']].rename(columns={'전체_점포_수': '업종별_점포수'})

# # 2) 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
# sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
# sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

# sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
# sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
# sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

# sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# # 3) 데이터 병합 (행정동 및 서비스 업종 기준)
# merged_df = pd.merge(store_subset, sales_subset, on=KEYS + ['서비스_업종_코드_명'], how='inner')
# print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 4. [업종별 점포수 x 소비 성별 비율] 서비스 업종별 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# gender_vars = ['남성_매출_비율', '여성_매출_비율']
# corr_matrix = pd.DataFrame(index=gender_vars)

# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 30: # 데이터 수가 충분한 업종만 대상
#         corrs = []
#         for g_var in gender_vars:
#             corr_val = group['업종별_점포수'].corr(group[g_var])
#             corrs.append(corr_val)
#         corr_matrix[industry] = corrs

# corr_matrix = corr_matrix.T
# corr_matrix = corr_matrix.dropna(how='all')

# print("\n=== 📈 [업종별 점포수 x 소비 성별 비율] 서비스 업종별 상관계수 매트릭스 생성 완료 ===")

# # ---------------------------------------------------------
# # 5. 컬러 음영 히트맵(Heatmap) 시각화 (숫자 표시 포함)
# # ---------------------------------------------------------
# plt.figure(figsize=(8, max(10, len(corr_matrix) * 0.4)))

# sns.heatmap(
#     corr_matrix, 
#     annot=True,            # 숫자를 함께 표시
#     fmt=".3f",             # 소수점 3자리까지 표시
#     annot_kws={"size": 9}, # 숫자 폰트 크기
#     cmap="coolwarm", 
#     center=0, 
#     cbar_kws={'label': '피어슨 상관계수 (r)'},
#     linewidths=0.3,
#     linecolor='white'
# )

# plt.title('가설 검증: 업종별 점포 수 vs 소비 성별 비율 상관계수 매트릭스 (히트맵)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('소비 성별 비율', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_업종별점포수_소비성별_히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# # 가설2) 성별 유동인구 수에 따라 업종별 소비 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 파일 로드 (통합 추정매출 파일 반영)
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# # 사용자가 미리 통합해 둔 추정매출 파일명 지정
# sales_filename = "서울시_상권분석서비스_추정매출_행정동_통합.csv"
# sales_filepath = os.path.join(DATA_DIR, sales_filename)

# if not os.path.exists(sales_filepath):
#     raise FileNotFoundError(f"통합 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

# print(f"📖 파일 로드 완료 [통합 추정매출]: {sales_filename}")
# try:
#     df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_sales = pd.read_csv(sales_filepath, encoding='cp949')
# df_sales.columns = df_sales.columns.str.strip()

# # 길단위인구 파일 로드 함수
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

# df_pop = load_file_by_keyword(DATA_DIR, "길단위인구-행정동")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 전체 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

# df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str).str.strip()
# df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str).str.strip()

# sales_quarters = set(df_sales['기준_년분기_코드'].unique())
# pop_quarters = set(df_pop['기준_년분기_코드'].unique())

# common_quarters = sorted(list(sales_quarters.intersection(pop_quarters)))

# # 2021년 1분기('20211')부터 2025년 4분기('20254')까지의 교집합 기간 필터링
# target_quarters = [q for q in common_quarters if '20211' <= q <= '20254']

# if not target_quarters:
#     raise ValueError("⚠️ 지정한 기간(20211 ~ 20254) 내에 겹치는 데이터가 없습니다.")

# print(f"🗓️ 최종 동기화 및 반영된 분석 기간: {target_quarters[0]} ~ {target_quarters[-1]} (총 {len(target_quarters)}개 분기)")

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
# df_pop = df_pop[df_pop['기준_년분기_코드'].isin(target_quarters)]

# # ---------------------------------------------------------
# # 3. 변수 가공 (성별 유동인구 수 및 업종별 소비 성별 비율 산출)
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 길단위인구 파일에서 행정동별 남성/여성 유동인구 합산 (수치 그대로 활용)
# pop_agg = df_pop.groupby(KEYS, as_index=False)[['남성_유동인구_수', '여성_유동인구_수']].sum()
# pop_subset = pop_agg[KEYS + ['남성_유동인구_수', '여성_유동인구_수']].copy()

# # 2) 통합 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
# sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
# sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

# sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
# sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
# sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

# sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# # 3) 데이터 병합 (행정동 및 분기 기준)
# merged_df = pd.merge(pop_subset, sales_subset, on=KEYS, how='inner')
# print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 4. [성별 유동인구 수 x 업종별 성별 매출비율] 서비스 업종별 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# pop_vars = ['남성_유동인구_수', '여성_유동인구_수']
# sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

# matrix_rows = []
# industries = []

# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 30: # 데이터 수가 충분한 업종만 대상
#         row_vals = []
#         for p_var in pop_vars:
#             for s_var in sales_gender_vars:
#                 corr_val = group[p_var].corr(group[s_var])
#                 row_vals.append(corr_val)
#         matrix_rows.append(row_vals)
#         industries.append(industry)

# col_names = ['남성유동수-남성매출', '남성유동수-여성매출', '여성유동수-남성매출', '여성유동수-여성매출']
# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
# corr_matrix = corr_matrix.dropna(how='all')

# print("\n=== 📈 [성별 유동인구 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 ===")

# # ---------------------------------------------------------
# # 5. 컬러 음영 히트맵(Heatmap) 시각화 (숫자 표시 포함)
# # ---------------------------------------------------------
# plt.figure(figsize=(10, max(10, len(corr_matrix) * 0.4)))

# sns.heatmap(
#     corr_matrix, 
#     annot=True,            # 숫자를 함께 표시
#     fmt=".3f",             # 소수점 3자리까지 표시
#     annot_kws={"size": 9}, # 숫자 폰트 크기
#     cmap="coolwarm", 
#     center=0, 
#     cbar_kws={'label': '피어슨 상관계수 (r)'},
#     linewidths=0.3,
#     linecolor='white'
# )

# plt.title('가설 검증: 성별 유동인구 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('유동인구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_성별유동인구수_소비성별_통합히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# # 가설3) 시간대에 따라 업종별 소비 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 통합 추정매출 파일 로드
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# sales_filename = "서울시_상권분석서비스_추정매출_행정동_통합.csv"
# sales_filepath = os.path.join(DATA_DIR, sales_filename)

# if not os.path.exists(sales_filepath):
#     raise FileNotFoundError(f"통합 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

# print(f"📖 파일 로드 완료 [통합 추정매출]: {sales_filename}")
# try:
#     df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_sales = pd.read_csv(sales_filepath, encoding='cp949')

# df_sales.columns = df_sales.columns.str.strip()

# # 업종 명칭 컬럼 감지
# industry_col_candidates = ['서비스_업종_코드_명', '업종_코드_명', '서비스업종코드명']
# industry_col = None
# for col in industry_col_candidates:
#     if col in df_sales.columns:
#         industry_col = col
#         break

# if not industry_col:
#     raise KeyError(f"⚠️ 업종 명칭을 나타내는 컬럼을 찾을 수 없습니다. (확인된 컬럼: {df_sales.columns.tolist()})")

# print(f"✅ 감지된 업종 명칭 컬럼: '{industry_col}'")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

# target_quarters = [str(y) + str(q) for y in range(2021, 2026) for q in range(1, 5)]
# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)].copy()

# print(f"🗓️ 반영된 분석 기간: 20211 ~ 20254 (총 데이터 건수: {len(df_sales):,}건)")

# if df_sales.empty:
#     raise ValueError("⚠️ 지정한 기간(20211 ~ 20254)에 해당하는 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 3. 변수 가공 (시간대별 매출 비율 및 성별 매출 비율 산출)
# # ---------------------------------------------------------
# # ★ 수정된 부분: 실제 파일의 컬럼명('시간대_00~06_매출_금액' 등)과 정확히 일치시킴
# time_cols = [
#     '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액', 
#     '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액'
# ]
# existing_time_cols = [col for col in time_cols if col in df_sales.columns]

# if not existing_time_cols:
#     raise KeyError(f"⚠️ 시간대별 매출 금액 컬럼을 찾을 수 없습니다. (확인된 컬럼: {df_sales.columns.tolist()})")

# # 각 행 단위에서 시간대별 비중 및 성별 비중 계산
# for col in existing_time_cols:
#     ratio_col_name = col.replace('_매출_금액', '_매출_비율')
#     df_sales[ratio_col_name] = np.where(
#         df_sales['당월_매출_금액'] > 0,
#         df_sales[col] / df_sales['당월_매출_금액'] * 100,
#         0
#     )

# time_ratio_cols = [col.replace('_매출_금액', '_매출_비율') for col in existing_time_cols]

# df_sales['총_성별매출'] = df_sales['남성_매출_금액'] + df_sales['여성_매출_금액']
# df_sales = df_sales[df_sales['총_성별매출'] > 0].copy()
# df_sales['남성_매출_비율'] = df_sales['남성_매출_금액'] / df_sales['총_성별매출'] * 100
# df_sales['여성_매출_비율'] = df_sales['여성_매출_금액'] / df_sales['총_성별매출'] * 100

# gender_cols = ['남성_매출_비율', '여성_매출_비율']

# # ---------------------------------------------------------
# # 4. [시간대별 매출 비율 x 소비 성별 비율] 업종별 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# matrix_rows = []
# industries = []

# col_labels = []
# for t_col in time_ratio_cols:
#     # 예쁜 라벨명 생성 (시간대_00~06_매출_비율 -> 00~06시)
#     time_label = t_col.replace('시간대_', '').replace('_매출_비율', '') + '시'
#     col_labels.append(f"{time_label} - 남성매출")
#     col_labels.append(f"{time_label} - 여성매출")

# grouped = df_sales.groupby(industry_col)
# print(f"📊 총 감지된 업종 개수: {len(grouped)}개")

# for industry, group in grouped:
#     if len(group) > 1: 
#         row_vals = []
#         for t_col in time_ratio_cols:
#             for g_col in gender_cols:
#                 if group[t_col].std() == 0 or group[g_col].std() == 0 or pd.isna(group[t_col].std()):
#                     corr_val = 0.0
#                 else:
#                     corr_val = group[t_col].corr(group[g_col])
#                     if pd.isna(corr_val):
#                         corr_val = 0.0
#                 row_vals.append(corr_val)
        
#         matrix_rows.append(row_vals)
#         industries.append(industry)

# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_labels)
# corr_matrix = corr_matrix.dropna(how='all')
# corr_matrix = corr_matrix.fillna(0)

# print(f"📊 최종 생성된 상관계수 매트릭스 형태: {corr_matrix.shape}")

# if corr_matrix.empty:
#     raise ValueError("⚠️ 여전히 상관계수 매트릭스가 비어 있습니다. 데이터프레임의 업종 컬럼 값을 확인해주세요.")

# # ---------------------------------------------------------
# # 5. 컬러 음영 히트맵(Heatmap) 시각화
# # ---------------------------------------------------------
# plt.figure(figsize=(14, max(10, len(corr_matrix) * 0.4)))

# sns.heatmap(
#     corr_matrix, 
#     annot=True,            
#     fmt=".3f",             
#     annot_kws={"size": 8}, 
#     cmap="coolwarm", 
#     center=0, 
#     cbar_kws={'label': '피어슨 상관계수 (r)'},
#     linewidths=0.2,
#     linecolor='white'
# )

# plt.title('가설 검증: 시간대별 매출 비율 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=14, fontweight='bold', pad=15)
# plt.xlabel('시간대별 매출 비중 vs 소비 성별 비율 조합', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_시간대별매출_소비성별_통합히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# 가설4) 남녀 상주인구 수에 따른 업종별 성별이 달라질 것이다

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 원본 폴더 경로 설정 및 파일 로드
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# 1) 통합 추정매출 파일 로드
sales_filename = "서울시_상권분석서비스_추정매출_행정동_통합.csv"
sales_filepath = os.path.join(DATA_DIR, sales_filename)

if not os.path.exists(sales_filepath):
    raise FileNotFoundError(f"통합 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

print(f"📖 파일 로드 완료 [통합 추정매출]: {sales_filename}")
try:
    df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_sales = pd.read_csv(sales_filepath, encoding='cp949')
df_sales.columns = df_sales.columns.str.strip()

# 2) 상주인구 파일 로드 함수
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

df_resident = load_file_by_keyword(DATA_DIR, "상주인구-행정동")

# ---------------------------------------------------------
# 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 동기화
# ---------------------------------------------------------
df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

df_resident['기준_년분기_코드'] = df_resident['기준_년분기_코드'].astype(str).str.strip()
df_resident['행정동_코드'] = df_resident['행정동_코드'].astype(str).str.strip()

sales_quarters = set(df_sales['기준_년분기_코드'].unique())
resident_quarters = set(df_resident['기준_년분기_코드'].unique())

common_quarters = sorted(list(sales_quarters.intersection(resident_quarters)))
target_quarters = [q for q in common_quarters if '20211' <= q <= '20254']

if not target_quarters:
    raise ValueError("⚠️ 지정한 기간(20211 ~ 20254) 내에 겹치는 데이터가 없습니다.")

print(f"🗓️ 최종 동기화 및 반영된 분석 기간: {target_quarters[0]} ~ {target_quarters[-1]} (총 {len(target_quarters)}개 분기)")

df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
df_resident = df_resident[df_resident['기준_년분기_코드'].isin(target_quarters)]

# ---------------------------------------------------------
# 3. 변수 가공 (남녀 상주인구 수 및 업종별 소비 성별 비율 산출)
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 상주인구 파일에서 행정동별 남성/여성 상주인구 합산
resident_agg = df_resident.groupby(KEYS, as_index=False)[['남성_상주인구_수', '여성_상주인구_수']].sum()
resident_subset = resident_agg[KEYS + ['남성_상주인구_수', '여성_상주인구_수']].copy()

# 2) 통합 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# 3) 데이터 병합 (행정동 및 분기 기준)
merged_df = pd.merge(resident_subset, sales_subset, on=KEYS, how='inner')
print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

if merged_df.empty:
    raise ValueError("병합된 데이터가 없습니다.")

# ---------------------------------------------------------
# 4. [남녀 상주인구 수 x 업종별 성별 매출비율] 서비스 업종별 상관계수 매트릭스 산출
# ---------------------------------------------------------
resident_vars = ['남성_상주인구_수', '여성_상주인구_수']
sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

matrix_rows = []
industries = []

for industry, group in merged_df.groupby('서비스_업종_코드_명'):
    if len(group) > 2: # 충분한 표본이 있는 업종만 대상
        row_vals = []
        for r_var in resident_vars:
            for s_var in sales_gender_vars:
                if group[r_var].std() == 0 or group[s_var].std() == 0 or pd.isna(group[r_var].std()):
                    corr_val = 0.0
                else:
                    corr_val = group[r_var].corr(group[s_var])
                    if pd.isna(corr_val):
                        corr_val = 0.0
                row_vals.append(corr_val)
        matrix_rows.append(row_vals)
        industries.append(industry)

col_names = ['남성상주수-남성매출', '남성상주수-여성매출', '여성상주수-남성매출', '여성상주수-여성매출']
corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
corr_matrix = corr_matrix.dropna(how='all')
corr_matrix = corr_matrix.fillna(0)

print(f"\n=== 📈 [남녀 상주인구 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 (크기: {corr_matrix.shape}) ===")

# ---------------------------------------------------------
# 5. 컬러 음영 히트맵(Heatmap) 시각화 (숫자 표시 포함)
# ---------------------------------------------------------
plt.figure(figsize=(10, max(10, len(corr_matrix) * 0.4)))

sns.heatmap(
    corr_matrix, 
    annot=True,            # 숫자를 함께 표시
    fmt=".3f",             # 소수점 3자리까지 표시
    annot_kws={"size": 9}, # 숫자 폰트 크기
    cmap="coolwarm", 
    center=0, 
    cbar_kws={'label': '피어슨 상관계수 (r)'},
    linewidths=0.3,
    linecolor='white'
)

plt.title('가설 검증: 남녀 상주인구 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('상주인구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
plt.ylabel('서비스 업종', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_남녀상주인구수_소비성별_통합히트맵.png")
plt.savefig(output_img, dpi=300)
print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
plt.show()
