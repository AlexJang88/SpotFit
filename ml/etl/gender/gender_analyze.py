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



# # 가설4) 남녀 상주인구 수에 따른 업종별 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 파일 로드
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# # 1) 통합 추정매출 파일 로드
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

# # 2) 상주인구 파일 로드 함수
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

# df_resident = load_file_by_keyword(DATA_DIR, "상주인구-행정동")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

# df_resident['기준_년분기_코드'] = df_resident['기준_년분기_코드'].astype(str).str.strip()
# df_resident['행정동_코드'] = df_resident['행정동_코드'].astype(str).str.strip()

# sales_quarters = set(df_sales['기준_년분기_코드'].unique())
# resident_quarters = set(df_resident['기준_년분기_코드'].unique())

# common_quarters = sorted(list(sales_quarters.intersection(resident_quarters)))
# target_quarters = [q for q in common_quarters if '20211' <= q <= '20254']

# if not target_quarters:
#     raise ValueError("⚠️ 지정한 기간(20211 ~ 20254) 내에 겹치는 데이터가 없습니다.")

# print(f"🗓️ 최종 동기화 및 반영된 분석 기간: {target_quarters[0]} ~ {target_quarters[-1]} (총 {len(target_quarters)}개 분기)")

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
# df_resident = df_resident[df_resident['기준_년분기_코드'].isin(target_quarters)]

# # ---------------------------------------------------------
# # 3. 변수 가공 (남녀 상주인구 수 및 업종별 소비 성별 비율 산출)
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 상주인구 파일에서 행정동별 남성/여성 상주인구 합산
# resident_agg = df_resident.groupby(KEYS, as_index=False)[['남성_상주인구_수', '여성_상주인구_수']].sum()
# resident_subset = resident_agg[KEYS + ['남성_상주인구_수', '여성_상주인구_수']].copy()

# # 2) 통합 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
# sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
# sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

# sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
# sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
# sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

# sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# # 3) 데이터 병합 (행정동 및 분기 기준)
# merged_df = pd.merge(resident_subset, sales_subset, on=KEYS, how='inner')
# print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 4. [남녀 상주인구 수 x 업종별 성별 매출비율] 서비스 업종별 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# resident_vars = ['남성_상주인구_수', '여성_상주인구_수']
# sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

# matrix_rows = []
# industries = []

# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 2: # 충분한 표본이 있는 업종만 대상
#         row_vals = []
#         for r_var in resident_vars:
#             for s_var in sales_gender_vars:
#                 if group[r_var].std() == 0 or group[s_var].std() == 0 or pd.isna(group[r_var].std()):
#                     corr_val = 0.0
#                 else:
#                     corr_val = group[r_var].corr(group[s_var])
#                     if pd.isna(corr_val):
#                         corr_val = 0.0
#                 row_vals.append(corr_val)
#         matrix_rows.append(row_vals)
#         industries.append(industry)

# col_names = ['남성상주수-남성매출', '남성상주수-여성매출', '여성상주수-남성매출', '여성상주수-여성매출']
# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
# corr_matrix = corr_matrix.dropna(how='all')
# corr_matrix = corr_matrix.fillna(0)

# print(f"\n=== 📈 [남녀 상주인구 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 (크기: {corr_matrix.shape}) ===")

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

# plt.title('가설 검증: 남녀 상주인구 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('상주인구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_남녀상주인구수_소비성별_통합히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# # 가설5) 직장인 남녀인구 수에 따른 업종별 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 원본 폴더 경로 설정 및 파일 로드
# # ---------------------------------------------------------
# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# # 1) 통합 추정매출 파일 로드
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

# # 2) 직장인구 파일 로드 함수
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

# df_worker = load_file_by_keyword(DATA_DIR, "직장인구-행정동")

# # 🔍 직장인구 파일의 실제 컬럼 확인 및 매핑 (실제 컬럼명 반영)
# print("📋 직장인구 파일 컬럼 목록:", df_worker.columns.tolist())

# male_col_candidates = ['남성_직장_인구_수', '남성_직장인구_수', '남성_직장인구', '남성직장인구수']
# female_col_candidates = ['여성_직장_인구_수', '여성_직장인구_수', '여성_직장인구', '여성직장인구수']

# male_worker_col = next((c for c in male_col_candidates if c in df_worker.columns), None)
# female_worker_col = next((c for c in female_col_candidates if c in df_worker.columns), None)

# if not male_worker_col or not female_worker_col:
#     male_worker_col = male_worker_col or next((c for c in df_worker.columns if '남성' in c and '직장' in c), None)
#     female_worker_col = female_worker_col or next((c for c in df_worker.columns if '여성' in c and '직장' in c), None)

# if not male_worker_col or not female_worker_col:
#     raise KeyError(f"⚠️ 직장인구 파일에서 남녀 직장인구 컬럼을 찾을 수 없습니다. (확인된 컬럼: {df_worker.columns.tolist()})")

# print(f"✅ 감지된 직장인구 컬럼 - 남성: '{male_worker_col}', 여성: '{female_worker_col}'")

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

# df_worker['기준_년분기_코드'] = df_worker['기준_년분기_코드'].astype(str).str.strip()
# df_worker['행정동_코드'] = df_worker['행정동_코드'].astype(str).str.strip()

# sales_quarters = set(df_sales['기준_년분기_코드'].unique())
# worker_quarters = set(df_worker['기준_년분기_코드'].unique())

# common_quarters = sorted(list(sales_quarters.intersection(worker_quarters)))
# target_quarters = [q for q in common_quarters if '20211' <= q <= '20254']

# if not target_quarters:
#     raise ValueError("⚠️ 지정한 기간(20211 ~ 20254) 내에 겹치는 데이터가 없습니다.")

# print(f"🗓️ 최종 동기화 및 반영된 분석 기간: {target_quarters[0]} ~ {target_quarters[-1]} (총 {len(target_quarters)}개 분기)")

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
# df_worker = df_worker[df_worker['기준_년분기_코드'].isin(target_quarters)]

# # ---------------------------------------------------------
# # 3. 변수 가공 (남녀 직장인구 수 및 업종별 소비 성별 비율 산출)
# # ---------------------------------------------------------
# KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# # 1) 직장인구 파일에서 행정동별 남성/여성 직장인구 합산
# worker_agg = df_worker.groupby(KEYS, as_index=False)[[male_worker_col, female_worker_col]].sum()
# worker_agg = worker_agg.rename(columns={male_worker_col: '남성_직장인구_수', female_worker_col: '여성_직장인구_수'})
# worker_subset = worker_agg[KEYS + ['남성_직장인구_수', '여성_직장인구_수']].copy()

# # 2) 통합 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
# sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
# sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

# sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
# sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
# sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

# sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# # 3) 데이터 병합 (행정동 및 분기 기준)
# merged_df = pd.merge(worker_subset, sales_subset, on=KEYS, how='inner')
# print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다.")

# # ---------------------------------------------------------
# # 4. [남녀 직장인구 수 x 업종별 성별 매출비율] 서비스 업종별 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# worker_vars = ['남성_직장인구_수', '여성_직장인구_수']
# sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

# matrix_rows = []
# industries = []

# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 2: # 충분한 표본이 있는 업종만 대상
#         row_vals = []
#         for w_var in worker_vars:
#             for s_var in sales_gender_vars:
#                 if group[w_var].std() == 0 or group[s_var].std() == 0 or pd.isna(group[w_var].std()):
#                     corr_val = 0.0
#                 else:
#                     corr_val = group[w_var].corr(group[s_var])
#                     if pd.isna(corr_val):
#                         corr_val = 0.0
#                 row_vals.append(corr_val)
#         matrix_rows.append(row_vals)
#         industries.append(industry)

# col_names = ['남성직장수-남성매출', '남성직장수-여성매출', '여성직장수-남성매출', '여성직장수-여성매출']
# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
# corr_matrix = corr_matrix.dropna(how='all')
# corr_matrix = corr_matrix.fillna(0)

# print(f"\n=== 📈 [남녀 직장인구 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 (크기: {corr_matrix.shape}) ===")

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

# plt.title('가설 검증: 남녀 직장인구 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('직장인구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설검증_남녀직장인구수_소비성별_통합히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# # 가설6) 1인가구 성별 수에 따른 업종별 소비 성별이 달라질 것이다

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
# plt.rc('axes', unicode_minus=False)

# # ---------------------------------------------------------
# # 1. 파일 경로 설정 및 데이터 로드
# # ---------------------------------------------------------
# SALES_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"
# SINGLE_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터"

# # 1) 통합 추정매출 파일 로드
# sales_filename = "서울시_상권분석서비스_추정매출_행정동_통합.csv"
# sales_filepath = os.path.join(SALES_DIR, sales_filename)

# if not os.path.exists(sales_filepath):
#     raise FileNotFoundError(f"통합 추정매출 파일을 찾을 수 없습니다: {sales_filepath}")

# print(f"📖 파일 로드 완료 [통합 추정매출]: {sales_filename}")
# try:
#     df_sales = pd.read_csv(sales_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_sales = pd.read_csv(sales_filepath, encoding='cp949')
# df_sales.columns = df_sales.columns.str.strip()

# # 2) 1인가구_전체 파일 로드
# single_filename = "1인가구_전체.csv"
# single_filepath = os.path.join(SINGLE_DIR, single_filename)

# if not os.path.exists(single_filepath):
#     raise FileNotFoundError(f"1인가구 파일을 찾을 수 없습니다: {single_filepath}")

# print(f"📖 파일 로드 완료 [1인가구_전체]: {single_filename}")
# try:
#     df_single = pd.read_csv(single_filepath, encoding='utf-8-sig')
# except UnicodeDecodeError:
#     df_single = pd.read_csv(single_filepath, encoding='cp949')
# df_single.columns = df_single.columns.str.strip()

# # ---------------------------------------------------------
# # 2. 기간(기준_년분기_코드) 데이터 정제 및 동기화
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()

# # 1인가구 데이터의 '기준_년월' (예: 202201)을 분기 코드로 변환 (예: 20221)
# df_single['기준_년월'] = df_single['기준_년월'].astype(str).str.strip()
# df_single['연도'] = df_single['기준_년월'].str[:4]
# df_single['월'] = df_single['기준_년월'].str[4:].astype(int)

# def month_to_quarter(m):
#     if m in [1, 2, 3]: return '1'
#     elif m in [4, 5, 6]: return '2'
#     elif m in [7, 8, 9]: return '3'
#     elif m in [10, 11, 12]: return '4'
#     return '1'

# df_single['분기'] = df_single['월'].apply(month_to_quarter)
# df_single['기준_년분기_코드'] = df_single['연도'] + df_single['분기']

# # 기간 동기화 (20211 ~ 20254 범위 내)
# target_quarters = [str(y) + str(q) for y in range(2021, 2026) for q in range(1, 5)]

# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)].copy()
# df_single = df_single[df_single['기준_년분기_코드'].isin(target_quarters)].copy()

# print(f"🗓️ 반영된 분석 기간 동기화 완료")

# # ---------------------------------------------------------
# # 3. 변수 가공 및 행정동 명칭(행정동_코드_명 / 행정동명) 기준 병합
# # ---------------------------------------------------------
# # 1인가구 데이터 성별 정제 (1: 남성, 2: 여성)
# df_single['성별_명'] = df_single['성별'].map({1: '남성_1인가구_수', 2: '여성_1인가구_수', '1': '남성_1인가구_수', '2': '여성_1인가구_수'})

# # 행정동명, 분기, 성별별 1인가구수 합산 (코드 대신 공통된 행정동 이름 기준 활용)
# single_agg = df_single.groupby(['기준_년분기_코드', '행정동명', '성별_명'], as_index=False)['1인가구수'].sum()

# # 기준4 반영: 반올림해서 정수로 치환
# single_agg['1인가구수'] = np.round(single_agg['1인가구수']).astype(int)

# # 남성과 여성 컬럼을 가로로 펼치기 (Pivot)
# single_pivot = single_agg.pivot_table(
#     index=['기준_년분기_코드', '행정동명'], 
#     columns='성별_명', 
#     values='1인가구수', 
#     fill_value=0
# ).reset_index()

# if '남성_1인가구_수' not in single_pivot.columns: single_pivot['남성_1인가구_수'] = 0
# if '여성_1인가구_수' not in single_pivot.columns: single_pivot['여성_1인가구_수'] = 0

# # 추정매출 파일에서 업종별 남성/여성 매출 비율 산출 (행정동_코드_명 기준)
# KEYS = ['기준_년분기_코드', '행정동_코드_명']
# sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
# sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

# sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
# sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
# sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

# sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# # 데이터 병합 (기준_년분기_코드 및 행정동 이름 기준)
# merged_df = pd.merge(
#     single_pivot, 
#     sales_subset, 
#     left_on=['기준_년분기_코드', '행정동명'], 
#     right_on=['기준_년분기_코드', '행정동_코드_명'], 
#     how='inner'
# )
# print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

# if merged_df.empty:
#     raise ValueError("병합된 데이터가 없습니다. 행정동 이름(행정동명 vs 행정동_코드_명)이나 기간을 확인해 주세요.")

# # ---------------------------------------------------------
# # 4. [1인가구 성별 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 산출
# # ---------------------------------------------------------
# single_vars = ['남성_1인가구_수', '여성_1인가구_수']
# sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

# matrix_rows = []
# industries = []

# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 2: # 충분한 표본이 있는 업종만 대상
#         row_vals = []
#         for s_var in single_vars:
#             for g_var in sales_gender_vars:
#                 if group[s_var].std() == 0 or group[g_var].std() == 0 or pd.isna(group[s_var].std()):
#                     corr_val = 0.0
#                 else:
#                     corr_val = group[s_var].corr(group[g_var])
#                     if pd.isna(corr_val):
#                         corr_val = 0.0
#                 row_vals.append(corr_val)
#         matrix_rows.append(row_vals)
#         industries.append(industry)

# col_names = ['남성1인가구수-남성매출', '남성1인가구수-여성매출', '여성1인가구수-남성매출', '여성1인가구수-여성매출']
# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
# corr_matrix = corr_matrix.dropna(how='all')
# corr_matrix = corr_matrix.fillna(0)

# print(f"\n=== 📈 [1인가구 성별 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 (크기: {corr_matrix.shape}) ===")

# # ---------------------------------------------------------
# # 5. 컬러 음영 히트맵(Heatmap) 시각화
# # ---------------------------------------------------------
# plt.figure(figsize=(10, max(10, len(corr_matrix) * 0.4)))

# sns.heatmap(
#     corr_matrix, 
#     annot=True,            # 숫자 표시
#     fmt=".3f",             # 소수점 3자리
#     annot_kws={"size": 9}, 
#     cmap="coolwarm", 
#     center=0, 
#     cbar_kws={'label': '피어슨 상관계수 (r)'},
#     linewidths=0.3,
#     linecolor='white'
# )

# plt.title('가설 검증: 1인가구 성별 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('1인가구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
# plt.ylabel('서비스 업종', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(SALES_DIR, "가설검증_1인가구성별수_소비성별_통합히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
# plt.show()



# ============================== 성별 비율이 아닌 성별 금액으로 가설 1~6 다시 코드 실행!! =========================


# # 가설1-1) 업종 수에 따라 업종별 소비 성별이 달라질 것이다(금액)

# # ---------------------------------------------------------
# # 0. 한글 폰트 설정 (Windows / Mac 대응)
# # ---------------------------------------------------------
# plt.rc('font', family='Malgun Gothic')
# plt.rc('axes', unicode_minus=False)

# DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# # 파일 안전하게 읽어오는 함수 (UTF-8 실패 시 CP949 자동 시도)
# def safe_read_csv(filepath):
#     try:
#         return pd.read_csv(filepath, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         return pd.read_csv(filepath, encoding='cp949')

# # 1) 통합 추정매출 파일 로드
# sales_filepath = os.path.join(DATA_DIR, "서울시_상권분석서비스_추정매출_행정동_통합.csv")
# df_sales = safe_read_csv(sales_filepath)
# df_sales.columns = df_sales.columns.str.strip()

# # 2) 점포 파일 로드
# store_files = glob.glob(os.path.join(DATA_DIR, "*점포*.csv"))
# if not store_files:
#     raise FileNotFoundError("점포 파일을 찾을 수 없습니다.")
# df_store = safe_read_csv(store_files[0])
# df_store.columns = df_store.columns.str.strip()

# print(f"📖 점포 파일 로드 완료: {os.path.basename(store_files[0])}")
# print(f"📋 점포 파일 전체 컬럼: {df_store.columns.tolist()}")

# # 🔍 점포 수 컬럼 자동 감지 로직
# store_col_candidates = ['점포_수', '점포수', '개소수']
# store_col = next((c for c in store_col_candidates if c in df_store.columns), None)

# if not store_col:
#     store_col = next((c for c in df_store.columns if '점포' in c), None)

# if not store_col:
#     raise KeyError(f"⚠️ 점포 파일에서 점포 수 관련 컬럼을 찾을 수 없습니다. (확인된 컬럼: {df_store.columns.tolist()})")

# print(f"✅ 감지된 점포 수 컬럼: '{store_col}'")

# # ---------------------------------------------------------
# # 2. 기간 동기화 및 데이터 정제
# # ---------------------------------------------------------
# df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
# df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()
# df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str).str.strip()
# df_store['행정동_코드'] = df_store['행정동_코드'].astype(str).str.strip()

# target_quarters = [str(y) + str(q) for y in range(2021, 2026) for q in range(1, 5)]
# df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
# df_store = df_store[df_store['기준_년분기_코드'].isin(target_quarters)]

# KEYS = ['기준_년분기_코드', '행정동_코드', '서비스_업종_코드_명']
# store_agg = df_store.groupby(KEYS, as_index=False)[store_col].sum()
# store_agg = store_agg.rename(columns={store_col: '점포_수'})

# sales_agg = df_sales.groupby(KEYS, as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()

# merged_df = pd.merge(store_agg, sales_agg, on=KEYS, how='inner')

# # ---------------------------------------------------------
# # 3. 상관계수 계산 및 히트맵 시각화
# # ---------------------------------------------------------
# matrix_rows, industries = [], []
# for industry, group in merged_df.groupby('서비스_업종_코드_명'):
#     if len(group) > 2:
#         r_male = group['점포_수'].corr(group['남성_매출_금액'])
#         r_female = group['점포_수'].corr(group['여성_매출_금액'])
#         matrix_rows.append([0 if pd.isna(r_male) else r_male, 0 if pd.isna(r_female) else r_female])
#         industries.append(industry)

# corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=['점포수-남성매출금액', '점포수-여성매출금액']).fillna(0)

# plt.figure(figsize=(8, max(8, len(corr_matrix) * 0.4)))
# sns.heatmap(
#     corr_matrix, 
#     annot=True, 
#     fmt=".3f", 
#     cmap="coolwarm", 
#     center=0, 
#     cbar_kws={'label': '피어슨 상관계수 (r)'}, 
#     linewidths=0.2, 
#     linecolor='white'
# )
# plt.title('가설1 검증: 업종별 점포 수 vs 남녀 소비 금액 (2021~2025)', fontsize=13, fontweight='bold', pad=15)
# plt.xlabel('점포 수 vs 남녀 소비 금액 조합', fontsize=11)
# plt.ylabel('서비스 업종', fontsize=11)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# plt.tight_layout()
# output_img = os.path.join(DATA_DIR, "가설1_점포수_소비금액_히트맵.png")
# plt.savefig(output_img, dpi=300)
# print(f"🖼️ 히트맵 이미지 저장 완료: {output_img}")
# plt.show()



# 가설2-1) 성별 유동인구 수에 따라 업종별 소비 성별이 달라질 것이다(금액)

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# 파일 안전하게 읽어오는 함수 (UTF-8 실패 시 CP949 자동 시도)
def safe_read_csv(filepath):
    try:
        return pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='cp949')

# ---------------------------------------------------------
# 1. 파일 로드 및 컬럼 자동 감지
# ---------------------------------------------------------
# 1) 통합 추정매출 파일 로드
sales_filepath = os.path.join(DATA_DIR, "서울시_상권분석서비스_추정매출_행정동_통합.csv")
df_sales = safe_read_csv(sales_filepath)
df_sales.columns = df_sales.columns.str.strip()

# 2) 유동인구 파일 로드
pop_files = glob.glob(os.path.join(DATA_DIR, "*길단위인구-행정동*.csv"))
if not pop_files:
    raise FileNotFoundError("길단위인구-행정동 파일을 찾을 수 없습니다.")
df_pop = safe_read_csv(pop_files[0])
df_pop.columns = df_pop.columns.str.strip()

print(f"📖 유동인구 파일 로드 완료: {os.path.basename(pop_files[0])}")
print(f"📋 유동인구 파일 전체 컬럼: {df_pop.columns.tolist()}")

# 유동인구 남녀 컬럼 자동 감지 로직
male_pop_candidates = ['남성_유동인구_수', '남성_유동인구', '남성유동인구수']
female_pop_candidates = ['여성_유동인구_수', '여성_유동인구', '여성유동인구수']

male_pop_col = next((c for c in male_pop_candidates if c in df_pop.columns), None)
female_pop_col = next((c for c in female_pop_candidates if c in df_pop.columns), None)

if not male_pop_col:
    male_pop_col = next((c for c in df_pop.columns if '남성' in c and '유동' in c), None)
if not female_pop_col:
    female_pop_col = next((c for c in df_pop.columns if '여성' in c and '유동' in c), None)

if not male_pop_col or not female_pop_col:
    raise KeyError(f"⚠️ 유동인구 파일에서 남녀 유동인구 수 컬럼을 찾을 수 없습니다. (확인된 컬럼: {df_pop.columns.tolist()})")

print(f"✅ 감지된 유동인구 컬럼 - 남성: '{male_pop_col}', 여성: '{female_pop_col}'")

# ---------------------------------------------------------
# 2. 기간 동기화 및 데이터 정제
# ---------------------------------------------------------
df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()
df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str).str.strip()
df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str).str.strip()

target_quarters = [str(y) + str(q) for y in range(2021, 2026) for q in range(1, 5)]
df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
df_pop = df_pop[df_pop['기준_년분기_코드'].isin(target_quarters)]

KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

pop_agg = df_pop.groupby(KEYS, as_index=False)[[male_pop_col, female_pop_col]].sum()
pop_agg = pop_agg.rename(columns={male_pop_col: '남성_유동인구_수', female_pop_col: '여성_유동인구_수'})

sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()

merged_df = pd.merge(pop_agg, sales_agg, on=KEYS, how='inner')

# ---------------------------------------------------------
# 3. 상관계수 계산 및 히트맵 시각화
# ---------------------------------------------------------
matrix_rows, industries = [], []
pop_vars = ['남성_유동인구_수', '여성_유동인구_수']
sales_vars = ['남성_매출_금액', '여성_매출_금액']

for industry, group in merged_df.groupby('서비스_업종_코드_명'):
    if len(group) > 2:
        row = []
        for p_var in pop_vars:
            for s_var in sales_vars:
                val = group[p_var].corr(group[s_var])
                row.append(0 if pd.isna(val) else val)
        matrix_rows.append(row)
        industries.append(industry)

col_names = ['남성유동수-남성매출금액', '남성유동수-여성매출금액', '여성유동수-남성매출금액', '여성유동수-여성매출금액']
corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names).fillna(0)

plt.figure(figsize=(10, max(10, len(corr_matrix) * 0.4)))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".3f", 
    cmap="coolwarm", 
    center=0, 
    cbar_kws={'label': '피어슨 상관계수 (r)'}, 
    linewidths=0.2, 
    linecolor='white'
)
plt.title('가설2 검증: 성별 유동인구 수 vs 업종별 남녀 소비 금액 (2021~2025)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('유동인구 성별 수 vs 업종별 남녀 소비 금액 조합', fontsize=11)
plt.ylabel('서비스 업종', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설2_유동인구_소비금액_히트맵.png")
plt.savefig(output_img, dpi=300)
print(f"🖼️ 히트맵 이미지 저장 완료: {output_img}")
plt.show()

