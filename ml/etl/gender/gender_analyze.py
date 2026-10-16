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



# 가설2) 성별 유동인구 수에 따라 업종별 소비 성별이 달라질 것이다

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 원본 폴더 경로 설정 및 파일 로드 (통합 추정매출 파일 반영)
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\1_상권 데이터\행정동"

# 사용자가 미리 통합해 둔 추정매출 파일명 지정
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

# 길단위인구 파일 로드 함수
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

df_pop = load_file_by_keyword(DATA_DIR, "길단위인구-행정동")

# ---------------------------------------------------------
# 2. 기간(기준_년분기_코드) 데이터 정제 및 20211 ~ 20254 전체 동기화
# ---------------------------------------------------------
df_sales['기준_년분기_코드'] = df_sales['기준_년분기_코드'].astype(str).str.strip()
df_sales['행정동_코드'] = df_sales['행정동_코드'].astype(str).str.strip()

df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str).str.strip()
df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str).str.strip()

sales_quarters = set(df_sales['기준_년분기_코드'].unique())
pop_quarters = set(df_pop['기준_년분기_코드'].unique())

common_quarters = sorted(list(sales_quarters.intersection(pop_quarters)))

# 2021년 1분기('20211')부터 2025년 4분기('20254')까지의 교집합 기간 필터링
target_quarters = [q for q in common_quarters if '20211' <= q <= '20254']

if not target_quarters:
    raise ValueError("⚠️ 지정한 기간(20211 ~ 20254) 내에 겹치는 데이터가 없습니다.")

print(f"🗓️ 최종 동기화 및 반영된 분석 기간: {target_quarters[0]} ~ {target_quarters[-1]} (총 {len(target_quarters)}개 분기)")

df_sales = df_sales[df_sales['기준_년분기_코드'].isin(target_quarters)]
df_pop = df_pop[df_pop['기준_년분기_코드'].isin(target_quarters)]

# ---------------------------------------------------------
# 3. 변수 가공 (성별 유동인구 수 및 업종별 소비 성별 비율 산출)
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 길단위인구 파일에서 행정동별 남성/여성 유동인구 합산 (수치 그대로 활용)
pop_agg = df_pop.groupby(KEYS, as_index=False)[['남성_유동인구_수', '여성_유동인구_수']].sum()
pop_subset = pop_agg[KEYS + ['남성_유동인구_수', '여성_유동인구_수']].copy()

# 2) 통합 추정매출 파일에서 업종별 남성/여성 매출 비율 산출
sales_agg = df_sales.groupby(KEYS + ['서비스_업종_코드_명'], as_index=False)[['남성_매출_금액', '여성_매출_금액']].sum()
sales_agg['총_성별매출'] = sales_agg['남성_매출_금액'] + sales_agg['여성_매출_금액']

sales_agg = sales_agg[sales_agg['총_성별매출'] > 0].copy()
sales_agg['남성_매출_비율'] = sales_agg['남성_매출_금액'] / sales_agg['총_성별매출'] * 100
sales_agg['여성_매출_비율'] = sales_agg['여성_매출_금액'] / sales_agg['총_성별매출'] * 100

sales_subset = sales_agg[KEYS + ['서비스_업종_코드_명', '남성_매출_비율', '여성_매출_비율']]

# 3) 데이터 병합 (행정동 및 분기 기준)
merged_df = pd.merge(pop_subset, sales_subset, on=KEYS, how='inner')
print(f"📊 최종 병합 데이터 크기: {merged_df.shape[0]}건")

if merged_df.empty:
    raise ValueError("병합된 데이터가 없습니다.")

# ---------------------------------------------------------
# 4. [성별 유동인구 수 x 업종별 성별 매출비율] 서비스 업종별 상관계수 매트릭스 산출
# ---------------------------------------------------------
pop_vars = ['남성_유동인구_수', '여성_유동인구_수']
sales_gender_vars = ['남성_매출_비율', '여성_매출_비율']

matrix_rows = []
industries = []

for industry, group in merged_df.groupby('서비스_업종_코드_명'):
    if len(group) > 30: # 데이터 수가 충분한 업종만 대상
        row_vals = []
        for p_var in pop_vars:
            for s_var in sales_gender_vars:
                corr_val = group[p_var].corr(group[s_var])
                row_vals.append(corr_val)
        matrix_rows.append(row_vals)
        industries.append(industry)

col_names = ['남성유동수-남성매출', '남성유동수-여성매출', '여성유동수-남성매출', '여성유동수-여성매출']
corr_matrix = pd.DataFrame(matrix_rows, index=industries, columns=col_names)
corr_matrix = corr_matrix.dropna(how='all')

print("\n=== 📈 [성별 유동인구 수 x 업종별 소비 성별 비율] 상관계수 매트릭스 생성 완료 ===")

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

plt.title('가설 검증: 성별 유동인구 수 vs 업종별 소비 성별 비율 상관계수 매트릭스 (2021~2025 통합)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('유동인구 성별 수 vs 소비 성별 비율 조합', fontsize=12)
plt.ylabel('서비스 업종', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_성별유동인구수_소비성별_통합히트맵.png")
plt.savefig(output_img, dpi=300)
print(f"\n🖼️ 히트맵 시각화 이미지 저장 완료: {output_img}")
plt.show()

