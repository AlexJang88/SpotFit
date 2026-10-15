import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 가설1) 유동인구 수가 적으면 폐업률이 높을것이다

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 공백이 제거된 안전한 데이터 폴더 경로 설정
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

# 유동인구 파일과 점포(폐업률) 파일 로드
df_flow = load_file_by_keyword(DATA_DIR, "길단위인구-행정동")
df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# ---------------------------------------------------------
# 2. 기간(기준_년분기_코드) 동기화 및 데이터 정제
# ---------------------------------------------------------
df_flow['기준_년분기_코드'] = df_flow['기준_년분기_코드'].astype(str)
df_flow['행정동_코드'] = df_flow['행정동_코드'].astype(str)

df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# 두 파일 간에 겹치는 기간(기준_년분기_코드) 교집합 추출
common_quarters = sorted(list(set(df_flow['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

df_flow = df_flow[df_flow['기준_년분기_코드'].isin(common_quarters)]
df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# ---------------------------------------------------------
# 3. 행정동 단위별 집계 (업종별 평균 폐업률 산출)
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 유동인구: 행정동별 총 유동인구 합계
flow_agg = df_flow.groupby(KEYS, as_index=False)['총_유동인구_수'].sum()

# 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (평균 폐업률 및 총 점포수 합계)
store_agg = df_store.groupby(KEYS, as_index=False).agg({
    '전체_점포_수': 'sum',
    '폐업_점포_수': 'sum',
    '폐업_률': 'mean'  # 업종별 평균 폐업률
})

# 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출
store_agg['행정동_전체_폐업률'] = np.where(
    store_agg['전체_점포_수'] > 0,
    store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
    0
)

# ---------------------------------------------------------
# 4. 데이터 병합 (Merge)
# ---------------------------------------------------------
merged_df = pd.merge(flow_agg, store_agg, on=KEYS, how='inner')
print(f"📊 최종 분석 표 크기 (전체 행정동 x 분기): {merged_df.shape[0]}건")

# ---------------------------------------------------------
# 5. 가설 검증 시각화 (산점도 및 회귀선)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

sns.regplot(
    data=merged_df, 
    x='총_유동인구_수', 
    y='행정동_전체_폐업률',
    scatter_kws={'alpha': 0.3, 'color': 'royalblue'},
    line_kws={'color': 'crimson', 'linewidth': 2.5}
)

plt.title('가설 검증: 행정동별 총 유동인구 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('총 유동인구 수', fontsize=12)
plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_유동인구_폐업률_상관시각화.png")
plt.savefig(output_img, dpi=300)
print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
plt.show()

# ---------------------------------------------------------
# 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# ---------------------------------------------------------
corr_val = merged_df['총_유동인구_수'].corr(merged_df['행정동_전체_폐업률'])
print(f"\n=== 📈 [유동인구 수] <-> [행정동 폐업률] 상관계수 결과 ===")
print(f"피어슨 상관계수 (r): {corr_val:.5f}")

if corr_val < 0:
    print("💡 해석: 유동인구 수가 많을수록 폐업률이 낮아지는(혹은 적을수록 높아지는) 경향이 있어 가설과 일치합니다.")
else:
    print("💡 해석: 유동인구 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")



# 가설2) 상주인구 수가 많으면 폐업률이 낮을것이다

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 파일이 위치한 원본 폴더 경로 설정
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

# 상주인구 파일과 점포(폐업률) 파일 로드
df_pop = load_file_by_keyword(DATA_DIR, "상주인구-행정동")
df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# ---------------------------------------------------------
# 2. 기준2: 기간(기준_년분기_코드) 동기화
# ---------------------------------------------------------
df_pop['기준_년분기_코드'] = df_pop['기준_년분기_코드'].astype(str)
df_pop['행정동_코드'] = df_pop['행정동_코드'].astype(str)

df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# 두 파일 간 공통으로 존재하는 기간(교집합) 추출
common_quarters = sorted(list(set(df_pop['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

df_pop = df_pop[df_pop['기준_년분기_코드'].isin(common_quarters)]
df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# ---------------------------------------------------------
# 3. 기준1 & 기준3: 모든 행정동별 집계 및 점포 업종별 평균 처리
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 상주인구: 행정동별 총 상주인구 합계
pop_agg = df_pop.groupby(KEYS, as_index=False)['총_상주인구_수'].sum()

# 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (업종별 평균 폐업률 산출)
store_agg = df_store.groupby(KEYS, as_index=False).agg({
    '전체_점포_수': 'sum',
    '폐업_점포_수': 'sum',
    '폐업_률': 'mean'  # 업종별 평균 폐업률
})

# 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출 (가중 효과 반영)
store_agg['행정동_전체_폐업률'] = np.where(
    store_agg['전체_점포_수'] > 0,
    store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
    0
)

# ---------------------------------------------------------
# 4. 데이터 병합 (Merge)
# ---------------------------------------------------------
merged_df = pd.merge(pop_agg, store_agg, on=KEYS, how='inner')
print(f"📊 최종 분석 표 크기 (서울시 모든 행정동 x 분기): {merged_df.shape[0]}건")

# ---------------------------------------------------------
# 5. 가설 검증 시각화 (산점도 및 회귀선)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

sns.regplot(
    data=merged_df, 
    x='총_상주인구_수', 
    y='행정동_전체_폐업률',
    scatter_kws={'alpha': 0.3, 'color': 'forestgreen'},
    line_kws={'color': 'darkorange', 'linewidth': 2.5}
)

plt.title('가설 검증: 행정동별 총 상주인구 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('총 상주인구 수', fontsize=12)
plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_상주인구_폐업률_상관시각화.png")
plt.savefig(output_img, dpi=300)
print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
plt.show()

# ---------------------------------------------------------
# 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# ---------------------------------------------------------
corr_val = merged_df['총_상주인구_수'].corr(merged_df['행정동_전체_폐업률'])
print(f"\n=== 📈 [상주인구 수] <-> [행정동 폐업률] 상관계수 결과 ===")
print(f"피어슨 상관계수 (r): {corr_val:.5f}")

if corr_val < 0:
    print("💡 해석: 상주인구 수가 많을수록 폐업률이 낮아지는 경향이 있어 가설과 일치합니다.")
else:
    print("💡 해석: 상주인구 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")



# 가설3) 집객시설 수가 많으면 폐업률이 낮을것이다

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ---------------------------------------------------------
plt.rc('font', family='Malgun Gothic') # Mac 사용 시 'AppleGothic'
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 파일이 위치한 원본 폴더 경로 설정
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

# 집객시설 파일과 점포(폐업률) 파일 로드
df_facility = load_file_by_keyword(DATA_DIR, "집객시설-행정동")
df_store = load_file_by_keyword(DATA_DIR, "점포-행정동")

# ---------------------------------------------------------
# 2. 기준2: 기간(기준_년분기_코드) 동기화
# ---------------------------------------------------------
df_facility['기준_년분기_코드'] = df_facility['기준_년분기_코드'].astype(str)
df_facility['행정동_코드'] = df_facility['행정동_코드'].astype(str)

df_store['기준_년분기_코드'] = df_store['기준_년분기_코드'].astype(str)
df_store['행정동_코드'] = df_store['행정동_코드'].astype(str)

# 두 파일 간 공통으로 존재하는 기간(교집합) 추출
common_quarters = sorted(list(set(df_facility['기준_년분기_코드']).intersection(set(df_store['기준_년분기_코드']))))
print(f"🗓️ 동기화된 분석 가능 기간: {common_quarters[0]} ~ {common_quarters[-1]} (총 {len(common_quarters)}개 분기)")

df_facility = df_facility[df_facility['기준_년분기_코드'].isin(common_quarters)]
df_store = df_store[df_store['기준_년분기_코드'].isin(common_quarters)]

# ---------------------------------------------------------
# 3. 기준1 & 기준3: 모든 행정동별 집계 및 점포 업종별 평균 처리
# ---------------------------------------------------------
KEYS = ['기준_년분기_코드', '행정동_코드', '행정동_코드_명']

# 1) 집객시설: 행정동별 총 집객시설 수 합계
facility_agg = df_facility.groupby(KEYS, as_index=False)['집객시설_수'].sum()

# 2) 점포 및 폐업률: 업종별 행을 행정동 단위로 통합 (업종별 평균 폐업률 산출)
store_agg = df_store.groupby(KEYS, as_index=False).agg({
    '전체_점포_수': 'sum',
    '폐업_점포_수': 'sum',
    '폐업_률': 'mean'  # 업종별 평균 폐업률
})

# 전체 점포 수 대비 총 폐업 점포 수 비율로 행정동 전체 폐업률 재산출 (가중 효과 반영)
store_agg['행정동_전체_폐업률'] = np.where(
    store_agg['전체_점포_수'] > 0,
    store_agg['폐업_점포_수'] / store_agg['전체_점포_수'] * 100,
    0
)

# ---------------------------------------------------------
# 4. 데이터 병합 (Merge)
# ---------------------------------------------------------
merged_df = pd.merge(facility_agg, store_agg, on=KEYS, how='inner')
print(f"📊 최종 분석 표 크기 (서울시 모든 행정동 x 분기): {merged_df.shape[0]}건")

# ---------------------------------------------------------
# 5. 가설 검증 시각화 (산점도 및 회귀선)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

sns.regplot(
    data=merged_df, 
    x='집객시설_수', 
    y='행정동_전체_폐업률',
    scatter_kws={'alpha': 0.3, 'color': 'purple'},
    line_kws={'color': 'crimson', 'linewidth': 2.5}
)

plt.title('가설 검증: 행정동별 집객시설 수 vs 폐업률', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('집객시설 수', fontsize=12)
plt.ylabel('행정동 전체 폐업률 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_img = os.path.join(DATA_DIR, "가설검증_집객시설_폐업률_상관시각화.png")
plt.savefig(output_img, dpi=300)
print(f"🖼️ 시각화 이미지 저장 완료: {output_img}")
plt.show()

# ---------------------------------------------------------
# 6. 피어슨 상관계수 (Pearson Correlation) 수치 산출
# ---------------------------------------------------------
corr_val = merged_df['집객시설_수'].corr(merged_df['행정동_전체_폐업률'])
print(f"\n=== 📈 [집객시설 수] <-> [행정동 폐업률] 상관계수 결과 ===")
print(f"피어슨 상관계수 (r): {corr_val:.5f}")

if corr_val < 0:
    print("💡 해석: 집객시설 수가 많을수록 폐업률이 낮아지는 경향이 있어 가설과 일치합니다.")
else:
    print("💡 해석: 집객시설 수와 폐업률 간의 관계가 예상과 다르게 나타났습니다.")
