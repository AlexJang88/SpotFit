import os
import pandas as pd


# # 1. 아파트 매매 데이터 행정동 매핑

# # 파일 경로 설정 (raw string 'r'을 사용하여 백슬래시 오류 방지)
# apt_sale_path = (
#     r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\2_부동산"
#     r" 데이터\apt_sale.csv"
# )
# kikmix_path = (
#     r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\KIKmix.20260916(말소코드포함).xlsx"
# )

# # 1. 파일 불러오기
# print("파일을 불러오는 중입니다...")
# df_apt = pd.read_csv(apt_sale_path, encoding="utf-8-sig")
# df_kik = pd.read_excel(kikmix_path, sheet_name="KIKmix")

# # 2. 말소된 동 제외 (말소일자가 NaN인 현존하는 동만 추출)
# df_kik_active = df_kik[df_kik["말소일자"].isna()].copy()

# # 3. 매핑을 위한 키 포맷 맞추기
# # apt_sale의 sggCd를 5자리 문자열로 변환
# df_apt["sggCd_str"] = df_apt["sggCd"].astype(str).str.zfill(5)

# # KIKmix의 법정동코드를 10자리 문자열로 변환 후 앞 5자리(시군구코드) 추출
# df_kik_active["법정동코드_str"] = (
#     df_kik_active["법정동코드"].astype(str).str.zfill(10)
# )
# df_kik_active["sggCd_prefix"] = df_kik_active["법정동코드_str"].str[:5]

# # 중복 제거 및 필요한 컬럼만 추출 ('동리명'은 법정동명, '읍면동명'은 행정동명)
# mapping_df = df_kik_active[
#     ["sggCd_prefix", "동리명", "행정동코드", "읍면동명"]
# ].drop_duplicates(subset=["sggCd_prefix", "동리명"])

# # 4. 시군구코드(sggCd_str)와 법정동명(umdNm)을 기준으로 병합 (Left Join)
# df_merged = pd.merge(
#     df_apt,
#     mapping_df,
#     left_on=["sggCd_str", "umdNm"],
#     right_on=["sggCd_prefix", "동리명"],
#     how="left",
# )

# # 5. 컬럼명 정돈 및 임시 컬럼 삭제
# df_merged = df_merged.rename(columns={"읍면동명": "행정동명"})
# df_merged = df_merged.drop(
#     columns=["sggCd_str", "sggCd_prefix", "동리명"], errors="ignore"
# )

# # 6. 결과 저장 (동일한 부동산 데이터 폴더 내에 저장)
# output_dir = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\2_부동산 데이터"
# output_path = os.path.join(output_dir, "apt_sale_with_hdong.csv")

# df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")

# print(f"매핑이 완료되었습니다! 저장된 파일 경로: {output_path}")
# print(f"총 데이터 행 수: {len(df_merged)}")
# print(
#     df_merged[
#         [
#             "sggCd",
#             "umdNm",
#             "jibun",
#             "aptNm",
#             "행정동코드",
#             "행정동명",
#         ]
#     ].head(5)
# )



# # 2. 연립/다세대(빌라) 매매 데이터 행정동 매핑

# # 파일 경로 설정 (경로에 맞게 수정하여 사용해 주세요)
# villa_sale_path = (
#     r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\2_부동산"
#     r" 데이터\villa_sale.csv"
# )
# kikmix_path = (
#     r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\KIKmix.20260916(말소코드포함).xlsx"
# )

# # 1. 파일 불러오기
# print("파일을 불러오는 중입니다...")
# df_villa = pd.read_csv(villa_sale_path, encoding="utf-8-sig")
# df_kik = pd.read_excel(kikmix_path, sheet_name="KIKmix")

# # 2. 말소된 동 제외 (말소일자가 NaN인 현존하는 동만 추출)
# df_kik_active = df_kik[df_kik["말소일자"].isna()].copy()

# # 3. 매핑 키 포맷 맞추기 (시군구코드 5자리, 법정동코드 10자리)
# df_villa["sggCd_str"] = df_villa["sggCd"].astype(str).str.zfill(5)
# df_kik_active["법정동코드_str"] = (
#     df_kik_active["법정동코드"].astype(str).str.zfill(10)
# )
# df_kik_active["sggCd_prefix"] = df_kik_active["법정동코드_str"].str[:5]

# # 중복 제거 및 필요한 매핑 컬럼 추출 ('동리명' = 법정동명, '읍면동명' = 행정동명)
# mapping_df = df_kik_active[
#     ["sggCd_prefix", "동리명", "행정동코드", "읍면동명"]
# ].drop_duplicates(subset=["sggCd_prefix", "동리명"])

# # 4. 시군구코드(sggCd)와 법정동명(umdNm)을 기준으로 병합 (Left Join)
# df_merged = pd.merge(
#     df_villa,
#     mapping_df,
#     left_on=["sggCd_str", "umdNm"],
#     right_on=["sggCd_prefix", "동리명"],
#     how="left",
# )

# # 5. 컬럼명 정돈 및 임시 컬럼 삭제
# df_merged = df_merged.rename(columns={"읍면동명": "행정동명"})
# df_merged = df_merged.drop(
#     columns=["sggCd_str", "sggCd_prefix", "동리명"], errors="ignore"
# )

# # 6. 결과 저장
# output_dir = os.path.dirname(villa_sale_path)
# output_path = os.path.join(output_dir, "villa_sale_with_hdong.csv")

# df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")

# print(f"매핑이 완료되었습니다! 저장된 파일 경로: {output_path}")
# print(f"총 데이터 행 수: {len(df_merged)}")
# print(
#     df_merged[
#         [
#             "sggCd",
#             "umdNm",
#             "jibun",
#             "mhouseNm",
#             "행정동코드",
#             "행정동명",
#         ]
#     ].head(5)
# )



# # 3. 단독/다가구 전/월세 데이터 행정동 매핑

# # 1. 경로 설정 (house_rent.csv는 '2_부동산 데이터' 폴더, KIKmix는 '상권 추천 서비스' 폴더)
# data_dir = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스"
# rent_csv_path = os.path.join(data_dir, r"2_부동산 데이터\house_rent.csv")
# kik_path = os.path.join(data_dir, "KIKmix.20260916(말소코드포함).xlsx")

# print("데이터 로딩 중...")

# # 2. 파일 존재 여부 확인 및 로드
# if os.path.exists(rent_csv_path):
#     rent_df = pd.read_csv(rent_csv_path, low_memory=False)
# else:
#     raise FileNotFoundError(f"'{rent_csv_path}' 경로에 house_rent.csv 파일이 존재하지 않습니다.")

# if os.path.exists(kik_path):
#     kik_df = pd.read_excel(kik_path, sheet_name=0)
# else:
#     raise FileNotFoundError(f"'{kik_path}' 경로에 KIKmix 마스터 파일이 존재하지 않습니다.")

# # 3. KIKmix 데이터 정제 (말소일자가 없는 현존하는 동만 필터링)
# active_kik = kik_df[kik_df['말소일자'].isna()].copy()

# # 4. 매핑용 키 생성 (법정동코드 앞 5자리 = 시군구코드 sggCd, 동리명 = 법정동명 umdNm)
# active_kik['sggCd'] = active_kik['법정동코드'].astype(str).str[:5].astype(int)
# active_kik['umdNm'] = active_kik['동리명']

# # 5. 1:N 매핑 시 데이터 폭증 방지를 위한 중복 제거 (첫 번째 행정동 대표값 선택)
# mapping = active_kik[['sggCd', 'umdNm', '행정동코드', '읍면동명']].dropna(subset=['읍면동명']).drop_duplicates()
# mapping_unique = mapping.drop_duplicates(subset=['sggCd', 'umdNm'], keep='first')

# # 6. house_rent 데이터와 병합 (Left Join)
# print("행정동 데이터 매핑 중...")
# merged_rent = pd.merge(rent_df, mapping_unique, on=['sggCd', 'umdNm'], how='left')

# # 7. 결과 확인
# total_rows = len(merged_rent)
# missing_hjd = merged_rent['행정동코드'].isna().sum()
# print(f"전체 데이터 행 수: {total_rows}")
# print(f"행정동 매핑 실패(결측) 행 수: {missing_hjd}")

# # 8. 결과 파일 저장 (house_rent.csv가 있던 '2_부동산 데이터' 폴더에 저장)
# output_dir = os.path.join(data_dir, "2_부동산 데이터")
# output_filename = os.path.join(output_dir, 'house_rent_with_hdong.csv')
# merged_rent.to_csv(output_filename, index=False, encoding='utf-8-sig')
# print(f"매핑된 파일 저장 완료: {output_filename}")



# 4. 오피스텔 전/월세 데이터 행정동 매핑

# 1. 파일 경로 설정
data_dir = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스"
officetel_csv_path = os.path.join(data_dir, r"2_부동산 데이터\officetel_rent.csv")
kik_path = os.path.join(data_dir, "KIKmix.20260916(말소코드포함).xlsx")

print("데이터 로딩 중...")

# 2. officetel_rent.csv 및 KIKmix 마스터 파일 로드
if os.path.exists(officetel_csv_path):
    officetel_df = pd.read_csv(officetel_csv_path, low_memory=False)
else:
    raise FileNotFoundError(f"'{officetel_csv_path}' 경로에 officetel_rent.csv 파일이 존재하지 않습니다.")

if os.path.exists(kik_path):
    kik_df = pd.read_excel(kik_path, sheet_name=0)
else:
    raise FileNotFoundError(f"'{kik_path}' 경로에 KIKmix 마스터 파일이 존재하지 않습니다.")

# 3. KIKmix 데이터 정제 (말소일자가 없는 현존하는 동만 필터링)
active_kik = kik_df[kik_df['말소일자'].isna()].copy()

# 4. 매핑용 키 생성 (법정동코드 앞 5자리 = 시군구코드 sggCd, 동리명 = 법정동명 umdNm)
active_kik['sggCd'] = active_kik['법정동코드'].astype(str).str[:5].astype(int)
active_kik['umdNm'] = active_kik['동리명']

# 5. 1:N 매핑 시 데이터 폭증 방지를 위한 중복 제거 (첫 번째 행정동 대표값 선택)
mapping = active_kik[['sggCd', 'umdNm', '행정동코드', '읍면동명']].dropna(subset=['읍면동명']).drop_duplicates()
mapping_unique = mapping.drop_duplicates(subset=['sggCd', 'umdNm'], keep='first')

# 6. officetel_rent 데이터와 병합 (Left Join)
print("행정동 데이터 매핑 중...")
merged_officetel = pd.merge(officetel_df, mapping_unique, on=['sggCd', 'umdNm'], how='left')

# 7. 결과 확인
total_rows = len(merged_officetel)
missing_hjd = merged_officetel['행정동코드'].isna().sum()
print(f"전체 데이터 행 수: {total_rows}")
print(f"행정동 매핑 실패(결측) 행 수: {missing_hjd}")

# 8. 결과 파일 저장 ('2_부동산 데이터' 폴더에 저장)
output_dir = os.path.join(data_dir, "2_부동산 데이터")
output_filename = os.path.join(output_dir, 'officetel_rent_with_hdong.csv')
merged_officetel.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"매핑된 파일 저장 완료: {output_filename}")

