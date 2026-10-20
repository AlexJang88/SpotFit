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


# 2. 연립/다세대(빌라) 매매 데이터 행정동 매핑

# 파일 경로 설정 (경로에 맞게 수정하여 사용해 주세요)
villa_sale_path = (
    r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\2_부동산"
    r" 데이터\villa_sale.csv"
)
kikmix_path = (
    r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\KIKmix.20260916(말소코드포함).xlsx"
)

# 1. 파일 불러오기
print("파일을 불러오는 중입니다...")
df_villa = pd.read_csv(villa_sale_path, encoding="utf-8-sig")
df_kik = pd.read_excel(kikmix_path, sheet_name="KIKmix")

# 2. 말소된 동 제외 (말소일자가 NaN인 현존하는 동만 추출)
df_kik_active = df_kik[df_kik["말소일자"].isna()].copy()

# 3. 매핑 키 포맷 맞추기 (시군구코드 5자리, 법정동코드 10자리)
df_villa["sggCd_str"] = df_villa["sggCd"].astype(str).str.zfill(5)
df_kik_active["법정동코드_str"] = (
    df_kik_active["법정동코드"].astype(str).str.zfill(10)
)
df_kik_active["sggCd_prefix"] = df_kik_active["법정동코드_str"].str[:5]

# 중복 제거 및 필요한 매핑 컬럼 추출 ('동리명' = 법정동명, '읍면동명' = 행정동명)
mapping_df = df_kik_active[
    ["sggCd_prefix", "동리명", "행정동코드", "읍면동명"]
].drop_duplicates(subset=["sggCd_prefix", "동리명"])

# 4. 시군구코드(sggCd)와 법정동명(umdNm)을 기준으로 병합 (Left Join)
df_merged = pd.merge(
    df_villa,
    mapping_df,
    left_on=["sggCd_str", "umdNm"],
    right_on=["sggCd_prefix", "동리명"],
    how="left",
)

# 5. 컬럼명 정돈 및 임시 컬럼 삭제
df_merged = df_merged.rename(columns={"읍면동명": "행정동명"})
df_merged = df_merged.drop(
    columns=["sggCd_str", "sggCd_prefix", "동리명"], errors="ignore"
)

# 6. 결과 저장
output_dir = os.path.dirname(villa_sale_path)
output_path = os.path.join(output_dir, "villa_sale_with_hdong.csv")

df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"매핑이 완료되었습니다! 저장된 파일 경로: {output_path}")
print(f"총 데이터 행 수: {len(df_merged)}")
print(
    df_merged[
        [
            "sggCd",
            "umdNm",
            "jibun",
            "mhouseNm",
            "행정동코드",
            "행정동명",
        ]
    ].head(5)
)

