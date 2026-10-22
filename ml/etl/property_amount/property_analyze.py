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



# # 4. 오피스텔 전/월세 데이터 행정동 매핑

# # 1. 파일 경로 설정
# data_dir = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스"
# officetel_csv_path = os.path.join(data_dir, r"2_부동산 데이터\officetel_rent.csv")
# kik_path = os.path.join(data_dir, "KIKmix.20260916(말소코드포함).xlsx")

# print("데이터 로딩 중...")

# # 2. officetel_rent.csv 및 KIKmix 마스터 파일 로드
# if os.path.exists(officetel_csv_path):
#     officetel_df = pd.read_csv(officetel_csv_path, low_memory=False)
# else:
#     raise FileNotFoundError(f"'{officetel_csv_path}' 경로에 officetel_rent.csv 파일이 존재하지 않습니다.")

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

# # 6. officetel_rent 데이터와 병합 (Left Join)
# print("행정동 데이터 매핑 중...")
# merged_officetel = pd.merge(officetel_df, mapping_unique, on=['sggCd', 'umdNm'], how='left')

# # 7. 결과 확인
# total_rows = len(merged_officetel)
# missing_hjd = merged_officetel['행정동코드'].isna().sum()
# print(f"전체 데이터 행 수: {total_rows}")
# print(f"행정동 매핑 실패(결측) 행 수: {missing_hjd}")

# # 8. 결과 파일 저장 ('2_부동산 데이터' 폴더에 저장)
# output_dir = os.path.join(data_dir, "2_부동산 데이터")
# output_filename = os.path.join(output_dir, 'officetel_rent_with_hdong.csv')
# merged_officetel.to_csv(output_filename, index=False, encoding='utf-8-sig')
# print(f"매핑된 파일 저장 완료: {output_filename}")




# 3-1. 단독/다가구 전/월세 데이터 지번 매핑 (원본파일에 지번 정보 없음) #############

import pandas as pd
import os
import io
import re

# # 1. 파일 경로 설정
# data_dir = r"C:\Users\M\Desktop\final_project\1_사전작업\3_상권 추천 서비스\2_부동산 데이터"
# rent_file_path = os.path.join(data_dir, "house_rent_with_hdong.csv") 
# building_file_path = os.path.join(data_dir, "서울시_건축물대장_표제부")
# if not os.path.exists(building_file_path):
#     if os.path.exists(building_file_path + ".csv"):
#         building_file_path = building_file_path + ".csv"

# print("데이터 로딩 중...")
# rent_df = pd.read_csv(rent_file_path, low_memory=False)

# # 건축물대장 안전 로드
# with open(building_file_path, 'rb') as f:
#     raw_data = f.read()

# decoded_data = raw_data.decode('cp949', errors='replace')
# building_df = pd.read_csv(
#     io.StringIO(decoded_data), 
#     low_memory=False, 
#     sep=',', 
#     on_bad_lines='skip'
# )

# print("건축물대장 키 가공 및 엄격한 지번 정제 중...")

# # 2. 키 및 지번 정제 (오염 방지)
# building_df['buildYear'] = building_df['사용승인일자'].astype(str).str[:4]
# building_df['buildYear'] = pd.to_numeric(building_df['buildYear'], errors='coerce')
# rent_df['buildYear'] = pd.to_numeric(rent_df['buildYear'], errors='coerce')

# building_df['totalFloorAr_round'] = pd.to_numeric(building_df['연면적'], errors='coerce').round(0)
# rent_df['totalFloorAr_round'] = pd.to_numeric(rent_df['totalFloorAr'], errors='coerce').round(0)

# building_df['umdNm'] = building_df['법정동코드명'].str.strip()
# rent_df['umdNm'] = rent_df['umdNm'].astype(str).str.strip()

# # 엄격한 지번 생성 및 유효성 검증 함수
# def make_clean_jibun(row):
#     try:
#         main_val = row['주지번']
#         sub_val = row['부지번']
        
#         # 주지번이 없거나 NaN이면 빈 값
#         if pd.isna(main_val) or str(main_val).strip() == '' or str(main_val) == '0':
#             return None
            
#         main_int = int(float(main_val))
        
#         # 🚨 핵심 방어선: 주지번이 1900~2030 사이의 연도 형태라면 지번이 아님! (잘못 매칭된 데이터 차단)
#         if 1900 <= main_int <= 2030:
#             return None
            
#         main_str = str(main_int)
        
#         # 부지번 처리
#         if pd.notna(sub_val) and str(sub_val).strip() != '' and str(sub_val) != '0' and str(sub_val) != '0000':
#             sub_int = int(float(sub_val))
#             if sub_int > 0:
#                 return f"{main_str}-{sub_int}"
                
#         return main_str
#     except:
#         return None

# building_df['parsed_jibun'] = building_df.apply(make_clean_jibun, axis=1)

# # 유효한 지번이 확실히 존재하는 데이터만 필터링
# building_clean = building_df.dropna(subset=['parsed_jibun']).copy()

# # 3. 오직 '1단계 정밀 매칭'만 수행 (확실한 데이터만 허용)
# print("정밀 매칭(법정동 + 건축년도 + 반올림 연면적) 수행 중...")

# # 동일 조건 내 중복이 발생할 경우 정확성을 위해 고유한 매칭만 추출하거나 첫 번째 유효값 사용
# bldg_subset = building_clean[
#     ['umdNm', 'buildYear', 'totalFloorAr_round', 'parsed_jibun']
# ].drop_duplicates(subset=['umdNm', 'buildYear', 'totalFloorAr_round'])

# merged_df = pd.merge(
#     rent_df,
#     bldg_subset,
#     on=['umdNm', 'buildYear', 'totalFloorAr_round'],
#     how='left',
#     suffixes=('', '_bldg')
# )

# # 4. 최종 결과 검증 및 저장
# total_rows = len(merged_df)
# valid_matched_count = merged_df['parsed_jibun'].notna().sum()

# print(f"\n[최종 정제 및 병합 결과]")
# print(f"- 전체 전월세 데이터 행 수: {total_rows:,}건")
# print(f"- 엄격하게 검증되어 올바른 지번이 매칭된 건수: {valid_matched_count:,}건 (매칭률: {(valid_matched_count/total_rows)*100:.2f}%)")

# # 결과 파일 저장
# output_filename = os.path.join(data_dir, 'house_rent_with_building_matched_clean.csv')
# merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
# print(f"클렌징 완료된 정밀 파일 저장: {output_filename}")



# 6. 연도별 전/월세 데이터 통합 (아파트, 연립/다세대, 단독/다가구, 오피스텔)

import os
import pandas as pd

# 1. 전/월세 데이터 파일 경로 설정
rent_path = r"C:\Users\M\Desktop\final_project\2_데이터분석\1_부동산 데이터\2_부동산 전월세 데이터"

# 2. 통합할 전/월세 연도별 파일 목록 리스트
rent_files = [
    "서울특별시_전월세가_2021.csv",
    "서울특별시_전월세가_2022.csv",
    "서울특별시_전월세가_2023.csv",
    "서울특별시_전월세가_2024.csv",
    "서울특별시_전월세가_2025.csv",
    "서울특별시_전월세가_2026.csv"
]

rent_dfs = []

# 3. 각 파일을 순회하며 로드 (인코딩 자동 예외 처리 적용)
for rent_file in rent_files:
    file_path = os.path.join(rent_path, rent_file)
    if os.path.exists(file_path):
        df = None
        # utf-8을 먼저 시도하고, 실패하면 cp949/euc-kr 시도
        for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"전/월세 데이터 로드 성공 ({enc}): {rent_file}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is not None:
            rent_dfs.append(df)
        else:
            print(f"파일 인코딩을 읽지 못했습니다: {rent_file}")
    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")

# 4. 데이터프레임 병합 및 저장
if rent_dfs:
    total_rent_df = pd.concat(rent_dfs, ignore_index=True)
    
    # 통합된 데이터를 새로운 CSV 파일로 저장
    output_filename = "total_rent_data.csv"
    total_rent_output = os.path.join(rent_path, output_filename)
    
    total_rent_df.to_csv(total_rent_output, index=False, encoding="utf-8-sig")
    print(f"\n[통합 완료] '{output_filename}' 파일로 저장되었습니다.")
    print(f"저장 경로: {total_rent_output}")
    print(f"총 통합 데이터 행 수: {len(total_rent_df):,}")
else:
    print("\n통합할 전/월세 데이터가 없습니다. 파일 경로와 파일명을 다시 확인해주세요.")

