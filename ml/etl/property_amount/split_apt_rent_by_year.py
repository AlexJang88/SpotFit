import pandas as pd
import os

# 1. 파일 경로 설정 (원래 원본 전/월세 파일 경로)
input_path = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw\아파트_전월세.csv"
output_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw"

print(f"데이터 로드 중: {input_path}")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"지정된 경로에 파일이 존재하지 않습니다: {input_path}")

# 2. 데이터 로드
df = pd.read_csv(input_path)

# 3. 계약연도 컬럼 확인 및 연도별 분할 저장
if '계약연도' not in df.columns:
    raise KeyError("데이터에 '계약연도' 컬럼이 존재하지 않습니다.")

years = df['계약연도'].dropna().unique()

for year in sorted(years):
    df_year = df[df['계약연도'] == year]
    filename = f"아파트_전월세_{int(year)}.csv"
    save_path = os.path.join(output_dir, filename)
    
    df_year.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {filename} (총 {len(df_year):,}건)")

print("\n>>> 아파트 전/월세 데이터의 2021년~2026년 연도별 파일 분할이 완료되었습니다!")
