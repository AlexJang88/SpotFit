import pandas as pd
import os

# 1. 파일 경로 설정
input_path = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw\아파트_매매.csv"
output_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw"

# 2. 데이터 로드
df = pd.read_csv(input_path)

# 3. 연도별(계약연도)로 분할하여 저장
years = df['계약연도'].dropna().unique()

for year in sorted(years):
    df_year = df[df['계약연도'] == year]
    filename = f"아파트_매매_{int(year)}.csv"
    save_path = os.path.join(output_dir, filename)
    
    df_year.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {filename} (총 {len(df_year):,}건)")

print("\n>>> 2021년부터 2026년까지 연도별 파일 분할이 완료되었습니다!")

