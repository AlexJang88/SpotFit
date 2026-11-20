import os
import geopandas as gpd
import pandas as pd

def create_seoul_dong_adjacency(shapefile_path):
    """
    서울시 행정동 경계 파일을 읽어 기하학적 접촉(touches/intersects)을 기반으로 
    인접 행정동을 매핑하고 그룹화 테이블을 생성하는 함수
    """
    print("1. 서울시 행정동 경계 파일(GeoJSON) 로드 중...")
    if not os.path.exists(shapefile_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {shapefile_path}")
        return None, None
        
    # GeoJSON 파일 로드
    gdf = gpd.read_file(shapefile_path)
    
    # 면적 및 거리 연산의 정확도를 위해 투영 좌표계(EPSG:5179 - 중부원점)로 변환
    if gdf.crs != "EPSG:5179":
        gdf = gdf.to_crs(epsg=5179)
    
    print("2. 공간 인접 관계(Queen 방식: 경계 또는 꼭짓점 맞닿음) 계산 중...")
    
    # 깃허브 오픈소스(raqoon886/Local_HangJeongDong 등)의 표준 컬럼명 자동 탐색
    code_col = 'adm_cd' if 'adm_cd' in gdf.columns else ('행정동코드' if '행정동코드' in gdf.columns else gdf.columns[0])
    name_col = 'adm_nm' if 'adm_nm' in gdf.columns else ('행정동명' if '행정동명' in gdf.columns else gdf.columns[1])
    
    adjacency_list = []
    
    # 각 행정동 폴리곤이 다른 폴리곤과 맞닿아 있는지(touches) 또는 겹치는지가(intersects) 확인
    for idx, row in gdf.iterrows():
        current_geom = row.geometry
        dong_code = row[code_col]
        dong_name = row[name_col]
        
        # 자기 자신을 제외하고 경계가 맞닿거나 교차하는 다른 동들 찾기
        neighbors = gdf[gdf.geometry.touches(current_geom) | gdf.geometry.intersects(current_geom)]
        neighbors = neighbors[neighbors.index != idx]
        
        neighbor_codes = neighbors[code_col].tolist()
        neighbor_names = neighbors[name_col].tolist()
        
        adjacency_list.append({
            '행정동_코드': dong_code,
            '행정동명': dong_name,
            '인접동_개수': len(neighbors),
            '인접동_코드목록': ", ".join(map(str, neighbor_codes)),
            '인접동_이름목록': ", ".join(map(str, neighbor_names))
        })
        
    # 데이터프레임으로 변환
    result_df = pd.DataFrame(adjacency_list)
    
    # 3. 결과 저장 (입력받은 폴더 경로와 동일한 곳에 저장)
    output_dir = os.path.dirname(shapefile_path)
    output_filename = "seoul_dong_adjacency_groups.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"-> [완료] 인접 그룹화 결과가 저장되었습니다: {output_path}")
    
    # [수정됨] 콤마 뒤에 포맷 지정자(d)를 명시하거나 괄호로 분리
    print(f"-> 총 처리된 행정동 수: {len(result_df):,d}개")
    
    return gdf, result_df

# --- 실행부 ---
if __name__ == "__main__":
    # 지정해주신 파일 경로 설정
    shapefile_path = r"C:\Users\M\Desktop\final_project\2_데이터분석\hangjeongdong_서울특별시.geojson"
    
    # 함수 실행
    gdf, df_result = create_seoul_dong_adjacency(shapefile_path)