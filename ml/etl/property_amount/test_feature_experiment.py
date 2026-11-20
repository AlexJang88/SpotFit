import os
import gc
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 외부(원격) MLflow 서버 주소 설정
MLFLOW_TRACKING_URI = "http://192.168.0.3:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run_test_experiment():
    csv_path = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw\아파트_매매.csv"
    model_save_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\ml\etl\property_amount\saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n==============================================")
    print(f"[아파트 매매 테스트] 행정동별 평당금액 예측 실험 시작")
    print(f"대상 파일 경로: {csv_path}")
    print(f"==============================================")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"지정된 경로에 데이터 파일이 존재하지 않습니다: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 2. 타깃 및 면적 컬럼 확인
    target_col = '거래금액(만원)' if '거래금액(만원)' in df.columns else ('물건금액(만원)' if '물건금액(만원)' in df.columns else '보증금(만원)')
    area_sqm_col = '전용면적(㎡)' if '전용면적(㎡)' in df.columns else '임대면적(㎡)'
    
    # 결측치 정제
    df = df.dropna(subset=[target_col, area_sqm_col, '자치구', '행정동'])
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[area_sqm_col] = pd.to_numeric(df[area_sqm_col], errors='coerce')
    df = df[(df[target_col] > 0) & (df[area_sqm_col] > 0)]
    
    # 파생 컬럼 생성 (전용면적(평) 및 평당금액)
    df['전용면적(평)'] = df[area_sqm_col] / 3.3058
    df['평당금액'] = df[target_col] / df['전용면적(평)']
    
    # 지정된 컬럼 정제
    df['계약연도'] = pd.to_numeric(df['계약연도'], errors='coerce').fillna(2024) if '계약연도' in df.columns else 2024
    df['층'] = pd.to_numeric(df['층'], errors='coerce').fillna(1) if '층' in df.columns else 1
    df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce').fillna(2000) if '건축년도' in df.columns else 2000
    df['건물 나이(년)'] = pd.to_numeric(df['건물 나이(년)'], errors='coerce').fillna(0) if '건물 나이(년)' in df.columns else 0

    # 3. 요청하신 6개 컬럼을 1개씩 순차 누적하는 Step 정의
    feature_steps = {
        "Step_1": ['계약연도'],
        "Step_2": ['계약연도', area_sqm_col],
        "Step_3": ['계약연도', area_sqm_col, '전용면적(평)'],
        "Step_4": ['계약연도', area_sqm_col, '전용면적(평)', '층'],
        "Step_5": ['계약연도', area_sqm_col, '전용면적(평)', '층', '건축년도'],
        "Step_6": ['계약연도', area_sqm_col, '전용면적(평)', '층', '건축년도', '건물 나이(년)']
    }
    
    y = df['평당금액']
    
    # 4. MLflow Experiment 지정 (아파트 매매 1개 실험)
    experiment_name = "apt-trade-dong-test"
    mlflow.set_experiment(experiment_name)
    
    # 5. 4가지 머신러닝 기법 정의 (hgb 추가)
    models = {
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, learning_rate=0.1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # 6. 순차 누적 피처 실험 및 모델 학습 루프
    for step_name, features in feature_steps.items():
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            continue
            
        X = df[valid_features].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            run_name_str = f"apt_trade_{step_name}_{model_name}"
            with mlflow.start_run(run_name=run_name_str):
                # 요구사항에 따른 태그 및 파라미터 로깅
                mlflow.set_tag("property_type", "apt-trade")
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("features_count", len(valid_features))
                mlflow.set_tag("features", ", ".join(valid_features))
                
                mlflow.log_params({
                    "property_type": "apt-trade",
                    "model_name": model_name,
                    "step_name": step_name,
                    "features_count": len(valid_features),
                    "features_list": valid_features
                })
                
                # 학습 및 예측
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 평가 지표 산출 (MAE, R2)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                
                # 모델 압축 저장 및 아티팩트 업로드
                model_filename = f"apt_trade_{step_name}_{model_name}.pkl"
                model_local_path = os.path.join(model_save_dir, model_filename)
                
                try:
                    joblib.dump(model, model_local_path, compress=3)
                    mlflow.log_artifact(model_local_path, artifact_path="model_files")
                    print(f"[APT-TRADE] [{step_name} ({len(valid_features)}개 피처)] [{model_name}] -> MAE: {mae:,.2f}, R2: {r2:.4f} (성공)")
                except Exception as e:
                    print(f"[APT-TRADE] [{step_name}] [{model_name}] -> MAE: {mae:,.2f}, R2: {r2:.4f} (⚠️ 업로드 경고: {e})")
                
                del model, y_pred
        gc.collect()

if __name__ == "__main__":
    run_test_experiment()
    print("\n>>> 아파트 매매 테스트 실험 및 MLflow 저장이 완료되었습니다!")

