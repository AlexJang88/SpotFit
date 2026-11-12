import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

# 1. 외부(원격) MLflow 서버 주소 설정
MLFLOW_TRACKING_URI = "http://192.168.0.3:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run_rent_experiment(property_type, csv_filename, experiment_name):
    """
    아파트 또는 빌라 전/월세 데이터에 대한 1개씩 순차 누적 피처 실험 (모델 압축 및 안정적 로깅)
    """
    base_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw"
    csv_path = os.path.join(base_dir, csv_filename)
    
    model_save_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\ml\etl\property_amount\saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n==============================================")
    print(f"[{property_type.upper()} 전/월세] 데이터 로드 및 순차 누적 실험 시작")
    print(f"대상 파일 경로: {csv_path}")
    print(f"MLflow 실험 이름: {experiment_name}")
    print(f"==============================================")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"지정된 경로에 데이터 파일이 존재하지 않습니다: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 2. 타깃(보증금) 및 필수 컬럼 결측치 정제
    df = df.dropna(subset=['보증금(만원)', '임대면적(㎡)', '자치구', '전월세구분'])
    df['계약연도'] = pd.to_numeric(df['계약연도'], errors='coerce')
    df['건물 나이(년)'] = pd.to_numeric(df['건물 나이(년)'], errors='coerce').fillna(df['건물 나이(년)'].median())
    df['층'] = pd.to_numeric(df['층'], errors='coerce').fillna(df['층'].median())
    df['월세(만원)'] = pd.to_numeric(df['월세(만원)'], errors='coerce').fillna(0)
    
    # 3. 범주형 인코딩
    for col in ['자치구', '행정동', '전월세구분']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # 4. 피처를 1개씩 순차 누적하는 Step 정의
    feature_steps = {
        "Step_1": ['임대면적(㎡)'],
        "Step_2": ['임대면적(㎡)', '전월세구분'],
        "Step_3": ['임대면적(㎡)', '전월세구분', '자치구'],
        "Step_4": ['임대면적(㎡)', '전월세구분', '자치구', '행정동'],
        "Step_5": ['임대면적(㎡)', '전월세구분', '자치구', '행정동', '층'],
        "Step_6": ['임대면적(㎡)', '전월세구분', '자치구', '행정동', '층', '건물 나이(년)'],
        "Step_7": ['임대면적(㎡)', '전월세구분', '자치구', '행정동', '층', '건물 나이(년)', '월세(만원)'],
        "Step_8": ['임대면적(㎡)', '전월세구분', '자치구', '행정동', '층', '건물 나이(년)', '월세(만원)', '계약연도']
    }
    
    y = df['보증금(만원)']
    
    # 5. MLflow Experiment 지정
    mlflow.set_experiment(experiment_name)
    
    # 6. 3가지 머신러닝 기법 정의
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, learning_rate=0.1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # 7. 순차 누적 피처 실험 및 모델 학습 루프
    for step_name, features in feature_steps.items():
        X = df[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{property_type}_{step_name}_{model_name}"):
                # 태그 및 파라미터 로깅
                mlflow.set_tag("features_count", len(features))
                mlflow.set_tag("features", ", ".join(features))
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("property_type", property_type)
                
                mlflow.log_params({
                    "step_name": step_name,
                    "features_count": len(features),
                    "features_list": features,
                    "model_name": model_name
                })
                
                # 모델 학습 및 예측
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 평가 지표 산출 (MAE, R2)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                
                # [개선된 모델 저장 방식] 압축(compress=3)을 적용하여 파일 용량을 최소화한 뒤 업로드 시도
                model_filename = f"{property_type}_{step_name}_{model_name}.pkl"
                model_local_path = os.path.join(model_save_dir, model_filename)
                
                try:
                    # 압축률을 높여 용량 축소 (multipart upload 이슈 회피 목적)
                    joblib.dump(model, model_local_path, compress=3)
                    mlflow.log_artifact(model_local_path, artifact_path="model_files")
                    print(f"[{property_type.upper()}] [{step_name} ({len(features)}개 피처)] [{model_name}] -> MAE: {mae:,.2f}, R2: {r2:.4f} (모델 저장 성공)")
                except Exception as e:
                    print(f"[{property_type.upper()}] [{step_name}] [{model_name}] -> MAE: {mae:,.2f}, R2: {r2:.4f} (⚠️ 서버 업로드 실패, 메트릭만 기록됨: {e})")

def main():
    # 1. 아파트 전/월세 실험 실행 (v2)
    run_rent_experiment(
        property_type="apt",
        csv_filename="아파트_전월세.csv",
        experiment_name="apt-rent-analysis-v2"
    )
    
    # 2. 빌라 전/월세 실험 실행 (v2)
    run_rent_experiment(
        property_type="villa",
        csv_filename="빌라_전월세.csv",
        experiment_name="villa-rent-analysis-v2"
    )
    print("\n>>> 모든 전/월세 순차 누적 피처 실험이 완료되었습니다!")

if __name__ == "__main__":
    main()
    
