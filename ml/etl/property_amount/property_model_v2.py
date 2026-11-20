import os
import hashlib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import mlflow
import mlflow.sklearn

def calculate_sha256(file_path):
    """파일의 SHA-256 해시값을 계산합니다."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def train_and_evaluate_combined(data_paths, model_type, tracking_uri):
    # 1. 여러 개의 데이터 파일을 로드하여 하나로 통합
    dfs = []
    combined_sha256s = []
    
    for path in data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
        sha = calculate_sha256(path)
        combined_sha256s.append(sha)
        print(f"데이터 로드: {path}")
        print(f"  - SHA-256: {sha}")
        df_temp = pd.read_csv(path)
        dfs.append(df_temp)
        
    # 데이터프레임 병합 (행 방향으로 결합)
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[통합 완료] 총 데이터 행 수: {len(df):,}")
    
    # 2. 타깃 컬럼 설정 (아파트/빌라 공통 매매가: dealAmount)
    target_col = 'dealAmount'
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target_col}'이 통합 데이터에 존재하지 않습니다.")
    
    # 쉼표 제거 및 숫자형 변환
    df[target_col] = pd.to_numeric(
        df[target_col].astype(str).str.replace(',', '', regex=False), 
        errors='coerce'
    )
    
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < initial_len:
        print(f"[알림] 유효하지 않은 가격 데이터 {initial_len - len(df):,}건을 제외했습니다.")
        
    y = df[target_col].astype(float)
    
    # 3. 학습 특징(Features) 선정 (공통 수치형 컬럼 추출 및 불필요한 문자열 컬럼 배제)
    exclude_cols = [target_col, '연결방식', '지번', '주소', '본번', '부번', 'Unnamed: 0']
    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
            
    print(f"학습에 사용되는 수치형 특징 컬럼 수: {len(feature_cols)}개")
    print(f"선정된 특징 컬럼 목록: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    
    # 4. MLflow 설정 (통합 모델 전용 Experiment 명시)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("house-price-combined-comparison")
    
    # 5. 교차 검증(CV) 설정 (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    
    # 모델 정의 (Random Forest는 대용량 통합 데이터 업로드 안정성을 위해 경량화 파라미터 적용)
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=30, 
            max_depth=15, 
            min_samples_leaf=10,
            random_state=42, 
            n_jobs=-1
        )
    elif model_type == 'hgb':
        model = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다 ('ridge', 'rf', 'hgb').")

    with mlflow.start_run() as run:
        # 태그 및 파라미터 기록
        mlflow.set_tag("owner", "model_developer")
        mlflow.set_tag("task_id", "real_estate_combined_forecast_02")
        mlflow.set_tag("review_status", "candidate")
        
        mlflow.log_param("data_paths", str(data_paths))
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_splits", 5)
        mlflow.log_param("features_count", len(feature_cols))
        
        # 교차 검증 수행
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            
            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)
            
            print(f"Fold {fold+1} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # 평균 성능 지표 기록
        mean_mae = np.mean(fold_maes)
        mean_rmse = np.mean(fold_rmses)
        mean_r2 = np.mean(fold_r2s)
        std_mae = np.std(fold_maes)
        
        mlflow.log_metric("cv_mean_mae", mean_mae)
        mlflow.log_metric("cv_std_mae", std_mae)
        mlflow.log_metric("cv_mean_rmse", mean_rmse)
        mlflow.log_metric("cv_mean_r2", mean_r2)
        
        print(f"\n[최종 통합 교차검증 결과]")
        print(f"Mean MAE: {mean_mae:.4f} (±{std_mae:.4f})")
        print(f"Mean RMSE: {mean_rmse:.4f}")
        print(f"Mean R2: {mean_r2:.4f}")
        
        # 전체 데이터로 최종 모델 재학습 및 MLflow 로깅 (보안 신뢰 타입 지정)
        model.fit(X, y)
        trusted_types = [
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "sklearn.tree._tree.Tree"
        ]
        mlflow.sklearn.log_model(
            model, 
            "model", 
            skops_trusted_types=trusted_types
        )
        print(f"MLflow Run ID 기록됨: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Real Estate Price Forecast Training")
    parser.add_argument("--data", type=str, nargs='+', required=True, help="학습할 데이터 경로들 (공백으로 구분하여 여러 개 입력)")
    parser.add_argument("--model", type=str, default="hgb", choices=["ridge", "rf", "hgb"], help="학습할 알고리즘 선택")
    parser.add_argument("--tracking-uri", type=str, default="http://192.168.0.3:5000", help="MLflow 서버 URI")
    
    args = parser.parse_args()
    train_and_evaluate_combined(args.data, args.model, args.tracking_uri)
