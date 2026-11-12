import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import mlflow
import mlflow.sklearn

def safe_read_csv(file_path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

def run_feature_experiment(data_path, experiment_name, tracking_uri):
    df = safe_read_csv(data_path)
    
    target_col = '거래금액(만원)'
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target_col}'이 존재하지 않습니다.")
        
    df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(float)
    
    # [수정] '주택유형'을 제외 컬럼에서 제거하여 피처 분석에 포함되도록 함
    # (문자열인 경우 모델 학습을 위해 자동으로 숫자형/원핫인코딩 처리 또는 수치형만 추출)
    exclude_cols = [
        target_col, '평당 금액(만원)', '지번', '등기일', '매수자', '매도자', 
        '행정동', '법정동', '자치구', '단지명', '건물명', '동', '계약일', '거래유형'
    ]
    
    # 만약 '주택유형'이 범주형(문자형) 문자열이라면 원핫 인코딩 처리 수행
    if '주택유형' in df.columns and not pd.api.types.is_numeric_dtype(df['주택유형']):
        df = pd.get_dummies(df, columns=['주택유형'], drop_first=True)
    
    # 후보 수치형 컬럼 추출 (원핫인코딩된 주택유형 컬럼들도 자동으로 포함됨)
    candidate_features = [
        c for c in df.columns 
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    print(f"=== [실험 시작: {experiment_name}] ===")
    print(f"후보 피처 목록 ({len(candidate_features)}개): {candidate_features}\n")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # 비교할 3가지 모델 알고리즘 정의
    models = {
        'hgb': HistGradientBoostingRegressor(random_state=42),
        'rf': RandomForestRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'et': ExtraTreesRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    }
    
    current_features = []
    
    for i, feature in enumerate(candidate_features):
        current_features.append(feature)
        X = df[current_features].fillna(0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"--- [Step {i+1}] 누적 피처: {current_features} (총 {len(current_features)}개) ---")
        
        # 각 피처 누적 단계마다 3개 모델 각각 실행 및 비교
        for model_name, model in models.items():
            maes, rmses, r2s = [], [], []
            
            with mlflow.start_run(run_name=f"step_{i+1}_{model_name}_with_{feature}"):
                mlflow.log_param("features_count", len(current_features))
                mlflow.log_param("features_list", ", ".join(current_features))
                mlflow.log_param("added_feature", feature)
                mlflow.log_param("model_type", model_name)
                
                for tr_idx, val_idx in kf.split(X, y):
                    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    preds = model.predict(X.iloc[val_idx])
                    maes.append(mean_absolute_error(y.iloc[val_idx], preds))
                    rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
                    r2s.append(r2_score(y.iloc[val_idx], preds))
                    
                mean_mae = np.mean(maes)
                mean_rmse = np.mean(rmses)
                mean_r2 = np.mean(r2s)
                
                mlflow.log_metric("cv_mean_mae", mean_mae)
                mlflow.log_metric("cv_mean_rmse", mean_rmse)
                mlflow.log_metric("cv_mean_r2", mean_r2)
                
                print(f"  [{model_name.upper()}] MAE: {mean_mae:,.2f} 만원 | R2: {mean_r2:.4f}")
                
                model.fit(X, y)
                mlflow.sklearn.log_model(
                    model, 
                    f"model_step_{i+1}_{model_name}",
                    skops_trusted_types=[
                        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", 
                        "sklearn.tree._tree.Tree"
                    ]
                )
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="데이터 파일 경로")
    parser.add_argument("--exp-name", required=True, help="MLflow Experiment 이름")
    parser.add_argument("--tracking-uri", default="http://192.168.0.3:5000")
    args = parser.parse_args()
    
    run_feature_experiment(args.data, args.exp_name, args.tracking_uri)
