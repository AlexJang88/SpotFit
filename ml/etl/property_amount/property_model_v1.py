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

def train_and_evaluate(data_path, model_type, tracking_uri):
    # 1. 데이터 로드 및 SHA-256 확인
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    data_sha256 = calculate_sha256(data_path)
    print(f"데이터 파일: {data_path}")
    print(f"데이터 SHA-256: {data_sha256}")
    
    df = pd.read_csv(data_path)
    print(f"데이터 로드 완료 (행 수: {len(df):,})")
    
    # 2. 타깃 컬럼 설정 (매매: dealAmount, 전월세: 보증금(만원) 등 완벽 대응)
    possible_targets = ['보증금(만원)', 'dealAmount', 'deposit', 'rentAmount', 'price', '보증금액', '보증금']
    target_col = None
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
            
    # 지정된 후보 중에 없다면 금액/가격 관련 키워드가 포함된 컬럼을 자동 탐색
    if target_col is None:
        amount_cols = [c for c in df.columns if 'amount' in c.lower() or 'price' in c.lower() or '금액' in c or '보증' in c]
        if amount_cols:
            target_col = amount_cols[0]
            
    if target_col is None or target_col not in df.columns:
        raise ValueError(f"적절한 타깃(가격/보증금) 컬럼을 찾을 수 없습니다. 전체 컬럼 목록: {df.columns.tolist()}")
        
    print(f"선택된 타깃 컬럼: {target_col}")
    
    # 쉼표(,)가 포함된 문자열 가격 데이터를 안전하게 숫자형(float)으로 변환
    df[target_col] = pd.to_numeric(
        df[target_col].astype(str).str.replace(',', '', regex=False), 
        errors='coerce'
    )
    
    # 결측치(NaN) 제거
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < initial_len:
        print(f"[알림] 유효하지 않은 타깃 데이터 {initial_len - len(df):,}건을 제외했습니다.")
        
    y = df[target_col].astype(float)
    
    # 3. 행정동 코드 및 이름 컬럼명 탐색 (데이터에 따라 이름이 다를 수 있으므로 유연하게 대처)
    dong_code_cols = [c for c in df.columns if '행정동_코드' in c or 'code' in c.lower() or 'adm_cd' in c.lower()]
    dong_name_cols = [c for c in df.columns if '행정동명' in c or 'name' in c.lower() or 'dong' in c.lower()]
    
    print(f"탐지된 행정동 관련 컬럼 - 코드: {dong_code_cols}, 이름: {dong_name_cols}")
    
    # 4. 학습용 특징(Features) 선정 및 불필요한 문자열 컬럼(연결방식, 지번 일치 등) 제외
    # - 타깃 컬럼 제외
    # - '연결방식', '주소', '지번' 등 불필요한 텍스트 컬럼 명시적 배제
    exclude_cols = [target_col, '연결방식', '지번', '주소', '본번', '부번', 'Unnamed: 0']
    
    # 수치형 컬럼이면서 제외 대상이 아닌 것들만 학습 특징으로 선택
    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        # 수치형 데이터이면서 결측치가 지나치게 많지 않은 컬럼 선택
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
            
    print(f"학습에 사용되는 수치형 특징 컬럼 수: {len(feature_cols)}개")
    print(f"선정된 특징 컬럼 목록: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    
    # 5. MLflow 설정 및 Experiment 지정
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("house-price-forecast-comparison")
    
    # 6. 교차 검증(CV) 설정 (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    
    # 모델 선택
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'rf':
        # 전월세 데이터처럼 거대한 데이터셋에서는 max_depth와 n_estimators를 제한하여 과도한 용량 증가 방지
        model = RandomForestRegressor(
            n_estimators=30,      # 트리의 개수를 줄여 모델 용량 대폭 축소
            max_depth=15,         # 트리의 최대 깊이 제한
            min_samples_leaf=10,  # 과적합 방지 및 리프 노드 크기 조절
            random_state=42, 
            n_jobs=-1
        )
    elif model_type == 'hgb':
        model = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다 ('ridge', 'rf', 'hgb').")

    with mlflow.start_run() as run:
        # 필수 태그 및 파라미터 기록
        mlflow.set_tag("owner", "model_developer")
        mlflow.set_tag("task_id", "real_estate_forecast_01")
        mlflow.set_tag("review_status", "candidate")
        
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("dataset_sha256", data_sha256)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_splits", 5)
        mlflow.log_param("random_seed", 42)
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
        
        print(f"\n[최종 교차검증 결과]")
        print(f"Mean MAE: {mean_mae:.4f} (±{std_mae:.4f})")
        print(f"Mean RMSE: {mean_rmse:.4f}")
        print(f"Mean R2: {mean_r2:.4f}")
        
        # 전체 데이터로 최종 모델 재학습 후 MLflow에 모델 아티팩트 로깅
        model.fit(X, y)
    
        # HGB와 Random Forest 등 트리 기반 모델들이 사용하는 내부 클래스를 모두 신뢰하도록 지정
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
    parser = argparse.ArgumentParser(description="Real Estate Price Forecast Training")
    parser.add_argument("--data", type=str, required=True, help="학습 데이터 경로")
    parser.add_argument("--model", type=str, default="hgb", choices=["ridge", "rf", "hgb"], help="학습할 알고리즘 선택")
    parser.add_argument("--tracking-uri", type=str, default="http://192.168.0.3:5000", help="MLflow 서버 URI")
    
    args = parser.parse_args()
    train_and_evaluate(args.data, args.model, args.tracking_uri)

