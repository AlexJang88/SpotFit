import os
import hashlib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
import mlflow
import mlflow.sklearn

def safe_read_csv(file_path):
    """한글 인코딩 문제를 방지하기 위해 utf-8-sig와 cp949를 순차적으로 시도하여 CSV를 읽습니다."""
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

def train_model(data_paths, model_type, tracking_uri, experiment_name):
    dfs = [safe_read_csv(p) for p in data_paths]
    
    # 메인 거래 데이터와 외부 요인(행정동표/연관도) 데이터 자동 분리
    main_df, ext_dfs = None, []
    for df in dfs:
        if 'dealAmount' in df.columns or '보증금(만원)' in df.columns:
            main_df = df
        else:
            ext_dfs.append(df)
            
    if main_df is None: 
        main_df = dfs[0]
        ext_dfs = dfs[1:]
    
    merged_df = main_df
    for ext_df in ext_dfs:
        # 공통 키('행정동_코드', '행정동', '자치구' 등)를 기준으로 스마트 병합
        common_keys = [c for c in merged_df.columns if c in ext_df.columns]
        if common_keys:
            print(f"공통 키 기준으로 외부 데이터 병합: {common_keys}")
            merged_df = pd.merge(merged_df, ext_df, on=common_keys, how='left')
        else:
            print(f"[알림] 공통 키가 없어 인덱스 기준으로 외부 데이터를 결합합니다.")
            merged_df = pd.concat([merged_df, ext_df.reset_index(drop=True)], axis=1)
            
    target_col = 'dealAmount' if 'dealAmount' in merged_df.columns else '보증금(만원)'
    merged_df[target_col] = pd.to_numeric(merged_df[target_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    merged_df = merged_df.dropna(subset=[target_col])
    y = merged_df[target_col].astype(float)
    
    exclude_cols = [target_col, '연결방식', '지번', '주소', '본번', '부번', 'Unnamed: 0', '행정동', '자치구']
    feature_cols = [c for c in merged_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged_df[c])]
    X = merged_df[feature_cols].fillna(0)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    
    if model_type == 'hgb': 
        model = HistGradientBoostingRegressor(random_state=42)
    elif model_type == 'rf': 
        model = RandomForestRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif model_type == 'et': 
        model = ExtraTreesRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError("지원하는 모델 타입: 'hgb', 'rf', 'et'")
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("features_count", len(feature_cols))
        
        for tr_idx, val_idx in kf.split(X, y):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict(X.iloc[val_idx])
            maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
            r2s.append(r2_score(y.iloc[val_idx], preds))
            
        mlflow.log_metric("cv_mean_mae", np.mean(maes))
        mlflow.log_metric("cv_mean_rmse", np.mean(rmses))
        mlflow.log_metric("cv_mean_r2", np.mean(r2s))
        print(f"[{experiment_name} | {model_type}] Mean MAE: {np.mean(maes):.4f} | Mean R2: {np.mean(r2s):.4f}")
        
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "sklearn.tree._tree.Tree"])
        print(f"MLflow Run ID 기록됨: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs='+', required=True)
    parser.add_argument("--model", default="hgb", choices=["hgb", "rf", "et"])
    parser.add_argument("--tracking-uri", default="http://192.168.0.3:5000")
    parser.add_argument("--exp-name", default="property-external-v3-individual", help="MLflow Experiment Name")
    args = parser.parse_args()
    train_model(args.data, args.model, args.tracking_uri, args.exp_name)

