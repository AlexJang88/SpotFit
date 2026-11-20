import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import mlflow
import mlflow.sklearn

def safe_read_csv(file_path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

def train_v8(data_paths, model_type, tracking_uri):
    dfs = [safe_read_csv(p) for p in data_paths]
    df = pd.concat(dfs, ignore_index=True)
    
    target_col = 'dealAmount'
    df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(float)
    
    drop_cols = [
        target_col, '_source_gu', '_source_month', '_source_page', '_source_total', 'buildYear',
        '연결방식', '지번', '주소', '본번', '부번', 'Unnamed: 0', '_source_service', '_fetched_at_utc', '_source_file'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(0)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("property-model-v8-combined-sales")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    
    if model_type == 'hgb': model = HistGradientBoostingRegressor(random_state=42)
    elif model_type == 'rf': model = RandomForestRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif model_type == 'ridge': model = Ridge(alpha=1.0)
    
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
        print(f"[v8 | {model_type}] Mean MAE: {np.mean(maes):.4f} | Mean R2: {np.mean(r2s):.4f}")
        
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model_v8", skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "sklearn.tree._tree.Tree"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs='+', required=True)
    parser.add_argument("--model", default="hgb", choices=["hgb", "rf", "ridge"])
    parser.add_argument("--tracking-uri", default="http://192.168.0.3:5000")
    args = parser.parse_args()
    train_v8(args.data, args.model, args.tracking_uri)
