import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
import mlflow
import mlflow.sklearn

def safe_read_csv(file_path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

def train_v11(data_paths, model_type, tracking_uri):
    ext_path = data_paths[-1]
    trade_paths = data_paths[:-1]
    
    ext_df = safe_read_csv(ext_path)
    trade_dfs = [safe_read_csv(p) for p in trade_paths]
    df = pd.concat(trade_dfs, ignore_index=True)
    
    common_keys = [c for c in df.columns if c in ext_df.columns]
    if common_keys: merged_df = pd.merge(df, ext_df, on=common_keys, how='left')
    else: merged_df = pd.concat([df, ext_df.reset_index(drop=True)], axis=1)
        
    target_col = 'dealAmount'
    merged_df[target_col] = pd.to_numeric(merged_df[target_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    merged_df = merged_df.dropna(subset=[target_col])
    y = merged_df[target_col].astype(float)
    
    drop_cols = [
        target_col, '_source_gu', '_source_month', '_source_page', '_source_total', 'buildYear',
        '연결방식', '지번', '주소', '본번', '부번', 'Unnamed: 0', '_source_service', '_fetched_at_utc', '_source_file',
        '행정동', '자치구'
    ]
    feature_cols = [c for c in merged_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(merged_df[c])]
    X = merged_df[feature_cols].fillna(0)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("property-model-v11-external-combined-sales")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    
    if model_type == 'hgb': model = HistGradientBoostingRegressor(random_state=42)
    elif model_type == 'rf': model = RandomForestRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif model_type == 'et': model = ExtraTreesRegressor(n_estimators=40, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    
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
        print(f"[v11 | {model_type}] Mean MAE: {np.mean(maes):.4f} | Mean R2: {np.mean(r2s):.4f}")
        
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model_v11", skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "sklearn.tree._tree.Tree"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs='+', required=True)
    parser.add_argument("--model", default="hgb", choices=["hgb", "rf", "et"])
    parser.add_argument("--tracking-uri", default="http://192.168.0.3:5000")
    args = parser.parse_args()
    train_v11(args.data, args.model, args.tracking_uri)
