import os
import joblib
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def train_and_save_rf_models():
    base_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw"
    model_save_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\models"
    os.makedirs(model_save_dir, exist_ok=True)
    
    mlflow.set_tracking_uri("http://192.168.0.3:5000/")
    redev_file = os.path.join(base_dir, "★(26년 6월기준) 서울시 정비사업 추진현황.xlsx")
    
    # 정비사업 데이터 로드 및 집계
    print("1. 서울시 정비사업 추진현황 데이터 로드 중...")
    if os.path.exists(redev_file):
        df_redev = pd.read_excel(redev_file, header=3)
        dong_col = next((c for c in df_redev.columns if '행정동명' in str(c)), None)
        if dong_col:
            df_redev_agg = df_redev.groupby(dong_col).size().reset_index(name='재개발_정비사업_구역수')
            df_redev_agg.rename(columns={dong_col: '행정동'}, inplace=True)
        else:
            df_redev_agg = pd.DataFrame(columns=['행정동', '재개발_정비사업_구역수'])
    else:
        df_redev_agg = pd.DataFrame(columns=['행정동', '재개발_정비사업_구역수'])

    ext_features = ['거리_to_GBD(km)', '거리_to_CBD(km)', '거리_to_YBD(km)', 
                    '접근성_score_GBD', '접근성_score_CBD', '접근성_score_YBD', '재개발_정비사업_구역수']

    # ==========================================
    # [Case 1] 아파트 매매 RandomForest 모델 학습 및 .joblib 저장
    # ==========================================
    print("\n--------------------------------------------------")
    print("🚀 [아파트 매매] 최종 RandomForest 모델 학습 및 .joblib 저장")
    print("--------------------------------------------------")
    mlflow.set_experiment("Feature_Improvement_Trade_All")
    
    trade_file = os.path.join(base_dir, "아파트_매매.csv")
    if os.path.exists(trade_file):
        df_trade = pd.read_csv(trade_file)
        t_col = next((c for c in ['물건금액(만원)', '거래금액(만원)', '거래금액'] if c in df_trade.columns), None)
        df_trade = df_trade.dropna(subset=[t_col, '전용면적(평)'])
        df_trade = df_trade[df_trade['전용면적(평)'] > 0]
        df_trade['평당금액'] = df_trade[t_col] / df_trade['전용면적(평)']
        
        if '행정동' in df_trade.columns and not df_redev_agg.empty:
            df_trade = pd.merge(df_trade, df_redev_agg, on='행정동', how='left')
            df_trade['재개발_정비사업_구역수'] = df_trade['재개발_정비사업_구역수'].fillna(0)
        else:
            df_trade['재개발_정비사업_구역수'] = 0

        trade_base_features = ['계약연도', '전용면적(㎡)', '전용면적(평)', '층', '건축년도', '건물 나이(년)']
        features_trade = [c for c in trade_base_features + ext_features if c in df_trade.columns]
        
        X_t = df_trade[features_trade].fillna(0)
        y_t = df_trade['평당금액']
        
        X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
        scaler_t = StandardScaler()
        X_tr_t_s = scaler_t.fit_transform(X_tr_t)
        X_te_t_s = scaler_t.transform(X_te_t)
        
        with mlflow.start_run(run_name="Trade_Final_RandomForest"):
            rf_trade = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_trade.fit(X_tr_t_s, y_tr_t)
            
            preds = rf_trade.predict(X_te_t_s)
            r2 = r2_score(y_te_t, preds)
            mae = mean_absolute_error(y_te_t, preds)
            
            mlflow.log_metric("final_r2", r2)
            mlflow.log_metric("final_mae", mae)
            
            # 💡 확장자를 .joblib으로 변경하여 저장
            local_model_path = os.path.join(model_save_dir, "trade_random_forest.joblib")
            local_scaler_path = os.path.join(model_save_dir, "trade_scaler.joblib")
            joblib.dump(rf_trade, local_model_path)
            joblib.dump(scaler_t, local_scaler_path)
            
            print(f"  [완료] 매매 모델 R²: {r2:.4f} | MAE: {mae:.1f}")
            print(f"  📂 로컬 모델 저장 경로: {local_model_path}")

    # ==========================================
    # [Case 2] 아파트 전월세 RandomForest 모델 학습 및 .joblib 저장
    # ==========================================
    print("\n--------------------------------------------------")
    print("🚀 [아파트 전월세] 최종 RandomForest 모델 학습 및 .joblib 저장")
    print("--------------------------------------------------")
    mlflow.set_experiment("Feature_Improvement_Rent_All")
    
    rent_file = os.path.join(base_dir, "아파트_전월세.csv")
    if os.path.exists(rent_file):
        df_rent = pd.read_csv(rent_file)
        r_col = next((c for c in ['보증금(만원)', '임대보증금(만원)', '보증금'] if c in df_rent.columns), None)
        df_rent = df_rent.dropna(subset=[r_col, '임대면적(평)'])
        df_rent = df_rent[df_rent['임대면적(평)'] > 0]
        
        if '전월세구분' in df_rent.columns:
            df_rent['전월세구분'] = df_rent['전월세구분'].map({'전세': 0, '월세': 1}).fillna(0)
            
        df_rent['평당보증금'] = df_rent[r_col] / df_rent['임대면적(평)']
        
        if '행정동' in df_rent.columns and not df_redev_agg.empty:
            df_rent = pd.merge(df_rent, df_redev_agg, on='행정동', how='left')
            df_rent['재개발_정비사업_구역수'] = df_rent['재개발_정비사업_구역수'].fillna(0)
        else:
            df_rent['재개발_정비사업_구역수'] = 0

        rent_base_features = ['계약연도', '전월세구분', '임대면적(㎡)', '임대면적(평)', '층', '건축년도', '건물 나이(년)']
        features_rent = [c for c in rent_base_features + ext_features if c in df_rent.columns]
        
        X_r = df_rent[features_rent].fillna(0)
        y_r = df_rent['평당보증금']
        
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        scaler_r = StandardScaler()
        X_tr_r_s = scaler_r.fit_transform(X_tr_r)
        X_te_r_s = scaler_r.transform(X_te_r)
        
        with mlflow.start_run(run_name="Rent_Final_RandomForest"):
            rf_rent = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_rent.fit(X_tr_r_s, y_tr_r)
            
            preds = rf_rent.predict(X_te_r_s)
            r2 = r2_score(y_te_r, preds)
            mae = mean_absolute_error(y_te_r, preds)
            
            mlflow.log_metric("final_r2", r2)
            mlflow.log_metric("final_mae", mae)
            
            # 💡 확장자를 .joblib으로 변경하여 저장
            local_model_path = os.path.join(model_save_dir, "rent_random_forest.joblib")
            local_scaler_path = os.path.join(model_save_dir, "rent_scaler.joblib")
            joblib.dump(rf_rent, local_model_path)
            joblib.dump(scaler_r, local_scaler_path)
            
            print(f"  [완료] 전월세 모델 R²: {r2:.4f} | MAE: {mae:.1f}")
            print(f"  📂 로컬 모델 저장 경로: {local_model_path}")

if __name__ == "__main__":
    train_and_save_rf_models()
    
