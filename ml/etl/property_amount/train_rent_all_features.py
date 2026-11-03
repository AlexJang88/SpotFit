import os
import json
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def run_rent_all_features():
    base_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\data\raw"
    output_dir = r"C:\Users\M\Desktop\final_project_develop\SpotFit\ml\etl\property_amount"
    
    # MLflow URI & Experiment 설정
    mlflow.set_tracking_uri("http://192.168.0.3:5000/")
    mlflow.set_experiment("Feature_Improvement_Rent_All")
    
    rent_file = os.path.join(base_dir, "아파트_전월세.csv")
    redev_file = os.path.join(base_dir, "★(26년 6월기준) 서울시 정비사업 추진현황.xlsx")
    
    if not os.path.exists(rent_file):
        print(f"❌ 파일을 찾을 수 없습니다: {rent_file}")
        return

    print("1. 서울시 정비사업 추진현황 데이터 로드 및 집계 중...")
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

    print("2. 아파트 전월세 데이터 로드 및 데이터 전처리 중...")
    df = pd.read_csv(rent_file)
    target_col = next((c for c in ['보증금(만원)', '임대보증금(만원)', '보증금'] if c in df.columns), None)
    
    df = df.dropna(subset=[target_col, '임대면적(평)'])
    df = df[df['임대면적(평)'] > 0]
    
    if '전월세구분' in df.columns:
        df['전월세구분'] = df['전월세구분'].map({'전세': 0, '월세': 1}).fillna(0)

    df['평당보증금'] = df[target_col] / df['임대면적(평)']
    
    if '행정동' in df.columns and not df_redev_agg.empty:
        df = pd.merge(df, df_redev_agg, on='행정동', how='left')
        df['재개발_정비사업_구역수'] = df['재개발_정비사업_구역수'].fillna(0)
    else:
        df['재개발_정비사업_구역수'] = 0

    base_features = ['계약연도', '전월세구분', '임대면적(㎡)', '임대면적(평)', '층', '건축년도', '건물 나이(년)']
    ext_features = ['거리_to_GBD(km)', '거리_to_CBD(km)', '거리_to_YBD(km)', 
                    '접근성_score_GBD', '접근성_score_CBD', '접근성_score_YBD', '재개발_정비사업_구역수']
    
    features_b = [c for c in base_features if c in df.columns]
    features_a = [c for c in base_features + ext_features if c in df.columns]

    X_b = df[features_b].fillna(0)
    X_a = df[features_a].fillna(0)
    y = df['평당보증금']

    X_tr_b, X_te_b, y_tr, y_te = train_test_split(X_b, y, test_size=0.2, random_state=42)
    X_tr_a, X_te_a, _, _ = train_test_split(X_a, y, test_size=0.2, random_state=42)

    scaler_b = StandardScaler()
    X_tr_b_s, X_te_b_s = scaler_b.fit_transform(X_tr_b), scaler_b.transform(X_te_b)

    scaler_a = StandardScaler()
    X_tr_a_s, X_te_a_s = scaler_a.fit_transform(X_tr_a), scaler_a.transform(X_te_a)

    models = {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)

    print("="*75)
    print("🚀 [아파트 전월세 통합] 내부 요인 vs 전체 요인(접근성+재개발) 모델 학습")
    print("="*75)

    best_model_name = None
    best_r2 = -999.0
    best_trained_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Rent_All_{name}"):
            model.fit(X_tr_b_s, y_tr)
            pred_b = model.predict(X_te_b_s)
            r2_b, mae_b = r2_score(y_te, pred_b), mean_absolute_error(y_te, pred_b)

            model.fit(X_tr_a_s, y_tr)
            pred_a = model.predict(X_te_a_s)
            r2_a, mae_a = r2_score(y_te, pred_a), mean_absolute_error(y_te, pred_a)

            if r2_a > best_r2:
                best_r2 = r2_a
                best_model_name = name
                best_trained_model = model

            mlflow.log_param("dataset_type", "Rent_Integrated")
            mlflow.log_param("model_name", name)
            mlflow.log_param("features_count_before", len(features_b))
            mlflow.log_param("features_count_after", len(features_a))
            mlflow.log_param("features_after_list", str(features_a))
            
            mlflow.log_metric("r2_before", r2_b)
            mlflow.log_metric("r2_after", r2_a)
            mlflow.log_metric("r2_diff", r2_a - r2_b)
            mlflow.log_metric("mae_before", mae_b)
            mlflow.log_metric("mae_after", mae_a)
            mlflow.log_metric("mae_diff", mae_a - mae_b)

            print(f"  [{name}] R²: {r2_b:.4f} ➡️ {r2_a:.4f} (Δ:{r2_a-r2_b:+.4f}) | MAE: {mae_b:.1f} ➡️ {mae_a:.1f} (Δ:{mae_a-mae_b:+.1f})")

    print(f"\n🏆 최고 성능 모델 선정: [{best_model_name}] (최종 R²: {best_r2:.4f})")
    print("4. 행정동별 현재, 1년 뒤(2027), 3년 뒤(2029) 예측 보증금 계산 중...")
    
    dong_results = {}
    if '행정동' in df.columns:
        grouped = df.groupby('행정동')
        for dong, group in grouped:
            curr_price = float(group['평당보증금'].mean())
            base_sample = group[features_a].mean().to_dict()
            
            # 1년 뒤 (2027년)
            sample_2027 = base_sample.copy()
            sample_2027['계약연도'] = 2027
            if '건물 나이(년)' in sample_2027:
                sample_2027['건물 나이(년)'] += 1
                
            # 3년 뒤 (2029년)
            sample_2029 = base_sample.copy()
            sample_2029['계약연도'] = 2029
            if '건물 나이(년)' in sample_2029:
                sample_2029['건물 나이(년)'] += 3

            df_pred = pd.DataFrame([sample_2027, sample_2029])[features_a].fillna(0)
            df_pred_s = scaler_a.transform(df_pred)
            preds = best_trained_model.predict(df_pred_s)

            dong_results[dong] = {
                "current_price_per_pyeong": round(curr_price, 1),
                "predicted_price_1y_2027": round(float(preds[0]), 1),
                "predicted_price_3y_2029": round(float(preds[1]), 1)
            }

    json_path = os.path.join(output_dir, "rent_predict_by_dong.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dong_results, f, ensure_ascii=False, indent=4)
    print(f"✅ 전월세 행정동별 예측 결과 저장 완료: {json_path}")

if __name__ == "__main__":
    run_rent_all_features()
