"""자체 생성한 joblib 또는 신뢰하는 MLflow 등록 모델로 아동 인구수를 추정한다."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['joblib','mlflow'],default='joblib')
    p.add_argument('--model-file',type=Path,default=root/'models/spotfit-child-count-elementary/v1/model.joblib')
    p.add_argument('--tracking-uri',default='sqlite:///'+(root/'artifacts/mlflow_child_subsets/tracking.db').as_posix())
    p.add_argument('--model-uri',default='models:/spotfit-child-count-elementary/1')
    p.add_argument('--input',type=Path,help='시설 컬럼을 포함한 CSV. 미지정이면 가상 예시 2행 사용')
    p.add_argument('--encoding',default='utf-8-sig',help='입력 CSV 인코딩. CP949 파일이면 cp949 지정')
    p.add_argument('--output',type=Path,help='새 결과 파일 경로. 기존 파일은 덮어쓰지 않음')
    a=p.parse_args()
    # 모델 파일 역직렬화는 코드를 실행할 수 있다. 외부 비신뢰 파일·URI를 사용하지 않는다.
    if a.backend=='joblib':
        import joblib
        bundle=joblib.load(a.model_file)
        model=bundle['model'];features=bundle['features']
    else:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(a.tracking_uri)
        model=mlflow.sklearn.load_model(a.model_uri)
        # 모델 학습 시 기록된 컬럼 순서를 사용한다. 버전마다 조합이 바뀔 수 있다.
        features=list(model.feature_names_in_)
    if list(model.feature_names_in_)!=list(features):
        raise ValueError('모델과 저장된 입력 컬럼 목록이 일치하지 않습니다.')
    if a.input:
        d=pd.read_csv(a.input,encoding=a.encoding,dtype={'dong_code':str})
    else:
        # 실제 지역 자료가 아닌 명시적인 데모이다. 다른 조합 모델에는 입력 CSV가 필요하다.
        d=pd.DataFrame({'example':['가상예시 A','가상예시 B'],'초등학교_수':[2.,3.],'약국_수':[10.,20.]})
        print('입력 CSV 미지정: 실제 지역이 아닌 가상 예시입니다.')
    missing=set(features)-set(d.columns)
    if missing: p.error(f'필수 컬럼 누락: {sorted(missing)}. --input CSV를 지정하세요.')
    if d.empty: p.error('입력 데이터가 비어 있습니다.')
    # 미상 값은 학습된 imputer가 처리하지만 잘못된 문자열·음수·무한대는 거부한다.
    x=d[features].apply(pd.to_numeric,errors='raise').astype(float)
    if np.isinf(x.to_numpy()).any() or x.lt(0).any().any():
        p.error('시설 수에는 음수 또는 무한대를 사용할 수 없습니다.')
    if x.isna().all(axis=1).any(): p.error('모든 입력 시설 값이 결측인 행은 예측하지 않습니다.')
    result=d.copy();result['predicted_child_population']=model.predict(x)
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    output=a.output or root/'artifacts/reports/child_predictions'/f'{a.backend}_{stamp}.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    # 소수 예측값은 회귀의 추정치이다. 저장 시 반올림하지 않아 검증 정밀도를 유지한다.
    result.to_csv(output,index=False,encoding='utf-8-sig',mode='x')
    print('사용 특징:',features)
    print(result.head(20).round(2).to_string(index=False))
    print('저장:',output)
    print('현재 아동 인구수 추정 실험입니다. 미래 예측이나 환경 점수가 아닙니다.')


if __name__=='__main__': main()
