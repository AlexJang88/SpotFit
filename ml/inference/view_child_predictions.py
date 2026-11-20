"""이미 계산된 테스트 예측 결과를 확인한다. 재학습이나 새 예측은 수행하지 않는다."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=root/'artifacts/child_feature_subsets/20261023_095054/elementary_test_predictions.csv')
    p.add_argument('--gu',help='자치구명 정확히 일치하는 행만 표시')
    p.add_argument('--dong',help='행정동명 정확히 일치하는 행만 표시')
    p.add_argument('--limit',type=int,default=20)
    a=p.parse_args()
    if a.limit<1: p.error('--limit은 1 이상이어야 합니다.')
    d=pd.read_csv(a.csv,dtype={'dong_code':str})
    if a.gu: d=d[d.gu==a.gu]
    if a.dong: d=d[d.dong==a.dong]
    if d.empty:
        print('조건에 맞는 테스트 행이 없습니다. 전체 행정동이 아닌 테스트 지역만 포함합니다.')
        return
    # 정답률 대신 명 단위 오차를 표시한다. 지역 필터를 적용하면 해당 부분집합의 지표이다.
    error=d.predicted-d.actual
    d=d.assign(error=error,absolute_error=error.abs())
    print(f'표시 대상 {len(d)}개 / MAE {error.abs().mean():.2f}명 / RMSE {np.sqrt((error**2).mean()):.2f}명')
    print(d.sort_values('absolute_error',ascending=False).head(a.limit).round(2).to_string(index=False))
    print('오차가 큰 순서입니다. 현재 인구 추정 실험이며 미래 예측·육아환경 점수가 아닙니다.')


if __name__=='__main__': main()
