# 교통사고 93차·94차 재구축 및 v1 배포

현재 데이터 구조를 기준으로 Raw 검증부터 `models/recommender/v1` 배포까지 하나의
명령으로 실행합니다.

```powershell
python scripts\rebuild_traffic_v93_v94_to_v1.py
```

## 실행 단계

1. `data/raw/93차 모델 데이터`의 46개 원천파일을 검사합니다.
2. `data/raw/94차 모델 데이터`에 동일한 기반파일 46개와 TOPIS 월별 108개가 있는지
   파일명과 SHA-256으로 검사합니다.
3. TOPIS 108개를 `data/interim/교통사고/94차/자치구별_교통량_2017_2025.csv`로
   다시 집계하고, 94차 Processed의 `추가교통량지표` 시트와 대조합니다.
4. 누수 검토가 끝난 고정 학습행렬인 `data/processed/교통사고_93차_데이터.xlsx`와
   `data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx`에서 2년·3년 모델을 다시 학습합니다.
5. `models/recommender/v1/traffic_accident_v93`, `traffic_accident_v94`에 서비스 모델,
   예측 CSV와 메타데이터를 생성합니다.
6. Joblib 로드, 모델 버전·예측기간, SHA-256, 2027·2028년 각 25개 자치구를 검사합니다.

Raw 원본은 출처 보존용이고 Processed는 모델에 실제 투입되는 고정 학습행렬입니다.
93차 원천 46개 전체를 매번 다시 해석하는 대신, 원천 완전성과 해시를 강제 검증한 뒤
검토 완료된 Processed를 학습 입력으로 사용합니다. 94차에서 새로 추가된 TOPIS 데이터는
Raw에서 실제로 다시 집계해 Processed와 값까지 대조합니다.

학습을 생략하고 기존 재현 모델을 다시 v1에 배포·검증할 때는 다음을 사용합니다.

```powershell
python scripts\rebuild_traffic_v93_v94_to_v1.py --skip-training
```
