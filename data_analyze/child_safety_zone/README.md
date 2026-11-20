# 어린이보호구역·아동 인구 분석 및 예측

## 분석 질문

행정동의 어린이보호구역 개수와 유형 구성이 다음 연도의 영유아·초등학생·중학생·고등학생
거주 인구 증가율과 관련이 있는지 분석합니다. 보호구역은 안전환경의 일부일 뿐이므로 결과를
육아 만족도 또는 보호구역 설치의 인과효과로 해석하지 않습니다.

## 1. 전처리 및 연관성 분석

```powershell
python data_analyze/child_safety_zone/analyze_child_safety_zone.py
```

처리 과정은 다음과 같습니다.

1. 파일 내부의 기준일을 읽어 2023년 6·12월, 2024년 6·12월, 2025년 12월,
   2026년 6월 스냅샷을 구분합니다.
2. `제1동/1동`, `1.2동/1·2동` 표기를 정규화하여 아동 인구 자료와 결합합니다.
3. 같은 연도에 6월과 12월을 동시에 학습하지 않고, 연간 대표값으로 12월 말만 사용합니다.
4. `t년 보호구역 → t+1년 아동 인구 로그 증감률` 목표를 생성합니다.
5. Pearson, Spearman, 부분 Spearman을 비교합니다.
6. 부분 Spearman에서는 시작 아동 수, 전체 인구, 자치구, 연도를 통제합니다.

전처리 패널은 다음 위치에 고정 저장되어 학습 코드가 재사용합니다.

```text
data/interim/child_safety_zone/safety_zone_snapshots.csv
data/interim/child_safety_zone/safety_zone_analysis_panel.csv
```

실행별 상관표·그래프·HTML 보고서는 다음 위치에 저장됩니다.

```text
data_analyze/child_safety_zone/results/YYYYMMDD_HHMMSS/
```

## 2. 모델 학습 및 비교

```powershell
python ml/training/train_child_safety_zone.py
```

비교 모델은 Dummy median, Linear Regression, Ridge, Random Forest, Extra Trees입니다.
모델 이름만 나열하지 않고 동일한 입력·목표·분할에서 비교합니다.

목표는 `log((다음 연도 아동 수+1)/(현재 아동 수+1))`입니다. 시작 인구가 0명 또는 몇 명인
행정동에서 일반 증감률이 수십 배로 폭증하는 문제를 완화합니다. 현재 해당 연령군 인구가
30명 미만인 표본은 행과 원본 값은 보존하고, 그 연령군의 학습 목표에서만 제외합니다.
원래 증감률 컬럼도 전처리 패널에 함께 남깁니다.

- 학습 및 모델 선택: 2023년 특징, 5-fold 교차검증
- 시간 외 테스트: 2024년 특징으로 2025년 증가율 예측
- 모델 선택 기준: 2023년 CV RMSE 평균 최소
- 최종 보고 지표: MAE, RMSE, R²
- 해석: 시간 외 테스트의 permutation importance

테스트 RMSE가 가장 좋은 모델을 사후 선택하지 않습니다. 그렇게 하면 테스트셋이 모델 선택에
사용되어 성능이 낙관적으로 평가되기 때문입니다.

결과는 다음 위치에 저장됩니다.

```text
artifacts/child_safety_zone/YYYYMMDD_HHMMSS/
├─ model_comparison.csv
├─ holdout_predictions.csv
├─ permutation_importance.csv
├─ model_metadata.json
├─ report.html
└─ models/
   ├─ child_safety_zone_infant.joblib
   ├─ child_safety_zone_elementary.joblib
   ├─ child_safety_zone_middle.joblib
   └─ child_safety_zone_high.joblib
```

## 현재 데이터의 결정적 한계

연말 실제 스냅샷은 2023·2024·2025년뿐이므로 학습 가능한 다음 연도 전환은 두 번입니다.
행정동 행 수가 많더라도 독립적인 시간 변화는 두 개뿐이어서 운영 모델로는 부족합니다.
2026년 6월 보호구역은 2026년 연말 자료가 아니고 2027년 아동 인구 정답도 없으므로 현재
현황에만 사용합니다. 2026년 연말 보호구역과 아동 인구가 확보되면 다시 학습해야 합니다.

또한 보호구역 수는 초등학교·유치원·어린이집 수와 현재 인구 규모를 반영합니다. 향후 모델에는
신규 주택 공급, 주택가격, 출생, 전입·전출, 전체 인구, 도로·사고 자료를 추가해야 합니다.
