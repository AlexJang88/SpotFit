# 교통사고 모델 스크립트 안내

교통사고 모델의 현재 기준은 안전등급 체계의 13차 기준모델과 19차 최우선 모델이다.
기존 1~12차 사고건수 실험은 비교·재현 목적으로 보존한다.

## 현재 사용하는 학습 스크립트

| 파일 | 역할 |
| --- | --- |
| `ml/train_traffic_accident_initial_v1.py` | 기존 1차 사고건수 기준모델과 공통 데이터 로더 |
| `ml/train_traffic_accident_incremental_models.py` | 기존 2~12차 사고건수 변수 확장 재현 |
| `ml/train_traffic_safety_grade_v13_v24.py` | 신규 13~24차 안전등급 학습·검증·MLflow 기록 |
| `ml/train_best_child_traffic_accident_rate.py` | 41차: 다음 연도 자치구별 어린이 인구 1만 명당 교통사고 피해자 수 예측 |

신규 안전등급 모델 전체 재현:

```powershell
python ml/train_traffic_safety_grade_v13_v24.py
```

특정 차수만 재현:

```powershell
python ml/train_traffic_safety_grade_v13_v24.py --versions 13 19
```

## 현재 사용하는 발표자료 스크립트

| 파일 | 역할 |
| --- | --- |
| `scripts/create_traffic_models_v1_v24_comparison_presentation.py` | 1~12차와 13~24차 비교 PPT 생성 |
| `scripts/create_traffic_models_v1_v5_presentation.py` | PPT 공용 스타일 함수와 과거 비교본 재현 |

## 데이터 구축 스크립트

교통사고 원천 수집·정제·통합 코드는 `scripts/traffic_data/`에 모았다. 모델 학습 때마다
실행하는 코드는 아니며 원천 데이터가 변경될 때만 단계별로 사용한다.

## 과거 스크립트

- `ml/archive/traffic_legacy/`: 초기 사고건수·가중점수 실험
- `scripts/archive/traffic_legacy/`: 과거 1~12차 발표자료와 일회성 내보내기

과거 스크립트는 삭제하지 않았지만 현재 모델 선정이나 새 발표자료에는 사용하지 않는다.
