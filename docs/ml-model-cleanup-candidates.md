# ML·모델 파일 정리 후보

- 점검일: 2026-11-05
- 기준: 현재 백엔드 참조, 학습 코드 의존성, 공식 문서상 현재 모델, 재생성 가능 여부
- 주의: 이 문서는 삭제 후보 분류이며 실제 파일은 삭제하지 않았다.

## 요약

| 영역 | 현재 크기 | 판단 |
|---|---:|---|
| `ml/` | 2.35MB | 소스는 작으므로 삭제보다 `archive` 분류가 적합 |
| `models/` | 35.60MB | 교통사고 13·19차 중심으로 축소 가능 |
| `artifacts/` | 150.52MB | 임시·구형 실험 산출물 다수, 가장 큰 정리 대상 |
| `model_registry/` | 사실상 0MB | 서비스 배포 구조이므로 유지 |

## 바로 삭제해도 영향이 거의 없는 후보

| 대상 | 크기 | 근거 |
|---|---:|---|
| `artifacts/ppt_review/` | 약 0.17MB | PPT 육안 검토용 임시 렌더링 이미지 |
| `ml/**/__pycache__/` | 약 0.12MB 이상 | Python이 자동 재생성하는 캐시 |
| `.tmp_mlflow_traffic/` | 임시 폴더 | 임시 MLflow 실행 흔적이며 현재 Git 검색에서 권한 오류 유발 |
| `tmpit*` | 임시 폴더 | 임시 생성 폴더로 현재 검색 시 권한 오류 유발 |

## 사용 참조가 없는 대용량 후보

| 대상 | 크기 | 확인 결과 |
|---|---:|---|
| `artifacts/models/regional_age_forecast_v1.joblib` | 약 70.01MB | 저장소 코드·백엔드·문서에서 파일명 참조가 발견되지 않음 |

이 파일은 현재 가장 큰 단일 정리 후보이다. 다만 외부에서 수동 사용 중인지 확인한 뒤 삭제해야 한다.

## 교통사고 모델 폴더

`models/안전/교통사고`에는 1~26차가 각각 7개 파일씩 존재하며 전체 약 35.6MB다.

### 유지 권장

| 차수 | 역할 | 크기 |
|---|---|---:|
| 13차 | 안전등급 체계의 기준모델 | 약 2.67MB |
| 19차 | 내부 순차검증 기준 현재 최우선 모델 | 약 3.66MB |

### 메타데이터만 유지 가능

- 1~12차: 과거 사고건수 실험이다. 다만 `13~24차` 재학습 코드가 각 차수의 `모델정보.json`에서 변수 구성을 읽으므로 `모델정보.json`은 유지해야 한다.
- 14~18차, 20~26차: 비교 실험 결과다. 최종 서비스가 직접 로드하지 않으며 학습 코드로 다시 생성할 수 있다.

따라서 공간 정리가 목적이면 다음 정책이 안전하다.

1. 13차와 19차 폴더는 전체 유지한다.
2. 1~12차는 `모델정보.json`만 유지하고 joblib·검증 CSV·보고서를 별도 보관하거나 삭제한다.
3. 14~18차와 20~26차는 `모델정보.json`과 핵심 비교 CSV만 유지하고 joblib을 정리한다.

13·19차를 제외한 모델 폴더 총량은 약 29.27MB다. 메타데이터를 남기고 joblib 중심으로 정리하면 대부분을 회수할 수 있다.

## `artifacts/` 정리 후보

| 폴더 | 크기 | 분류 |
|---|---:|---|
| `artifacts/mlflow_traffic_safety_grade/` | 38.27MB | 현재 13~26차 실험 추적 기록; 유지 권장 또는 외부 백업 후 정리 |
| `artifacts/mlflow_traffic_accident/` | 22.28MB | 과거 1~12차 실험; archive 또는 삭제 후보 |
| `artifacts/traffic_accident_next_year/` | 9.13MB | 레거시 사고건수 실험; 삭제 후보 |
| `artifacts/traffic_accident_safety_standard/` | 5.47MB | 레거시 안전기준 실험; 삭제 후보 |
| `artifacts/traffic_accident_weighted_score/` | 0.07MB | 레거시 가중점수 실험; 삭제 후보 |
| `artifacts/reports/` | 5.08MB | 발표·보고서가 섞여 있어 개별 확인 후 정리 |

레거시 교통사고 산출물 네 종류를 정리하면 약 36.95MB를 회수할 수 있다.

## `ml/` 소스 분류

### 유지

- `ml/train_traffic_safety_grade_v13_v24.py`
- `ml/train_traffic_accident_initial_v1.py`
- `ml/train_traffic_accident_incremental_models.py`
- `ml/training/train_crime_safety_*`
- 현재 실제로 다시 학습하거나 추론하는 아동·상권 모델 코드
- `ml/inference/`의 실제 실행 진입점

### archive에 두는 것이 적합

- `ml/archive/traffic_legacy/`: 이미 올바르게 분리되어 있음
- `ml/training/AptSaleRF_v*.py`, `AptSaleXGB_v*.py`, `AptSaleRidge_v*.py`: 버전별 실험 스크립트가 다수이며 현재 서비스 진입점에서 직접 참조하지 않음
- `ml/etl/property_amount/property_model_v1.py`~`v11.py`: 과거 실험 계열
- 절대경로가 남은 `failure_rate`, `gender`, 일부 `property_amount` 분석 코드

소스 코드는 전체가 2.35MB뿐이므로 용량 절감 효과는 작다. 삭제보다 `ml/archive/` 아래로 이동해 현재 코드와 과거 실험을 구분하는 편이 안전하다.

## 빈 구조 폴더

다음 폴더는 `.gitkeep` 외에 실질적인 코드가 없다.

- `ml/collectors`
- `ml/evaluation`
- `ml/features`
- `ml/pipelines`
- `ml/tests`
- `ml/validation`

프로젝트 표준 구조로 사용할 계획이 없다면 제거 가능하지만 용량 절감 효과는 없다.

## 현재 백엔드가 실제 사용하는 모델

- 범죄안전 예측: `artifacts/crime_safety_forecast/.../crime_safety_direct_best_ml_exception.joblib` 또는 MLflow Registry
- 일반 모델 배포: `model_registry/` manifest와 MinIO 객체 저장소
- 교통사고 `models/안전/교통사고/*`: 현재 FastAPI 서비스가 직접 로드하지 않음
- `models/recommender`: `upload_minio.py`의 업로드 대상으로 문서화되어 있음

## 권장 정리 순서

1. 임시 캐시와 `artifacts/ppt_review` 삭제
2. 참조 없는 `regional_age_forecast_v1.joblib`의 외부 사용 여부 확인 후 삭제
3. 레거시 교통사고 artifacts 약 36.95MB 삭제 또는 압축 보관
4. 교통사고 모델은 13·19차 전체 유지, 나머지는 메타데이터 중심으로 축소
5. 아파트 버전별 스크립트와 구형 property model을 `ml/archive/`로 이동

위 순서에서 1~4를 적용하면 대략 130MB 이상을 정리할 수 있다. 정확한 회수량은 메타데이터 보존 범위에 따라 달라진다.
