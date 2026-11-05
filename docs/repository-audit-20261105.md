# SpotFit 저장소 전수 점검 결과

- 점검일: 2026-11-05
- 범위: 백엔드, 프론트엔드, ML·ETL, 데이터 분석, 문서, 발표자료 생성 코드
- 원칙: 모델 정의나 사용자 데이터를 임의 변경하지 않고 재현성·실행성·내용 일관성을 우선 확인

## 결론

현재 주력 백엔드와 교통사고·범죄 모델 설명은 서로 일치한다. Python 소스 전체도 문법 오류 없이 컴파일된다. 가장 큰 품질 문제는 주력 코드보다 `ml/etl` 아래에 남은 과거 실험 스크립트의 절대경로·사설 서버 주소·즉시 실행 구조다.

교통사고 모델의 현재 프로젝트 내 지위는 다음과 같다.

- 13차: 안전기준표 기본변수 4개를 사용한 비교 기준모델
- 19차: 내부 순차검증 등급 정확도가 가장 높은 현재 최우선 후보
- 25·26차: 재난약자·차량통행속도 추가 실험이며 19차를 대체하지 못함
- 13~26차 등급: 경감지표를 제외하고 재환산한 상대등급이며 공식 지역안전지수 등급이 아님

## 이번에 바로 정리한 항목

1. 임시 작업 폴더를 `.gitignore`에 추가해 검색·상태 확인 시 권한 오류와 노이즈를 줄였다.
2. 개발 의존성에 `pytest`를 명시해 백엔드 테스트 실행 조건을 분명히 했다.
3. 중복된 교통사고 13차 단독 요약 생성기를 제거했다.
4. 교통사고 발표자료는 13차 단독 설명과 13→19차 개선 과정 자료를 명확히 구분했다.
5. 확보 데이터 출처는 `docs/data-source-inventory.md`에 확인 수준과 한계를 함께 기록했다.

## 검사 결과

| 검사 | 결과 | 비고 |
|---|---|---|
| Python 전체 문법 컴파일 | 통과 | `backend`, `ml`, `scripts`, `data_analyze`, `upload_minio.py` |
| 백엔드 애플리케이션 import | 통과 | FastAPI 앱과 라우터 로딩 확인 |
| 백엔드 pytest | 환경 미설치로 미실행 | `requirements-dev.txt`에 pytest 추가 |
| 프론트엔드 TypeScript 검사 | 환경 미설치로 미실행 | 현재 PC에서 `npm` 명령을 찾지 못함 |
| 문서·코드 한글 깨짐 검색 | 이상 없음 | 콘솔 CP949 출력만 깨졌고 실제 파일명·파일 내용은 정상 |
| 교통사고 모델 메타데이터 | 일치 | 13차 기준, 19차 내부 CV 최우선 |

## 우선순위별 남은 문제

### 높음: 실행 재현성을 깨는 레거시 절대경로

활성 소스 범위에서 과거 개인 PC 경로가 48곳 발견됐다. 주요 파일은 다음과 같다.

- `ml/etl/property_amount/train_rent_all_features.py`
- `ml/etl/property_amount/train_and_save_final_models.py`
- `ml/etl/property_amount/test_rent_experiment.py`
- `ml/etl/property_amount/test_feature_experiment.py`
- `ml/etl/property_amount/feature_experiment_pipeline2.py`
- `ml/etl/property_amount/split_apt_rent_by_year.py`
- `ml/etl/property_amount/split_apt_trade_by_year.py`
- `ml/etl/failure_rate/failure_rate_analyze.py`
- `ml/etl/gender/gender_analyze.py`
- `ml/etl/property_amount/Grouping_administrative_dongs.py`
- `ml/etl/property_amount/property_analyze.py`

이 파일들은 현재 프로젝트 경로가 아닌 `C:\Users\M\Desktop\final_project...`를 직접 참조한다. 일부는 고정된 사설 MLflow 주소 `192.168.0.3:5000`도 사용한다. 데이터 파일명과 파이프라인 계약을 먼저 확정한 뒤 다음 방식으로 일괄 전환해야 한다.

```python
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("SPOTFIT_PROPERTY_DATA_DIR", ROOT / "data" / "processed" / "부동산"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///...")
```

경로만 기계적으로 바꾸면 현재 최종 파일명과 레거시 스크립트의 예상 파일명이 달라 잘못된 데이터를 읽을 수 있으므로, 모델 의미를 보존하기 위해 자동 치환하지 않았다.

### 높음: 테스트 실행 환경

- Python 3.14 환경에는 `pytest`가 설치되어 있지 않았다.
- Node.js/npm이 PATH에 없어 프론트 타입검사와 빌드를 실행하지 못했다.
- MinIO 통합 테스트는 별도 테스트 버킷을 만들므로 MinIO가 실행된 환경에서 수행해야 한다.

권장 검증 순서:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest backend/tests -q
cd frontend
npm ci
npm run typecheck
npm run build
```

### 중간: 레거시 분석 코드 구조

일부 분석 파일은 import 시 곧바로 대용량 CSV를 읽고 분석을 실행한다. 함수와 `main()`으로 감싸지 않아 재사용·테스트가 어렵다. 특히 `gender_analyze.py`는 긴 주석 처리 실험과 실제 실행 코드가 한 파일에 혼재한다.

권장 조치:

- 실행부를 `if __name__ == "__main__":` 아래로 이동
- 입출력 경로를 명령행 인자로 전환
- 실험별 파일을 분리하고 종료된 실험은 `ml/archive`로 이동
- 공통 데이터 로더와 인코딩 처리를 한 모듈로 통합

### 중간: 모델 검증 표현

- 교통사고 19차는 내부 CV 등급 정확도 56%로 최우선이다.
- 2025 단일 최종검증 정확도만 보면 다른 차수가 더 높을 수 있지만, 이 값으로 모델을 다시 선택하면 최종검증 누수가 된다.
- 따라서 발표자료에는 항상 “내부 순차검증 기준 최우선”이라고 표현해야 한다.
- 13~26차 등급은 공식 지역안전지수 등급이 아니라 경감지표 제외 상대등급이다.

### 중간: 데이터 출처·계보

다음 원본은 출처 메타데이터 보완이 필요하다.

- 부동산 매매 최종 CSV의 직접 원본·다운로드일·SHA-256
- 서울시 물가상승률 파일의 정확한 통계표 URL
- 어린이 교통사고 파일의 TAAS 조회 조건
- 범죄 CSV의 공식 URL과 경찰서→자치구 매핑 규칙

자세한 내용은 `docs/data-source-inventory.md`를 따른다.

### 낮음: 생성 스크립트와 산출물 관리

발표자료 생성기는 `scripts/`에, 생성된 PPT는 바탕화면이나 `artifacts/presentations/`에 두는 규칙을 권장한다. 일회성 렌더링·검토 스크립트는 검토가 끝나면 제거한다.

## 안전하게 다음에 진행할 최적화

1. 부동산 ETL의 실제 사용 파일을 확정하고 11개 레거시 파일의 경로·MLflow 설정을 일괄 표준화한다.
2. `gender_analyze.py`와 `property_analyze.py`를 실행 가능한 작은 모듈로 분리한다.
3. 개발 의존성을 설치한 환경에서 백엔드·프론트 전체 검증을 통과시킨다.
4. 원본 데이터 매니페스트에 URL·다운로드일·SHA-256·생성 스크립트를 연결한다.
5. 교통사고 19차 이후 모델은 같은 내부 CV 기준과 추가 독립연도로만 승격한다.
