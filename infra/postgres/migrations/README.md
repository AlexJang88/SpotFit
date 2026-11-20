# SpotFit ERD v2 DB 생성·초기데이터 SQL

PostgreSQL **17** / PostGIS **3.5** 기준입니다. 기존 Compose `postgres` 서비스의 애플리케이션 DB(기본 `spot_fit`) 안에 **`spotfit` 스키마**를 생성합니다. Airflow·MLflow 내부 테이블, 기존 `public` 업무 테이블은 수정하지 않습니다. 회원가입·로그인 테이블은 없습니다.

> 이 변경은 SQL과 실행 도구를 제공합니다. 운영 DB 적용, 기존 CSV/JSON을 읽는 백엔드의 DB 전환, Airflow DAG 구현은 자동으로 수행되지 않습니다. `docker compose up`만으로 기존 DB에 적용되지 않습니다. 아래 명령으로 명시적으로 적용하세요.

## 1. 파일·실행 순서

| 파일 | 역할 |
|---|---|
| `000_install.sql` | **실행 진입점**. 트랜잭션·마이그레이션 버전·오류 중단·하위 SQL 실행 |
| `001_schema.sql` | ERD 업무 테이블 33개 + 모델 유형 카탈로그, FK·인덱스·제약·트리거 |
| `002_reference_seed.sql` | 모델 5종, 출처 6종, 점수·등급·추천 정책 |
| `003_seoul_area_seed.sql` | 서울시 1 + 자치구 25 + 행정동 427, 이름 별칭, PostGIS 경계 |
| `090_verify.sql` | 테이블·기준정보·지역 수·경계 유효성 검증 |
| `tests/constraints.sql` | **격리 테스트 전용**. 테스트 이력·예측값은 마지막에 ROLLBACK |

최종 테이블 수는 **35개**입니다: ERD의 33개 + `ml_model_type` + `schema_migration`.

`000_install.sql`은 `psql`의 `\ir`, `\gset`, `\if`를 사용하므로 **psql로 실행**하세요. SQL 편집기에 전체를 붙여 넣거나 stdin으로 파일 내용만 전송하면 상대경로 포함이 실패할 수 있습니다. 파일 경로를 지정한 `-f` 방식으로 실행합니다. 개별 `001_schema.sql`을 따로 재실행하지 마세요.

## 2. Ubuntu 서버에서 실행 — 권장

레포지토리를 최신화하고 `postgres` 컨테이너가 실행 중인지 확인한 뒤:

```bash
cd /home/spotfit/SpotFit

# 먼저 대상 DB 이름만 확인: DB 쓰기 없음
python3 scripts/apply_spotfit_database.py

# 최초 테이블 생성 + 초기데이터 입력 + 검증
python3 scripts/apply_spotfit_database.py --apply

# 이후 상태 확인만 수행
python3 scripts/apply_spotfit_database.py --verify
```

Python 표준 라이브러리만 사용하므로 pip 설치가 필요 없습니다. Docker를 실행할 수 있는 사용자여야 합니다. 비밀번호는 코드나 명령 인자에 넣지 않고 컨테이너의 `POSTGRES_USER`, `POSTGRES_DB`와 로컬 DB 인증 설정을 사용합니다. `.env`의 비밀값을 출력하지 않습니다.

Compose 파일·프로젝트 이름을 별도로 사용한다면 **현재 DB 컨테이너를 만든 동일한 설정**을 지정합니다. 예:

```bash
python3 scripts/apply_spotfit_database.py -f compose.yaml -f compose.dev.yaml -p spot_fit --apply
```

이 명령은 컨테이너를 새로 시작하지 않습니다. 잘못된 프로젝트를 지정해 실행 중인 `postgres` 서비스를 찾지 못하면 중단합니다. 기본·시스템·Airflow·MLflow DB를 대상으로 지정하면 실행 도구가 거부합니다.

## 3. Windows PowerShell에서 실행

```powershell
Set-Location C:\Users\M\Desktop\JUS\SpotFit
python scripts/apply_spotfit_database.py
python scripts/apply_spotfit_database.py --apply
python scripts/apply_spotfit_database.py --verify
```

SQL 파일을 컨테이너에 **바이트 그대로 복사**하므로 PowerShell의 파이프 문자 인코딩 차이로 한글이 깨지는 문제를 피합니다. `Get-Content ... | psql` 방식은 사용하지 않아도 됩니다.

## 4. Python 없이 SQL만 실행

Ubuntu Bash + 현재 Compose 환경:

```bash
docker compose cp ./infra/postgres/migrations postgres:/tmp/spotfit-db-sql-v1
docker compose exec -T postgres sh -c \
  'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/spotfit-db-sql-v1/000_install.sql'
```

같은 복사 경로를 여러 번 쓰면 Docker의 디렉터리 복사 규칙에 따라 하위 폴더로 복사될 수 있습니다. 반복 적용은 임의의 전용 임시 경로를 사용하는 Python 실행 도구가 편리합니다.

Docker 없이 PostgreSQL에 직접 연결할 때:

```bash
psql -X -h 127.0.0.1 -p 5432 -U spot_fit -d spot_fit -W \
  -v ON_ERROR_STOP=1 -f infra/postgres/migrations/000_install.sql
```

DB 자체는 미리 존재해야 합니다. 기존 Compose는 `POSTGRES_DB`로 DB를 생성하므로 새 DB 생성 SQL을 넣지 않았습니다. 빈 일반 PostgreSQL 서버라면 관리자가 먼저 대상 DB를 생성하고 `postgis`, `btree_gist` 확장을 설치할 권한을 부여해야 합니다. 기존 운영 DB는 적용 전에 별도 백업을 권장합니다. **볼륨 삭제나 `docker compose down -v`는 필요하지 않습니다.**

## 5. 초기데이터의 의미

- **모델 유형 5개:** 범죄노출, 교통사고, 아파트 매매, 전세, 월세. 모델 바이너리나 학습 실행은 가짜로 생성하지 않습니다.
- **정책:** 범죄 가중합 안전점수 50% + 교통안전 50%, 25개 구를 5개씩 A~E, `#526e68` 색상에 등급별 투명도, 예산 내 구별 최저가 동 하나씩 최대 5곳 추천.
- **지역:** 레포지토리의 `frontend/public/seoul-dong.geojson`과 `admin_boundary_registry.json`에서 가져온 **2026-10-01 스냅샷**. 자치구·동 이름과 코드를 임의 생성하지 않았습니다.
- **코드:** `adm_cd2`의 행정안전부 10자리 행정동 코드 사용. 8자리 SGIS 코드 `adm_cd`와 혼용하지 않습니다. 자치구 5자리, 시 2자리.
- **경계:** `MultiPolygon`, SRID 4326. 원본 경계는 `ST_MakeValid` 후 폴리곤만 추출·정규화하며, 자치구·시 경계는 **같은 행정동 스냅샷을 합집합**해서 생성합니다. 출처·라이선스와 원본 SHA256은 생성 SQL 머리말에 기록합니다.
- **기간:** 초기 `valid_from=2026-10-01`은 확인된 스냅샷의 시작일이지, 동의 실제 최초 설립일이 아닙니다. 2015년 등 과거 학습자료를 넣으려면 당시 지역 버전과 매핑을 먼저 확보해야 합니다. 과거 자료를 현재 동 코드에 무조건 연결하면 안 됩니다.
- **출처 URL:** 일부는 원천 포털 주소입니다. 자동 다운로드 API URL이나 고정 공표일이 확인된 것으로 간주하지 않습니다.
- 실제 원자료, 아파트 단지·거래, 모델 버전·성능, 2027/2028 예측값은 **별도 ETL 적재 대상**이며 초기에는 비어 있습니다.

지역 SQL 재생성은 다음 명령입니다. 공개 경계 파일이 변경되면 단순 재생성·덮어쓰기 대신 별도 마이그레이션으로 이전 버전 종료와 신규 버전 삽입을 설계해야 합니다.

```bash
python3 scripts/build_postgres_area_seed.py
```

## 6. ERD ↔ 실제 테이블

대문자 ERD 엔티티를 `spotfit` 스키마 아래 소문자 이름으로 옮겼습니다. 예: `ML_MODEL_VERSION` → `spotfit.ml_model_version`.

| 영역 | 테이블 |
|---|---|
| 수집·품질 | `meta_data_source`, `meta_ingestion_run`, `meta_data_asset_version`, `meta_pipeline_run`, `meta_pipeline_input`, `meta_data_quality_result` |
| 지역·공고 | `ref_admin_area`, `ref_admin_area_alias`, `ref_boundary_notice`, `ref_boundary_notice_area` |
| 정형 원자료 | `core_apartment`, `core_property_transaction`, `core_population_stat`, `core_crime_stat`, `core_traffic_accident_stat`, `core_facility_snapshot` |
| Feature·모델 | `feature_dataset_version`, `ml_training_run`, `ml_model_version`, `ml_model_metric`, `ml_model_feature`, `ml_model_deployment` |
| 서비스 | `service_property_prediction`, `service_safety_prediction`, `service_scoring_policy`, `service_grade_policy`, `service_neighborhood_score`, `service_recommendation_policy`, `service_recommendation_request`, `service_recommendation_result` |
| 익명 통계·운영 | `analytics_service_event`, `ops_audit_log`, `ops_service_incident` |
| 보조 | `ml_model_type`, `schema_migration` |

논리 ERD를 구현하면서 추가한 물리 필드:

- 정책 `policy_code`: 초기데이터 재실행 시 같은 정책을 식별.
- 모델 `validation_passed`: 파이프라인이 사전 확정한 성능 기준을 통과했는지 기록. DB가 영역별 임계값을 임의로 추정하지 않습니다.
- 지역 `code_system`, `source_snapshot`: 코드 출처와 스냅샷 식별.
- 등급 `fill_opacity`: 현재 프론트의 색상 투명도 보존.
- 가격 `deposit_amount`, `prediction_kind`: 월세 보증금 조건, 미래가격/대표 현재가 구분.
- 안전 `prediction_kind`: 미래 예측과 교통 2025 참고값 구분.
- 동네점수 `score_batch_key`: 한 번의 25개 구 점수 산출 묶음(UUID). 각 행마다 별도 UUID가 아니라 같은 배치에는 같은 값을 사용합니다.

## 7. 데이터 입력 시 주의할 규칙

1. **금액은 만원**, 면적은 **㎡**, 단가는 **만원/㎡**입니다. 기존 평당값은 3.305785로 나눠 적재합니다. 가격 예측의 월세 `predicted_amount`는 월세 금액, `deposit_amount`는 조건 보증금입니다.
2. 범죄 `predicted_value`에는 Y를, `safety_score`에는 Feature 가중합 안전점수를 넣습니다. 같은 값이 아닙니다.
3. `service_neighborhood_score.prediction_year`는 범죄의 목표연도입니다. 교통 참고연도는 `traffic_prediction_id`로 연결된 행에 별도로 보존하므로 2027 범죄 + 2025 교통을 지원합니다.
4. 모델 학습 이력 → 모델 버전 → 성능·Feature → 검증 승인 → Champion → 배포 이력 순으로 입력합니다. 배포 시 성공한 학습, 품질 통과 데이터, CV/HOLDOUT/TEST 지표, 검증 승인과 Champion이 필요합니다. 점수에 쓰는 Feature 가중치가 있으면 합계 100%도 검사합니다.
5. 모델당 1개의 Champion만 가능한 것이 아니라 **모델 유형·예측기간당 1개**입니다. 전환 시 기존 배포를 종료하고 기존 Champion을 해제한 뒤 새 모델을 활성화합니다.
6. 예측·점수·추천 결과는 수정하지 않는 스냅샷입니다. 갱신 시 새 모델 버전·새 예측·새 산출 배치를 만들고 과거 결과를 보존합니다. 사용된 모델 Feature와 모델 정의도 덮어쓰지 않습니다.
7. DB는 가중합, 지역·모델 유형, 가격과 예산, 구별 중복 등을 검사합니다. **구별 최저가 후보 탐색·전체 25개 구 정렬·적절한 모델 선정·배포 파일 교체는 애플리케이션/학습 파이프라인 책임**입니다.
8. 익명 요청 이력은 선택 저장입니다. `expires_at`만 설정한다고 자동 삭제되지는 않습니다. 추후 Airflow/스케줄러가 만료 요청을 삭제하면 관련 추천 결과는 CASCADE 삭제되고 익명 이벤트의 요청 FK는 NULL이 됩니다.
9. 로그·JSON 속성에 비밀번호, 토큰, 개인 식별자·정확한 IP를 넣지 않아야 합니다. 입력 API에서 속성 허용목록을 적용해야 하며, JSON 컬럼이 개인정보를 자동 제거해 주지는 않습니다.

## 8. 반복 실행·오류·버전 관리

- 전체 설치·seed·검증은 **하나의 트랜잭션**입니다. 중간 오류 시 중단되고 해당 실행은 커밋되지 않습니다.
- 같은 초기 SQL 재실행 시 DDL은 `schema_migration` 버전으로 건너뛰고 초기 행은 `ON CONFLICT DO NOTHING`으로 중복 입력을 막습니다. 초기 경계가 NULL인 부모 지역만 경계를 채웁니다.
- 기존 정책·지역 행을 seed 값으로 덮어쓰지 않습니다. 기본 정책과 데이터가 맞지 않으면 검증이 실패할 수 있으므로 원인을 먼저 확인합니다.
- 신규 스키마 변경은 기존 `001_schema.sql`을 고치는 방식이 아니라 **별도 버전 마이그레이션**으로 작성해야 합니다. 테이블 일부만 있는 불완전한 수동 설치를 임의 삭제·복원하지 않습니다.
- 적용을 위해 사용자/패스워드 생성이나 기존 계정 권한 변경을 자동 수행하지 않습니다. 별도 애플리케이션 계정 사용 시 필요한 `USAGE`·테이블·시퀀스 권한은 관리자가 부여하세요. `PUBLIC`에 스키마 권한을 공개하지 않습니다.
- `000_install.sql`을 `infra/postgres/init`에 두지 않았습니다. 기존 운영 환경에서 `pull`이나 컨테이너 재생성만으로 예상치 못한 업무 DB 변경이 일어나지 않게 하기 위함입니다.

## 9. 검증

```bash
python3 scripts/test_spotfit_database.py
```

이미 존재하는 `postgis/postgis:17-3.5-alpine` 이미지로 테스트 전용 컨테이너를 생성합니다. 외부 네트워크·공개 포트·영구 볼륨 없이 동작하며, 종료 시 해당 테스트 컨테이너만 제거합니다. 기존 Compose 서비스와 DB는 사용하지 않습니다.

2026-11-16 검증 결과:

- 최초 생성·초기데이터 입력 성공, 같은 설치 SQL 2회 실행 성공.
- ERD 엔티티 33개 전부 존재 + 보조 2개, 서울 지역 453건과 유효한 경계 확인.
- **제약조건 테스트 22건 통과**: 기간 중첩, 잘못된 부모, 외래키, 가중치, 예산, NaN, 배포 검증, Champion 중복, 예측기간, 구/동 구분, 가중합, 스냅샷 변경, NULL 포함 가격키 중복, 추천 금액·동일 구 중복, 만료 요청 연관 삭제 등.
- 테스트 행은 ROLLBACK되어 초기 모델·예측 업무 테이블은 비어 있음.

결과는 로컬 `artifacts/database_sql_validation/`에 저장됩니다. 이 폴더는 기존 `.gitignore`에 의해 Git에서 제외됩니다.
