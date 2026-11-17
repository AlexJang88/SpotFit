# SpotFit PostgreSQL ERD v2

기준일: 2026-11-13  
범위: 비회원 서비스, 데이터 수집·전처리, 5개 최종모델, 예측·추천·운영정보

이 문서는 논리 ERD이며 아직 PostgreSQL Migration을 실행한 것은 아니다. 회원가입을 제공하지 않으므로 사용자, 비밀번호, 개인 프로필, 개인 즐겨찾기 테이블은 포함하지 않는다.

### 실행 가능한 SQL (2026-11-16 추가)

[DB 생성·초기데이터 실행 안내](../../infra/postgres/migrations/README.md)와 [설치 SQL](../../infra/postgres/migrations/000_install.sql)을 제공합니다. ERD 업무 테이블 33개와 보조 테이블 2개, 서울 25개 구·427개 행정동 및 기준 정책을 생성합니다. 임시 PostGIS에서 검증했으며, **운영 DB 적용과 백엔드의 DB 조회 전환은 별도**입니다. 물리 설계에서 추가한 필드·제약조건은 실행 안내에 정리했습니다.

## 시각화 파일

- [전체 구조 PNG](erd_visuals/spotfit_erd_full.png)
- [전체 구조 SVG](erd_visuals/spotfit_erd_full.svg)
- [모델·서비스 상세 PNG](erd_visuals/spotfit_erd_service_detail.png)
- [모델·서비스 상세 SVG](erd_visuals/spotfit_erd_service_detail.svg)
- [브라우저 확대·스크롤 보기](erd_visuals/spotfit_erd_visualization.html)

![SpotFit 전체 ERD](erd_visuals/spotfit_erd_full.png)

![SpotFit 모델·서비스 상세 ERD](erd_visuals/spotfit_erd_service_detail.png)

## 저장소 역할

| 저장소 | 저장 대상 |
|---|---|
| PostgreSQL·PostGIS | 표준 데이터, 행정구역, 모델 메타데이터, 예측값, 점수, 추천 결과, 정책, 익명 통계 |
| MinIO `spotfit-raw` | 수집한 원본 CSV·XLS·XLSX·JSON |
| MinIO `spotfit-curated` | 대용량 전처리 결과와 학습용 Parquet |
| MinIO `spotfit-models` | Joblib 등 서비스 모델 파일 |
| MLflow | 실험 파라미터, 성능지표, Run·Registry 버전, 모델 Artifact 연결 |
| 파일 로그 | API 전체 로그와 예외 Stack Trace |

## 전체 데이터 흐름

```mermaid
flowchart LR
    A[공공데이터 API·파일] --> B[Airflow 수집]
    B --> C[(MinIO Raw)]
    B --> D[(PostgreSQL Meta·Staging)]
    C --> E[Airflow 전처리]
    D --> E
    E --> F[(PostgreSQL Core·Feature)]
    E --> G[(MinIO Curated)]
    F --> H[모델 학습·검증]
    G --> H
    H --> I[MLflow]
    H --> J[(MinIO Models)]
    I --> K[(PostgreSQL ML·Service)]
    J --> K
    K --> L[FastAPI]
    L --> M[React·Leaflet]
```

## 1. 데이터 수집·행정구역·표준 데이터 ERD

```mermaid
erDiagram
    META_DATA_SOURCE ||--o{ META_INGESTION_RUN : executes
    META_INGESTION_RUN ||--o{ META_DATA_ASSET_VERSION : creates
    META_PIPELINE_RUN ||--o{ META_PIPELINE_INPUT : consumes
    META_DATA_ASSET_VERSION ||--o{ META_PIPELINE_INPUT : used_by
    META_PIPELINE_RUN ||--o{ META_DATA_QUALITY_RESULT : checks
    META_DATA_ASSET_VERSION ||--o{ META_DATA_QUALITY_RESULT : receives

    REF_ADMIN_AREA o|--o{ REF_ADMIN_AREA : parent_of
    REF_ADMIN_AREA ||--o{ REF_ADMIN_AREA_ALIAS : has
    REF_BOUNDARY_NOTICE ||--o{ REF_BOUNDARY_NOTICE_AREA : contains
    REF_ADMIN_AREA ||--o{ REF_BOUNDARY_NOTICE_AREA : affected

    REF_ADMIN_AREA ||--o{ CORE_APARTMENT : contains
    REF_ADMIN_AREA ||--o{ CORE_PROPERTY_TRANSACTION : occurs_in
    REF_ADMIN_AREA ||--o{ CORE_POPULATION_STAT : measures
    REF_ADMIN_AREA ||--o{ CORE_CRIME_STAT : measures
    REF_ADMIN_AREA ||--o{ CORE_TRAFFIC_ACCIDENT_STAT : measures
    REF_ADMIN_AREA ||--o{ CORE_FACILITY_SNAPSHOT : contains
    CORE_APARTMENT ||--o{ CORE_PROPERTY_TRANSACTION : has

    META_DATA_ASSET_VERSION ||--o{ CORE_PROPERTY_TRANSACTION : provenance
    META_DATA_ASSET_VERSION ||--o{ CORE_POPULATION_STAT : provenance
    META_DATA_ASSET_VERSION ||--o{ CORE_CRIME_STAT : provenance
    META_DATA_ASSET_VERSION ||--o{ CORE_TRAFFIC_ACCIDENT_STAT : provenance
    META_DATA_ASSET_VERSION ||--o{ CORE_FACILITY_SNAPSHOT : provenance

    META_DATA_SOURCE {
        bigint source_id PK
        varchar source_code UK
        varchar source_name
        varchar provider
        varchar source_url
        varchar update_cycle
        boolean is_active
    }
    META_INGESTION_RUN {
        bigint ingestion_run_id PK
        bigint source_id FK
        varchar airflow_dag_id
        varchar airflow_run_id
        varchar reference_period
        timestamptz started_at
        timestamptz finished_at
        varchar status
        text error_summary
    }
    META_DATA_ASSET_VERSION {
        bigint asset_version_id PK
        bigint ingestion_run_id FK
        varchar object_uri
        varchar checksum_sha256
        bigint file_size
        bigint row_count
        timestamptz collected_at
        boolean is_current
    }
    META_PIPELINE_RUN {
        bigint pipeline_run_id PK
        varchar pipeline_type
        varchar airflow_dag_id
        varchar airflow_run_id UK
        varchar code_commit
        varchar status
        timestamptz started_at
        timestamptz finished_at
    }
    META_PIPELINE_INPUT {
        bigint pipeline_run_id PK,FK
        bigint asset_version_id PK,FK
        varchar input_role PK
    }
    META_DATA_QUALITY_RESULT {
        bigint quality_result_id PK
        bigint pipeline_run_id FK
        bigint asset_version_id FK
        varchar check_name
        varchar severity
        boolean passed
        jsonb details
        timestamptz checked_at
    }
    REF_ADMIN_AREA {
        bigint area_id PK
        varchar area_code
        varchar area_name
        varchar area_level
        bigint parent_area_id FK
        date valid_from
        date valid_to
        boolean is_current
        geometry boundary
    }
    REF_ADMIN_AREA_ALIAS {
        bigint alias_id PK
        bigint area_id FK
        varchar alias_name
        date valid_from
        date valid_to
    }
    REF_BOUNDARY_NOTICE {
        bigint notice_id PK
        varchar notice_type
        varchar title
        date announced_at
        date effective_date
        varchar status
        text source_url
        timestamptz collected_at
    }
    REF_BOUNDARY_NOTICE_AREA {
        bigint notice_id PK,FK
        bigint area_id PK,FK
        varchar relation_type PK
    }
    CORE_APARTMENT {
        bigint apartment_id PK
        bigint area_id FK
        varchar apartment_code UK
        varchar apartment_name
        varchar road_address
        int built_year
        geometry location
    }
    CORE_PROPERTY_TRANSACTION {
        bigint transaction_id PK
        bigint apartment_id FK
        bigint area_id FK
        bigint asset_version_id FK
        varchar transaction_type
        date contract_date
        numeric exclusive_area_sqm
        numeric sale_amount
        numeric deposit_amount
        numeric monthly_rent
        varchar source_transaction_key UK
    }
    CORE_POPULATION_STAT {
        bigint population_stat_id PK
        bigint area_id FK
        bigint asset_version_id FK
        int reference_year
        varchar age_group
        bigint population
    }
    CORE_CRIME_STAT {
        bigint crime_stat_id PK
        bigint area_id FK
        bigint asset_version_id FK
        int reference_year
        varchar crime_type
        bigint occurrence_count
    }
    CORE_TRAFFIC_ACCIDENT_STAT {
        bigint traffic_stat_id PK
        bigint area_id FK
        bigint asset_version_id FK
        int reference_year
        varchar accident_type
        bigint accident_count
        bigint casualty_count
    }
    CORE_FACILITY_SNAPSHOT {
        bigint facility_snapshot_id PK
        bigint area_id FK
        bigint asset_version_id FK
        varchar facility_type
        date reference_date
        bigint facility_count
    }
```

### 행정구역 설계 원칙

- `REF_ADMIN_AREA`는 현재 행정동만 덮어쓰지 않고 `valid_from`, `valid_to`로 과거 버전을 보존한다.
- 분동·통합처럼 한 공고에 여러 구역이 연결될 수 있으므로 공고와 지역은 `REF_BOUNDARY_NOTICE_AREA`로 다대다 연결한다.
- `relation_type`은 `OLD`, `NEW`, `AFFECTED` 중 하나를 사용한다.
- 경계는 PostGIS `MultiPolygon`, 좌표계는 서비스 표준 SRID 하나로 통일한다.

## 2. Feature·모델·예측·추천 ERD

```mermaid
erDiagram
    META_PIPELINE_RUN ||--o{ FEATURE_DATASET_VERSION : produces
    FEATURE_DATASET_VERSION ||--o{ ML_TRAINING_RUN : trains_with
    META_PIPELINE_RUN ||--o{ ML_TRAINING_RUN : executes
    ML_TRAINING_RUN ||--o{ ML_MODEL_VERSION : creates
    ML_MODEL_VERSION ||--o{ ML_MODEL_METRIC : evaluates
    ML_MODEL_VERSION ||--o{ ML_MODEL_FEATURE : uses
    ML_MODEL_VERSION ||--o{ ML_MODEL_DEPLOYMENT : deploys

    ML_MODEL_VERSION ||--o{ SERVICE_PROPERTY_PREDICTION : predicts
    ML_MODEL_VERSION ||--o{ SERVICE_SAFETY_PREDICTION : predicts
    REF_ADMIN_AREA ||--o{ SERVICE_PROPERTY_PREDICTION : located_in
    REF_ADMIN_AREA ||--o{ SERVICE_SAFETY_PREDICTION : located_in
    CORE_APARTMENT ||--o{ SERVICE_PROPERTY_PREDICTION : priced

    SERVICE_SCORING_POLICY ||--o{ SERVICE_GRADE_POLICY : defines
    SERVICE_SCORING_POLICY ||--o{ SERVICE_NEIGHBORHOOD_SCORE : calculates
    SERVICE_SAFETY_PREDICTION ||--o{ SERVICE_NEIGHBORHOOD_SCORE : crime_component
    SERVICE_SAFETY_PREDICTION ||--o{ SERVICE_NEIGHBORHOOD_SCORE : traffic_component
    REF_ADMIN_AREA ||--o{ SERVICE_NEIGHBORHOOD_SCORE : receives

    SERVICE_RECOMMENDATION_POLICY ||--o{ SERVICE_RECOMMENDATION_REQUEST : applies
    SERVICE_RECOMMENDATION_REQUEST ||--o{ SERVICE_RECOMMENDATION_RESULT : returns
    SERVICE_NEIGHBORHOOD_SCORE ||--o{ SERVICE_RECOMMENDATION_RESULT : ranks
    SERVICE_PROPERTY_PREDICTION ||--o{ SERVICE_RECOMMENDATION_RESULT : prices
    REF_ADMIN_AREA ||--o{ SERVICE_RECOMMENDATION_RESULT : recommends

    FEATURE_DATASET_VERSION {
        bigint feature_dataset_id PK
        bigint pipeline_run_id FK
        varchar domain
        varchar feature_version
        date reference_date
        varchar storage_uri
        varchar schema_hash
        bigint row_count
        boolean quality_passed
        timestamptz created_at
    }
    ML_TRAINING_RUN {
        bigint training_run_id PK
        bigint feature_dataset_id FK
        bigint pipeline_run_id FK
        varchar model_type
        varchar mlflow_run_id UK
        varchar algorithm
        varchar status
        timestamptz started_at
        timestamptz finished_at
    }
    ML_MODEL_VERSION {
        bigint model_version_id PK
        bigint training_run_id FK
        varchar model_type
        varchar version
        int prediction_horizon
        varchar artifact_uri
        varchar model_hash
        varchar status
        boolean is_champion
        timestamptz created_at
    }
    ML_MODEL_METRIC {
        bigint metric_id PK
        bigint model_version_id FK
        varchar dataset_split
        varchar metric_name
        numeric metric_value
    }
    ML_MODEL_FEATURE {
        bigint model_feature_id PK
        bigint model_version_id FK
        varchar feature_name
        numeric importance
        numeric weight_pct
        varchar direction
        int display_order
    }
    ML_MODEL_DEPLOYMENT {
        bigint deployment_id PK
        bigint model_version_id FK
        varchar environment
        varchar status
        timestamptz deployed_at
        timestamptz retired_at
    }
    SERVICE_PROPERTY_PREDICTION {
        bigint property_prediction_id PK
        bigint model_version_id FK
        bigint apartment_id FK
        bigint area_id FK
        varchar transaction_type
        date reference_date
        date prediction_date
        numeric exclusive_area_sqm
        numeric predicted_amount
        numeric predicted_price_per_sqm
        timestamptz created_at
    }
    SERVICE_SAFETY_PREDICTION {
        bigint safety_prediction_id PK
        bigint model_version_id FK
        bigint area_id FK
        varchar safety_domain
        int reference_year
        int prediction_year
        numeric predicted_value
        numeric safety_score
        int safety_rank
        varchar safety_grade
        timestamptz created_at
    }
    SERVICE_SCORING_POLICY {
        bigint scoring_policy_id PK
        varchar policy_name
        numeric crime_weight
        numeric traffic_weight
        date valid_from
        date valid_to
        boolean is_active
    }
    SERVICE_GRADE_POLICY {
        bigint grade_policy_id PK
        bigint scoring_policy_id FK
        varchar grade
        int minimum_rank
        int maximum_rank
        varchar color
        int display_order
    }
    SERVICE_NEIGHBORHOOD_SCORE {
        bigint neighborhood_score_id PK
        bigint area_id FK
        bigint scoring_policy_id FK
        bigint crime_prediction_id FK
        bigint traffic_prediction_id FK
        int reference_year
        int prediction_year
        numeric crime_score
        numeric traffic_score
        numeric total_score
        int total_rank
        varchar total_grade
        timestamptz created_at
    }
    SERVICE_RECOMMENDATION_POLICY {
        bigint recommendation_policy_id PK
        varchar policy_name
        int recommendation_count
        boolean one_dong_per_district
        boolean exclude_missing_price
        varchar sort_rule
        boolean is_active
    }
    SERVICE_RECOMMENDATION_REQUEST {
        uuid request_id PK
        bigint recommendation_policy_id FK
        varchar transaction_type
        numeric minimum_budget
        numeric maximum_budget
        numeric preferred_area_sqm
        int crime_prediction_year
        timestamptz requested_at
        timestamptz expires_at
    }
    SERVICE_RECOMMENDATION_RESULT {
        uuid request_id PK,FK
        int recommendation_rank PK
        bigint area_id FK
        bigint property_prediction_id FK
        bigint neighborhood_score_id FK
        numeric predicted_amount
        numeric neighborhood_score
        varchar neighborhood_grade
        jsonb reason_snapshot
    }
```

### 모델 유형 코드

| `model_type` | 모델 |
|---|---|
| `CRIME_EXPOSURE` | 아동 지역범죄 노출 모델 |
| `TRAFFIC_ACCIDENT` | 어린이 교통사고·교통안전 모델 |
| `APARTMENT_SALE` | 아파트 매매가격 모델 |
| `APARTMENT_JEONSE` | 아파트 전세가격 모델 |
| `APARTMENT_MONTHLY_RENT` | 아파트 전월세가격 모델 |

### 비회원 추천 요청 보존 원칙

- 추천 요청에는 이름, 이메일, 전화번호, 비밀번호, 정확한 IP를 저장하지 않는다.
- 예산은 필요하면 정확한 값 대신 구간으로 저장할 수 있다.
- `expires_at` 이후 요청·결과를 자동 삭제하는 보존정책을 적용한다.
- 이전 추천 결과 재현이나 통계가 필요 없으면 `SERVICE_RECOMMENDATION_REQUEST`, `RESULT`는 저장하지 않고 요청 즉시 계산해 반환해도 된다.

## 3. 익명 분석·운영 ERD

```mermaid
erDiagram
    REF_ADMIN_AREA ||--o{ ANALYTICS_SERVICE_EVENT : references
    SERVICE_RECOMMENDATION_REQUEST o|--o{ ANALYTICS_SERVICE_EVENT : generated_by
    META_PIPELINE_RUN o|--o{ OPS_AUDIT_LOG : concerns
    ML_MODEL_VERSION o|--o{ OPS_AUDIT_LOG : concerns
    META_PIPELINE_RUN o|--o{ OPS_SERVICE_INCIDENT : fails_in

    ANALYTICS_SERVICE_EVENT {
        bigint event_id PK
        uuid request_id FK
        bigint area_id FK
        varchar event_name
        varchar transaction_type
        varchar budget_band
        varchar area_band
        jsonb event_properties
        timestamptz occurred_at
    }
    OPS_AUDIT_LOG {
        bigint audit_id PK
        varchar actor_type
        varchar action
        varchar target_type
        varchar target_id
        jsonb before_value
        jsonb after_value
        varchar request_id
        timestamptz created_at
    }
    OPS_SERVICE_INCIDENT {
        bigint incident_id PK
        bigint pipeline_run_id FK
        varchar incident_type
        varchar severity
        varchar status
        text summary
        timestamptz occurred_at
        timestamptz resolved_at
    }
```

### 익명 이벤트 예시

- `SEARCH_AREA`
- `SUBMIT_RECOMMENDATION`
- `OPEN_REGION_DETAIL`
- `CHANGE_BUDGET_BAND`
- `CHANGE_AREA_PRESET`
- `CLICK_RECOMMENDED_AREA`

익명 이벤트에는 개인을 식별할 수 있는 값 대신 거래유형, 예산구간, 면적구간, 지역코드와 발생시각만 저장한다.

## 핵심 제약조건

1. 활성 Champion은 동일 `model_type`·예측기간당 하나만 허용한다.
2. 모델 배포 전 필수 성능지표와 데이터 품질검사가 모두 통과해야 한다.
3. `SERVICE_NEIGHBORHOOD_SCORE`에는 계산 당시 정책과 모델 버전을 추적할 수 있어야 한다.
4. 범죄·교통 점수 가중치 합은 1 또는 100%가 되어야 한다.
5. 추천 결과는 요청 당시 가격·점수 Snapshot을 보존해 이후 모델 갱신에도 재현 가능해야 한다.
6. 동일 원본 체크섬은 중복 수집하지 않는다.
7. 원본 데이터와 모델 Binary는 PostgreSQL에 저장하지 않는다.
8. 행정구역 유효기간이 겹치거나 부모 지역이 순환하지 않도록 검증한다.

`staging` 테이블은 외부 원천마다 컬럼 구성이 달라 이 논리 ERD에서는 생략한다. 실제 구현 시 각 원천 Schema를 거의 그대로 보존하는 별도 테이블로 생성한다.

## 1차 구현 대상

현재 서비스에 먼저 필요한 테이블은 다음과 같다.

1. `REF_ADMIN_AREA`
2. `REF_BOUNDARY_NOTICE`, `REF_BOUNDARY_NOTICE_AREA`
3. `META_DATA_SOURCE`, `META_INGESTION_RUN`, `META_DATA_ASSET_VERSION`
4. `META_PIPELINE_RUN`, `META_DATA_QUALITY_RESULT`
5. `CORE_APARTMENT`, `CORE_PROPERTY_TRANSACTION`
6. `FEATURE_DATASET_VERSION`
7. `ML_TRAINING_RUN`, `ML_MODEL_VERSION`, `ML_MODEL_METRIC`, `ML_MODEL_FEATURE`
8. `SERVICE_PROPERTY_PREDICTION`, `SERVICE_SAFETY_PREDICTION`
9. `SERVICE_SCORING_POLICY`, `SERVICE_GRADE_POLICY`, `SERVICE_NEIGHBORHOOD_SCORE`
10. `SERVICE_RECOMMENDATION_POLICY`

추천 요청 이력과 익명 이용통계는 서비스 분석이 필요해지는 시점에 2차로 추가해도 된다.
