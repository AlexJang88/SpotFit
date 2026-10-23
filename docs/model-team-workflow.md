# SpotFit 모델 협업·버전 관리 규칙

이 문서는 여러 팀원이 동시에 모델을 실험할 때 MLflow, PostgreSQL, MinIO의 역할을
명확히 나누고 모델명·버전·파일 충돌을 방지하기 위한 규칙입니다.

## 1. 구성요소별 역할

| 구성요소 | 저장 내용 | 관리 원칙 |
| --- | --- | --- |
| MLflow Tracking | Experiment, Run, 파라미터, 성능, 태그, 선정 근거 | 학습 코드에서 MLflow API로 기록 |
| PostgreSQL `mlflow` DB | MLflow 실험·Run·등록 모델·버전 메타데이터 | 직접 수정 금지 |
| MLflow 아티팩트용 MinIO | Run 모델, 보고서, 입력 예시, 의존성 | MLflow가 저장·조회 |
| 서비스 모델용 MinIO | 배포 대상으로 최종 선정된 champion 모델만 저장 | `model-tools upload`로 1회 발행 |
| `model_registry/` | 서비스 MinIO 객체 경로·객체 버전 ID·SHA-256 | 업로드 도구가 생성 |
| `artifacts/`, `models/` | 로컬 실험 결과와 업로드 전 모델 | 임시 작업·검증 공간 |

PostgreSQL에는 모델 파일이 아니라 메타데이터가 저장됩니다. 실제 모델 파일은 MinIO에
저장됩니다. MLflow 아티팩트 버킷은 실험 재현용이고 서비스 모델 버킷은 최종 배포용입니다.
학습 중간 모델과 비교 후보는 서비스 모델용 MinIO에 올리지 않습니다.

```text
학습 코드
  └─ MLflow HTTP API
       ├─ PostgreSQL: Experiment / Run / Model Version 정보
       └─ MLflow MinIO: Run 아티팩트와 Logged Model

최종 선정된 champion 모델만
  └─ model-tools
       ├─ 서비스 MinIO: 불변 모델 파일
       └─ model_registry: 위치·버전·SHA-256 정보
```

## 2. 모델 식별자와 버전

### Experiment

같은 목적의 비교 실험은 같은 Experiment 이름을 사용합니다.

```text
child-population-algorithm-comparison
house-price-forecast-comparison
```

Experiment 이름에 팀원 이름이나 `v1`을 붙이지 않습니다. 개별 실행은 자동 생성되는 Run ID로 구분합니다.

### Registered Model

모델명은 담당자 이름이 아니라 서비스 목적과 타깃으로 정합니다.

```text
spotfit-child-count-elementary
spotfit-child-count-infant
spotfit-house-price
```

### MLflow Model Version

같은 Registered Model 이름에 등록하면 MLflow가 Version 1, 2, 3을 자동으로 부여합니다.
팀원이 번호를 직접 지정하거나 예상하지 않습니다.

```text
spotfit-child-count-elementary
  ├─ Version 1
  ├─ Version 2
  └─ Version 3
```

서로 다른 Registered Model은 각각 Version 1부터 시작할 수 있습니다.

### 최종 배포 MinIO Version

MinIO 업로드의 `--version v1`, `v2`는 최종 배포 버전입니다. MLflow의 후보 버전마다
증가시키지 않고 실제 배포 모델이 교체될 때만 새 버전을 발급합니다. MLflow Version 번호와
자동으로 연결되지 않습니다.

```text
spotfit-child-count-elementary/v1
spotfit-house-price/v1
```

모델명이 다르므로 각각 `v1`을 사용할 수 있습니다. 같은 모델명·같은 버전에 다른 파일을
올리지 않습니다. 재학습 결과는 새로운 버전을 사용합니다.

| 항목 | 예시 | 생성 방식 |
| --- | --- | --- |
| Experiment | `child-population-algorithm-comparison` | 목적별 고정 |
| Run ID | `abc123...` | MLflow 자동 생성 |
| Registered Model | `spotfit-child-count-elementary` | 타깃별 고정 |
| MLflow Version | `3` | MLflow 자동 생성 |
| MinIO Version | `v2` | 최종 배포 모델 교체 시 발급 |
| 모델 별칭 | `candidate`, `champion` | 승인 후 변경 |

## 3. 역할 분리

| 역할 | 책임 |
| --- | --- |
| 데이터 담당 | 학습 데이터 기준일·행정동 매핑·파일 SHA-256 관리 |
| 모델 개발자 | 모델 비교, MLflow Run 생성, 결과와 한계 기록 |
| 검증 담당 | 누수·분할·기준 모델·교차검증·선정 근거 검토 |
| 모델 릴리스 담당 | 최종 champion 확인, 서비스 MinIO 발행, 배포 이력 관리 |
| 서비스 담당 | 승인 모델 다운로드, 입력 검증, 배포와 롤백 |

- 모든 팀원은 Experiment와 Run을 생성할 수 있습니다.
- 모델 개발자는 비교 근거를 바탕으로 Registry 후보 버전을 등록합니다.
- 검증 담당자가 확인한 최종 한 버전에만 `champion` 별칭을 지정합니다.
- 서비스 MinIO에는 champion으로 확정된 모델만 릴리스 담당자가 업로드합니다.
- 후보·실험·보류 모델은 서비스 MinIO에 업로드하지 않습니다.
- 다른 팀원의 Run·모델·아티팩트를 삭제하거나 수정하지 않습니다.
- 암호·키·토큰을 MLflow 파라미터·태그·설명에 기록하지 않습니다.

## 4. 학습 전 데이터 고정

비교 실험은 데이터 경로를 반드시 명시합니다. 최신 폴더 자동 선택에 의존하면 팀원마다
다른 데이터가 선택될 수 있습니다. 현재 전처리 결과는 `data/interim`에서 관리합니다.

```powershell
$trainingData = 'data/interim/child_population/facility_child_counts_20261023_093757_426625/analysis_panel.csv'
```

학습 전 다음 정보를 기록합니다.

- 데이터 상대 경로와 SHA-256
- 기준연도와 생성일
- 행정동 수
- 타깃과 연령 구간 정의
- 입력 후보와 결측 처리 정책
- 학습·검증·테스트 또는 교차검증 분할 파일 SHA-256
- 난수 seed

같은 seed라도 데이터 행 순서가 달라지면 분할 결과가 달라질 수 있습니다. `cv_split.csv`의
행정동 코드까지 비교해야 같은 실험 조건인지 확인할 수 있습니다.

## 5. 공유 MLflow 연결

공유 실험은 로컬 SQLite가 아니라 Ubuntu MLflow HTTP 주소를 지정합니다.
학습 PC가 PostgreSQL에 직접 접속하지 않습니다. MLflow 서버가 PostgreSQL과 MinIO 연결을 담당합니다.

```powershell
$trackingUri = 'http://192.168.0.3:5000'
```

MLflow 버전이 팀원 PC와 서버에서 다르면 새 API 또는 DB 스키마 차이로 오류가 발생할 수 있습니다.
`requirements-dev.txt`와 `requirements-server.txt`의 MLflow 버전을 같은 값으로 유지합니다.

```powershell
python -c "import mlflow; print(mlflow.__version__)"
```

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml \
  exec mlflow mlflow --version
```

MLflow UI를 서버 IP로 접속할 때에는 해당 주소를 allowed hosts와 CORS origin에 명시합니다.
보안 미들웨어 전체 비활성화나 `*` 허용은 사용하지 않습니다.

## 6. 모델 학습과 알고리즘 비교

개발 환경 설치:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

초등 학령기 모델 비교 예시:

```powershell
python ml/training/train_child_feature_subsets.py `
  --data $trainingData `
  --ages elementary `
  --folds 5 `
  --tracking-uri $trackingUri
```

기본 비교 대상:

- 평균값 예측: 시설 정보를 사용하지 않는 기준 모델
- Ridge: 선형 관계와 단순성 확인
- 얕은 결정트리: 설명 가능한 비선형 규칙 확인
- RandomForest: 여러 트리 결합 효과 확인
- ExtraTrees: 무작위 분할 방식의 안정성 확인
- HistGradientBoosting: 순차 오차 보완 효과 확인

모든 후보는 동일한 개발 표본과 교차검증 폴드를 사용합니다. 시설 상위 5개는 각 폴드의
학습 부분에서만 다시 선정해 검증 데이터 누수를 막습니다.

## 7. 필수 평가 결과

모델 선정 전 다음 자료가 있어야 합니다.

| 파일·기록 | 의미 |
| --- | --- |
| `comparison.csv` | 알고리즘·조합별 CV 성능과 실행 시간 |
| `fold_results.csv` | 폴드별 점수와 실제 시설 조합 |
| `selection_reasons.json` | 선정 또는 보류 이유 |
| `report.html` | 전체 비교 요약 |
| `cv_split.csv` | 지역 분할 재현 정보 |
| MLflow Run | 파라미터·지표·태그·모델 아티팩트 |

평가 지표:

- MAE: 평균적으로 몇 명 차이가 나는지 표시하며 낮을수록 좋음
- RMSE: 큰 오차에 더 큰 페널티를 주며 낮을수록 좋음
- R²: 기준 분산 대비 설명력이며 높을수록 좋고 음수가 될 수 있음
- MAE 표준편차: 폴드에 따른 성능 변동
- 기준 모델 대비 개선율
- 학습·예측 시간

MAE가 가장 낮다는 이유만으로 복잡한 모델을 선택하지 않습니다. 현재 비교 규칙은 최저 MAE의
1 표준오차 범위에서 단순한 알고리즘과 적은 특징을 우선합니다. 평균 기준 대비 MAE 5% 이상 개선하고
전체 폴드 중 80% 이상에서 기준보다 좋아야 탐색 후보로 유지합니다.

이 기준은 통계적 유의성 검정이 아니라 사전에 정한 운영 기준입니다. 조건을 통과하지 못하면
`selection_deferred`로 기록하고 공식 모델 선정을 보류합니다.

기존 테스트를 반복해서 확인하며 모델을 변경하지 않습니다. 이미 확인한 테스트는 참고 결과이며,
최종 성능을 주장하려면 새로운 독립 데이터가 필요합니다.

## 8. MLflow 기록 규칙

### 부모 Run 필수 항목

- 데이터·분할 SHA-256
- 기준연도와 표본 수
- 폴드 수와 seed
- 타깃·연령 정의
- 후보 알고리즘 목록
- 모델 선정 규칙
- 담당자와 작업 번호
- 미래 예측 또는 인과모델이 아니라는 제한

### 후보 Run 필수 항목

- 알고리즘과 파라미터
- 특징 순위 조합과 폴드별 실제 시설
- CV MAE 평균·표준편차, RMSE, R²
- 기준 모델 대비 개선율과 폴드 승리 횟수
- `exploratory_candidate` 또는 `selection_deferred`
- 한국어 선정·보류 이유
- 입력 예시와 패키지 버전

공통 태그:

```text
owner
task_id
dataset_sha256
split_sha256
review_status
selection_status
```

코드 주석은 MLflow 설명에 자동 반영되지 않으므로 선정 이유와 제한사항을 별도로 기록합니다.

## 9. PostgreSQL 관리 규칙

PostgreSQL의 `mlflow` DB에는 다음 메타데이터가 저장됩니다.

- `experiments`: Experiment 정보
- `runs`: 실행 상태와 시간
- `params`: 학습 파라미터
- `metrics`, `latest_metrics`: 평가 지표
- `tags`: Run 태그와 설명
- `logged_models`: MLflow에 저장한 모델 정보
- `registered_models`: 공식 등록 모델 이름
- `model_versions`: 등록 모델 버전

테이블 이름은 MLflow 버전에 따라 달라질 수 있으므로 먼저 `information_schema.tables`를 확인합니다.

PostgreSQL에서는 읽기 전용 조회만 허용합니다. 모델 등록이나 버전 변경을 위해 직접
`INSERT`, `UPDATE`, `DELETE`하지 않습니다. 모든 변경은 MLflow API를 사용해야 메타데이터와
MinIO 아티팩트 경로가 일관되게 유지됩니다.

MLflow 화면 오류가 발생하면 DB를 고치기 전에 다음을 확인합니다.

1. MLflow 서버 로그
2. 클라이언트와 서버 MLflow 버전
3. allowed hosts와 CORS 설정
4. PostgreSQL 스키마 마이그레이션 상태

DB 마이그레이션은 반드시 백업 후 서버 담당자가 수행합니다.

## 10. 모델 검토와 MLflow Registry 버전 관리

검증 담당자는 다음을 확인합니다.

1. 데이터 누수와 중복 행정동이 없는가
2. 특징 선정과 결측 대체가 폴드 학습 내부에서 수행됐는가
3. 모든 모델이 같은 데이터·분할·평가 지표로 비교됐는가
4. 평균 기준 모델보다 안정적으로 개선됐는가
5. 특정 폴드나 특정 지역에만 성능이 의존하지 않는가
6. 선정·탈락 이유가 근거 수치와 함께 기록됐는가
7. 서비스에서 표현하려는 목표와 모델 타깃이 일치하는가

모델 개발자는 비교 기준을 통과한 후보를 `runs:/<run_id>/model` URI로 Registry에 등록합니다.
후보 모델 파일은 MLflow 아티팩트 저장소에, 버전 메타데이터는 PostgreSQL에 저장됩니다.
MLflow가 반환한 실제 버전 번호를 기록하고 번호를 미리 가정하지 않습니다.

등록 버전 필수 태그:

```text
validation_status=candidate
source_run_id=<Run ID>
dataset_sha256=<데이터 해시>
split_sha256=<분할 해시>
deployment_status=not_deployed
```

`selection_deferred` 모델은 공식 후보로 등록하지 않습니다. 비교 목적으로 등록해야 한다면
`exploratory_not_for_deployment` 태그와 설명을 반드시 붙입니다.

후보 버전은 여러 개 존재할 수 있습니다. 알고리즘·특징·데이터가 달라질 때마다 새 Run과
새 MLflow Model Version으로 관리하고 기존 버전을 덮어쓰지 않습니다. 이 단계에서는 서비스
MinIO 업로드를 수행하지 않습니다.

최종 모델은 단순 정확도 하나가 아니라 다음 기준으로 선정합니다.

1. 동일 데이터·분할에서 기준 모델과 여러 알고리즘을 비교했는가
2. 교차검증 평균과 표준편차, 폴드별 결과가 안정적인가
3. 기준 모델 대비 개선 폭이 서비스에 의미 있는가
4. 입력 컬럼과 전처리를 서비스에서 동일하게 재현할 수 있는가
5. 추론 시간과 모델 크기가 운영 환경에 적절한가
6. 모델 목적과 제한사항이 사용자에게 표시할 내용과 일치하는가

검증 담당자가 확인한 최종 한 버전에만 `champion` 별칭을 지정합니다. 나머지는 MLflow에서
candidate·보류·폐기 상태로 남기고 서비스 MinIO에는 올리지 않습니다.

## 11. 최종 champion 모델만 서비스 MinIO에 업로드

MLflow Registry 등록과 서비스 MinIO 업로드는 별도 단계입니다. Registry에는 여러 후보 버전이
있을 수 있지만 서비스 MinIO에는 최종 선정된 champion 한 버전만 올립니다.

```text
모델명: spotfit-child-count-elementary
최종 MinIO 배포 버전: v2
MLflow Run ID: <run_id>
MLflow Version: <자동 부여 번호>
담당자: <이름>
선정 근거: <CV·독립 평가·기준 모델 비교 요약>
상태: champion_approved
```

같은 모델명·같은 버전에 다른 파일을 올리지 않습니다. MinIO 버전은 최종 배포 모델이 교체될
때만 증가합니다. 후보 A·B·C를 비교했다고 MinIO v1·v2·v3을 각각 만들지 않습니다.

MLflow의 정확한 champion Version에서 모델을 내려받거나 동일 Run의 검증된 파일을 사용해
새 로컬 경로에 준비합니다. Run ID와 SHA-256을 확인한 뒤 업로드합니다.

```powershell
$modelName = 'spotfit-child-count-elementary'
$releaseVersion = 'v2'
$releaseDir = "artifacts/models/$modelName/$releaseVersion"

if (Test-Path -LiteralPath $releaseDir) {
  throw '이미 존재하는 버전입니다. 새 버전을 사용하세요.'
}

New-Item -ItemType Directory -Path $releaseDir | Out-Null
Copy-Item -LiteralPath '검토완료-model.joblib' -Destination "$releaseDir/model.joblib"
```

서비스 모델 하나를 업로드합니다.

```powershell
docker compose --env-file .env.dev -f compose.dev.yaml run --rm model-tools upload `
  --file "/artifacts/models/$modelName/$releaseVersion/model.joblib" `
  --name $modelName `
  --version $releaseVersion
```

업로드가 성공해야 다음 manifest가 생성됩니다.

```text
model_registry/spotfit-child-count-elementary/v2.json
```

manifest에는 MinIO 버킷, 객체 키, 객체 버전 ID, SHA-256과 파일 크기가 포함됩니다.
원격 객체 없이 manifest를 수동으로 만들지 않습니다.

`upload_minio.py`는 최종 추천 모델을 위한 `models/recommender` 하위만 처리합니다. 이 폴더에는
실험·후보 모델을 넣지 않고 배포가 확정된 추천 모델만 둡니다. 다른 종류의 최종 모델은 위의
`model-tools upload` 명령으로 하나씩 올립니다. 아동 인구 모델을 `models/recommender`에 복사해
우회 업로드하지 않습니다.

## 12. 최종 모델의 MLflow와 MinIO 연결 기록

모델마다 다음 매핑을 유지합니다.

| 항목 | 예시 |
| --- | --- |
| Registered Model | `spotfit-child-count-elementary` |
| MLflow Version | `3` |
| Source Run ID | `abc123...` |
| MinIO Version | `v2` |
| Manifest | `model_registry/spotfit-child-count-elementary/v2.json` |
| SHA-256 | manifest의 모델 해시 |
| 선정 상태 | `champion` |

MinIO 업로드 후 champion MLflow Model Version의 `minio_release_version` 태그를 실제 버전으로
갱신하고 `deployment_status=published`를 기록합니다.
MLflow Version 2와 MinIO `v2`처럼 숫자가 같아도 자동으로 같은 모델이라고 판단하지 않습니다.
Run ID와 SHA-256을 함께 확인합니다.

## 13. 모델 별칭과 배포

- `candidate`: 검증 중인 버전이며 서비스 MinIO에 올리지 않음
- `champion`: 최종 사용이 승인되어 서비스 MinIO 발행 대상인 버전
- 별칭 변경은 릴리스 담당자만 수행
- 별칭 변경 전후 버전을 작업 기록에 남김
- 버전 없는 `latest` 모델 자동 선택 금지

MLflow 별칭을 바꿔도 실행 중인 API 모델은 자동으로 교체되지 않습니다. champion 결정 후
서비스 MinIO 발행과 배포를 별도로 수행해야 합니다. 서비스는 승인한
manifest 또는 명시적인 Registry Version을 내려받고 재로딩·배포해야 합니다.

배포 전 확인:

- 모델 파일 SHA-256
- 입력 컬럼명과 순서
- 입력 자료형과 결측 처리
- 학습 패키지 버전
- 대표 샘플 예측
- 이전 champion으로의 롤백 가능 여부

## 14. 실패 처리와 롤백

### 학습 실패

실패한 Run을 삭제하지 않습니다. 오류 원인과 `FAILED` 상태를 남기고 수정 후 새 Run을 생성합니다.

### Registry 등록 실패

PostgreSQL을 직접 수정하지 않습니다. Run의 모델 아티팩트 존재 여부와 MLflow 서버 로그를 확인하고
새 API 요청으로 처리합니다.

### MinIO 업로드 실패

manifest를 수동 생성하지 않습니다. 원격 객체와 로컬 해시를 확인합니다. 다른 후보 모델로
대체 업로드하거나 서로 다른 파일로 같은 버전을 덮어쓰지 않습니다.

### 배포 실패

이전 champion과 manifest를 보존합니다. 이전 승인 버전으로 되돌린 뒤 API를 재시작하고
대표 샘플 예측을 다시 확인합니다.

공유 환경에서 금지하는 작업:

- PostgreSQL MLflow DB 직접 수정·초기화
- MinIO 버킷 또는 객체 버전 임의 삭제
- 다른 팀원의 Run·Registered Model 삭제
- 기존 manifest 내용 변경
- 공유 서버에서 `docker compose down -v`

## 15. 체크리스트

### 모델 개발자

- [ ] 승인된 데이터와 분할을 사용했다.
- [ ] 데이터·분할 SHA-256을 기록했다.
- [ ] 기준 모델과 여러 알고리즘을 같은 조건으로 비교했다.
- [ ] 폴드별 성능과 실제 특징을 저장했다.
- [ ] 선정·보류 이유와 한계를 MLflow에 기록했다.
- [ ] 테스트 결과를 보고 모델을 다시 고르지 않았다.
- [ ] 후보를 MLflow Registry에서 버전으로 관리했다.
- [ ] 후보 모델을 서비스 MinIO에 직접 업로드하지 않았다.

### 검증 담당

- [ ] 누수·중복·결측 처리를 검토했다.
- [ ] CV 평균과 표준편차를 확인했다.
- [ ] 기준 모델 대비 개선을 확인했다.
- [ ] 특정 지역이나 폴드에 대한 의존을 확인했다.
- [ ] 서비스 사용 가능 여부를 승인 또는 보류했다.

### 릴리스 담당

- [ ] MLflow가 부여한 실제 Version을 기록했다.
- [ ] 검증 완료된 champion 모델인지 확인했다.
- [ ] champion 한 버전만 서비스 MinIO 발행 대상으로 선택했다.
- [ ] 최종 배포 MinIO 버전을 발급했다.
- [ ] 동일 버전을 덮어쓰지 않았다.
- [ ] manifest와 SHA-256을 확인했다.
- [ ] MLflow Version과 MinIO Version을 Run ID·해시로 연결했다.
- [ ] candidate/champion 변경 이력을 남겼다.
- [ ] 후보 버전 수와 관계없이 최종 모델만 MinIO에 올렸다.

### 서비스 담당

- [ ] 승인된 정확한 모델 버전을 사용한다.
- [ ] 입력 스키마와 패키지 버전을 확인했다.
- [ ] 배포 후 대표 예측과 로그를 확인했다.
- [ ] 이전 모델로 롤백할 수 있다.

## 16. 아직 자동화되지 않은 부분

다음 항목은 시스템이 완전히 강제하지 않으므로 담당자 절차로 관리합니다.

- 담당자와 작업 번호 자동 태깅
- 최종 MinIO 버전 발행 잠금
- Registry 등록과 별칭 변경 권한 분리
- MLflow 모델과 서비스 MinIO 파일의 SHA-256 자동 비교
- 검증 승인 후 자동 배포
- 새로운 독립 테스트 데이터 확보

관련 문서: [모델 저장 안내](model-storage.md),
[MLflow 조합 실험](child-feature-subsets-mlflow.md),
[예측 사용법](child-prediction-usage.md).
