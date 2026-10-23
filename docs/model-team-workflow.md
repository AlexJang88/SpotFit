# 모델 학습·MLflow·MinIO 팀 협업 규칙

작성 기준: 2026-10-23. 현재 구현을 기준으로 한 팀 운영 제안입니다.
아래 추가된 12절이 기존 RF-only 실행·선정 안내보다 우선합니다.
담당자 지정·승인 절차는 팀 합의 후 적용하며, 아래 규칙이 코드나 서버 권한으로
모두 강제되는 것은 아닙니다. 이 문서 작성으로 서버 업로드·배포는 수행하지 않습니다.

## 1. 저장소별 역할

| 구분 | 역할 | 팀 공유 방법 |
| --- | --- | --- |
| Git | 코드, 의존성, 문서, MinIO manifest | 기능 브랜치 → PR 리뷰 → 병합 |
| data/processed | 분석·학습 입력 | 기준 파일·해시를 합의하고 별도로 공유 |
| artifacts/child_feature_subsets | 로컬 조합 모델·평가 CSV·보고서 | Git 제외, MLflow 아티팩트로 기록 |
| MLflow Tracking | Run, 설정, 성능, 코드, 모델 아티팩트 | 공유 Ubuntu 서버 사용 |
| MLflow Registry | 선정 모델 버전·태그·별칭 | 등록 담당자 관리 |
| models/<name>/<version>/model.joblib | MinIO 업로드 전 로컬 패키지 | Git 제외 |
| MinIO 모델 버킷 | 배포·공유할 불변 모델 파일 | 기존 model-tools로 업로드 |
| model_registry/<name>/<version>.json | 객체 경로·객체 버전 ID·SHA-256 | 업로드 성공 후 Git PR |

MLflow 아티팩트 저장소와 서비스용 MinIO 모델 버킷은 별도입니다.
같은 모델 파일이 실험 보존용과 공유용으로 각각 존재할 수 있습니다.
MLflow 버전 3과 MinIO v3가 같은 모델이라는 보장은 없습니다.
반드시 Run ID·파일 SHA-256·manifest 경로로 연결을 기록합니다.

## 2. 전체 절차

```text
데이터·코드 기준 고정
  → 조합 학습 및 검증 비교(MLflow Run)
  → 검증 기준으로 후보 선정
  → 테스트 평가 및 리뷰
  → MLflow Registry 등록 / 로컬 모델 내보내기
  → MinIO 업로드 및 manifest 생성
  → manifest·검토 기록 PR
  → 별도 승인 후 서비스 배포·검증
```

현재 스크립트는 학습·테스트 평가를 한 번에 수행합니다. --register-best를 주면
리뷰 전이라도 Registry에 등록되므로 탐색용으로만 취급합니다. 등록은 배포가 아닙니다.
champion 자동 지정, 추론 API 자동 교체, CI/CD 승인은 구현되어 있지 않습니다.

## 3. 작업 시작 전 체크

1. 자신의 변경을 보존하고 작업 트리를 확인한 뒤 최신 main에서 기능 브랜치를 만듭니다.
2. `feature/ml-<작업>-<담당자>`처럼 담당자를 구분합니다. main 직접 push는 하지 않습니다.
3. 동일 학습 파일 수정이 필요하면 담당자를 먼저 정합니다. 다른 팀원의 변경을 덮어쓰지 않습니다.
4. 데이터 경로, SHA-256, 연도, 타깃, 결측 정책, 동 분할, 비교 지표를 합의합니다.
5. 비교 실험은 반드시 --data를 지정합니다. 생략 시 최신 로컬 폴더를 자동 선택하므로
   팀원마다 다른 데이터가 선택될 수 있습니다.
6. 데이터 행 순서도 유지합니다. 같은 seed라도 행 순서가 달라지면 분할이 달라집니다.
   split.csv의 행정동 목록이 동일한지 확인합니다.
7. MLflow 서버 주소와 권한을 확인합니다. .env.dev 값을 이 학습 스크립트가 자동으로
   읽는 것은 아닙니다. --tracking-uri를 명시합니다.

## 4. 학습 실행 예시

프로젝트 루트의 PowerShell에서, 공유 서버가 직접 접근 가능한 경우:

```powershell
# 실제 승인된 서버 주소와 데이터 경로로 바꿉니다.
$trackingUri = 'http://서버IP:5000'
$trainingData = 'data/processed/child_population/승인된결과폴더/analysis_panel.csv'
python ml/training/train_child_feature_subsets.py --data $trainingData --tracking-uri $trackingUri --ages elementary
```

직접 접속이 제한된 서버는 별도 터미널에서 다음 터널을 유지합니다.

```powershell
ssh -N -L 5002:127.0.0.1:5000 사용자@서버IP
```

이때 --tracking-uri는 http://127.0.0.1:5002입니다. 서버 공개 범위·방화벽은
서버 담당자가 관리하며, 인증 없이 인터넷에 MLflow를 공개하지 않습니다.

현재 실험 이름은 child-population-feature-subsets로 고정되어 있습니다.
Run 이름은 subsets_<시각>, 자식은 <age>_<조합번호>_<요소수>features입니다.
동일 이름의 Run이 존재해도 Run ID로 구분됩니다. 이름을 고유 키로 사용하지 않습니다.
각 개인 PC에서는 같은 출력 폴더 충돌을 피하도록 이 스크립트를 한 번에 하나만 실행합니다.
여러 팀원이 로컬 SQLite DB를 네트워크 드라이브로 공유하지 말고 공유 MLflow를 사용합니다.
서버 최초 실험 생성은 담당자가 먼저 한 번 수행한 뒤 팀원이 사용합니다.

## 5. Run에 남겨야 할 설명

현재 자동 기록과 수동 보완을 구분합니다.

| 항목 | 현재 코드 |
| --- | --- |
| 데이터 연도·해시·분할 수·seed | 부모 Run 자동 |
| 연령·선택 요소·RF 설정·검증 점수 | 자식 Run 자동 |
| 테스트·평균예측 기준 성능 | 선정된 자식 Run 자동 |
| 코드·분할·순위·보고서 | 아티팩트 자동 |
| 담당자·Git commit·이슈 번호 | 자동 아님, UI 태그로 보완 |
| 상세 모델 설명·변경 이유 | 자동 아님, UI 설명 또는 PR 기록 |
| MinIO 내보내기 경로·해시 연결 | minio_export.csv 로컬 생성, MLflow 자동 첨부 아님 |

팀 공통 수동 태그: owner, git_commit, issue, dataset_id, purpose,
review_status. 부모 Run에 필수 기록하고 선정 자식 Run에도 연결합니다.
기존 코드에 없는 --owner, --experiment-name 옵션을 임의로 사용하지 않습니다.

설명 템플릿:

```text
담당자 / 검토자:
이슈 / Git commit / 미커밋 변경 여부:
학습 데이터 경로·SHA-256 / 기준연도:
타깃·연령 정의 / 입력 요소 / 결측 정책:
분할 기준 / 검증 MAE / 테스트 MAE·RMSE·R²:
기준 모델과 비교 / 이전 후보 대비 변경점:
MLflow 서버 식별명 / experiment / parent_run_id / winner_run_id:
MLflow 모델명·버전:
MinIO 모델명·버전 / manifest 경로 / 모델 파일 SHA-256:
제약·금지 용도 / 승인 상태 / 롤백 대상:
```

비밀번호·키·토큰·접속 문자열의 비밀 부분을 태그나 설명에 적지 않습니다.
코드 주석은 MLflow 설명 칸에 자동으로 표시되지 않습니다.

## 6. 평가·선정 규칙

- 학습 자료에서만 상위 5개 특징·결측 대체값을 구합니다.
- 같은 연령군, 데이터, 동 분할, 기준 모델에서만 조합 성능을 직접 비교합니다.
- 검증 MAE로 선정하고 테스트 점수로 조합을 다시 고르지 않습니다.
- 반복 실험으로 이미 본 테스트에 맞추게 되면 최종 성능 확인용 추가 홀드아웃이 필요합니다.
- 실패한 Run은 삭제하지 않고 원인을 남깁니다. 다른 사람의 Run·모델·아티팩트를 수정하지 않습니다.
- 전체 인구수 규모와 시설 수의 상관을 육아환경의 질이나 인과효과로 표현하지 않습니다.
- 현재 자료는 시설의 과거 값이 동일합니다. 현재 인구 추정 결과를 미래 인구 예측으로 발표하지 않습니다.
- 현재 영유아·초등 선정 모델은 평균값 기준보다 MAE가 나쁩니다. Registry 등록을
  서비스 채택으로 해석하지 않으며, 배포는 보류하고 추가 검증합니다.

## 7. 버전 충돌 방지

### 실험과 등록

실험은 팀원들이 병렬 진행할 수 있지만 같은 모델의 공식 Registry 등록은 담당자 1명이
직렬로 진행합니다. 숫자 버전은 MLflow가 할당하며 반환된 실제 번호를 기록합니다.
다른 사람의 v1을 자신의 v1로 가정하지 않습니다. champion/candidate 별칭 변경은
릴리스 담당자만 수행하고 변경 전후 버전을 기록합니다. 현재 권한 강제 여부는 별도 확인이 필요합니다.

### MinIO 내보내기·공유

팀원 각자 v2를 사용하면 충돌하므로 개발 후보는 다음처럼 담당자·작업번호를 넣습니다.

```text
dev-memberA-issue123-20261023-01
```

현재 내보내기 코드는 영문·숫자·하이픈·밑줄만 허용합니다. 등록 담당자는 공식
v1/v2/...를 작업 이슈에서 먼저 예약합니다. Git의 로컬 중복 검사만으로 서로 다른
PC의 동시 업로드를 막을 수 없으므로 **동일 모델·버전 업로드는 담당자만** 수행합니다.
이미 등록한 모델 바이트·manifest는 수정하지 않고 새 버전을 생성합니다.

학습 없이 기존 선정 모델 내보내기:

```powershell
python ml/training/export_child_models.py --results artifacts/child_feature_subsets/20261023_095054 --version dev-memberA-issue123-20261023-01
```

학습·Registry 등록·내보내기를 함께 하는 명령은 등록 담당자만 사용합니다.

```powershell
python ml/training/train_child_feature_subsets.py --data $trainingData --tracking-uri $trackingUri --register-best --export-version dev-memberA-issue123-20261023-01
```

이 명령도 MinIO 공유 모델 업로드·manifest 생성·서비스 활성화는 수행하지 않습니다.

## 8. MinIO 업로드와 Git PR

models는 기본 model-tools 컨테이너 마운트가 아니므로 파일을 artifacts/models의
새 후보 경로로 복사해 업로드합니다. 다음 예시는 초등 모델 하나만 업로드합니다.

```powershell
$modelName = 'spotfit-child-count-elementary'
$candidateVersion = 'dev-memberA-issue123-20261023-01'
$candidateDir = "artifacts/models/$modelName/$candidateVersion"
if (Test-Path -LiteralPath $candidateDir) { throw '새 버전 경로를 사용하세요.' }
New-Item -ItemType Directory -Path $candidateDir | Out-Null
Copy-Item -LiteralPath "models/$modelName/$candidateVersion/model.joblib" -Destination "$candidateDir/model.joblib"
docker compose --env-file .env.dev -f compose.dev.yaml run --rm model-tools upload --file "/artifacts/models/$modelName/$candidateVersion/model.joblib" --name $modelName --version $candidateVersion
```

성공 후 생성된 model_registry/<name>/<version>.json과 로컬 파일 해시를 확인합니다.
업로드 실패 시 JSON을 수동으로 꾸며 만들지 않습니다. 같은 원인으로 실패하면
기존 원격 객체 유무를 담당자가 점검하고 새 버전 정책에 따라 처리합니다.
upload_minio.py는 data도 업로드하므로 모델 하나 공유에는 위 전용 CLI를 우선 사용합니다.
버킷 생성·버전 관리 초기화(init)는 서버 담당자가 준비합니다. 팀원이 운영 설정을 임의 변경하지 않습니다.

PR에는 코드·문서·의존성·실제 업로드된 manifest만 추가합니다.
원본 데이터, joblib, SQLite, .env*, 인증키는 커밋하지 않습니다.
manifest 충돌은 한쪽 내용을 임의 선택하지 말고 다른 버전을 배정합니다.
PR 설명에 5절 템플릿을 작성하고 최소 1명 이상 검토합니다.
Git commit 해시가 필요한 경우 깨끗한 학습 코드 커밋을 먼저 만들고 그 커밋으로 학습합니다.

## 9. 역할 분담 제안 (5명)

| 역할 | 책임 |
| --- | --- |
| 데이터 담당 | 원천·기준일·행정동 매핑·데이터 해시와 입력 스냅샷 |
| 모델 담당 | 학습 코드·조합 실험·Run 설명 |
| 검증 담당 | 데이터 누수·기준 모델·분할·테스트 해석 리뷰 |
| 저장소/릴리스 담당 | Registry 등록·버전 예약·MinIO 업로드·별칭·롤백 |
| 서비스 담당 | 입력 스키마·모델 로딩·API/화면·배포 후 동작 검증 |

서버 운영 담당은 저장소 담당이 겸임할 수 있습니다. 담당자 부재 시 대행자를 이슈에
명시합니다. 작성자가 혼자 모델 검증과 서비스 승인까지 완료하지 않습니다.

## 10. 배포·복구 규칙

- 로컬 SQLite나 PC file:// 경로를 공유 서버 모델 URI처럼 사용하지 않습니다.
- 서비스는 승인한 manifest 또는 정확한 Registry 버전을 사용합니다. 버전 없는 최신 모델 자동 선택 금지.
- joblib 파일은 {model, features, target} 사전이고 MLflow sklearn 모델은 Pipeline입니다.
  로더를 혼용하지 않고 입력 컬럼·자료형을 검증합니다.
- 별칭 변경은 실행 중인 API를 자동 갱신하지 않습니다. 별도 재로딩/배포 및 샘플 예측 확인이 필요합니다.
- 이전 모델·manifest·별칭 이력을 유지해 이전 승인 버전으로 롤백합니다.
- 공유 서버에서 임의 down -v, 버킷 삭제, Registry 버전 삭제, DB 초기화를 하지 않습니다.
- MLflow DB와 아티팩트를 함께 백업하고 서비스 모델 버킷·manifest도 보존합니다.

## 11. 현재 자동화되지 않은 사항

owner/commit 설명 자동 입력, 개인별 실험 이름 옵션, 공식 버전 예약 잠금,
서버 권한별 등록 제한, export CSV의 MLflow 첨부, PR 승인 기반 배포는 미구현입니다.
지금은 담당자·리뷰 절차로 운영하며 자동 충돌 방지를 보장하지 않습니다.
이 문서는 기존 코드 동작을 변경하지 않습니다.

관련 문서: [프로젝트 README](../README.md), [개발 워크플로우](development-workflow.md),
[모델 저장 안내](model-storage.md), [조합 실험](child-feature-subsets-mlflow.md),
[예측 사용법](child-prediction-usage.md).

## 12. 알고리즘 비교·선정 근거 필수 규칙 (개정)

기본 학습 진입점은 여러 알고리즘 비교로 변경했습니다. 과거 RF 조합만 재현할
때만 --legacy-rf-only를 지정합니다. 이전 절과 다른 문서의 --register-best,
--export-version은 legacy 모드 전용이며 새 비교에서는 지원하지 않습니다.
새 결과는 기존 export_child_models.py가 받는 형식이 아니므로 그 도구로 내보내지 않습니다.

```powershell
python ml/training/train_child_feature_subsets.py --data data/processed/child_population/facility_child_counts_20261023_093757_426625/analysis_panel.csv --ages elementary --folds 5
```

- Dummy 평균 예측을 기준으로 Ridge, 얕은 결정트리, RandomForest, ExtraTrees,
  HistGradientBoosting을 동일 개발 표본·폴드에서 비교합니다.
- 기존 테스트 85개 동은 제외하고 학습+검증 337개 동을 개발 데이터로 사용합니다.
- 폴드마다 학습 부분에서만 상관 상위 5개를 다시 선정합니다. 31개 조합은 고정
  시설명이 아니라 순위 위치 조합입니다. 각 폴드 실제 시설명을 반드시 함께 확인합니다.
- 비교 규모는 연령당 5종×31조합×5폴드=775회 fit입니다. 4연령이면 3,100회입니다.
- 동일 설정 예산으로 비교하며 파라미터는 탐색 전에 고정합니다. 최적 튜닝 결과라고 표현하지 않습니다.
- 선정은 최저 평균 MAE+그 후보의 표준오차 이내에서 단순성 순서(Ridge→tree→RF→ET→HGB),
  적은 특징 수, 낮은 MAE 순입니다. 이 순서는 모델 우열의 사실이 아닌 사전 정한 운영 규칙입니다.
- 기준 대비 MAE 5% 이상 개선 및 80% 이상 폴드 승리를 모두 만족해야 탐색 후보로 유지합니다.
  아니면 공식 선정 보류입니다. 5%·80%는 통계적 유의성 문턱이 아닙니다.
- 기준 통과 후보도 독립 최종 검증 전에는 서비스 배포하지 않습니다. 구 단위 홀드아웃은 미구현입니다.
- 각 조합 MAE 평균·표준편차, RMSE, R², 학습/예측 시간, 폴드별 시설, 선정/탈락 사유를 저장합니다.
- 많은 조합을 같은 CV로 선택한 점수는 낙관적일 수 있습니다. 성능 확정에는 중첩 검증 또는 새 독립 자료가 필요합니다.
- 기존 테스트는 이미 봤으므로 기본 미평가입니다. --evaluate-reference-test를 명시하면
  참고 평가로만 저장하며 그 결과로 후보를 재선정하지 않습니다.
- 결과는 artifacts/child_algorithm_comparison, MLflow 실험은 child-population-algorithm-comparison입니다.
  선정 후보의 Run 설명에 한국어 선정·보류 이유를 기록하고 표준 모델을 보존합니다.
- Registry 자동 등록, champion 변경, 기존 v1 덮어쓰기, MinIO 자동 업로드는 하지 않습니다.
  candidate.joblib의 존재는 배포 승인 표시가 아닙니다.
- PR에는 report.html, comparison.csv, fold_results.csv, selection_reasons.json의 MLflow 링크와
  데이터·분할 해시를 첨부합니다. 알고리즘 이름만으로 선정 이유를 대신하지 않습니다.
