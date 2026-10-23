# 아동 인구수 시설 조합 실험과 MLflow

> 2026-10-23 개정: 기본 진입점은 다중 알고리즘 교차검증으로 변경했습니다.
> 아래 RF-only 명령은 과거 재현용이며 `--legacy-rf-only`를 추가해야 합니다.
> 새 실행은 `python ml/training/train_child_feature_subsets.py --data <analysis_panel.csv> --ages elementary --folds 5`입니다.
> `--register-best`, `--export-version`은 새 비교 모드에서 지원하지 않습니다.
> 새 결과를 기존 exporter에 전달하지 마세요. 선정/보류 근거와 변경 규칙은
> [팀 규칙 12절](model-team-workflow.md#12-알고리즘-비교선정-근거-필수-규칙-개정)을 참고하세요.

## README의 공유 저장 규칙 (중요)

로컬 MLflow는 실험용 기록입니다. 프로젝트 공식 공유 모델은 README대로
MinIO에 저장하고, 업로드 성공 후 생성된 model_registry JSON을 Git으로 관리합니다.
MLflow Model Registry 버전과 MinIO manifest의 v1/v2는 별개이며 자동 동기화되지 않습니다.

이번 우승 모델 4개는 models/spotfit-child-count-<연령>/v1/model.joblib에 복사했습니다.
원본 124개 조합은 실험 폴더에 보존합니다. SHA-256과 run_id 연결은 결과 폴더의
minio_export.csv에 있으며 실제 업로드 완료를 뜻하지 않습니다.

다음 학습에서 내보내기를 함께 하려면:

```powershell
python ml/training/train_child_feature_subsets.py --register-best --export-version v2
```

이미 학습한 결과만 내보내기(재학습 없음):

```powershell
python ml/training/export_child_models.py --results artifacts/child_feature_subsets/20261023_095054 --version v1
```

README의 `python upload_minio.py`는 models뿐 아니라 data 전체도 업로드합니다.
모델 하나만 업로드하려면 공유 개발 Compose의 기존 CLI를 사용하세요:

```powershell
New-Item -ItemType Directory -Force artifacts/models/spotfit-child-count-infant/v1
Copy-Item models/spotfit-child-count-infant/v1/model.joblib artifacts/models/spotfit-child-count-infant/v1/model.joblib
docker compose --env-file .env.dev -f compose.dev.yaml run --rm model-tools upload --file /artifacts/models/spotfit-child-count-infant/v1/model.joblib --name spotfit-child-count-infant --version v1
```

다른 연령군도 infant를 elementary/middle/high로 바꾸면 됩니다.
기존 캐시가 있다면 덮어쓰기 전에 같은 파일인지 확인하세요.
업로드가 성공해야 model_registry/<이름>/v1.json이 생성됩니다.
아직 업로드·공유 서버 연결·자동 활성화는 수행하지 않았습니다.

## 실행

프로젝트 루트에서 실행합니다. 개발 PC에는 `requirements-dev.txt`를 설치합니다.

```powershell
python ml/training/train_child_feature_subsets.py --register-best
```

기본값은 프로젝트 artifacts/mlflow_child_subsets/tracking.db에 SQLite 실험을
저장합니다. .env를 읽거나 원격 서버에 자동 접속하지 않습니다.
4개 연령군 × 31개 부분집합 = 124개 RandomForest 모델을 학습합니다.
각 연령군의 학습 상관 상위 5개를 따로 선정하므로 후보 목록은 서로 다릅니다.
결측 관측률 70% 미만/상수 지표는 제외하며 선정표도 저장합니다.

## 평가 설계

- 최신 단일 연도(이번 자료 2025년)의 422개 행정동을 seed=42로 분할합니다.
- 학습 253 / 검증 84 / 테스트 85개 동이며 모든 조합이 같은 분할을 사용합니다.
- 학습 데이터에서만 상위 특징과 결측 대체 중앙값을 계산합니다.
- 동일한 RF(80 trees, min_samples_leaf=5)로 조합 효과를 비교합니다.
- 검증 MAE가 최소인 조합을 고른 후 그 모델만 테스트합니다.
- MAE와 RMSE는 명 단위로 낮을수록 좋고, R²는 높을수록 좋습니다.
  R²는 정답률이 아니며 음수가 될 수 있습니다.
- 평균 인구만 예측하는 DummyRegressor와 테스트 MAE를 비교합니다.
- 입력은 선택한 시설뿐입니다. 행정동 코드/연령 인구/전체 인구는 입력하지 않습니다.
- 이 실험은 현재 인구 추정입니다. 미래 인구·육아환경 점수·인과효과가 아닙니다.
  시설 과거 값이 동일하고 실제 스냅샷 시점이 불명확한 한계가 남습니다.
  인접 동의 공간 유사성 때문에 구 단위 홀드아웃보다 낙관적일 수 있습니다.
- 조합 선택은 단일 검증셋 기준이므로 불안정할 수 있습니다. 후속 작업은 학습 내
  교차검증과 구 단위 홀드아웃입니다. 테스트를 보고 조합을 바꾸지 마세요.

## 로컬 MLflow 화면

프로젝트 루트에서:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow_child_subsets/tracking.db --host 127.0.0.1 --port 5001
```

http://127.0.0.1:5001 에서 Experiments → child-population-feature-subsets 선택.
부모 Run 아래 124개 자식 Run이 있습니다. age를 동일하게 필터링한 뒤
validation_mae 오름차순으로 정렬하거나 Compare를 사용하세요.
서로 다른 연령군 MAE만으로 모델 우열을 비교하지 않습니다.
각 Run은 features, feature_count, 설정, 검증 성능, 모델 joblib를 저장합니다.
joblib는 신뢰하는 자체 생성 파일만 로드하세요.
우승 Run만 test_*와 baseline_test_* 및 MLflow model을 추가로 가집니다.
나머지 Run에 테스트 수치가 없는 것은 의도적인 평가 누수 방지입니다.

## 버전 관리

Run은 한 번의 학습 실험 기록이며 Model Version과 다릅니다.
--register-best를 넣으면 연령별 다음 이름으로 우승 모델을 등록합니다.

- spotfit-child-count-infant
- spotfit-child-count-elementary
- spotfit-child-count-middle
- spotfit-child-count-high

같은 이름으로 재등록하면 v1, v2 등 버전이 증가합니다. 31개 조합을 모두
서비스 버전으로 승격하지 않고 Run으로 보존하고, 선정 모델만 등록합니다.
각 버전에는 탐색 실험임을 나타내는 태그가 붙습니다. 자동 배포하지 않습니다.

```python
import mlflow
from mlflow import MlflowClient

# 학습한 저장소와 같은 URI를 사용해야 동일한 버전을 조회할 수 있습니다.
mlflow.set_tracking_uri("sqlite:///artifacts/mlflow_child_subsets/tracking.db")
client = MlflowClient()
name = "spotfit-child-count-infant"
# 검토한 버전 번호를 지정합니다. 아래 1은 예시입니다.
client.set_registered_model_alias(name, "candidate", "1")
model = mlflow.sklearn.load_model(f"models:/{name}@candidate")
# 데이터프레임에는 모델에 기록된 시설 컬럼이 정확히 있어야 합니다.
# predictions = model.predict(input_dataframe)
```

승인 후 champion 별칭을 지정할 수 있고 이전 버전으로 별칭을 되돌려 롤백할
수 있습니다. 별칭 변경 자체는 실행 중인 API 모델을 자동으로 교체하지 않습니다.
API 재로딩/배포 절차가 필요합니다. 사용자 시설 선택 서비스에도 이 모델이
그대로 점수식이 되는 것은 아닙니다(출력은 인구수).

## Docker 서버에 기록하기

compose.yaml의 MLflow는 컨테이너 내부 http://mlflow:5000,
호스트 기본 포트 5000으로 설정되어 있습니다. 기본 바인딩은 127.0.0.1입니다.
서버에서 프로젝트와 데이터가 준비된 Python 환경으로:

```bash
python ml/training/train_child_feature_subsets.py --tracking-uri http://127.0.0.1:5000 --register-best
```

다른 PC에서 접근하려면 SSH 터널 등 승인된 접근 경로가 필요합니다.
인증 없이 공개 인터넷에 포트를 열지 마세요. 서버 환경에서 필요한 입력 CSV는
--data로 명시할 수 있습니다. 로컬 실험이 서버에 자동 복사되지는 않습니다.
위 명령은 새 학습과 새 서버 Run을 생성합니다. 원격 연결 실패 시 로컬로
조용히 우회하지 않고 실패하므로 저장 위치를 혼동하지 않습니다.

공식 문서:
- https://www.mlflow.org/docs/latest/registry/
- https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
