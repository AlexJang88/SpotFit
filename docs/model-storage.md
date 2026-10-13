# MinIO 모델 저장·공유

학습 코드와 `model_registry/<모델명>/<버전>.json`은 Git에 저장하고, 학습된 파일은
MinIO의 `spot-fit-models` 버킷에 저장합니다. `artifacts/`의 실제 파일은 계속 Git에서 제외합니다.
버킷명은 S3 규칙상 밑줄을 사용할 수 없어 프로젝트명 `spot_fit` 대신 `spot-fit-models`를 사용합니다.

## 1. 로컬 저장소 준비

프로젝트 루트에서 실행합니다. Windows PowerShell·macOS 터미널에서 같은 명령을 사용합니다.
기존 `.env`도 기본값으로 동작하므로 덮어쓸 필요가 없습니다.

```sh
docker compose up --build -d
docker compose --profile tools build model-tools
docker compose --profile tools run --rm model-tools init
```

`init`은 모델 버킷을 생성하고 객체 버전 관리를 켭니다. 다시 실행해도 기존 모델을 삭제하지 않습니다.
새 버킷에는 공개 접근 정책을 설정하지 않습니다. MinIO 콘솔은 http://localhost:9001 이며
로컬 로그인 정보는 `.env`의 `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`입니다.

## 2. 학습된 모델 업로드

먼저 실제 학습 결과를 `artifacts/models/recommender.joblib`에 저장했다고 가정합니다.
아래 명령은 학습을 수행하지 않으며 파일이 있어야 합니다. `.pt`, `.onnx`, `.pkl` 등 다른
형식도 파일 경로를 바꿔 사용할 수 있습니다. 여러 파일로 구성된 모델은 하나의 아카이브로 묶습니다.

```sh
docker compose --profile tools run --rm model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
```

컨테이너의 `/artifacts`는 로컬 프로젝트의 `artifacts` 폴더입니다. 명령에는 Windows의
`C:\...` 경로 대신 위와 같은 컨테이너 경로를 사용합니다.
파일 업로드가 성공하면 `model_registry/recommender/v1.json`이 생성됩니다.

JSON에는 모델명, 버전, 파일명, 버킷, 객체 키, MinIO 객체 버전 ID, SHA-256, 파일 크기가 들어갑니다.
서버 주소나 비밀번호는 들어가지 않습니다. 파일은 임시 복사본으로 업로드하며 모델 크기만큼의
추가 임시 디스크 공간이 필요합니다. 같은 모델명·버전의 JSON은 덮어쓰지 않으므로 재학습한 모델은
`v2`처럼 새 버전을 사용합니다. Git에서 JSON 변경을 함께 리뷰하세요.

```sh
git add model_registry/recommender/v1.json
git commit -m "model: register recommender v1"
git push
```

## 3. Git에 등록한 모델 다운로드

```sh
docker compose --profile tools run --rm model-tools download --manifest /model_registry/recommender/v1.json
```

파일은 `artifacts/models/recommender/v1/recommender.joblib`에 저장됩니다.
JSON이 지정한 **정확한 MinIO 객체 버전**을 가져오고 크기·SHA-256을 검증합니다.
정상 캐시가 있으면 재다운로드하지 않으며, 실패한 다운로드는 기존 파일을 덮어쓰지 않습니다.
이 과정은 모델 파일을 역직렬화하거나 추론 코드를 실행하지 않습니다.

## 백엔드에서 사용

모델을 사용할 서비스 코드에서 다음 함수를 호출할 수 있습니다. 다운로드는 동기 함수이므로
매 요청마다 호출하기보다 모델 준비 단계 또는 별도 작업에서 실행합니다.

```python
import os
from pathlib import Path

from app.repositories.model_store import ModelStore
from app.services.model_artifacts import download_model

model_file = download_model(
    ModelStore.from_env(),
    Path(os.environ["MODEL_REGISTRY_DIR"]) / "recommender/v1.json",
    Path(os.environ["MODEL_CACHE_DIR"]),
)
# 선택한 ML 프레임워크로 model_file을 로드하고 추론을 구현합니다.
```

백엔드 캐시는 `/model-cache`의 Docker 볼륨에, CLI 다운로드는 로컬 `artifacts/models`에 저장합니다.
아직 학습된 모델이나 추론 API가 없으므로 서버 시작 시 특정 모델을 자동으로 내려받지는 않습니다.

## Ubuntu MinIO를 개발 PC에서 사용하기

Windows·macOS의 Docker 컨테이너가 Ubuntu MinIO에 연결하도록 하려면 [VirtualBox 배포 안내](virtualbox-deployment.md)의
“개발 PC에서 서버 MinIO 사용” 순서를 따라 내부망 endpoint와 개발 전용 자격 증명을 설정합니다. 개발 PC의 `.env`에는
서버 `.env.server` 전체 대신 `DEV_S3_ENDPOINT_URL`, `DEV_S3_MODEL_BUCKET`, `DEV_S3_ACCESS_KEY_ID`,
`DEV_S3_SECRET_ACCESS_KEY`만 복사합니다. `compose.minio-server.yaml`은 FastAPI 백엔드와 `model-tools`를
Ubuntu 내부망 endpoint로 지정합니다. MinIO API 포트는 Ubuntu 방화벽에서 신뢰하는 내부망만 허용합니다. 현재
S3 endpoint는 HTTP이므로 이 방식은 신뢰된 내부망에서만 사용하고 root 자격 증명은 개발 PC에 복사하지 않습니다.

같은 내부망에서 `192.168.0.3:9000` 포트가 열리는지 확인한 뒤 공용 서버 버킷을 준비하고 모델을 관리합니다.

```sh
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools build model-tools
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools init
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools download --manifest /model_registry/recommender/v1.json
```

manifest에는 서버 MinIO의 버킷·객체 키·버전 ID가 기록됩니다. 따라서 다른 개발 PC도 같은 endpoint, 버킷,
접근 가능한 계정을 사용해야 파일을 검증해 내려받을 수 있습니다. Git에는 manifest만 올라가며 실제 모델 파일은
서버 MinIO가 보관합니다. FastAPI 컨테이너도 이 override를 적용해 실행하면 동일한 서버 모델 버킷을 사용합니다.
Airflow/MLflow artifact 저장은 기본 Compose의 MinIO를 계속 사용합니다.

새 개발 PC용 계정을 발급할 때는 Ubuntu에서 server Compose의 `--profile admin run --rm minio-dev-bootstrap`을
실행하고, 각 PC의 `.env`에 별도 계정을 안전하게 전달합니다. Root 자격 증명을 개발 PC에 복사하지 않습니다.

## 통합 검증

```sh
docker compose --profile tools run --rm --entrypoint python model-tools -m unittest discover -s tests/integration -p test_model_artifacts.py -v
```

테스트는 임의 이름의 전용 버킷을 생성해 업로드·다운로드·버전 고정·손상 검증을 확인하고
그 테스트 버킷만 정리합니다. 버킷 생성·삭제 권한이 있는 로컬 개발 계정으로 실행하세요.

[MinIO Python SDK](https://github.com/minio/minio-py/tree/7.2.20)
