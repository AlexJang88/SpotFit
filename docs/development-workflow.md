# SpotFit 개발 워크플로우

이 문서는 Windows·macOS 개발 PC에서 작업하고 GitHub로 코드를 공유한 뒤, 내부망 Ubuntu 서버에 반영하는 흐름을 설명합니다. 기본 개발 Compose의 MinIO는 PC마다 별도 저장소입니다. 따라서 GitHub는 코드와 모델 메타데이터를 공유하고, 모델 파일은 사용할 MinIO에 별도로 업로드해야 합니다.

## 1. 개발 PC 최초 준비

Docker Desktop(Windows/macOS) 또는 Docker Engine(Linux)과 Compose 플러그인을 설치한 뒤, 저장소를 받습니다.

```sh
git clone https://github.com/AlexJang88/SpotFit.git
cd SpotFit
```

환경 파일은 각 PC에서 따로 만들고 Git에 올리지 않습니다.

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```sh
# macOS / Linux
cp .env.example .env
```

`.env`에 로컬 개발용 비밀번호와 키를 설정하고, 프로젝트 루트에서 구성을 검사한 뒤 시작합니다.

```sh
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

화면은 `http://localhost:8080`, API 문서는 `http://localhost:8080/api/docs`, Airflow는 `http://localhost:8081`, MLflow는 `http://localhost:5000`에서 확인합니다. MinIO API는 `http://localhost:9000`, 콘솔은 `http://localhost:9001`입니다. 로컬 MinIO 로그인 정보는 해당 PC의 `.env`에 있습니다. Airflow 암호는 다음 명령으로 확인할 수 있습니다.

```sh
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

## 2. 매일 작업할 때

작업 전 최신 코드를 받고, 기능별 브랜치를 만들어 작업합니다.

```sh
git switch main
git pull --ff-only origin main
git switch -c feature/기능-이름
```

소스는 컨테이너에 연결되어 있으므로 프런트엔드는 HMR, 백엔드는 자동 재시작됩니다. 상태와 로그를 보는 명령은 다음과 같습니다.

```sh
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f minio
```

`Ctrl+C`는 로그 보기만 종료합니다. 개발 스택을 멈추려면 다음을 실행합니다.

```sh
docker compose down
```

이 명령은 PostgreSQL·MinIO 볼륨을 유지합니다. `docker compose down -v`는 DB와 업로드한 모델을 포함한 데이터 볼륨을 삭제하므로 의도적으로 전체 데이터를 초기화할 때만 사용합니다.

## 3. 라이브러리를 추가할 때

컨테이너 안에서 일회성으로 `pip install`하는 방식은 재생성하면 사라집니다. 패키지 선언 파일을 수정하고 이미지를 다시 빌드해, 다른 개발자와 Ubuntu에서도 같은 의존성을 재현할 수 있게 합니다.

### React / TypeScript 패키지

프로젝트 루트에서 실행하며, 패키지 이름은 실제 사용할 패키지로 바꿉니다.

```sh
docker compose exec frontend npm install 패키지명
docker compose exec frontend npm install --save-dev 개발도구명
```

`frontend/package.json`과 `frontend/package-lock.json`이 갱신됩니다. 둘 다 커밋합니다. 새 브랜치나 다른 PC는 `docker compose up` 시 `npm ci`로 lock 파일 기준 설치합니다. 설치 후에도 해석되지 않으면 프런트엔드 컨테이너를 다시 시작합니다.

```sh
docker compose restart frontend
docker compose exec frontend npm run build
```

### FastAPI 전용 Python 패키지

API에서만 사용하는 패키지는 `backend/requirements.txt`에 버전 범위를 적습니다. 예를 들어 `httpx>=0.27,<1.0`을 추가했다면:

```sh
docker compose up --build -d backend
```

### 학습·데이터 처리 패키지

백엔드 추론과 Airflow 학습 작업이 함께 사용하는 NumPy, Pandas, scikit-learn 등의 패키지는 `ml/requirements.txt`에 추가합니다. 백엔드와 Airflow 이미지가 이 파일을 사용하므로 둘 다 재빌드해야 합니다.

```sh
docker compose up --build -d backend airflow mlflow
```

이후 모델 CLI를 쓸 때는 도구 이미지도 다시 빌드합니다.

```sh
docker compose --profile tools build model-tools
```

Airflow 전용 provider처럼 Airflow에만 필요한 의존성은 Airflow 버전과 호환성을 확인해 `infra/airflow/Dockerfile`의 별도 프로젝트 패키지 설치 단계에 추가합니다. Dockerfile 첫 Airflow 설치 단계의 공식 constraints 설정을 제거하거나 프로젝트 패키지 전체에 무심코 적용하지 마세요. Airflow와 ML 패키지의 버전 충돌을 막기 위해 현재 설치 단계가 분리되어 있습니다.

의존성 선언을 바꾼 뒤에는 함께 작업하는 사람에게 어떤 파일이 바뀌었는지 알려주고 해당 이미지를 재빌드하게 합니다. requirements만 바꾸고 실행 중인 컨테이너에 설치하는 것으로 끝내지 않습니다.

## 4. MinIO에 데이터를 저장할 때

MinIO는 S3 호환 객체 저장소입니다. 기본 개발 Compose에서는 앱 컨테이너끼리 `http://minio:9000`으로 통신합니다. Ubuntu 저장소를 개발 PC가 사용할 때는 `compose.minio-server.yaml`이 백엔드와 도구의 endpoint를 Ubuntu 내부망 IP로 바꿉니다. 버킷은 데이터 종류별로 나눕니다.

현재 제공된 `model-tools` CLI는 **모델 파일**의 버전 업로드·다운로드를 지원합니다. 임의의 원본 데이터셋을 업로드하는 `data upload` 명령은 아직 없습니다. 원본 데이터는 기본적으로 `data/raw`, 가공 데이터는 `data/interim`·`data/processed`에 로컬로 두며 Git에서 제외됩니다. 데이터셋 공유가 필요하면 콘솔에서 전용 버킷(예: `spot-fit-data`)을 만들고 업로드하거나, 추후 데이터용 CLI/수집 파이프라인을 추가합니다. 모델 버킷에 데이터셋을 섞지 마세요.

### 모델 버킷 초기화

프로젝트 루트에서 도구 이미지를 만들고 한 번 실행합니다. 초기화는 `spot-fit-models` 버킷을 만들고 객체 버전 관리를 활성화합니다.

```sh
docker compose --profile tools build model-tools
docker compose --profile tools run --rm model-tools init
```

### 모델 파일 업로드와 manifest 커밋

학습 코드가 먼저 파일을 예를 들어 `artifacts/models/recommender.joblib`에 저장해야 합니다. `.pt`, `.onnx` 등도 `--file` 경로를 바꾸면 됩니다. 이 폴더는 컨테이너의 `/artifacts`로 연결되어 있으므로 명령에는 호스트의 Windows 경로 대신 컨테이너 경로를 씁니다.

```sh
docker compose --profile tools run --rm model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
```

성공하면 모델 바이트는 MinIO에, 메타데이터 manifest는 `model_registry/recommender/v1.json`에 기록됩니다. manifest에는 버킷·객체 키·객체 버전 ID·크기·SHA-256이 포함됩니다. 새 학습 결과는 기존 버전을 덮어쓰지 말고 `v2`처럼 새 버전으로 올립니다.

```sh
git add model_registry/recommender/v1.json
git commit -m "model: register recommender v1"
git push -u origin feature/기능-이름
```

모델 파일(`.joblib`, `.pt` 등)은 Git에 추가하지 않습니다. Git에는 코드와 manifest를 올립니다.

### manifest에 기록된 모델 다운로드

저장소를 받은 개발자는 같은 MinIO에 접근 가능해야 합니다. 버킷 초기화 후 다음 명령을 실행하면 버전 ID를 지정해 내려받고 크기와 SHA-256을 검증합니다.

```sh
docker compose --profile tools run --rm model-tools download --manifest /model_registry/recommender/v1.json
```

기본 저장 위치는 `artifacts/models/recommender/v1/recommender.joblib`입니다. MLflow의 `spot-fit-mlflow` 버킷은 학습 실행 기록의 artifact용이며, 위 명령의 `spot-fit-models`와 목적이 다릅니다.

### 개발 PC에서 Ubuntu MinIO 사용

기본 `.env`의 MinIO는 로컬 개발용입니다. Ubuntu MinIO의 모델 버킷을 FastAPI와 CLI에서 사용하려면 Ubuntu에서 내부망의 S3 API 접근을 허용하고 제한된 개발 계정을 만듭니다. 개발 PC `.env`에는 `DEV_S3_*` 값만 설정합니다. 설정 절차와 방화벽 규칙은 [VirtualBox 배포 안내](virtualbox-deployment.md#개발-pc에서-서버-minio-사용)에 있습니다.

다른 터미널에서 프런트엔드와 백엔드를 서버 MinIO 설정으로 실행할 수 있습니다.

```sh
docker compose -f compose.yaml -f compose.minio-server.yaml up --build -d
```

이 구성은 개발 PC의 FastAPI 백엔드와 모델 CLI를 Ubuntu의 `spot-fit-models`에 연결합니다. 기본 Compose의 로컬 MinIO는
Airflow/MLflow artifact 저장을 위해 함께 실행될 수 있습니다. MinIO API는 같은 내부망에서 직접 접속하므로 개발 중 SSH 터널이 필요하지 않습니다. MinIO 관리 콘솔만 기존 SSH 터널로 접속합니다.

별도 터미널에서 Ubuntu MinIO를 지정해 버킷 초기화·업로드·다운로드를 실행합니다. `--no-deps`는 로컬 MinIO를 대신 시작하는 것을 막습니다.

```sh
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools build model-tools
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools init
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
docker compose -f compose.yaml -f compose.minio-server.yaml --profile tools run --rm --no-deps model-tools download --manifest /model_registry/recommender/v1.json
```

개발 서버를 실행할 때도 두 compose 파일을 함께 사용하면 백엔드가 서버의 `spot-fit-models` 버킷을 사용합니다. 기본 Compose가 시작하는 MinIO는 Airflow/MLflow 개발용으로 남아 있을 수 있습니다. Git에는 manifest를 커밋하고 모델 바이트는 Ubuntu MinIO에 저장합니다. 같은 bucket/모델명/version을 여러 개발자가 동시에 쓰지 않도록 새 버전을 조정하세요.

## 5. 변경사항을 GitHub와 Ubuntu에 반영

기능 구현과 필요한 검증을 마치면 변경 파일을 확인해 브랜치에 커밋하고 push합니다.

```sh
git status
git add 경로/파일
git commit -m "feat: 변경 내용"
git push -u origin feature/기능-이름
```

GitHub에서 리뷰·병합한 뒤 Ubuntu에서 `main`을 가져옵니다. Ubuntu의 `.env.server`는 Git에 올리지 않으며 서버에 그대로 유지합니다.

```bash
cd /home/spotfit/SpotFit
git switch main
git pull --ff-only origin main
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

의존성을 추가한 변경이면 `up --build -d`가 해당 이미지도 다시 만듭니다. 로그는 다음처럼 확인합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs -f backend
```

## 6. Git에 올릴 파일 확인

커밋 전 `git status`로 다음 기준을 확인합니다.

- 커밋: 애플리케이션 코드, Compose/Dockerfile, `frontend/package.json`과 `package-lock.json`, Python requirements, SQL 마이그레이션, 문서, 모델 manifest JSON
- 커밋 금지: `.env`, `.env.server`, 비밀번호·키, `data/`의 사용자 데이터, `artifacts/`의 모델 파일, `node_modules`, Docker 볼륨

의존성과 manifest는 플랫폼 간 재현을 위해 함께 공유하고, 데이터·모델 바이트는 MinIO나 별도 승인된 파일 전달 경로로 공유합니다.
