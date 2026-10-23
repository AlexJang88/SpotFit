# SpotFit 개발 워크플로우

Windows·macOS 개발 PC에서 앱을 개발하고 같은 내부망 Ubuntu 서버의 PostgreSQL, MinIO, Airflow, MLflow를 함께 쓰는 절차입니다. 개발 PC에서는 프런트엔드·백엔드·Nginx만 Docker로 실행합니다. 개발 PC에 PostgreSQL, MinIO, Airflow, MLflow를 설치하거나 띄울 필요가 없습니다. 서버가 켜져 있고 개발 PC에서 해당 포트로 접속 가능해야 합니다.

## Compose 구성

| 목적 | Compose 명령에 지정하는 파일 | 실행되는 주요 서비스 |
| --- | --- | --- |
| 개발 PC에서 공유 Ubuntu 인프라 사용 | `compose.dev.yaml` | Nginx, React/Vite, FastAPI |
| Ubuntu 테스트 서버 | `compose.yaml` + `compose.server.yaml` | 앱, PostgreSQL/PostGIS, MinIO, Airflow, MLflow |
| PC만으로 독립적인 전체 개발 스택 실행 | `compose.yaml` | 위 서비스 전체 |

일상적인 공유 서버 개발에는 `compose.dev.yaml`을 사용합니다. `compose.yaml`은 전체 스택을 로컬에서 독립 실행하고 싶을 때 사용합니다.

## 1. 개발 PC 최초 준비

Docker Desktop(Windows/macOS) 또는 Docker Engine과 Compose 플러그인을 설치하고 저장소를 받습니다.

```sh
git clone https://github.com/AlexJang88/SpotFit.git
cd SpotFit
```

환경 파일을 만들고 `.env.dev`에 Ubuntu 서버의 실제 접속 정보와 자격 증명을 입력합니다. `.env.dev`는 Git에 올라가지 않습니다.

```powershell
# Windows PowerShell
Copy-Item .env.dev.example .env.dev
```

```sh
# macOS / Linux
cp .env.dev.example .env.dev
```

`.env.dev`의 `POSTGRES_PASSWORD`는 서버 `.env.server`의 값과 같게 설정합니다. `S3_SECRET_ACCESS_KEY`에는 서버에서 제한된 개발 계정을 만들 때 설정한 `DEV_S3_SECRET_ACCESS_KEY` 값을 넣습니다. 서버 root MinIO 키는 개발 PC로 복사하지 않습니다. 기본 서버 주소가 다르면 `POSTGRES_HOST`, `S3_ENDPOINT_URL`, `MLFLOW_TRACKING_URI`를 실제 Ubuntu IP로 수정합니다.

서버가 실행 중이고 내부망 접속이 허용된 상태에서 구성을 검사하고 개발 앱을 시작합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml config --quiet
docker compose --env-file .env.dev -f compose.dev.yaml up --build -d
docker compose --env-file .env.dev -f compose.dev.yaml ps
```

개발 PC의 화면은 `http://localhost:8080`, API 문서는 `http://localhost:8080/api/docs`입니다. Airflow와 MLflow는 각각 `http://192.168.0.3:8081`, `http://192.168.0.3:5000`에서 서버에 직접 접속합니다. MinIO S3 API는 `http://192.168.0.3:9000`입니다. MinIO 관리 콘솔은 서버에서 localhost에만 바인딩되어 있으므로 필요한 경우 SSH 터널을 사용합니다. 실제 주소는 `.env.dev`와 [VirtualBox 배포 안내](virtualbox-deployment.md)에 맞춥니다.

현재 FastAPI에는 PostgreSQL을 실제 조회하는 기능이 아직 구현되어 있지 않습니다. Compose는 DB 접속 환경변수만 전달합니다. DB 연결 코드와 마이그레이션을 구현한 뒤 실제 DB 연동을 검증해야 합니다. `/api/spot_fit` 생존 확인은 DB나 MinIO 연결 검사가 아닙니다.

## 2. 매일 작업과 종료

작업 전 최신 코드를 받고 기능 브랜치를 만듭니다.

```sh
git switch main
git pull --ff-only origin main
git switch -c feature/기능-이름
```

소스는 개발 컨테이너에 연결되어 있어 프런트엔드는 HMR, 백엔드는 자동 재시작됩니다. 상태·로그 확인에는 다음 명령을 사용합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml ps
docker compose --env-file .env.dev -f compose.dev.yaml logs -f backend
docker compose --env-file .env.dev -f compose.dev.yaml logs -f frontend
```

`Ctrl+C`는 로그 보기만 끝냅니다. 개발 PC의 앱 컨테이너를 중지할 때는 다음 명령을 실행합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml down
```

이 구성은 DB·MinIO·Airflow·MLflow를 내리지 않고 개발 앱만 종료합니다. 공유 서버는 계속 실행됩니다.

## 3. 라이브러리를 추가할 때

컨테이너 안에서 일회성으로 설치하지 말고 저장소의 선언 파일을 수정하고 이미지를 다시 빌드합니다.

### React / TypeScript 패키지

```sh
docker compose --env-file .env.dev -f compose.dev.yaml exec frontend npm install 패키지명
docker compose --env-file .env.dev -f compose.dev.yaml exec frontend npm install --save-dev 개발도구명
docker compose --env-file .env.dev -f compose.dev.yaml exec frontend npm run build
```

갱신된 `frontend/package.json`과 `frontend/package-lock.json`을 함께 커밋합니다. 새 브랜치나 다른 PC에서는 컨테이너 시작 시 `npm ci`가 lock 파일에 맞춰 설치합니다.

### Python 패키지

Python 의존성은 루트의 두 파일로만 관리합니다.

- `requirements-server.txt`: FastAPI, 학습·추론, MLflow, MinIO 등 서버 런타임
- `requirements-dev.txt`: 서버 전체 + EDA·시각화·Excel·공간 분석

개발 PC는 `python -m pip install -r requirements-dev.txt`를 사용합니다.
서버에 필요한 패키지는 `requirements-server.txt`에 추가하고 이미지를 다시 빌드합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml up --build -d backend
```

백엔드 이미지 재빌드:

```sh
docker compose --env-file .env.dev -f compose.dev.yaml up --build -d backend
```

Ubuntu 서버:

```sh
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d airflow
```

MLflow가 사용하는 Python 패키지도 바뀌면 서버의 MLflow 이미지를 포함해 재빌드합니다. Airflow 전용 provider는 `infra/airflow/Dockerfile`의 별도 프로젝트 패키지 설치 단계에 추가하고, Airflow 버전과 호환성을 확인합니다.

모델 CLI 패키지는 같은 백엔드 Dockerfile을 사용하므로 필요한 시점에 도구 이미지를 다시 빌드합니다.

## 4. Ubuntu MinIO에 모델 업로드·다운로드

Ubuntu의 `spot-fit-models` 버킷은 버전 관리가 켜진 모델 저장소입니다. Ubuntu 서버에서 개발 전용 MinIO 계정을 먼저 만들고, `.env.dev`의 S3 키를 그 계정에 맞춥니다. 설정과 내부망 방화벽은 [VirtualBox 배포 안내](virtualbox-deployment.md#개발-pc에서-서버-minio-사용)를 따릅니다. 모델 파일은 MinIO에, 버전·체크섬 manifest는 `model_registry/`에 두고 manifest만 Git으로 공유합니다.

도구 이미지를 빌드하고 버킷을 초기화합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml --profile tools build model-tools
docker compose --env-file .env.dev -f compose.dev.yaml --profile tools run --rm model-tools init
```

학습 결과 파일을 `artifacts/models/recommender.joblib`에 저장한 다음 업로드합니다. 컨테이너에서 보이는 `/artifacts` 경로를 사용합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml --profile tools run --rm model-tools upload --file /artifacts/models/recommender.joblib --name recommender --version v1
```

성공하면 모델 바이트는 Ubuntu MinIO에, 메타데이터는 `model_registry/recommender/v1.json`에 기록됩니다. 같은 모델 버전을 덮어쓰지 말고 새 학습 결과는 `v2`처럼 새 버전으로 올립니다.

```sh
git add model_registry/recommender/v1.json
git commit -m "model: register recommender v1"
git push -u origin feature/기능-이름
```

다른 개발 PC에서 manifest를 받은 뒤 같은 서버 계정으로 모델 파일을 검증해 내려받습니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml --profile tools run --rm model-tools download --manifest /model_registry/recommender/v1.json
```

현재 `model-tools`는 모델 파일 관리 명령을 제공합니다. 임의 데이터셋 업로드 CLI는 아직 없으며, 사용자·공공 데이터는 `data/`에 로컬 보관하거나 별도 데이터 적재 기능을 구현합니다. MLflow 실험 artifact는 Ubuntu MLflow 설정과 그 서버 저장소가 관리합니다.

## 5. GitHub에서 Ubuntu 서버로 반영

검증한 변경을 브랜치에 push하고 GitHub에서 병합합니다. Ubuntu의 `.env.server`는 서버에만 둡니다. 병합된 새 버전을 서버에서 가져와 빌드·실행합니다.

```bash
cd /home/spotfit/SpotFit
git switch main
git pull --ff-only origin main
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

로그 확인:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs -f backend
```

## 6. Git에 올릴 파일

커밋 대상은 앱 코드, Compose/Dockerfile, 프런트 lock 파일, Python requirements, SQL 마이그레이션, 문서, 모델 manifest입니다. `.env.dev`, `.env.server`, 비밀번호·키, `data/` 사용자 데이터, `artifacts/` 모델 파일, `node_modules`, Docker 볼륨은 커밋하지 않습니다.
