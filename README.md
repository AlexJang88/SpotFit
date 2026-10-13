# spot_fit

React + TypeScript, Python FastAPI 기반의 ML/DL 서비스 개발용 초기 구조입니다.
개발 환경은 Docker Compose로 실행하며, Linux 배포를 위한 소스와 인프라 설정을 분리합니다.
현재 구현 범위는 기본 화면, API 생존 확인, 개발 컨테이너 설정, MinIO 모델 업로드·다운로드입니다. DB 모델·연결,
추천·예측 로직, 데이터 파이프라인, 인증은 각 폴더에서 구현할 예정입니다.

## 디렉토리 구조

```text
spot_fit/
├── frontend/                   # React + TypeScript (Vite)
│   ├── public/                 # 그대로 제공할 정적 파일
│   ├── src/
│   │   ├── models/             # Model: 도메인 타입, 클라이언트 상태
│   │   ├── controllers/        # Controller: 화면 동작·상태를 연결하는 hooks
│   │   ├── views/              # View: React 화면
│   │   │   ├── pages/
│   │   │   ├── components/
│   │   │   └── layouts/
│   │   ├── services/           # HTTP API 호출
│   │   ├── routes/             # 화면 라우팅
│   │   ├── styles/
│   │   └── assets/
│   └── tests/
├── backend/                    # Python FastAPI
│   ├── app/
│   │   ├── models/             # Model: 도메인·ORM 엔티티
│   │   ├── controllers/        # Controller: FastAPI 라우터
│   │   ├── views/              # View: API 응답 모델·표현
│   │   ├── schemas/            # 요청 검증용 모델
│   │   ├── services/           # 비즈니스 로직·모델 추론
│   │   ├── repositories/       # DB·객체 저장소 접근
│   │   ├── core/               # 환경설정·보안·로깅
│   │   ├── db/                 # DB 엔진·세션
│   │   └── main.py
│   ├── migrations/versions/    # DB 마이그레이션 (도구 설정은 추후 추가)
│   └── tests/{unit,integration}/
├── ml/
│   ├── collectors/            # 공공데이터·Open API·허용된 크롤링
│   ├── etl/                   # Pandas 기반 정제·변환
│   ├── validation/            # 데이터 품질 검증
│   ├── features/              # 피처 엔지니어링
│   ├── training/              # Scikit-learn·XGBoost·선택적 PyTorch
│   ├── evaluation/            # 평가 지표·모델 비교
│   ├── pipelines/             # 수집 → 정제 → 학습 실행 흐름
│   ├── notebooks/             # EDA 실험
│   └── tests/
├── infra/
│   ├── nginx/                 # 개발 프록시 설정·인증서 보관 위치
│   ├── postgres/init/         # 새 DB 볼륨 최초 시작 시 실행할 SQL
│   ├── minio/                 # 공식 소스로 MinIO 이미지 빌드
│   ├── airflow/dags/          # 선택적 워크플로 도입 위치
│   └── mlflow/                # 선택적 모델 추적 설정 위치
├── data/{raw,interim,processed}/ # 로컬 데이터 (Git 제외)
├── artifacts/{models,reports}/  # 학습 모델·평가 결과 (Git 제외)
├── model_registry/             # 모델 버전·저장 위치·해시 JSON (Git 포함)
├── scripts/                   # 공통 작업 스크립트
├── docs/architecture/         # 아키텍처 문서
├── .github/workflows/         # 추후 GitHub Actions 추가
├── compose.yaml               # 개발 환경
├── .env.example               # 공유 가능한 환경변수 예시
├── .gitignore
├── .gitattributes             # OS 간 줄바꿈 통일
└── .editorconfig              # UTF-8·들여쓰기 통일
```

React는 전통적인 서버 MVC 프레임워크가 아니므로 역할에 맞춘 MVC 형태로 구성합니다.
화면은 `frontend/src/views`, API의 JSON 표현은 `backend/app/views`에서 담당합니다.
컨트롤러는 요청과 응답을 연결하고 실제 처리 로직은 서비스에 둡니다.
학습 작업은 웹 요청 처리와 분리하여 `ml`에서 실행합니다. 기본 Compose에는 Airflow,
MLflow, Scikit-learn, XGBoost, Pandas, NumPy, PyTorch, Transformers가 포함됩니다.

## 공유 Ubuntu 서버를 사용하는 개발 환경

일상적인 개발은 개발 PC에서 프런트엔드·백엔드·Nginx만 실행하고 Ubuntu 서버의 PostgreSQL, MinIO, Airflow, MLflow에 연결합니다. 개발 PC에 이 인프라 서비스를 별도로 설치하거나 실행할 필요가 없습니다. Docker Engine/Desktop와 Compose v2 이상, 같은 내부망의 Ubuntu 서버가 필요합니다.

```powershell
Copy-Item .env.dev.example .env.dev
```

macOS/Linux에서는 `cp .env.dev.example .env.dev`를 실행합니다. `.env.dev`에서 Ubuntu IP와 서버 PostgreSQL 암호, 개발 전용 MinIO 계정을 설정한 다음 실행합니다.

```sh
docker compose --env-file .env.dev -f compose.dev.yaml config --quiet
docker compose --env-file .env.dev -f compose.dev.yaml up --build -d
```

개발 화면은 `http://localhost:8080`, API 문서는 `http://localhost:8080/api/docs`입니다. Ubuntu의 Airflow·MLflow·MinIO는 각각 서버의 `8081`, `5000`, `9000` 포트로 사용합니다. 상세한 설정과 명령은 [개발 워크플로우](docs/development-workflow.md)를 참고하세요. 현재 백엔드에는 DB 조회 기능이 아직 구현되어 있지 않아 DB 접속 환경만 준비되어 있습니다.

## 로컬 전체 스택 개발 환경 (선택)

공유 서버를 사용하지 않고 PC에서 모든 인프라를 독립 실행하려면 기본 `compose.yaml`을 사용할 수 있습니다. Windows는 Docker Desktop의 Linux 컨테이너 모드를 사용합니다. 호스트에 Node.js와 Python을 설치할 필요는 없습니다. 먼저 `.env.example`을 `.env`로 복사하고 비밀번호와 키를 채웁니다. 이 로컬 모드의 PostgreSQL·MinIO 데이터는 각 PC에 따로 저장됩니다.

```powershell
Copy-Item .env.example .env
```

macOS/Linux에서는 `cp .env.example .env`를 실행합니다.

MinIO 커뮤니티 버전은 [공식 안내](https://github.com/minio/minio#source-only-distribution)에 따라
`infra/minio/Dockerfile`에서 고정된 공식 소스 태그를 빌드합니다. 최초 빌드는 Go 의존성 다운로드와
컴파일로 몇 분 이상 걸릴 수 있으며, 이후에는 Docker 빌드 캐시를 재사용합니다.
호스트에 Go를 설치할 필요는 없습니다. Windows와 Intel Mac에서는 amd64, Apple Silicon Mac에서는
arm64용으로 각 환경에서 빌드합니다.

| 서비스 | 기본 접속 주소 |
| --- | --- |
| React 화면 (Nginx 경유) | http://localhost:8080 |
| API 문서 | http://localhost:8080/api/docs |
| API 생존 확인 | http://localhost:8080/api/spot_fit |
| PostgreSQL + PostGIS | localhost:5432 |
| MinIO S3 API | http://localhost:9000 |
| MinIO 콘솔 | http://localhost:9001 |
| Airflow | http://localhost:8081 (계정은 `.env` 참조) |
| MLflow | http://localhost:5000 |

포트를 변경하면 표의 접속 포트도 변경합니다. 컨테이너 내부에서는 DB를 `postgres:5432`,
객체 저장소를 `minio:9000`으로 접근합니다. 프런트엔드 API 호출은 `/api/...` 상대 경로를
사용합니다. 모델 버킷은 아래 모델 관리 명령으로 초기화하며, 수집 데이터 버킷은 별도로 구성합니다.
`/api/spot_fit`는 API 프로세스 생존만 확인하고 DB·MinIO 연결은 검사하지 않습니다.

### 로컬 데이터와 모델을 MinIO에 업로드

Python 의존성을 최초 한 번 설치한 뒤 프로젝트 루트에서 실행합니다. Docker 명령은 필요하지 않습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r scripts/upload-requirements.txt
python3 upload_minio.py
```

`upload_minio.py`는 `.env.dev`의 S3 설정을 읽고 `data` 아래 모든 일반 파일을 업로드합니다.
`data/raw/a.csv`는 `raw/a.csv`, `data/processed/a.parquet`는 `processed/a.parquet`로 저장됩니다.
숨김 파일·폴더(`.gitkeep`, `.DS_Store` 등)와 심볼릭 링크는 제외합니다.
동일한 객체 경로가 이미 존재하면 내용이 달라도 업로드하지 않습니다. 파일 삭제는 원격에 반영하지 않습니다.
기본 버킷은 `S3_DATA_BUCKET`, 미설정 시 `S3_MODEL_BUCKET`입니다. 버킷은 미리 생성되어 있어야 하며 읽기·쓰기 권한이 필요합니다.

모델은 아래 구조로 넣으면 같은 명령으로 자동 등록합니다. 각 버전 폴더에는 모델 파일 하나를 넣습니다.
여러 파일이 필요한 모델은 ZIP 등 하나의 아카이브로 묶으세요.

```text
models/
└── recommender/
    ├── v1/model.pkl
    └── v2/model.pkl
```

모델 파일은 기존 방식대로 MinIO의 `모델명/버전/SHA-256/파일명`에 저장됩니다.
JSON은 `model_registry/recommender/v1.json`처럼 생성되며 모델명·버전·버킷·객체 경로·객체 버전 ID·SHA-256·크기를 담습니다.
모델 버킷은 `.env.dev`의 `S3_MODEL_BUCKET`을 사용하며 MinIO 객체 버전 관리가 활성화되어 있어야 합니다.
기존 모델 관리 CLI의 `init`으로 초기화할 수 있습니다.

```bash
docker compose --env-file .env.dev -f compose.dev.yaml run --rm model-tools init
python3 upload_minio.py
```

등록된 동일 모델은 건너뛰고, 같은 모델명·버전의 내용이 바뀌면 오류를 내므로 `v2`처럼 새 버전 폴더를 사용하세요.
MinIO 객체만 있고 로컬 JSON이 없으면 원격 파일의 SHA-256을 검증하고 JSON을 복원하며 다시 업로드하지 않습니다.
생성된 `model_registry/` JSON은 Git에 커밋하고 모델 파일은 커밋하지 않습니다. 모델 활성화는 별도입니다.
다른 버킷이나 설정 파일은 `--bucket 버킷이름`, `--env-file .env.server`로 지정할 수 있습니다.
동일 경로에 대한 동시 실행은 피하세요. 존재 확인과 업로드는 별도 요청이므로 동시 실행 시 덮어쓰기가 발생할 수 있습니다.

Airflow와 MLflow는 PostgreSQL의 별도 `airflow`, `mlflow` 데이터베이스를 사용합니다.
Airflow 계정은 `.env`의 `AIRFLOW_ADMIN_USER`이며, 첫 실행 시 생성되는 비밀번호를 확인하려면
`docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated`를 실행합니다.
학습 의존성은 [ml/requirements.txt](ml/requirements.txt)에서 관리하고 백엔드와 Airflow 이미지 양쪽에 설치합니다.
Airflow·MLflow 웹 포트는 호스트의 `127.0.0.1`에만 열립니다.

PostgreSQL 컨테이너는 PostgreSQL 17과 PostGIS 3.5가 포함된 이미지를 사용합니다. 새 DB 볼륨을
처음 만들 때 `postgis`와 `postgis_topology` 확장을 자동으로 활성화합니다. 기존 PostgreSQL 볼륨을
유지하면서 PostGIS 이미지로 변경한 경우에는 컨테이너를 다시 만든 뒤 DB에서 확장을 한 번 활성화합니다.

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
SELECT PostGIS_Full_Version();
```

소스 코드는 bind mount되어 React HMR과 FastAPI reload가 적용됩니다.
`node_modules`는 Docker named volume에 저장해 macOS·Windows·Linux 네이티브 패키지가
섞이지 않도록 했습니다. 의존성 변경 후에는 `docker compose up --build -d`를 다시 실행합니다.
Windows/macOS 파일 변경 감지를 위해 기본 polling을 활성화했습니다.

```sh
docker compose logs -f
docker compose exec frontend npm run build
docker compose down
```

`docker compose down`은 DB·MinIO 데이터를 보존합니다. `down -v`는 데이터 볼륨까지
삭제하므로 초기화가 필요한 경우에만 사용합니다. 기존 DB 볼륨의 비밀번호는 `.env` 변경만으로
갱신되지 않습니다.

## MinIO로 학습 모델 관리

모델 파일은 MinIO에, 모델명·버전·저장 위치·SHA-256이 담긴 JSON은 `model_registry/`에 저장합니다.
로컬 저장소를 처음 준비할 때 실행합니다.

```sh
docker compose --profile tools build model-tools
docker compose --profile tools run --rm model-tools init
```

업로드·다운로드 및 백엔드 연동은 [모델 저장 안내](docs/model-storage.md)를 참고하세요.
공유 서버 개발 Compose는 Ubuntu MinIO를 사용합니다. 내부망 S3 API 접근과 제한된 계정 구성은 [개발 PC 서버 인프라 안내](docs/virtualbox-deployment.md#개발-pc에서-ubuntu-서버-인프라-사용)를 참고하세요. 로컬 전체 스택 구성은 PC별 저장소를 사용합니다.
Git에는 manifest만 올라가고 모델 파일은 MinIO에 저장됩니다.

일상적인 브랜치·커밋·Ubuntu 반영 절차, 패키지 추가 위치와 재빌드 명령, 모델을 Ubuntu MinIO에 등록하는 방법은 [개발 워크플로우](docs/development-workflow.md)를 참고하세요.

## GitHub · macOS · Windows 협업

- `.gitignore`는 OS 메타데이터, IDE 설정, 환경변수·개인 키, 의존성, 캐시, 데이터·모델 산출물을 제외합니다.
- `.env.example`, `.env.dev.example`, `.env.server.example`, `frontend/package-lock.json`, Python 의존성 명세와 마이그레이션은 커밋합니다.
- `.gitattributes`로 텍스트 파일은 LF, `.bat`·`.cmd`만 CRLF를 사용합니다.
- 빈 폴더는 `.gitkeep` 또는 Python `__init__.py`로 유지합니다.
- 파일명과 import의 대소문자를 정확히 맞춥니다. 대소문자만 다른 파일은 만들지 않습니다.
  대소문자만 바꿀 때는 `git mv old_name temp_name` 후 `git mv temp_name NewName`을 사용합니다.
- `node_modules`, 가상환경, DB 볼륨을 다른 OS에서 복사하지 않고 각 환경에서 재생성합니다.
- 이미 추적된 파일에는 `.gitignore`가 소급 적용되지 않습니다. 초기 커밋 전에 `git status`로 확인합니다.

이 초기 구성은 Git 저장소 초기화, 원격 저장소 연결, 커밋·push를 수행하지 않습니다.
Linux 운영 배포에서 변경할 항목은 [배포 안내](docs/deployment.md)를 참고하세요.
Windows PC의 VirtualBox Ubuntu VM에서 내부망 테스트 서버로 실행하려면
[VirtualBox 배포 안내](docs/virtualbox-deployment.md)를 참고하세요. Ubuntu에서는
`.env.server.example`을 `.env.server`로 복사해 VM 고정 IP와 서버 전용 비밀번호를 설정합니다.

## 참고 문서

- [Vite 시작하기](https://vite.dev/guide/): Node.js 버전 요구사항과 개발 서버
- [FastAPI 컨테이너 배포](https://fastapi.tiangolo.com/deployment/docker/): Python 이미지 기반 빌드
- [Compose 시작 순서](https://docs.docker.com/compose/how-tos/startup-order/): 의존 서비스와 상태 검사
