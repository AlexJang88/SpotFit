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
학습 작업은 웹 요청 처리와 분리하여 `ml`에서 실행합니다. Airflow·MLflow·학습 라이브러리는
폴더만 준비했으며 기본 Compose에는 포함하지 않았습니다.

## 개발 환경 실행

Docker Engine/Desktop와 Docker Compose v2 이상이 필요합니다. Windows는 Docker Desktop의
Linux 컨테이너 모드를 사용합니다. 호스트에 Node.js와 Python을 설치할 필요는 없습니다.

1. 환경 파일을 복사합니다.

   Windows PowerShell:
   ```powershell
   Copy-Item .env.example .env
   ```

   macOS / Linux:
   ```bash
   cp .env.example .env
   ```

2. `.env`에서 로컬 개발용 비밀번호와 필요한 포트를 설정합니다.
3. 프로젝트 루트에서 실행합니다.

   ```sh
   docker compose config --quiet
   docker compose up --build -d
   docker compose ps
   ```

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

포트를 변경하면 표의 접속 포트도 변경합니다. 컨테이너 내부에서는 DB를 `postgres:5432`,
객체 저장소를 `minio:9000`으로 접근합니다. 프런트엔드 API 호출은 `/api/...` 상대 경로를
사용합니다. 모델 버킷은 아래 모델 관리 명령으로 초기화하며, 수집 데이터 버킷은 별도로 구성합니다.
`/api/spot_fit`는 API 프로세스 생존만 확인하고 DB·MinIO 연결은 검사하지 않습니다.

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
현재는 각 PC의 로컬 MinIO를 사용합니다. 맥북·Windows 간 모델 공유에는 공용 MinIO 서버가 필요하며,
Git만으로 모델 파일이 공유되지는 않습니다.

## GitHub · macOS · Windows 협업

- `.gitignore`는 OS 메타데이터, IDE 설정, 환경변수·개인 키, 의존성, 캐시, 데이터·모델 산출물을 제외합니다.
- `.env.example`, `frontend/package-lock.json`, Python 의존성 명세와 마이그레이션은 커밋합니다.
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
