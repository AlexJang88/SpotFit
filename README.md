# SpotFit 배포 가이드

SpotFit은 서울 지역의 아파트 가격, 아동범죄노출, 어린이 교통안전을 함께 비교하고 예산에 맞는 동네를 추천하는 서비스입니다.

이 문서는 새로운 Ubuntu 서버에 SpotFit을 배포하는 순서를 설명합니다. 서버와 네트워크가 정상이고, 모델·데이터 파일을 전달받았다고 가정합니다. 아래 명령은 별도 설명이 없으면 프로젝트 루트인 `/home/spotfit/SpotFit`에서 실행합니다.

## 1. 배포 후 실행되는 서비스

| 서비스 | 역할 | 기본 주소 |
| --- | --- | --- |
| SpotFit | 프런트엔드와 API | `http://서버IP:8080` |
| Airflow | 데이터 수집·갱신 파이프라인 | `http://서버IP:8081` |
| MLflow | 모델 실험·버전 기록 | `http://서버IP:5000` |
| MinIO | 데이터·모델 객체 저장소 | API `9000`, 콘솔 `9001` |
| PostgreSQL/PostGIS | 서비스·파이프라인 DB | `5432` |

SpotFit 화면은 Nginx, React/Vite, FastAPI로 구성되며 모든 구성요소는 Docker Compose로 실행됩니다.

## 2. 서버 준비

권장 환경은 Ubuntu와 Docker Compose v2입니다.

### 2.1 기본 프로그램 설치

```bash
sudo apt update
sudo apt install -y git curl ca-certificates openssl nano python3 python3-venv python3-pip
```

### 2.2 Docker 설치

Docker와 Compose가 이미 설치되어 있다면 이 단계는 건너뜁니다.

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker spotfit
```

`spotfit`이 실제 배포 계정이 아니라면 마지막 명령의 사용자명을 변경합니다. 명령 실행 후 SSH에서 로그아웃하고 다시 접속합니다.

```bash
docker --version
docker compose version
```

Docker로 배포하므로 서버에 Node.js나 머신러닝 라이브러리를 별도로 설치할 필요는 없습니다. 이미지 빌드 과정에서 필요한 라이브러리가 자동 설치됩니다.

## 3. 프로젝트 받기

```bash
cd /home/spotfit
git clone --branch develop https://github.com/AlexJang88/SpotFit.git
cd /home/spotfit/SpotFit
git branch --show-current
```

배포 대상 브랜치가 변경되었다면 `develop` 대신 지정된 브랜치를 사용합니다.

## 4. 서버 환경변수 설정

### 4.1 환경파일 생성

```bash
cd /home/spotfit/SpotFit
cp .env.server.example .env.server
chmod 600 .env.server
```

### 4.2 비밀번호 생성

다음 명령을 네 번 실행해 PostgreSQL, MinIO, Airflow, 모델관리 API에 서로 다른 값을 사용합니다.

```bash
openssl rand -hex 24
```

Airflow Fernet 키는 다음 명령으로 생성합니다.

```bash
python3 -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

### 4.3 `.env.server` 작성

```bash
nano .env.server
```

아래 내용을 기준으로 작성합니다. `192.168.0.3`은 실제 서버 IP로 바꾸고, `CHANGE_ME_...` 값은 앞에서 생성한 값으로 바꿉니다.

```dotenv
# SpotFit
APP_BIND_ADDRESS=192.168.0.3
APP_PORT=8080

# PostgreSQL
POSTGRES_DB=spot_fit
POSTGRES_USER=spot_fit
POSTGRES_PASSWORD=CHANGE_ME_POSTGRES_PASSWORD
POSTGRES_BIND_ADDRESS=127.0.0.1
POSTGRES_PORT=5432

# MinIO
MINIO_ROOT_USER=spot_fit_admin
MINIO_ROOT_PASSWORD=CHANGE_ME_MINIO_PASSWORD
MINIO_API_BIND_ADDRESS=127.0.0.1
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# S3 / 모델 저장소
S3_ENDPOINT_URL=http://minio:9000
S3_MODEL_BUCKET=spot-fit-models
S3_REGION=us-east-1
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=

# MLflow
MLFLOW_BIND_ADDRESS=127.0.0.1
MLFLOW_PORT=5000
MLFLOW_ARTIFACT_BUCKET=spot-fit-mlflow

# Airflow
AIRFLOW_BIND_ADDRESS=127.0.0.1
AIRFLOW_PORT=8081
AIRFLOW_UID=1000
AIRFLOW_ADMIN_USER=spotfit
AIRFLOW_FERNET_KEY=CHANGE_ME_AIRFLOW_FERNET_KEY
AIRFLOW_SECRET_KEY=CHANGE_ME_AIRFLOW_SECRET_KEY

# 모델 생성·갱신
MODEL_AUTO_BOOTSTRAP=1
MODEL_TRAINING_API_KEY=CHANGE_ME_MODEL_TRAINING_API_KEY

# 애플리케이션 로그
APP_LOG_LEVEL=INFO
APP_LOG_RETENTION_DAYS=30
APP_TIMEZONE=Asia/Seoul
WATCH_USE_POLLING=false

# 데이터 수집 API
DATA_GO_KR_API_KEY=발급받은_공공데이터포털_키
ECOS_API_KEY=발급받은_ECOS_키
```

`AIRFLOW_UID`는 다음 명령의 출력값으로 설정합니다.

```bash
id -u
```

`S3_ACCESS_KEY_ID`와 `S3_SECRET_ACCESS_KEY`를 비워두면 서버 Compose가 MinIO 관리자 계정을 사용합니다. 별도의 MinIO 서비스 계정을 사용한다면 두 항목에 해당 접근키를 입력합니다.

Airflow, MLflow, MinIO 콘솔을 다른 PC에서 직접 접속해야 한다면 각 `*_BIND_ADDRESS`도 실제 서버 IP로 바꿉니다. SSH 터널로 관리 화면에 접속할 경우에는 `127.0.0.1`을 유지합니다.

### 4.4 VWorld 지도 키 설정

```bash
nano frontend/.env.local
```

```dotenv
VITE_VWORLD_API_KEY=발급받은_VWorld_웹지도_키
```

VWorld에서 실제 서비스 주소 또는 도메인을 사용 주소로 등록합니다.

## 5. 모델과 데이터 배치

모델과 데이터는 Git에 포함되지 않으므로 담당자에게 전달받아 아래 경로에 복사합니다.

```text
/home/spotfit/SpotFit/
├── models/recommender/v1/
├── data/raw/
├── data/processed/
├── data/interim/
├── artifacts/
└── model_registry/
```

최소한 서비스 최종 모델 폴더인 `models/recommender/v1`은 폴더 구조를 유지한 상태로 전체 복사합니다.

```text
models/recommender/v1/
├── apart_sale_final/
├── apart_jeonse_final/
├── apart_wolse_final/
├── traffic_accident_v94/
└── child_crime_exposure/
```

아동범죄노출 모델에 `current_release.json`과 `releases/`가 있다면 모두 함께 복사합니다. 모델 파일뿐 아니라 같은 폴더의 메타데이터, 예측 CSV, Feature 점수 파일도 필요합니다.

상세보기의 과거 가격·안전 그래프에는 다음 결과 파일이 사용됩니다.

```text
data/processed/property_price_history/annual_by_dong.csv
data/processed/property_price_history/annual_monthly_by_dong.csv
data/processed/neighborhood_reference/annual_safety_reference.csv
```

결과 파일을 서버 시작 시 생성하려면 다음 입력 파일을 함께 복사합니다.

```text
data/raw/부동산/아파트_매매_2010_2026.csv
data/raw/부동산/아파트_전월세_2010_2026.csv
data/processed/child_crime_exposure/borough_panel/panel.csv
data/raw/94차 모델 데이터/교통사고/자치구별_어린이교통사고_*.xlsx
```

누락 모델 자동 생성에는 각 모델의 원본·전처리 입력이 필요합니다. 교통사고 모델은 다음 파일도 사용합니다.

```text
data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx
```

## 6. Compose 설정 확인

```bash
cd /home/spotfit/SpotFit
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
```

명령이 출력 없이 끝나면 환경변수와 Compose 문법 확인이 완료된 것입니다.

## 7. 컨테이너 빌드 및 실행

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps -a
```

처음 실행할 때는 Docker 이미지와 Python·Node 라이브러리를 설치하므로 시간이 걸릴 수 있습니다. 다음 컨테이너들이 실행됩니다.

- `nginx`
- `frontend`
- `backend`
- `postgres`
- `minio`
- `mlflow`
- `airflow`

`tracking-db-init`과 `mlflow-init`은 초기화 작업을 마친 후 `Exited (0)`으로 종료되는 것이 정상입니다.

컨테이너가 준비될 때까지 상태를 확인합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps -a
```

## 8. SpotFit DB 테이블과 초기데이터 생성

PostgreSQL 컨테이너가 `healthy`가 된 후 실행합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml exec -T postgres mkdir -p /tmp/spotfit-migrations

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml cp ./infra/postgres/migrations/. postgres:/tmp/spotfit-migrations/

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml exec -T postgres sh -c 'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/spotfit-migrations/000_install.sql'
```

`000_install.sql`이 필요한 SQL을 순서대로 실행하므로 `infra/postgres/migrations` 아래 SQL을 각각 실행할 필요가 없습니다.

설치 결과를 검증합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml exec -T postgres sh -c 'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/spotfit-migrations/090_verify.sql'
```

이 과정에서 `spotfit` 스키마, 서비스 테이블, 모델 유형, 서울시 행정구역, 기본 점수·등급·추천 정책이 생성됩니다.

## 9. 배포 확인

서버 IP가 `192.168.0.3`인 경우 다음 주소를 확인합니다.

| 확인 항목 | 주소 |
| --- | --- |
| SpotFit 화면 | `http://192.168.0.3:8080` |
| API 문서 | `http://192.168.0.3:8080/api/docs` |
| API 실행 상태 | `http://192.168.0.3:8080/api/spot_fit` |
| 서비스 모델 상태 | `http://192.168.0.3:8080/api/recommender/model-status` |
| 모델 자동생성 상태 | `http://192.168.0.3:8080/api/models/status` |

서버 터미널에서도 확인할 수 있습니다.

```bash
curl -fsS http://192.168.0.3:8080/api/spot_fit
curl -fsS http://192.168.0.3:8080/api/recommender/model-status
curl -fsS http://192.168.0.3:8080/api/models/status
curl -fsS http://192.168.0.3:8080/api/neighborhoods/catalog
```

마지막으로 화면에서 다음 기능을 확인합니다.

1. 서울시 지도와 자치구·행정동 검색
2. 매매·전세·월세 가격 조회
3. 예산 범위와 면적 선택
4. 아동범죄노출·교통안전 비율 조절
5. 2027년·2028년 예측연도 변경
6. 예산에 맞는 추천 동네 5곳 조회
7. 자치구·행정동 상세정보와 가격 변화 그래프
8. 자치구 동네점수와 과거 안전 참고지표

## 10. 관리 화면 접속

관리 포트를 `127.0.0.1`로 설정한 경우 로컬 PC 터미널에서 SSH 터널을 엽니다.

```bash
ssh -L 8081:127.0.0.1:8081 -L 5000:127.0.0.1:5000 -L 9001:127.0.0.1:9001 spotfit@192.168.0.3
```

SSH 연결을 유지한 상태에서 브라우저로 접속합니다.

| 관리 화면 | 주소 |
| --- | --- |
| Airflow | `http://localhost:8081` |
| MLflow | `http://localhost:5000` |
| MinIO 콘솔 | `http://localhost:9001` |

Airflow 초기 사용자와 비밀번호는 다음 로그에서 확인합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs airflow
```

MinIO는 `.env.server`의 `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`로 로그인합니다.

## 11. Airflow 파이프라인 활성화

Airflow에 로그인한 뒤 사용할 DAG를 활성화합니다.

| DAG | 실행 주기 | 기능 |
| --- | --- | --- |
| `property_refresh_pipeline` | 매월 말일 23:00 | 부동산 수집·전처리·모델 갱신 |
| `crime_refresh_pipeline` | 매월 말일 23:00 | 범죄 데이터 수집·전처리·2년/3년 모델 학습·검증·서비스 반영 |
| `traffic_refresh_collection` | 매월 말일 23:00 | 교통사고 TOPIS 자료 수집·연간 준비 상태 확인 |

범죄 파이프라인의 상세 수집 범위는 [범죄 데이터 갱신 파이프라인](docs/crime-refresh-pipeline.md)을 참고합니다. MDIS 신규 연도 자료는 직접 추출해 기존 경로에 추가하고, 유흥주점·단란주점 API는 공공데이터포털에서 활용신청을 완료해야 합니다.

## 12. 서버 재부팅 후 실행

`compose.server.yaml`의 상시 서비스에는 재시작 정책이 설정되어 있어 Docker 서비스가 시작되면 다시 실행됩니다. 수동으로 실행하려면 다음 명령을 사용합니다.

```bash
cd /home/spotfit/SpotFit
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps -a
```

## 13. 코드 업데이트 배포

서버에서 새 코드를 반영할 때는 다음 순서로 실행합니다.

```bash
cd /home/spotfit/SpotFit
git pull --ff-only origin develop

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up -d --force-recreate backend frontend nginx

docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps -a
```

새로운 환경변수가 추가됐다면 `.env.server.example`과 기존 `.env.server`를 비교해 실제 값을 추가합니다. 새로운 모델·데이터가 배포 대상이라면 Git pull과 별도로 동일한 경로에 복사합니다. DB 변경이 포함됐다면 8절의 설치·검증 SQL을 다시 실행합니다.

## 14. 로그 확인

전체 로그:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs --tail=200
```

백엔드 실시간 로그:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs -f --tail=100 backend
```

프런트엔드·Nginx 로그:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs --tail=100 frontend nginx
```

파일로 저장되는 백엔드 로그:

```text
logs/backend/spotfit-api.log
```

## 15. 서비스 중지와 재실행

중지:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml stop
```

재실행:

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up -d
```

## 16. 배포 완료 체크리스트

- [ ] Docker와 Docker Compose가 설치되어 있다.
- [ ] 프로젝트 `develop` 브랜치를 받았다.
- [ ] `.env.server`의 IP, 비밀번호, API 키를 설정했다.
- [ ] `frontend/.env.local`에 VWorld 키를 설정했다.
- [ ] `models/recommender/v1` 전체 모델 묶음을 복사했다.
- [ ] `data/raw`, `data/processed` 등 서비스 입력 데이터를 복사했다.
- [ ] Compose 설정 검사를 통과했다.
- [ ] 전체 컨테이너를 빌드하고 실행했다.
- [ ] SpotFit DB 테이블과 초기데이터를 생성하고 검증했다.
- [ ] 모델 상태 API에서 모델 준비 상태를 확인했다.
- [ ] 실제 화면에서 가격·점수·추천·상세 그래프를 확인했다.
- [ ] Airflow에서 필요한 DAG를 활성화했다.

## 공개 운영 전 참고

현재 Compose는 내부망 시연을 위한 HTTP·Vite 개발 서버 구성입니다. 인터넷에 공개하는 운영 서비스로 전환할 때는 HTTPS와 도메인, 정적 프런트엔드 빌드, 관리 포트 접근제어, 비밀키 관리, 데이터베이스·MinIO·모델 백업을 추가로 구성해야 합니다.
