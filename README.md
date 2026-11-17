# SpotFit

서울 지역의 주거 가격, 아동 범죄 노출, 어린이 교통안전 정보를 제공하는 React·FastAPI 서비스입니다.

## 1차 서버 실행

Ubuntu 서버에서 Docker Compose v2를 사용합니다. `.env.server.example`을 `.env.server`로 복사하고 실제 주소, 비밀번호, 키를 설정한 뒤 실행합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

기본 앱 포트는 `8080`이며 API 생존 확인 경로는 `/api/spot_fit`, API 문서는 `/api/docs`입니다. 운영 배포 전환에 필요한 항목은 [배포 안내](docs/deployment.md)에 정리했습니다.

## 배포에 필요한 파일

Git에는 애플리케이션 코드와 환경변수 예시가 있습니다. `data/`, `models/`, `artifacts/`의 실제 내용은 Git에서 제외되므로 서버로 별도 전달해야 합니다.

- `data/raw/`, `data/processed/`: 서비스 모델 생성에 필요한 원천·전처리 입력. 교통사고 v94는 `data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx`를 사용합니다.
- `models/recommender/v1/`: 추천 서비스의 최종 모델과 예측 CSV. 서버 시작 시 누락된 모델을 복구하는 기능이 있지만, 관련 입력 파일을 먼저 준비해야 합니다.
- `.env.server`: 서버 전용 설정과 비밀키. Git에 올리지 않습니다.

`data/interim/`의 교통사고 v94 파일은 최종 모델 단독 재학습에 필요하지 않습니다. 다른 도메인의 전처리·모델 복구 경로는 배포 전에 서버에서 확인하세요.

## 확인

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs --tail=100 backend
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml logs --tail=100 frontend
```

`/api/spot_fit`은 API 프로세스 생존만 확인합니다. 모델 준비 상태와 추천 API는 실제 데이터·모델이 준비된 상태에서 별도로 확인해야 합니다.

현재 `compose.yaml`과 `compose.server.yaml`은 1차 시연용 구성으로, 프런트엔드 개발 서버와 HTTP 프록시를 사용합니다. 공개 운영 배포에는 정적 프런트엔드 빌드, HTTPS, 비밀키·백업 설정을 추가해야 합니다.
