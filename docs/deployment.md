# SpotFit 1차 서버 배포 확인

현재 `compose.yaml`과 `compose.server.yaml`은 Ubuntu 시연 서버용 구성입니다. 프런트엔드 Vite 개발 서버와 HTTP 프록시를 사용하므로 공개 운영 환경의 보안 설정을 완료한 구성은 아닙니다.

## 실행 전

1. `.env.server.example`을 `.env.server`로 복사하고 실제 서버 주소와 비밀키를 설정합니다. `.env.server`는 Git에 올리지 않습니다.
2. `data/raw/`, `data/processed/`의 필요한 입력과 `models/recommender/v1/`의 최종 산출물을 서버에 전달합니다. 이 폴더들의 실제 파일은 Git에서 제외됩니다.
3. 서버에 모델 파일이 없다면 시작 시 백그라운드에서 복구를 시도합니다. 주거 모델은 주거 전처리 입력, 교통 v94는 `data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx`, 아동 범죄 모델은 해당 raw/processed 입력이 필요합니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml config --quiet
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml up --build -d
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml ps
```

## 실행 후

`GET /api/spot_fit`은 API 생존 확인용입니다. 모델과 추천 서비스는 아래 경로에서 별도로 확인합니다.

```text
GET /api/recommender/model-status
GET /api/recommender/district-forecasts?horizon=2
GET /api/recommender/district-forecasts?horizon=3
```

`model-status`의 `readiness`에서 `property`, `traffic`, `child_crime`가 모두 `ready: true`인지 확인합니다. 하나라도 `false`이면 해당 모델 파일이나 입력 데이터, 백엔드 로그를 확인합니다. 추천 API는 필요한 모델이 준비되지 않으면 503을 반환할 수 있습니다.

## 공개 운영 전환 시 필요한 작업

- 프런트엔드를 `npm ci && npm run build`로 빌드하고 생성된 정적 파일을 Nginx에서 제공합니다.
- 백엔드 소스 bind mount와 개발용 설정을 제거하고, HTTPS와 HTTP 리다이렉트를 설정합니다.
- PostgreSQL·MinIO 포트 접근을 제한하고 별도 서비스 계정 및 백업 정책을 적용합니다.
- 검증된 의존성 버전과 컨테이너 이미지 버전을 고정합니다.
