# 최종 서비스 구현·배포 안내

## 화면과 추천 규칙

- 진입 화면: `frontend/src/views/pages/FinalServicePage.tsx` (기존 HomePage는 보존).
- PPT의 회녹색·민트 톤: #354241 / #617B78 / #8EB3AD / #E9F5F1.
- 사용자는 매매/전세, 총예산(만원), 선택적으로 면적(㎡), 범죄 예측연도(2027/2028)를 입력한다.
- 아동범죄노출·교통안전 영역 선택은 제거했다. 두 영역을 항상 사용한다.
- 영역 간 비율은 서비스 기본 가정인 50:50이다. 학습에서 추출한 Feature 중요도와는 다르다.
- 범죄는 2025 기준 2년/3년 모델의 실제 `service_forecast.csv` 결과를 사용한다. Y가 낮은 순으로 5개 구씩 100/80/60/40/20점이다.
- 교통안전은 현재 사용 가능한 2025년 결과를 안전점수 높은 순으로 동일한 5분위 점수로 환산한다. 같은 원점수에는 같은 순위/점수를 준다.
- 종합점수 = 범죄안전 점수 × 0.5 + 교통안전 점수 × 0.5. 종합점수 → 범죄안전 점수 → 자치구명 순으로 정렬한 25개 구를 5개씩 A–E로 표시한다. 따라서 동점 점수라도 등급 경계에서는 등급이 다를 수 있다.
- 예산 이하 행정동을 종합점수 → 범죄안전 점수 → 낮은 가격 → 지역명 순으로 최대 5개 추천한다. 같은 구의 여러 동이 포함될 수 있다. 후보가 부족하면 실제 개수만 반환한다.
- 면적 미입력: 기존 가격 모델의 행정동 대표 총액. 입력: 대표 평당가격 × 면적 / (400/121). 후자는 면적 환산 추정값이지 아파트 개별 매물 예측이 아니다.
- 행정동 검색/선택: 해당 동 가격 + 소속 자치구 점수. 모든 동이 독립적인 범죄·교통 예측을 가진 것처럼 표시하지 않는다.
- 자치구 가격은 가격 자료가 있는 하위 동의 단순평균이다. 가격이 없는 동은 검색 가능하나 가격 정보 없음으로 표시하고 추천에서 제외한다.

## 데이터 한계 (화면에도 명시)

범죄 예측은 2027/2028년이지만 교통안전은 2025년 참고 결과다. 동일 미래연도 두 모델을 결합한 서비스는 아직 아니다. 최신 교통 모델의 2027/2028 결과를 연결하려면 결과 파일·필드·점수 방향을 확인하고 어댑터를 교체해야 한다.

부동산 가격은 최신 거래 특성 기반의 대표 추정값이며 미래 가격 시계열 예측이 아니다. 현재 스냅샷의 매매 자료연도는 2026년, 전세는 2024년이다. 실제 매물 목록/거래 가능 여부를 보증하지 않는다.

## API

- `GET /api/neighborhoods/catalog?horizon=2` — 전체 가격·구 점수·기준연도·경고.
- `GET /api/neighborhoods/recommendations?budget_manwon=50000&transaction=sale&horizon=2` — TOP 5. 선택 인자 `area_m2=84`, `limit=1..5`.
- `GET /api/neighborhoods/detail?district=강남구&neighborhood=역삼1동&horizon=3&transaction=rent` — 행정동 상세.
- `horizon`은 2 또는 3, `transaction`은 sale 또는 rent. 잘못된 값 422, 없는 지역 404, 모델 산출물 누락/이상 503. 원인은 백엔드 로그에 기록한다.
- 기존 범죄 API는 유지한다. 새 화면은 위 신규 API를 사용한다.

## 서버에 필요한 파일

Git으로 소스만 전달하면 모델/가격 자료는 누락될 수 있다. 다음 로컬 파일은 별도 동기화한다.

1. `models/recommender/v1/child_crime_exposure/2y/`와 `3y/`: 각 `child_crime_exposure_2y.joblib` 또는 `child_crime_exposure_3y.joblib`, `service_forecast.csv`, `model_metadata.json`.
2. `models/recommender/v1/traffic_accident/2025년_최종검증_예측결과.csv`.
3. 부동산은 아래 중 하나:
   - 로컬 추론: `apart_sale/model_bundle.joblib`, `apart_rent/model_bundle.joblib`와 `data/raw/아파트_매매.csv`, `data/raw/아파트_전월세.csv`.
   - 원자료가 없는 배포 환경: `models/recommender/v1/property_service_forecasts.json`.

현재 가격 스냅샷은 사용자가 지정한 기존 서비스 `http://192.168.0.3:8080/api/recommender/district-forecasts`에서 내려받은 실제 422개 동의 모델 출력이다. 합성 가격이 아니며 출처와 추출시간을 JSON에 보존한다. 새 서비스 기동 시 원격 서버에 다시 접속하지 않는다.

가격 갱신은 원자료와 모델이 있는 환경에서 `python scripts/export_property_service_catalog.py`를 실행하고 생성 파일을 배포한다. 신뢰하는 기존 모델 API에서 추출할 경우만 `--source-url`을 명시한다. 스냅샷을 교체한 뒤 백엔드를 재시작하여 캐시를 갱신한다.

지도 배경에는 `frontend/.env`의 `VITE_VWORLD_API_KEY`가 필요하다. 없는 경우 지도 배경 누락 안내와 행정경계만 표시한다. 키를 Git에 커밋하지 않는다.

기존 Docker 배포 설정/비밀정보를 유지한 상태에서 소스와 위 자료를 전달하고 `docker compose up -d --build backend frontend nginx`로 반영한다. Compose 개발 구성에서 프론트엔드는 소스 마운트를 사용하며 운영 구성은 해당 배포 문서를 따른다. 이번 작업에서는 원격 서버를 재시작하거나 배포하지 않았다.

## 확인

- API 테스트: `PYTHONPATH=backend python -m pytest backend/tests/integration/test_neighborhood_recommendations.py -q` (pytest/httpx 설치 필요; PowerShell은 `$env:PYTHONPATH='backend'`).
- 프론트엔드: `cd frontend`, `npm ci`, `npx tsc --noEmit`, `npm run build`.
- 브라우저 검증 스크립트: `scripts/verify_final_service_ui.py`. 기본 대상은 로컬 5181 포트. TOP 5, 행정동 상세, 2028 전환, 후보 없음, 전세·면적, 모바일 표시를 검증한다.
- 이번 검증 결과: API 26개 통과, TypeScript 검사/배포 빌드 통과. 실제 모델 산출물로 브라우저 검증 통과.
