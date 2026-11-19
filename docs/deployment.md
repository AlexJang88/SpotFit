# SpotFit 1차 서버 배포 확인

## 2026년 자료 기반 2028·2029년 모델 자동 교체 상태

현재 자동 수집되는 것은 TOPIS 월별 교통량뿐입니다. 이 DAG의 성공은 모델 학습이나 배포 성공을 뜻하지 않습니다. 2026년 연간 원본·정답 데이터, 교통·범죄·부동산 세 모델의 2028년 및 2029년 예측, 통합 추천 결과가 모두 검증되기 전에는 새 모델을 서비스에 교체하지 않습니다.

범죄 정답은 **사건 발생지 기준 서울 25개 자치구별 수치**가 필요합니다. 서울시 5대 범죄 발생현황 통계는 구별 수치가 경찰관서 주소지 기준이라고 명시하므로 이를 대체 정답으로 사용하지 않습니다. 장소 유형별 통계도 자치구 발생지 수치를 제공하지 않습니다. 별도의 경찰청 `범죄 발생 지역별 통계`는 2025년 자료에 서울 25개 자치구 컬럼과 자동 변환 OpenAPI가 있어 후보로 확인됐습니다. 그러나 2026년판은 아직 게시되지 않았고, 자치구 컬럼이 사건 발생지 기준임을 원본 정의로 확인한 뒤에만 정답 데이터로 승인합니다. 그 전까지 범죄 모델 재학습과 통합 자동 배포는 차단 상태입니다.

현재 저장소의 교통 v94 학습·발행 코드와 범죄·부동산 학습 코드는 2025년 또는 기존 파일 형식에 고정된 부분이 남아 있습니다. 따라서 현 파이프라인을 2026년 데이터에 실행해 서비스 모델이 자동 교체된다고 간주해서는 안 됩니다. 공식 API 계약과 인증키를 확정한 뒤 세 수집기·전처리기·재학습기·후보 검증·원자적 교체·재시작 후 API 검증을 구현하고 통합 테스트해야 합니다. 인증키는 Git에 저장하지 않고 서버 환경변수로 주입합니다.

참고 공식 자료: [서울시 5대 범죄 발생현황](https://data.seoul.go.kr/dataList/DT201004L030001/S/2/datasetView.do), [경찰청 범죄 발생 지역별 통계](https://www.data.go.kr/data/3074462/fileData.do), [서울경찰청 5대범죄 발생 장소별 현황](https://www.data.go.kr/data/15054737/fileData.do).

### 공식 API 인증키 준비

공공데이터포털에 로그인하여 아래 네 서비스의 활용신청을 각각 완료하고 발급된 서비스키를 서버의 Git 비추적 `.env.server` 파일 `DATA_GO_KR_API_KEY`에 설정합니다. 개발 PC에서 테스트할 때는 `.env.dev`에 설정합니다. 키를 채팅·Git·문서에 붙여 넣지 않습니다. 서비스별로 서로 다른 키가 발급되면 수집기를 연결할 때 환경변수를 분리해야 하므로 키 값 대신 그 사실만 공유합니다.

- [한국도로교통공단 지자체별 대상 교통사고 통계](https://www.data.go.kr/data/15056770/openapi.do)
- [경찰청 범죄 발생 지역별 통계](https://www.data.go.kr/data/3074462/fileData.do) — 2026년판과 사건 발생지 기준 여부 확인 전에는 학습 정답으로 사용하지 않음
- [국토교통부 아파트 매매 실거래가 자료](https://www.data.go.kr/data/15126469/openapi.do)
- [국토교통부 아파트 전월세 실거래가 자료](https://www.data.go.kr/data/15126474/openapi.do)

TOPIS 월별 교통량 수집에는 현재 별도 API 키가 필요하지 않습니다. 위 키를 입력해도 현재 DAG가 자동으로 네 API를 호출하거나 모델을 교체하는 것은 아닙니다. 수집기·학습·검증·승격 구현 및 실데이터 검증이 별도로 필요합니다.

## 교통 모델 신규 연도 데이터 수집 (구축 1단계)

Airflow의 `traffic_refresh_collection` DAG는 매월 말일 23:00(한국시간)에 TOPIS 월별
교통량 원본을 `data/raw/traffic_model_refresh/topis/<연도>/`로 수집합니다. 연도가
바뀌면 이전 연도의 빠진 월도 다시 확인합니다. 이미 저장된 원본과 다른 내용이 공개되면
덮어쓰지 않고 실패하여 운영자가 개정 여부를 확인하게 합니다. `data/interim`은 사용하지
않습니다.

Ubuntu 서버의 `.env.server`에서 `AIRFLOW_UID`를 `id -u` 결과로 설정하고,
`data/raw`가 그 UID로 쓰기 가능한지 확인해야 합니다. DAG의 연간 준비 상태는 다음처럼
확인할 수 있습니다.

```bash
docker compose --env-file .env.server -f compose.yaml -f compose.server.yaml exec airflow \
  python -m scripts.traffic_data.refresh_readiness --year 2026
```

현재 자동 수집기는 TOPIS만 다룹니다. 교통사고, 사고다발지역, 인구, 학교, 건강 자료의
2026년 수집·전처리 코드와 새 연도 교통·범죄 모델 학습 및 통합 서비스 승격은 아직
연결되지 않았습니다. 준비 상태가 `false`일 때는 기존 2027·2028년 서비스 모델을
유지해야 합니다. 이 DAG의 성공은 데이터 수집 성공일 뿐 신규 통합 모델 배포 성공을
뜻하지 않습니다.

## 부동산 모델 월별 자동 갱신 (매매 V64 · 전세 V33 · 월세 전환율)

Airflow의 `property_refresh_pipeline` DAG가 매월 말일 23:00(한국시간)에 아래 순서로 실행됩니다.

1. **수집** (Airflow): 국토교통부 아파트 매매·전월세 실거래가 API와 한국은행 ECOS 주택담보대출 금리를 `data/raw/부동산/refresh/`로 내려받습니다. 신고 기한(30일)을 고려해 오늘 기준 두 달 전까지를 확정 월로 보고, 확정 월 중 최근 3개월은 매번 다시 받아 늦은 신고·취소를 반영합니다.
2. **전처리·재학습·교체** (백엔드 `POST /api/models/property/refresh`): 받은 달을 기존 원본 CSV(`아파트_매매_2010_2026.csv` 등)에 병합 → 입력 파일 3개(`data/processed/부동산/`) 재생성 → 모델 3개 재학습 → 검증 → `models/recommender/v1/apart_*_final` 교체 → 서비스 캐시 갱신.
3. **준비 확인(재학습 전)**: 매매·전월세·금리 세 자료가 모두 확정된 마지막 달(오늘 기준 두 달 전)까지 있고, 그 달이 서비스 중인 모델의 기준 월보다 새로울 때만 병합·재학습이 실행됩니다. 하나라도 모자라거나 새로운 달이 없으면 아무 파일도 바꾸지 않고 정상 종료하며, 사유는 `artifacts/property_refresh/last_run.json`의 `process.reasons`에 남습니다 (`process.skipped: true`).
4. **검증에서 걸러지는 경우**: 예측이 있는 동네가 10% 넘게 줄거나, 현재 가격 중앙값이 이전의 0.7~1.5배를 벗어나거나, 예측값이 0 이하·결측이면 새 모델을 버리고 기존 모델을 유지한 채 DAG가 실패합니다. 월세는 학습 마지막 날이 과거로 가거나 학습 건수가 10% 넘게 줄면 같습니다.

필요한 인증키 (서버 `.env.server`, 개발 PC는 `.env.dev`):

| 환경변수 | 발급처 | 용도 |
|---|---|---|
| `DATA_GO_KR_API_KEY` | 공공데이터포털 [아파트 매매 실거래가 자료](https://www.data.go.kr/data/15126469/openapi.do), [아파트 전월세 실거래가 자료](https://www.data.go.kr/data/15126474/openapi.do) 활용신청 (같은 계정이면 키 하나) | 실거래가 수집 |
| `ECOS_API_KEY` | 한국은행 [ECOS Open API](https://ecos.bok.or.kr/api/) 인증키 신청 | 주택담보대출 금리 수집 |
| `MODEL_TRAINING_API_KEY` | 직접 정한 긴 임의 문자열 (이미 있음) | DAG가 백엔드 갱신 API를 호출할 때 |

수동 실행: `python -m scripts.property_data.run_property_refresh --stage collect|process|all` (저장소 루트). 결과는 `artifacts/property_refresh/last_run.json`, 실행·학습 로그는 `artifacts/property_refresh/refresh.log`와 `artifacts/property_model_bootstrap.log`에 남습니다. 병합으로 원본 CSV가 바뀌기 전의 파일은 `data/raw/부동산/_backup/`에 하나씩 보관합니다.

도로명주소 DB(`rnaddrkor_seoul.txt`), K-apt 단지 정보, 상권분석 행정동 영역, 한강공원 위치는 거의 바뀌지 않는 자료라 자동 수집하지 않습니다. 행정동이 개편되면 이 파일들을 직접 교체해야 하며, 새 행정동 코드를 찾지 못한 거래는 `last_run.json`의 `no_dong` 건수로 보고됩니다.

현재 `compose.yaml`과 `compose.server.yaml`은 Ubuntu 시연 서버용 구성입니다. 프런트엔드 Vite 개발 서버와 HTTP 프록시를 사용하므로 공개 운영 환경의 보안 설정을 완료한 구성은 아닙니다.

## 실행 전

1. `.env.server.example`을 `.env.server`로 복사하고 실제 서버 주소와 비밀키를 설정합니다. `.env.server`는 Git에 올리지 않습니다.
2. `data/raw/`, `data/processed/`의 필요한 입력과 `models/recommender/v1/`의 최종 산출물을 서버에 전달합니다. 이 폴더들의 실제 파일은 Git에서 제외됩니다.
3. 서버에 모델 파일이 없다면 시작 시 백그라운드에서 복구를 시도합니다. 주거 모델은 주거 전처리 입력, 교통 v94는 `data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx`, 아동 범죄 모델은 해당 raw/processed 입력이 필요합니다.

과거 안전 참고지표 그래프를 표시하려면 `data/processed/child_crime_exposure/borough_panel/panel.csv`와 `data/raw/94차 모델 데이터/교통사고/자치구별_어린이교통사고_2017_2021.xlsx`, `자치구별_어린이교통사고_2022_2025.xlsx`도 서버에 있어야 합니다. 백엔드 시작 시 소스보다 오래되었거나 없는 `data/processed/neighborhood_reference/annual_safety_reference.csv`를 자동 생성합니다. 수동 생성은 `python scripts/build_historical_safety_reference.py`입니다. 이 그래프는 매년 서울 25개 자치구 내 범죄 노출률·어린이 교통사고율의 5분위 상대순위 참고값으로, 2027·2028년 예측 동네점수와 동일한 산식이 아닙니다.

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
