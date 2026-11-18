# 범죄 데이터 갱신 파이프라인

## 구성

`crime_refresh_pipeline` DAG: 매월 6일 05:00 Asia/Seoul, catchup 없음, 동시 실행 1개.

공개 다운로드 → 원본 스냅샷/출처·SHA256 기록 → 별도 작업공간 전처리 → 입력 가용성 검증 →
2·3년 모델 과거 시점 재학습 검증 → 최종 재학습/점수 산출 → 타 모델 연도 검증 → 원자적 서비스 전환.

Airflow는 표준 라이브러리로 수집만 수행합니다. 학습은 pandas/sklearn이 있는 백엔드의
`POST /api/models/crime/refresh`에서 별도 프로세스로 실행합니다.
조회 GET도 `X-Model-Admin-Key` 인증이 필요합니다. `.env`의 `MODEL_TRAINING_API_KEY`를
백엔드/Airflow에 동일하게 전달하며 기존 compose 설정을 사용합니다.

## 자동 수집 범위와 남은 자료

| 원본 | 현재 방식 | 자동 반영 |
| --- | --- | --- |
| [5대 범죄](https://data.seoul.go.kr/bsp/wgs/dataView/data300View/481.do) | 서울 데이터 허브 CSV 전체 다운로드, 무인증 | 연도·25구·5유형 검증 후 반영 |
| [1세별 주민등록인구](https://jumin.mois.go.kr/ageStatYear.do) | 행안부 공개 CSV 다운로드 폼, 무인증, 완료연도 12월 | 기존 최종연도부터 직전 완료연도까지 받아 중간 연도 누락 방지. 서울 구·동 합계 검증 |
| [인구이동](https://data.seoul.go.kr/bsp/wgs/dataView/data300View/64.do) | 서울 CSV 전체 다운로드, 무인증 | 헤더행 제거·연도형식 통일 후 반영 |
| [사업체·종사자 밀도](https://stat.eseoul.go.kr/statHtml/statHtml.do?orgId=201&tblId=DT_201004_C010014) | 서울 통계표 CSV, 무인증 | 2014~2024년 25구 확인, 사업체수·밀도·종사자밀도·평균종사자 |
| [학생변동](https://stat.eseoul.go.kr/statHtml/statHtml.do?orgId=201&tblId=DT_201004_G010004) | 서울 통계표 CSV, 무인증 | 2012~2024년 25구 확인, 학교급별 전출입·도내외 |
| [가구형태별 가구 및 가구원](https://stat.eseoul.go.kr/statHtml/statHtml.do?orgId=201&tblId=DT_201004_B010006) | 서울 통계표 CSV, 무인증 | 2015~2025년 25구. 기존 KOSIS 원본과 모델 입력 5항목 수치 일치 확인 |
| [출생·사망](https://stat.eseoul.go.kr/statHtml/statHtml.do?orgId=201&tblId=DT_201004_A050018) | 서울 통계표 월별 CSV, 무인증 | 2014~2025년 25구. 12개월이 모두 존재할 때만 연간 합산 |
| MDIS 전국사업체조사 | 기존 로그인 추출본 사용 | 신규 연도는 직접 추출 필요; 무인 다운로드 미구현 |
| [유흥주점](https://www.data.go.kr/data/15154890/openapi.do) | 공공데이터포털 `/1741000/entertainment_bars/info` | 현재 키로 서울 25개 인허가기관 4,993건 다운로드 확인 |
| [단란주점](https://www.data.go.kr/data/15154883/openapi.do) | 공공데이터포털 `/1741000/singing_bars/info` | 수집 코드 구현. 현재 키는 HTTP 403/서비스 이용권한 미등록이므로 활용신청 확인 필요; 기존 CSV 유지 |

2026-11-17 실제 다운로드 기준입니다. 유흥주점의 서비스 루트만 호출하면 코드 12가 반환되지만,
공식 상세기능 `/info`는 기존 `DATA_GO_KR_API_KEY`로 정상 동작합니다. 이 때문에 서비스 자체가 폐기되었다고 판단하면 안 됩니다.
단란주점은 별도 활용신청 승인이 필요합니다. 키는 `.env`에만 두고 출력/보고서/로그에 기록하지 않습니다.
서울 기관 코드 조건 `cond[OPN_ATMY_GRP_CD::EQ]`, 페이지당 100건으로 전체 페이지를 검증합니다.
기본 안전 제한은 종류별 250개 페이지 요청이며, 같은 날 `python -m scripts.child_crime.collect_refresh --resume`으로
실패한 수집을 이어갈 수 있습니다. 재시도 요청도 제공기관의 트래픽에 포함됩니다.
인허가일·폐업일로 연말 영업 여부를 재구성하며, 현재 주소를 과거 주소로 사용하는 한계와 폐업일 불명 제외 건수를 기록합니다.

즉, **파이프라인 실행 자동화와 모든 원본의 무인 갱신은 다릅니다.** 현재 8종의 자동 수집과
전처리/재학습/조건부 배포를 연결했습니다. 단란주점은 이용권한 확인, MDIS는 직접 추출이 남았습니다.
초기 조사 때 받은 `households.csv`(데이터 허브 9번)는 가구원 통계이므로 사용하지 않습니다.
실제 가구구조는 `household_structure.csv`를 사용합니다.

## 실행

Windows/Ubuntu 공통, 프로젝트 루트에서 ML 의존성이 설치된 Python으로 실행:

```sh
python -m scripts.child_crime.run_refresh --stage all
python -m scripts.child_crime.run_refresh --stage process --no-promote
python -m scripts.child_crime.run_refresh --stage process --force
```

`--no-promote`: 전체 검증·학습을 하되 서비스 포인터를 바꾸지 않습니다.
`--force`: 동일 원본이라도 재검증합니다. 데이터/품질 검증을 우회하지 않습니다.
완료연도 자료가 필요하며 월별 인구를 연말 인구 대신 자동 사용하지 않습니다.
원본을 수동 갱신할 때는 기존 동일한 파일명/열 구조를 유지해야 합니다.
MDIS 신규 연도 파일은 기존 `*_연간자료_*.csv` 명명 규칙/파일설계서 구조를 따릅니다.
관리자 POST API는 수집이 아닌 **전처리 이후 단계**를 실행합니다. Airflow는 먼저 수집하고
`{"collection_run":"수집실행ID"}`를 POST하여 해당 스냅샷을 명시적으로 지정합니다.
본문 없는 POST/CLI process는 최신 수집을 사용합니다.

서버는 변경 코드를 pull한 뒤 백엔드/Airflow를 재생성하고 Airflow에서 DAG를 활성화합니다.
단순 restart는 Python 의존성을 설치하지 않습니다. 실제 기존 컨테이너에서 `openpyxl` 누락을 확인했으므로
반드시 현재 `requirements-server.txt`로 `--build`하세요. 작업 시작 전 필수 의존성 검사도 수행합니다.
모델·raw·processed는 git 대상이 아니므로 서버에도 원본 파일과 현재 모델이 있어야 합니다.
Airflow UID와 백엔드 사용자 모두 공유 raw 경로에 읽기/쓰기 권한이 필요합니다.
주점 외 공개 수집기에는 추가 API 키가 필요하지 않습니다. DAG가 기본적으로 일시정지 상태일 수 있으므로 UI에서 확인하세요.
로컬 검증 시 Airflow 컨테이너는 실행 중이 아니었습니다. 아래 예시는 운영 compose 사용 시 명령입니다.
서버에서 실제 사용하는 `--env-file`, `-f compose.server.yaml` 등을 동일하게 추가해야 합니다.

```sh
docker compose up -d --build backend airflow
docker compose exec airflow airflow dags list-import-errors
docker compose exec airflow airflow dags unpause crime_refresh_pipeline
docker compose exec airflow airflow dags trigger crime_refresh_pipeline
```

Airflow 실행 결과와 `artifacts/crime_refresh/last_run.json`의 `process.state`를 확인합니다.
`promoted`는 서비스 전환, `skipped`는 입력 동일, `awaiting_companion_models`는 타 모델의 연도 불일치입니다.
마지막 상태를 새로운 모델의 배포 성공으로 해석하지 않습니다.

## 저장/복구

- 다운로드: `data/raw/child_crime_exposure/refresh/<run_id>/`, 최신 수집 포인터 `latest.json`.
- 전처리/학습입력/검증: `data/processed/child_crime_exposure/refresh/<run_id>/`.
- 최종 후보: `models/recommender/v1/child_crime_exposure/releases/<run_id>/`.
- 서비스 전환: 같은 폴더의 `current_release.json` 하나를 원자적으로 교체. 2·3년 모델을 함께 전환.
- 전환 전에는 기존 `2y/`, `3y/`를 그대로 사용. 이전 배포는 삭제하지 않음.
- 관리자 기록: `artifacts/crime_refresh/refresh.log`, `last_run.json`.
- MDIS 대용량 파일은 가능하면 읽기용 하드링크로 작업공간에 연결하고, 전환 전 원본 해시를 재검증.
  지원하지 않는 파일시스템에서는 복사하므로 디스크 여유 공간이 필요합니다. 실행 이력 자동 삭제는 하지 않습니다.

수집·전처리 실패/결측/검증 미통과 시 기존 서비스 포인터는 유지됩니다.
프로세스 간 OS 잠금으로 중복 갱신을 방지합니다. 백엔드 재시작으로 작업 추적이 끊기면 DAG는 실패 처리합니다.
로그에는 실행 시작/종료 UTC 시각을 기록합니다.

## 검증/모델 정책

- 마지막 완료 라벨연도 기준 3개 target year, 각 25구. 학습 라벨은 해당 예측 origin year 이하만 사용.
- 같은 기간 Persistence와 Spearman·순위 MAE·5분위 일치율·상위20% 적중률을 비교해 저장.
- 배포 기준: Spearman ≥ max(0.8, Persistence−0.01), 순위 MAE ≤ min(3, Persistence+0.5), 5분위 ≥50%.
- 현재 최종 모델의 Feature/기법/하이퍼파라미터를 고정하고 재학습. 전체 조합 재탐색은 하지 않음.
- 2년 가중치: 재학습 Ridge 절대 표준화 계수 정규화. 3년 가중치: **기존 확정 통합 영향력 공식 유지**.
  3년 통합 영향력을 신규 데이터로 재추정하는 로직은 이번 자동화에 포함하지 않았고 metadata에 명시.
- 기존 metadata 성능은 과거 실험 수치. 신규 실행 성능은 `validation.json`을 확인해야 함.
- 최근 3개년 검증은 운영 회귀검증입니다. 기존 모델 선택에 쓰였던 연도와 겹치므로 새 독립 테스트 성능이라고 주장하지 않습니다.
- 새 연도 범죄·인구가 맞아도 다른 Feature의 공표지연 입력이 부족하면 배포하지 않음.
- 교통/아파트와 예측연도가 맞지 않으면 범죄만 새 연도로 교체해 통합 서비스를 깨뜨리지 않음.

### 기존 원자료 결측 주의

2014~2025년 범죄 원본의 일부 소수 범죄 발생/검거 셀은 빈 값입니다. 기존 모델은 groupby sum에서 이를
0 기여로 집계했습니다. 이번에는 이를 `legacy_missing_counts.json`에 별도 기록하고 기존 재현에만 허용합니다.
이는 공식적으로 확인된 실제 0건이라는 뜻이 아니며, 신규 연도 결측은 차단합니다.
기존 연도 수정 공표값이 달라지면 자동 덮어쓰기하지 않고 검토를 요구합니다.

### 로컬 검증 기록 (2026-11-17)

고정 설정 재학습의 2023~2025년 목표연도 검증: 2년 모델 Spearman 0.9713, 순위 MAE 1.23,
5분위 일치율 81.33%; 3년 모델 Spearman 0.9756, 순위 MAE 1.12, 5분위 일치율 84.00%.
두 모델 모두 운영 품질 기준을 통과했습니다. 이는 새 Feature 탐색에 따른 개선 결과가 아닙니다.

관련 단위·통합 테스트 78개 통과. 로컬 백엔드 재시작 후 관리자 GET과 2·3년 지역 조회 모두 HTTP 200 확인.
Windows에서 서비스 포인터 전환까지 확인했고, Linux Docker에서는 누락된 `openpyxl` 설치 후 같은 격리 입력으로
전처리·4개 origin 재학습·최종 산출물 검증을 재개해 통과했습니다. Linux 검증은 서비스를 교체하지 않았으며
결과는 `artifacts/crime_refresh/linux_validation.json`에 있습니다. 최초 의존성 실패 기록은 보존합니다.
Airflow 이미지/컨테이너는 로컬에 없어 실제 DAG 예약 실행은 미검증입니다.
