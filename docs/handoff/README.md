# SpotFit — 다른 PC / 새 Codex 세션 인수인계

기준: **2026-11-10, Asia/Seoul**. 이 문서는 완료 여부가 검증된 내용과 사용자 희망 사항을 구분한다.
절대경로를 그대로 복사하지 말고 새 PC의 저장소 루트 기준으로 해석한다.

## 새 Codex에 전달할 시작 프롬프트

```text
이 저장소의 docs/handoff/README.md, PROJECT_STATE.md, MODEL_RESULTS.json,
TRANSFER_MANIFEST.json을 읽고 이전 작업을 이어받아줘.
Git 브랜치와 변경 파일, 로컬 모델·데이터 유무를 먼저 확인해줘.
문서의 완료/실험/미구현 구분과 모델 성능 비교 조건을 유지해줘.
기존 변경을 덮어쓰거나 자동으로 학습·커밋·푸시하지 말고,
인수인계 상태를 간단히 요약한 뒤 내가 요청하는 다음 작업을 진행해줘.
```

## 읽는 순서

1. 이 파일: 이동·복원 방법, 안전한 시작 절차.
2. `PROJECT_STATE.md`: 프로젝트 맥락, 현재 모델/API/UI, 주의사항, 최근 발표 설명.
3. `MODEL_RESULTS.json`: 저장된 평가 결과에서 추출한 수치. 다른 타깃/기간끼리 개선율을 계산하지 않는다.
4. `TRANSFER_MANIFEST.json`: 원본 PC의 경로·파일크기·선별 체크섬·Git 상태·도구 버전.
5. 실제 코드와 모델 메타데이터: 문서와 불일치하면 원인을 확인하고 최신 증거를 우선한다.

## 인수인계 ZIP의 의미

`artifacts/handoff/SpotFit_Codex_handoff_20261110.zip`은 **대화 맥락과 최근 작업물 이동용**이다.
전체 실행환경 백업이나 모든 모델·학습 데이터 백업은 아니다.

ZIP 구성:

- `workspace/docs/handoff/`: 인수인계 문서와 결과 요약, manifest.
- `workspace/scripts/`: 최근 PPT 생성·렌더링·확인 스크립트와 인수인계 패키징 스크립트.
- `workspace/frontend/...`: 미커밋 UI 수정 파일 2개 스냅샷.
- `workspace/artifacts/presentations/`: 완성 프로젝트 PPT 15장, 범죄 개선 PPT 5장 및 일부 시각 검증 자료.
- `workspace/artifacts/...`, `workspace/models/...`: 허용 목록에 지정한 소형 보고서·메타데이터·범죄 예측 CSV,
  자치구 집계 panel과 priority Feature CSV(개별 원자료가 아닌 학습용 집계표).
- `external/SPOT_FIT_발표자료.pptx`: 사용자가 준 원본 발표 틀.
- `git_tracked_changes.patch`: 당시 추적 파일의 미커밋 diff. untracked 파일은 위 스냅샷으로 별도 포함.

**원자료, joblib 모델, .env/비밀키, 브라우저 프로필, Docker 볼륨, node_modules는 포함하지 않는다.**
모델 metadata/CSV가 들어 있어도 joblib이 없으면 현재 범죄 API의 준비 검사는 실패한다.
원본 PC에도 현재 `data/raw/아동범죄`와 `data/raw/아동인프라` 폴더는 없었다.
옛 경로만 보고 전처리 재현이 가능하다고 가정하지 말고 백업/실제 이동 위치를 확인한다.

## 다른 PC로 옮기는 절차

1. Git 저장소를 clone/fetch하고 필요한 브랜치·커밋을 확인한다.
   - 기록 시 브랜치: `feature/frontend_us`
   - 기록 시 HEAD: `983cfa0` (`feature/crime_model_us` 병합 커밋)
   - 커밋/푸시는 이 인수인계 작업에서 수행하지 않았다.
   - Git 원격에 이 HEAD와 최근 파일이 있는지는 새 PC에서 확인한다.
2. ZIP을 저장소 밖의 별도 임시 폴더에 먼저 푼다.
3. `workspace/` 내용을 저장소와 비교한 다음 필요한 파일만 복사/병합한다.
   - 이미 있는 코드·데이터·PPT를 일괄 덮어쓰지 않는다.
   - `HomePage.tsx`는 추적 파일의 수정본, `readability-refresh.css`는 새 파일이다. 둘을 함께 반영해야 한다.
   - 패치와 같은 파일의 스냅샷을 중복 적용하지 않는다. 패치는 먼저 `git apply --check`로 검사한다.
4. 아래 표의 대용량 산출물을 별도로 이동한다. `data/`, `models/`, `artifacts/`는 Git ignore 대상이다.
5. `.env.example`를 참고해 새 PC용 `.env`를 별도로 만든다. 실제 키는 안전한 채널로 전달한다.
6. 필요한 기능만 읽기 전용으로 확인한다. 누락 파일이 있다고 즉시 전체 재학습하지 않는다.

| 이어갈 작업 | 별도 이동 대상 |
|---|---|
| PPT 편집만 | ZIP의 PPT·스크립트·보고서, 원본 템플릿, 네모고딕 글꼴 설치 |
| 범죄 API 실행 | `models/recommender/v1/child_crime_exposure/` 전체(joblib 포함) |
| 모든 도메인 서비스 | `models/recommender/v1/` 전체 + 각 서비스가 읽는 입력 자료·예측 CSV |
| 현재 1년 모델 재학습 | `artifacts/child_crime_borough_v2/panel.csv` (이 ZIP에 집계표 포함) |
| 현재 2년 모델 재탐색 | 위 panel + `feature_groups.json`, `artifacts/child_population_priority_v1/priority_features.csv`, 해당 `feature_groups.json` (이 ZIP에 포함) |
| 2년 산출물 재배포 | `models/experiments/child_crime_exposure_2y_2027_v1/`와 `artifacts/child_crime_exposure_2y_2027_v1/` 전체 |
| 전처리부터 재현 | 관련 `data/raw/`, `data/interim/`, `data/processed/`, 원천·실험 산출물 전체 |
| 과거 모델 개선 실험 재현 | PROJECT_STATE에 명시된 각 실험 `artifacts/`, `models/experiments/`와 그 입력 자료 |
| 기존 DB / MinIO까지 복구 | 별도의 DB 백업·MinIO 객체 및 볼륨 백업 필요. 이 ZIP으로 복구되지 않음 |

MinIO에 모델이나 데이터가 모두 올라가 있다고 확인된 것은 아니다. `scripts/minio_data_sync.py`는
설정된 데이터 버킷을 `data/`로 내려받는 도구이며, 모델·실험 전체를 자동 복원하는 도구가 아니다.
기본적으로 원격에 없는 로컬 파일을 삭제하지 않는다. 실행 전 `--help`와 대상 버킷을 확인한다.

## 환경 준비 참고

- `requirements-dev.txt`는 `-r requirements-server.txt`를 이미 포함한다. 개발 설치에는 둘을 따로 설치할 필요가 없다.
- Ubuntu의 externally-managed-environment 오류는 venv를 만들어 설치한다. `--break-system-packages`로 우회하지 않는다.
- 설치 구문은 `python -m pip install -r requirements-dev.txt`다. `-r`을 빼지 않는다.
- Python 3.14에서 모든 패키지/기존 joblib이 호환된다고 가정하지 않는다. 원본 패키지 버전은 manifest에 기록했다.
- PPT 생성은 `python-pptx`, `pandas`; 이미지 점검은 Pillow; 렌더링은 Windows 데스크톱 PowerPoint COM을 사용했다.
- PPT용 패키지가 현재 requirements에 모두 선언되어 있지는 않다. 독립 PPT 작업용 venv라면
  `python -m pip install python-pptx pandas Pillow`로 준비할 수 있다.
- 프론트엔드 package.json은 Node `>=22.12.0`을 요구한다.
- 모델 파일은 신뢰 가능한 원본에서만 가져온다. joblib/pickle 로드는 코드를 실행할 수 있다.

Ubuntu 예시(서버에 Python/venv 패키지가 준비되어 있다는 전제):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

## 안전한 재개 확인

```text
git status --short
git branch --show-current
git log -1 --oneline
docker compose ps
```

서비스 실행 후 읽기 전용 확인:

```text
GET /api/safety/crime/model-status
GET /api/safety/crime/district-forecasts?horizon=1
GET /api/safety/crime/district-forecasts?horizon=2
GET /api/safety/crime/category-scores?horizon=1
```

`python scripts/verify_spotfit_deck_evidence.py`는 `http://127.0.0.1:8080`의 API를 읽고
로컬 점검 JSON을 작성한다. 먼저 `artifacts/presentations/template_review/` 폴더가 있어야 한다.
호스트/포트가 달라지면 스크립트를 조정한다. `docker compose exec -T frontend npm run typecheck`도 확인 가능하다.

## PPT 이어서 만들 때

- 기준 PPT: `artifacts/presentations/SPOT_FIT_프로젝트발표_템플릿유지_20261110.pptx`
- 범죄 모델 추가 5장: `artifacts/presentations/SPOT_FIT_범죄모델_개선과정_5장_20261110.pptx`
- 원본: ZIP의 `external/SPOT_FIT_발표자료.pptx`
- 글꼴: `네모고딕`, `네모고딕 Medium`, `네모고딕 Light` — 폰트 파일은 ZIP에 없음.
- 원본 슬라이드 비율/크기: 20 × 11.25 inch(16:9).
- 기존 생성 스크립트의 `SOURCE`에는 원본 PC의 Desktop 절대경로가 있다. 새 PC에서 실제 원본 경로로 수정한 뒤 실행한다.
- 범죄 5장 생성 스크립트는 완성 15장 PPT와 `build_spotfit_project_deck.py`의 helper를 재사용한다.
- 덮어쓰기 우려가 있으면 스크립트의 `DEST`를 새 파일명으로 바꾼다. 기존 수동 편집본을 덮어쓰지 않는다.
- 파일 내 발표자 노트에 출처와 해석 주의사항을 넣었다. 슬라이드 본문만 읽고 모델 상태를 추정하지 않는다.

## 이후 갱신

`python scripts/package_spotfit_handoff.py`는 manifest/모델 수치 요약/허용 목록 ZIP을 새로 작성한다.
현재 날짜의 버전 파일을 덮어쓸 수 있으므로 외부에서 편집한 인수인계본은 다른 이름으로 보관한다.
문서의 해석/완료 상태는 자동 갱신되지 않으므로 실제 코드 변경 시 PROJECT_STATE도 함께 갱신한다.
