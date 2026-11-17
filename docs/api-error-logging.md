# API 예외 처리와 관리자 로그

## 적용 범위

- 모든 API 요청에는 `X-Request-ID`가 응답 헤더로 반환된다.
- 입력 검증 오류는 422, 없는 리소스는 404, 관리자 인증 실패는 403, 중복 실행은 409로 반환된다.
- 모델·데이터 장애는 503, 처리하지 못한 오류는 500으로 반환된다.
- 5xx 응답에는 파일 경로나 스택 트레이스를 노출하지 않는다. 전체 원인은 관리자 로그에만 저장한다.
- 오류 본문은 기존 `detail` 필드를 유지하면서 `error.code`, `error.status`, `error.request_id`, `error.timestamp`를 제공한다.
- 프론트엔드는 `detail`과 `request_id`를 공통으로 읽어 사용자 안내문과 문의번호를 표시한다. 서버에 연결할 수 없으면 네트워크 확인 안내를 표시한다.

## 로그 파일

Docker Compose에서는 Windows와 Ubuntu 모두 프로젝트의 `logs/backend/spotfit-api.log`에 저장된다. 개발 PC용 `compose.dev.yaml`은 동시에 실행되는 로컬 스택과 충돌하지 않도록 `spotfit-api-dev.log`를 사용한다. 컨테이너 내부 디렉터리는 `/var/log/spotfit`이며, 컨테이너를 재생성해도 호스트 파일은 유지된다.

로그는 한 줄에 JSON 한 건으로 저장되며 다음 내용을 포함한다.

- ISO 8601 시각과 UTC 오프셋(기본 `Asia/Seoul`)
- 심각도, 로거 이름, 요청 ID
- HTTP 메서드, 경로, 상태 코드, 처리 시간, 클라이언트 IP
- 5xx 예외의 전체 스택 트레이스
- 서버 시작·종료, 모델 자동 복구·재학습 성공 또는 실패

자정마다 파일을 교체하고 기본 30일간 보관한다. 설정값은 다음과 같다.

```ini
APP_LOG_LEVEL=INFO
APP_LOG_RETENTION_DAYS=30
APP_TIMEZONE=Asia/Seoul
MODEL_TRAINING_API_KEY=충분히_긴_관리자_비밀키
```

Docker를 사용하지 않고 Windows 또는 Ubuntu에서 직접 실행하면 `APP_LOG_DIR`을 지정할 수 있다. 지정하지 않으면 실행 디렉터리 아래 `logs/backend`를 사용한다.

## 관리자 로그 조회 API

`MODEL_TRAINING_API_KEY`가 설정된 서버에서만 사용할 수 있다.

```bash
curl -H "X-Model-Admin-Key: 실제_관리자키" \
  "http://서버주소:8080/api/admin/logs?level=ERROR&limit=200&date=2026-11-12"
```

- `level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` 중 하나이며 해당 단계 이상을 반환한다.
- `limit`: 1~1000, 기본 200
- `date`: 선택값, `YYYY-MM-DD`
- 결과는 최신순이다.

키가 없으면 503으로 비활성화되고, 틀린 키는 403을 반환한다. 키를 URL 파라미터나 Git에 저장하지 않는다.

Ubuntu에서 파일을 직접 확인할 수도 있다.

```bash
tail -f logs/backend/spotfit-api.log
docker compose logs -f backend
```

Windows PowerShell에서는 다음과 같이 확인한다.

```powershell
Get-Content .\logs\backend\spotfit-api.log -Wait
```
