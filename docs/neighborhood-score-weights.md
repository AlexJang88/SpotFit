# 사용자별 동네점수 반영 비율

상단 **비율** 아이콘(접근성 이름: 점수 반영 비율 설정)에서 범죄·교통 비율을 변경한다. 기존 색상과 접이식 설정 패널을 사용한다.

- 기본값: 범죄 50% / 교통 50%.
- 슬라이더: 1% 단위, 0~100%, 두 비율의 합은 항상 100%.
- 빠른 선택: 교통 중심 30:70 / 균형 50:50 / 범죄 중심 70:30.
- 기본값 초기화 제공. 브라우저 localStorage(`spotfit.crime-weight-percent.v1`)에 저장하며 로그인·DB 변경은 없다. 저장 불가 환경에서는 현재 방문 동안만 유지한다.
- 종합점수 = 범죄 안전점수 × 범죄 비율 + 교통 안전점수 × 교통 비율.
- 각 모델의 Feature 가중치와 예측값은 변경하지 않으며 재학습하지 않는다.
- 자치구 점수·순위·A~E 등급·지도 색상·추천·상세 점수를 같은 비율로 갱신한다. 25개 구를 정렬해 5개씩 등급을 부여한다. 기존 동점 규칙(범죄 안전점수, 자치구명)은 유지한다.
- 추천은 기존대로 예산 범위 내 구별 최저가 행정동 1곳을 고른 뒤 변경된 구 순위로 최대 5곳을 선정한다. 행정동은 소속 구 점수를 사용한다.
- 요청은 debounce/AbortController로 처리하고 변경 직후 이전 결과는 지운다. API 오류 시 안내와 기존 재시도 기능을 사용한다.

## API

다음 GET API에 선택 쿼리 `crime_weight`, `traffic_weight`를 추가했다. 백분율이 아닌 **0~1** 값이다.

- `/api/neighborhoods/catalog`
- `/api/neighborhoods/recommendations`
- `/api/neighborhoods/detail`

예: `/api/neighborhoods/catalog?horizon=2&crime_weight=0.7&traffic_weight=0.3`

둘 다 생략하면 0.5/0.5, 하나만 전달하면 나머지는 `1 - 전달값`이다. 둘 다 전달할 때 합이 1이 아니거나 범위를 벗어나거나 NaN/Infinity이면 422를 반환한다. 응답 `weights`는 실제 적용 비율이다. 요청별로 계산하므로 다른 사용자에게 영향을 주지 않는다.

## 검증과 배포

- API: `python -m pytest tests/integration/test_neighborhood_recommendations.py tests/unit/test_neighborhood_year_gate.py -q` (backend 작업 디렉터리).
- 프론트엔드: `npm run build` (frontend 작업 디렉터리).
- 브라우저: 개발 서버 8080 실행 후 `python scripts/verify_score_weights_browser.py` (Playwright 및 Microsoft Edge 필요).
- 배포 시 백엔드와 프론트엔드 코드를 함께 반영한다. DB 마이그레이션, 모델 재생성, 추가 환경변수는 필요 없다.
