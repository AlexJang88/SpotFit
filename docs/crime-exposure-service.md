# 아동 지역범죄 노출 서비스

## 서비스 모델

- Y: `log(1 + 서울 자치구 5대범죄 발생건수 / 0~17세 아동인구 × 10,000)`
- 1년: 2025년 입력 → 2026년 예측
- 2년: 2025년 입력 → 2027년 예측
- 위험순위 1위는 노출지수가 가장 높은 자치구다.
- 사용자 안전점수는 노출위험이 낮은 순서로 5개 구씩 100, 80, 60, 40, 20점이며 등급은 A~E다.

## API

- `GET /api/safety/crime/district-forecasts?horizon=1`
- `GET /api/safety/crime/district-forecasts?horizon=2`
- `GET /api/safety/crime/model-status`
- `POST /api/safety/crime/models/train`

재학습은 운영 자원을 많이 사용하므로 `MODEL_TRAINING_API_KEY`를 서버 `.env`에 설정하고 `X-Model-Admin-Key` 헤더로 전달해야 한다. 키가 없으면 API는 의도적으로 503을 반환한다.

```bash
curl -X POST "http://127.0.0.1:8080/api/safety/crime/models/train" \
  -H "X-Model-Admin-Key: 서버에 설정한 키"
```

모델이 이미 있으면 재학습하지 않고 상태만 반환한다. 강제 재학습은 `?force=true`를 사용한다. 실행 상태는 `model-status`에서 확인하며 로그는 `artifacts/crime_service_training.log`에 저장된다.

## 재생성 순서

1. 1년 모델 또는 예측 파일이 없으면 `ml/training/train_service_crime_exposure_1y.py` 실행
2. 2년 실험 모델이 없으면 `ml/training/optimize_exposure_2y_2027.py` 실행
3. `scripts/child_crime/export_service_crime_forecasts.py`로 `recommender/v1` 서비스 산출물 생성
4. 백엔드 예측 캐시 초기화

API는 동시에 한 건만 학습하도록 잠금 처리되어 있다. 서버 시작 요청과 학습 요청은 분리되어 있으므로 모델 누락 상태에서도 API 서버 자체는 정상 기동한다.
