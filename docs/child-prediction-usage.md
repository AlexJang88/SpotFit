# 아동 인구 모델 결과 확인·예측

프로젝트 루트에서 실행합니다. 명령은 .env를 읽지 않습니다.
기존 결과·모델을 보존하며 재학습과 원격 업로드는 수행하지 않습니다.

## 1. 이미 평가된 테스트 결과 확인

```powershell
python ml/inference/view_child_predictions.py
python ml/inference/view_child_predictions.py --gu 도봉구 --limit 30
python ml/inference/view_child_predictions.py --dong 쌍문4동
```

actual은 실제 6~11세 인구, predicted는 추정 인구, error는 예측-실제,
absolute_error는 절대오차입니다. 테스트 85개 동만 있으며 검색되지 않는 동도 있습니다.
--csv로 다른 연령군의 테스트 결과를 선택할 수 있습니다.

## 2. 로컬 joblib로 새 예측

```powershell
python ml/inference/predict_child_population.py --backend joblib
```

입력이 없으면 초등학교 2/3개, 약국 10/20개의 가상 예시를 사용합니다.
실제 행정동을 예측하려면 UTF-8 CSV에 초등학교_수, 약국_수 컬럼을 넣습니다.
행정동명·코드 등 추가 열은 결과에 보존됩니다.

```powershell
python ml/inference/predict_child_population.py --input my_facilities.csv
```

기본 elementary v1 외 모델은 --model-file로 변경합니다. 필요한 컬럼은 모델마다
다르며 프로그램이 모델에 저장된 컬럼 순서로 입력을 정렬합니다.
신뢰할 수 있는 자체 모델만 로드하세요. pickle/joblib는 임의 코드를 실행할 수 있습니다.

## 3. MLflow 등록 모델로 새 예측

```powershell
python ml/inference/predict_child_population.py --backend mlflow
```

Ubuntu MLflow에 해당 모델이 등록돼 있고 SSH 터널을 5002 포트로 연결했다면:

```powershell
python ml/inference/predict_child_population.py --backend mlflow --tracking-uri http://127.0.0.1:5002 --model-uri models:/spotfit-child-count-elementary/1 --input my_facilities.csv
```

서버의 모델 버전 번호를 실제 등록 상태에 맞게 지정합니다. 서버 접속은 로컬
모델을 업로드하지 않습니다. 로컬 UI의 5001 포트와 SSH 터널 5002를 구분하세요.

결과는 artifacts/reports/child_predictions에 UTF-8 BOM CSV로 저장합니다.
--output으로 경로를 정할 수 있지만 기존 파일은 덮어쓰지 않습니다.
학습 패널 전체를 입력할 수도 있지만 학습에 쓰인 동의 예측은 테스트 성능이 아닙니다.
elementary v1은 테스트 MAE 약 432명으로 평균값 기준 모델보다 성능이 낮습니다.
환경 점수·미래 인구 예측 또는 실제 정책 판단에 사용하지 마세요.
