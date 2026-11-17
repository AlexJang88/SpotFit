# 범죄노출 모델 데이터 정리

기준일: 2026-11-17. 현재 서비스 모델은 2025년 입력으로 2027년(2년), 2028년(3년)을 예측한다.

## 폴더 구조

```text
data/
  raw/
    .gitkeep
    child_crime_exposure/          # 실제 사용 원본 38개, 하위 폴더 없이 한 곳에 보관
  processed/
    .gitkeep
    child_crime_exposure/
      borough_panel/              # 구·연도 패널, MDIS 집계 캐시, 그룹·출처·커버리지
      population_features/        # 연령 코호트·학생 전입/전출 및 공표시차 처리
      training/
        2y.csv                    # 225행, 기준연도 2015~2023 → 목표 2017~2025
        3y.csv                    # 200행, 기준연도 2015~2022 → 목표 2018~2025
      inference/
        latest_features.csv       # 2025년 서울 25개 자치구, 라벨 없는 예측 입력
      dataset_manifest.json
  interim/
    .gitkeep                      # 현재 범죄 파이프라인은 interim 미사용
```

모든 raw 경로는 `data/raw/child_crime_exposure/`를 기준으로 한다.

## 보존한 원본 38개

| 자료 | 파일 패턴 또는 이름 | 개수 | 사용 |
|---|---|---:|---|
| 5대범죄 | 서울시_5대범죄_2014_2025_원자료.csv | 1 | Y, 범죄건수, 유형별 이력 |
| 연령별 주민등록인구 | *연령별인구현황_연간.csv | 16 | 2010~2025, 0~17세 인구·인구구조·코호트 |
| 가구구조 | 101_DT_1JC1501_20261106110150.xls | 1 | 3년 모델·점수 |
| 사업체·종사자 밀도 | 사업체 및 종사자 밀도.xls | 1 | 3년 모델·점수 |
| 인구이동 | 서울시 인구이동(2015년 이후) 통계.csv | 1 | 3년 모델·점수 |
| 출생·사망 | 출생사망인구수_*.xls | 3 | 2014~2025, 3년 모델·점수 |
| 학생 전입·전출 | 학생변동상황(전출·전입)_20261110103410.xlsx | 1 | 3년 앙상블의 직접 예측 부분 |
| 유흥·단란주점 | 서울시 *주점영업 인허가 정보.csv | 2 | 개폐업일 기준 과거 영업 수·3년 모델·점수 |
| MDIS 사업체 | 2014~2024 *_연간자료_*.csv | 11 | 산업·조직형태 비율 |
| MDIS 코드북 | 파일설계서*.xlsx | 1 | 자치구 코드 매핑 |
| 합계 | | **38** | |

모델 목표값: `log(1 + 5대범죄 발생건수 / 0~17세 아동인구 × 10,000)`.
실제 아동 피해 건수가 아닌 지역 범죄노출 대리지표다.

2년 모델은 범죄·인구·Y의 현재값과 과거값·변화량을 사용한다. 3년 모델과 서비스 점수에 필요한 보조 원본을 합쳐 보존했다. 원본 값은 수정하지 않았으며 이동 전후 SHA-256을 확인했다.

## 삭제·이동 범위

- raw/interim/processed의 범죄 최종 모델에 필요 없는 파일 2,042개(12,673,287,418바이트)를 영구 삭제했다.
- 다른 분야의 원본·전처리, 기존 중간 산출물, 중복 원본, MDIS 2010~2013, 미채택 스트레스·학급당 학생수·종사상지위 자료 등을 삭제했다.
- 기존 `artifacts/child_crime_borough_v2`의 전처리 자료는 `processed/child_crime_exposure/borough_panel`로 이동했다. 실험 보고서·성능 비교 기록은 artifacts에 보존했다.
- 기존 `artifacts/child_population_priority_v1` 전처리 자료는 `processed/child_crime_exposure/population_features`로 이동했다.
- models의 모델·메타데이터·서비스 배포용 예측 CSV는 유지했다. 배포용 CSV는 전처리 입력이 아닌 서비스 산출물이다.
- 삭제한 데이터의 전체 백업은 만들지 않았다. 파일 목록·크기·보존 원본 해시와 검증 결과는 `artifacts/data_cleanup_20261117/`에 있다.
- 다른 모델의 원본이 삭제되어 해당 모델을 처음부터 재생성하려면 원본을 다시 확보해야 한다. 이 정리는 로컬 파일에 적용되며, git에서 무시하는 데이터가 서버에서 자동으로 이동·삭제되는 것은 아니다.

## 전처리 재생성

```bash
python scripts/child_crime/prepare_service_data.py
```

Windows/Ubuntu에서 같은 Python 스크립트를 사용한다. 남긴 raw 38개로 패널·학습용·예측용 데이터를 생성한다. MDIS 집계 캐시가 없으면 2014~2024 원본을 다시 집계한다.

## 선택된 모델의 재학습·복구

```bash
python scripts/child_crime/rebuild_service_crime_models.py
```

모델이나 배포용 예측이 없으면 전처리를 준비한 후 최종 선택 조합으로 재학습한다. 재점수화 파일만 없으면 배포 모델을 이용해 점수를 재생성한다. 고정 설정은 `ml/config/child_crime_exposure_locked.json`에 보존했다.

```bash
python scripts/child_crime/train_locked_service_models.py
# 별도 폴더로 재현을 확인할 경우
python scripts/child_crime/train_locked_service_models.py --output-root artifacts/check_crime_models
```

위 복구는 현재 승인된 2025 입력 버전에 대한 재현이다. 새로운 기준연도에 배포하려면 새 데이터·성능 검증과 고정 설정 갱신이 필요하다. 기존 CV 수치를 새 모델의 성능으로 재사용하지 않는다.

## 검증 결과

- 기존 패널과 재생성 패널의 공통 입력값 일치: 400행.
- 기존 population feature와 재생성 공통 입력값 일치: 400행.
- 배포 모델로 새 입력을 사용했을 때 2년·3년 예측값 차이 0.
- 원본에서 전처리 후 고정 설정으로 새로 학습한 2년·3년 모델도 기존 예측값 차이 0.
- 두 모델의 25개 구 점수·등급·순위 유지, 가중치 합계 100%.
