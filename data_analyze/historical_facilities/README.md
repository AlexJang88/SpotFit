# 시설 데이터 5개년 확보 작업

분석 기간은 완료된 연도인 2021~2025년이다. 2026년 원본은 현재 상태 확인용으로만 사용한다.

## 디렉터리

- `data/raw/1년치만 있는 데이터`: 수집된 2026년 현재 원본
- `data/raw/5년치_시설/지역아동센터`: 아동권리보장원 2021~2025년 말 공식 주소록
- `data/raw/5년치_시설/관광지점입장객_대체지표`: 한국문화관광연구원 2021~2025년 공식 통계집
- `data/processed/historical_facilities/reconstructed_2021_2025`: 현재 원본의 날짜 필드로 재구성한 연말 스냅샷
- `data/processed/historical_facilities/official_2021_2025`: 실제 연도별 공식 원본을 정규화한 CSV
- `source_manifest.csv`: 항목별 공식 출처, 확보 방식 및 한계
- `official_download_manifest.csv`: 내려받은 공식 원본의 파일별 URL·크기·SHA-256
- `collection_inventory.csv`: 분석 시 우선 사용할 9개 카테고리 경로와 지표·품질 한계

재구성 자료는 실제 과거에 내려받은 원본이 아니다. 각 출력 행의 `quality_flag`와
`reconstruction_basis`를 분석에 반드시 함께 사용한다.

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
python data_analyze/historical_facilities/build_reconstructed_snapshots.py
python data_analyze/historical_facilities/normalize_official_child_centers.py
python data_analyze/historical_facilities/normalize_official_tourism.py
python data_analyze/historical_facilities/verify_collection.py
```

## 해석 주의

- CCTV, 공원, 학교 및 통학구역은 현재 목록에 남아 있는 기록만 과거로 역산한다. 철거·폐지·폐교 기록은 누락될 수 있다.
- 모범음식점은 재지정 이력을 완전한 기간 자료로 제공하지 않으므로 과거 지정 상태가 근사치다.
- 병원 자료는 병원 인허가 자료이며 전체 의원을 포함하지 않는다.
- 통행불편지역은 해당 연도 조사 기록이지 연말 현재 재고가 아니다.
- 관광 명소 목록은 과거를 복원할 날짜 필드가 없다. 대신 주요관광지점 입장객 통계집과 서울 지점별 공식 통계표를 대체 수요지표로 보관한다. 지점 합계는 서울 전체 관광객 수가 아니며 연도별 보고 지점 수도 달라질 수 있다.
- 지역아동센터는 아동권리보장원에서 공개한 실제 연도 말 주소록을 사용하며 현재 스냅샷 역산 결과보다 우선한다.
