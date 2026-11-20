"""Build a Korean data dictionary from the versioned SpotFit DDL snapshot."""
from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/erd/physical_schema/schema_snapshot_ddl.json"
OUTPUT = ROOT / "docs/SpotFit_DB_전체_테이블_정의서.xlsx"

TABLES = {
    "schema_migration": ("DB 스키마 변경 이력", "적용한 DB 설치·변경 SQL 버전과 적용 시각을 기록"),
    "ml_model_type": ("모델 유형 기준", "범죄·교통·매매·전세·월세 등 모델의 종류와 목표값 정의"),
    "meta_data_source": ("원본 데이터 출처", "데이터 제공 기관·원천 URL·갱신주기를 관리"),
    "meta_ingestion_run": ("원본 수집 실행 이력", "출처별 수집 시도, 대상 기간, 성공·실패와 오류 기록"),
    "meta_data_asset_version": ("원본 파일 버전", "수집된 원본 파일의 위치·해시·크기·행 수 기록"),
    "meta_pipeline_run": ("데이터 파이프라인 실행", "전처리·학습 파이프라인 실행 상태와 코드 버전 기록"),
    "meta_pipeline_input": ("파이프라인 입력 연결", "실행에 사용한 원본 파일 버전과 입력 역할 연결"),
    "meta_data_quality_result": ("데이터 품질검사 결과", "검사 항목별 통과 여부와 상세 결과 기록"),
    "ref_admin_area": ("행정구역 기준", "시·구·동 코드, 상위 지역, 유효기간 및 경계 관리"),
    "ref_admin_area_alias": ("행정구역 별칭", "검색용 행정구역 대체 명칭과 적용 기간 관리"),
    "ref_boundary_notice": ("행정구역 변경 공고", "분동·합동·명칭 변경 등의 공고와 시행일 관리"),
    "ref_boundary_notice_area": ("변경 공고 대상 지역", "행정구역 변경 공고와 영향받는 지역의 연결"),
    "core_apartment": ("아파트 기준 정보", "아파트 코드·주소·준공연도·좌표 관리"),
    "core_property_transaction": ("부동산 실거래", "매매·전세·월세 거래 금액과 면적·계약일 관리"),
    "core_population_stat": ("인구 통계", "지역·연도·연령대별 인구수 관리"),
    "core_crime_stat": ("범죄 통계", "지역·연도·범죄 유형별 발생 건수 관리"),
    "core_traffic_accident_stat": ("교통사고 통계", "지역·연도·사고 유형별 사고·사상자 수 관리"),
    "core_facility_snapshot": ("시설 현황", "지역·시점·시설 유형별 개수 관리"),
    "feature_dataset_version": ("학습 데이터셋 버전", "모델 입력 Feature 데이터셋의 버전·저장 위치·품질 기록"),
    "ml_training_run": ("모델 학습 실행", "사용 데이터·알고리즘·학습 상태·MLflow 실행 기록"),
    "ml_model_version": ("모델 버전", "학습 결과물의 버전·예측기간·파일 해시·검증 상태 기록"),
    "ml_model_metric": ("모델 성능지표", "모델 버전별 학습·검증·테스트 성능값 기록"),
    "ml_model_feature": ("모델 입력변수·영향력", "모델에 사용한 Feature의 중요도·가중치·위험 방향 기록"),
    "ml_model_deployment": ("모델 배포 이력", "개발·스테이징·운영 환경별 모델 배포 상태와 시각 기록"),
    "service_property_prediction": ("아파트 가격 예측", "지역·아파트·거래 유형별 예측 금액과 면적 관리"),
    "service_safety_prediction": ("범죄·교통안전 예측", "지역별 범죄노출·교통사고 예측값과 점수·등급 관리"),
    "service_scoring_policy": ("동네점수 산정 정책", "범죄·교통안전 반영 비율과 정책 유효기간 관리"),
    "service_grade_policy": ("동네점수 등급 정책", "A~E 등급의 순위 범위·표시 색상 관리"),
    "service_neighborhood_score": ("동네 종합점수", "지역의 범죄·교통 점수를 합산한 총점·순위·등급 관리"),
    "service_recommendation_policy": ("동네 추천 정책", "추천 개수·지역 중복·가격 누락 처리·정렬 규칙 관리"),
    "service_recommendation_request": ("동네 추천 요청", "사용자의 거래 유형·예산·면적·예측연도 요청 기록"),
    "service_recommendation_result": ("동네 추천 결과", "추천 순위별 지역·가격 예측·동네점수와 추천 이유 기록"),
    "analytics_service_event": ("서비스 사용 이벤트", "검색·추천 요청·상세보기 등 익명 사용 행동 기록"),
    "ops_audit_log": ("운영 감사 로그", "시스템·관리자의 주요 작업과 변경 전후 값 기록"),
    "ops_service_incident": ("서비스 장애 이력", "장애 유형·심각도·상태·해결 시각 기록"),
}

ID_NAMES = {
    "alias": "행정구역 별칭", "apartment": "아파트", "area": "행정구역", "asset_version": "원본 파일 버전",
    "audit": "감사 기록", "crime_prediction": "범죄 예측", "crime_stat": "범죄 통계",
    "deployment": "모델 배포", "event": "서비스 이벤트", "facility_snapshot": "시설 현황",
    "feature_dataset": "학습 데이터셋", "grade_policy": "등급 정책", "incident": "서비스 장애",
    "ingestion_run": "원본 수집 실행", "metric": "성능지표", "model_feature": "모델 입력변수",
    "model_version": "모델 버전", "neighborhood_score": "동네 종합점수", "notice": "행정구역 변경 공고",
    "parent_area": "상위 행정구역", "pipeline_run": "파이프라인 실행", "population_stat": "인구 통계",
    "property_prediction": "부동산 가격 예측", "quality_result": "품질검사 결과",
    "recommendation_policy": "추천 정책", "request": "추천 요청", "safety_prediction": "안전 예측",
    "scoring_policy": "점수 산정 정책", "source": "데이터 출처", "traffic_prediction": "교통 예측",
    "traffic_stat": "교통사고 통계", "training_run": "모델 학습 실행", "transaction": "부동산 거래",
}

COLUMNS = {
    "accident_count": "해당 지역·연도·유형의 교통사고 건수", "accident_type": "교통사고 유형",
    "action": "감사 대상 작업 내용", "actor_type": "작업 주체 구분(시스템·관리자·파이프라인)",
    "after_value": "변경 후 값(JSON)", "age_group": "집계 대상 연령대", "airflow_dag_id": "실행한 Airflow DAG 식별자",
    "airflow_run_id": "Airflow DAG 실행 식별자", "algorithm": "학습에 사용한 알고리즘명",
    "alias_name": "행정구역 검색용 별칭", "announced_at": "변경 공고 발표일",
    "apartment_code": "아파트 고유 코드", "apartment_name": "아파트 이름",
    "applied_at": "DB 변경 SQL 적용 시각", "area_band": "면적 구간", "area_code": "공식 행정구역 코드",
    "area_level": "행정구역 단계(시·구·동)", "area_name": "행정구역명",
    "artifact_uri": "학습된 모델 파일의 저장 위치", "audit_id": "감사 로그 식별자",
    "before_value": "변경 전 값(JSON)", "boundary": "행정구역 공간 경계(다중 폴리곤)",
    "budget_band": "예산 구간", "built_year": "아파트 준공연도",
    "casualty_count": "교통사고 사상자 수", "check_name": "데이터 품질검사 항목명",
    "checked_at": "품질검사 시각", "checksum_sha256": "원본 파일 무결성 검사용 SHA-256 해시",
    "code_commit": "실행 당시 코드 커밋 식별자", "code_system": "행정구역 코드 체계",
    "collected_at": "원본 파일 수집 시각", "color": "등급 표시 색상 코드",
    "contract_date": "부동산 거래 계약일", "created_at": "레코드 생성 시각",
    "crime_prediction_year": "범죄노출 예측 대상 연도", "crime_score": "범죄안전 영역 점수",
    "crime_type": "범죄 유형", "crime_weight": "동네점수 중 범죄 영역 반영 비율",
    "dataset_split": "성능 측정 데이터 구간(학습·CV·검증 등)", "deployed_at": "모델 배포 시각",
    "deposit_amount": "전세·월세 보증금(만원)", "description": "설명 문구",
    "details": "품질검사 상세 결과(JSON)", "direction": "Feature 값 증가 시 위험 방향",
    "display_name": "사용자 표시용 모델명", "display_order": "표시 순서",
    "domain": "모델 또는 Feature의 업무 영역", "effective_date": "행정구역 변경 시행일",
    "environment": "모델 배포 환경(개발·스테이징·운영)", "error_summary": "실패 원인 요약",
    "event_name": "사용자 행동 이벤트명", "event_properties": "이벤트 상세 속성(JSON)",
    "exclude_missing_price": "가격 정보가 없는 지역 추천 제외 여부",
    "exclusive_area_sqm": "아파트 전용면적(㎡)", "expires_at": "추천 요청 기록 만료 시각",
    "facility_count": "해당 지역·유형의 시설 개수", "facility_type": "시설 분류",
    "feature_name": "모델 입력변수명", "feature_version": "Feature 가공 버전",
    "file_size": "원본 파일 크기(바이트)", "fill_opacity": "지도 등급 색상 채우기 투명도",
    "finished_at": "실행 종료 시각", "grade": "등급(A~E)",
    "importance": "모델 입력변수 중요도", "incident_type": "서비스 장애 유형",
    "input_role": "파이프라인에서 원본 파일이 맡는 입력 역할",
    "is_active": "현재 사용 중인 정책·출처 여부", "is_champion": "현재 채택된 대표 모델 여부",
    "is_current": "현재 유효한 지역·파일 버전 여부", "location": "아파트 위치 좌표",
    "maximum_budget": "사용자가 정한 최대 예산(만원)", "maximum_rank": "등급에 해당하는 최대 순위",
    "metric_name": "성능지표 이름", "metric_value": "성능지표 측정값",
    "minimum_budget": "사용자가 정한 최소 예산(만원)", "minimum_rank": "등급에 해당하는 최소 순위",
    "mlflow_run_id": "MLflow 학습 실행 식별자", "model_hash": "모델 파일 무결성 검사용 SHA-256 해시",
    "model_type": "모델 종류 코드", "monthly_rent": "월세 금액(만원)",
    "neighborhood_grade": "추천 시점의 동네 등급", "neighborhood_score": "추천 시점의 동네점수",
    "notice_type": "행정구역 변경 유형(분리·통합·개명 등)", "object_uri": "원본 파일 저장 경로 또는 URI",
    "occurred_at": "이벤트·장애 발생 시각", "occurrence_count": "범죄 발생 건수",
    "one_dong_per_district": "자치구당 행정동 하나만 추천할지 여부",
    "passed": "품질검사 통과 여부", "pipeline_type": "파이프라인 종류",
    "policy_code": "정책 고유 코드", "policy_name": "정책 표시명",
    "population": "해당 지역·연도·연령대의 인구수", "predicted_amount": "예측한 거래 금액(만원)",
    "predicted_price_per_sqm": "예측한 ㎡당 가격(만원)", "predicted_value": "안전 모델의 원 예측값",
    "prediction_date": "가격 예측 대상일", "prediction_horizon": "기준시점에서 예측시점까지의 연수",
    "prediction_kind": "예측·추정·참고값 구분", "prediction_year": "안전·동네점수 예측 대상 연도",
    "preferred_area_sqm": "사용자 희망 전용면적(㎡)", "provider": "데이터 제공 기관",
    "quality_passed": "데이터셋 품질검사 통과 여부", "reason_snapshot": "추천 사유의 당시 값(JSON)",
    "recommendation_count": "한 번에 제공할 추천 지역 수", "recommendation_rank": "추천 목록 순위",
    "reference_date": "자료·예측의 기준일", "reference_period": "원본 수집 대상 기간",
    "reference_year": "통계·예측의 기준연도", "relation_type": "공고와 지역의 관계(기존·신규 등)",
    "requested_at": "추천 요청 시각", "resolved_at": "서비스 장애 해결 시각",
    "retired_at": "모델 배포 종료 시각", "road_address": "아파트 도로명주소",
    "row_count": "파일·데이터셋 행 수", "safety_domain": "안전 영역(범죄노출·교통사고)",
    "safety_grade": "안전 영역 예측 등급", "safety_rank": "안전 영역 내 자치구 순위",
    "safety_score": "안전 영역 예측점수", "sale_amount": "매매 거래 금액(만원)",
    "schema_hash": "Feature 데이터셋 컬럼 구조 해시", "score_batch_key": "같은 점수 산출 묶음 식별자",
    "severity": "품질오류·장애의 심각도", "sort_rule": "추천 결과 정렬 규칙",
    "source_code": "원본 데이터 출처 코드", "source_name": "원본 데이터 출처명",
    "source_snapshot": "행정구역 기준자료 스냅샷 식별자",
    "source_transaction_key": "원천 거래자료의 중복 제거용 고유키",
    "source_url": "원본 데이터 제공 URL", "started_at": "실행 시작 시각",
    "status": "실행·모델·장애 등의 처리 상태", "storage_uri": "가공된 학습 데이터셋 저장 경로",
    "summary": "장애 또는 작업 내용 요약", "target_description": "모델이 예측하는 목표값 설명",
    "target_id": "감사 대상 레코드 식별자", "target_type": "감사 대상 개체 종류",
    "title": "공고 또는 장애 제목", "total_grade": "동네 종합등급(A~E)",
    "total_rank": "동네 종합점수의 자치구 순위", "total_score": "범죄·교통 점수 가중합 총점",
    "traffic_score": "교통안전 영역 점수", "traffic_weight": "동네점수 중 교통 영역 반영 비율",
    "transaction_type": "부동산 거래 유형(매매·전세·월세)", "update_cycle": "원본 데이터의 갱신주기",
    "valid_from": "지역·정책 유효 시작일", "valid_to": "지역·정책 유효 종료일",
    "validation_passed": "모델 검증 통과 여부", "version": "스키마 또는 모델 버전",
    "weight_pct": "모델 입력변수의 점수 가중치(%)",
}

OVERRIDES = {
    ("service_recommendation_result", "neighborhood_score"): "추천 시점에 확정된 동네 종합점수",
    ("service_recommendation_result", "predicted_amount"): "추천에 사용한 아파트 예측금액(만원)",
    ("service_recommendation_result", "area_id"): "추천된 행정동 식별자",
    ("service_neighborhood_score", "area_id"): "동네점수를 산정한 자치구 식별자",
    ("service_safety_prediction", "area_id"): "범죄·교통안전 예측 대상 자치구 식별자",
    ("service_property_prediction", "area_id"): "가격 예측 대상 지역 식별자",
    ("core_property_transaction", "area_id"): "거래 발생 지역 식별자",
    ("ref_boundary_notice_area", "area_id"): "변경 공고의 영향 지역 식별자",
    ("service_recommendation_request", "request_id"): "추천 요청 고유 식별자(UUID)",
    ("service_recommendation_result", "request_id"): "이 결과가 속한 추천 요청 식별자(UUID)",
    ("service_neighborhood_score", "crime_prediction_id"): "범죄노출 안전 예측 결과 참조키",
    ("service_neighborhood_score", "traffic_prediction_id"): "교통사고 안전 예측 결과 참조키",
    ("service_neighborhood_score", "total_grade"): "점수 정책에 따라 확정된 동네 종합등급",
    ("service_grade_policy", "grade"): "점수 정책에 속하는 등급(A~E)",
    ("ref_boundary_notice_area", "relation_type"): "공고 대상 지역의 관계(이전·신규·영향 등)",
}


def describe(table: str, column: str) -> str:
    if (table, column) in OVERRIDES:
        return OVERRIDES[(table, column)]
    if column in COLUMNS:
        return COLUMNS[column]
    if column.endswith("_id"):
        stem = column[:-3]
        if stem in ID_NAMES:
            return ID_NAMES[stem] + " 식별자"
    raise KeyError(f"컬럼 설명 누락: {table}.{column}")


def main():
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    tables = snapshot["tables"]
    assert len(tables) == len(TABLES), "DDL 테이블 수가 변경되어 정의서 검토가 필요합니다"
    wb = Workbook()
    summary = wb.active
    summary.title = "테이블 목록"
    detail = wb.create_sheet("컬럼 정의")
    summary.append(["스키마", "테이블명", "한글명", "테이블 역할", "컬럼 수", "PK", "FK 수"])
    detail.append(["스키마", "테이블명", "테이블 한글명", "컬럼 순서", "컬럼명", "한글 역할", "자료형", "키", "NULL 허용", "참조 테이블·컬럼", "기본값"])

    for table in tables:
        name = table["name"]
        if name not in TABLES:
            raise KeyError(f"테이블 한글명 누락: {name}")
        korean, role = TABLES[name]
        constraints = table["constraints"]
        pk = {col for c in constraints if c["kind"] == "p" for col in c["columns"]}
        fk = {col: f"{c['target_schema']}.{c['target_table']}({', '.join(c['target_columns'])})"
              for c in constraints if c["kind"] == "f" for col in c["columns"]}
        summary.append([table["schema"], name, korean, role, len(table["columns"]),
                        ", ".join(sorted(pk)), sum(c["kind"] == "f" for c in constraints)])
        for col in table["columns"]:
            column = col["name"]
            keys = "/".join(key for key, yes in (("PK", column in pk), ("FK", column in fk)) if yes)
            detail.append([table["schema"], name, korean, col["position"], column,
                           describe(name, column), col["type"], keys, "예" if col["nullable"] else "아니요",
                           fk.get(column, ""), col.get("default") or ""])

    for sheet, widths in ((summary, [14, 43, 27, 75, 12, 30, 10]),
                          (detail, [14, 43, 27, 12, 39, 70, 28, 10, 13, 50, 40])):
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for idx, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(idx)].width = width
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="073C38")
            cell.font = Font(name="맑은 고딕", color="FFFFFF", bold=True)
            cell.alignment = Alignment(vertical="center")
        sheet.row_dimensions[1].height = 28
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="맑은 고딕", size=10)
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                if cell.row % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="F1F7F4")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(f"{OUTPUT} | tables={len(tables)} columns={detail.max_row-1}")


if __name__ == "__main__":
    main()
