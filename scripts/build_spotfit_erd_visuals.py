"""Build standalone SVG/HTML ERD visuals for the SpotFit database design."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture" / "erd_visuals"
FONT = "Pretendard, Noto Sans KR, Malgun Gothic, Arial, sans-serif"


def t(x, y, value, size=16, fill="#16383B", weight=400, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def rounded(x, y, w, h, fill, stroke="#D4E3DF", radius=18, width=1.5, css=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" class="{css}"/>'
    )


def path(points, color="#6E9F97", width=3, dashed=False, marker=True):
    commands = [f"M {points[0][0]} {points[0][1]}"]
    for x, y in points[1:]:
        commands.append(f"L {x} {y}")
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    end = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash}{end}/>'
    )


def defs():
    return f"""
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#002D35" flood-opacity="0.12"/>
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#6E9F97"/>
  </marker>
  <style>
    .card {{ filter: url(#shadow); }}
  </style>
</defs>
"""


def base(width, height, title, subtitle, body):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{defs()}
<rect width="100%" height="100%" fill="#F5F8F7"/>
<rect x="0" y="0" width="100%" height="18" fill="#00D084"/>
{t(54, 68, title, 34, '#002D35', 800)}
{t(54, 102, subtitle, 17, '#577572', 400)}
<line x1="54" y1="124" x2="{width - 54}" y2="124" stroke="#D5E3DF" stroke-width="2"/>
{body}
<text x="{width - 54}" y="{height - 24}" font-family="{FONT}" font-size="13" fill="#78908D" text-anchor="end">SpotFit · 2026-11-13 · 비회원 서비스 기준</text>
</svg>
"""


SCHEMA = {
    "META": ("#E7F4F0", "#007F73"),
    "REFERENCE": ("#E9F2F5", "#266A78"),
    "CORE": ("#EEF6ED", "#3A7B4F"),
    "FEATURE": ("#FFF5DD", "#9B6A00"),
    "ML": ("#F0ECF8", "#675095"),
    "SERVICE": ("#E7F8F0", "#00885C"),
    "ANALYTICS": ("#F8EFE8", "#9A5B2F"),
    "OPS": ("#F2F1F0", "#5D6563"),
}


def schema_box(x, y, w, h, schema, description, tables, columns=2):
    fill, accent = SCHEMA[schema]
    out = [rounded(x, y, w, h, "#FFFFFF", stroke="#CFE0DB", radius=22, width=1.5, css="card")]
    out.append(rounded(x, y, w, 68, fill, stroke=fill, radius=22, width=0))
    out.append(t(x + 24, y + 30, schema, 19, accent, 800))
    out.append(t(x + 24, y + 53, description, 13, "#577572", 400))
    gap, top = 12, y + 86
    chip_w = (w - 48 - gap * (columns - 1)) / columns
    chip_h = 56
    for index, (name, label) in enumerate(tables):
        row, col = divmod(index, columns)
        cx = x + 24 + col * (chip_w + gap)
        cy = top + row * (chip_h + 11)
        out.append(rounded(cx, cy, chip_w, chip_h, "#FAFCFB", stroke="#D8E6E2", radius=11, width=1))
        out.append(t(cx + 13, cy + 23, name, 12, "#143C3D", 700))
        out.append(t(cx + 13, cy + 43, label, 11, "#6A8581", 400))
    return "\n".join(out)


def build_overview():
    width, height = 2400, 1800
    body = []
    body.append(schema_box(54, 154, 450, 494, "META", "Airflow 실행·데이터 버전·품질", [
        ("DATA_SOURCE", "원천·갱신주기"), ("INGESTION_RUN", "수집 실행이력"),
        ("DATA_ASSET_VERSION", "MinIO 파일 버전"), ("PIPELINE_RUN", "전처리·학습 실행"),
        ("PIPELINE_INPUT", "입력 데이터 계보"), ("DATA_QUALITY_RESULT", "자동 품질검사"),
    ]))
    body.append(schema_box(54, 694, 450, 494, "REFERENCE", "행정구역·경계 변경 이력", [
        ("ADMIN_AREA", "시·구·동 유효기간"), ("ADMIN_AREA_ALIAS", "이전·대체 명칭"),
        ("BOUNDARY_NOTICE", "변경 공고"), ("NOTICE_AREA", "분동·통합 연결"),
    ]))
    body.append(schema_box(544, 154, 510, 1034, "CORE", "정제·표준화된 공통 데이터", [
        ("APARTMENT", "아파트·위치"), ("PROPERTY_TRANSACTION", "매매·전세·전월세"),
        ("POPULATION_STAT", "연령별 인구"), ("CRIME_STAT", "5대범죄 발생"),
        ("TRAFFIC_ACCIDENT_STAT", "교통사고·사상자"), ("FACILITY_SNAPSHOT", "시설 연도별 현황"),
    ], columns=1))
    body.append(schema_box(1094, 154, 510, 474, "FEATURE", "모델 학습 Snapshot", [
        ("FEATURE_DATASET_VERSION", "도메인·Feature 버전"),
        ("PROPERTY_FEATURES", "부동산 공통 Feature"),
        ("CRIME_FEATURES", "범죄노출 Feature"),
        ("TRAFFIC_FEATURES", "교통안전 Feature"),
    ], columns=1))
    body.append(schema_box(1094, 674, 510, 514, "ML", "학습·평가·배포 버전", [
        ("TRAINING_RUN", "학습 실행"), ("MODEL_VERSION", "5개 모델 버전"),
        ("MODEL_METRIC", "성능지표"), ("MODEL_FEATURE", "Feature·가중치"),
        ("MODEL_DEPLOYMENT", "Champion·배포이력"),
    ], columns=1))
    body.append(schema_box(1644, 154, 702, 1034, "SERVICE", "API가 직접 조회하는 서비스 데이터", [
        ("PROPERTY_PREDICTION", "매매·전세·전월세 예측"), ("SAFETY_PREDICTION", "범죄·교통 예측"),
        ("SCORING_POLICY", "범죄·교통 반영비중"), ("GRADE_POLICY", "A~E 기준·색상"),
        ("NEIGHBORHOOD_SCORE", "통합 동네점수"), ("RECOMMENDATION_POLICY", "추천 규칙"),
        ("RECOMMENDATION_REQUEST", "비회원 조건 Snapshot"), ("RECOMMENDATION_RESULT", "추천지역 5개"),
    ]))
    body.append(schema_box(544, 1240, 1060, 410, "ANALYTICS", "개인식별정보 없는 서비스 이용 통계", [
        ("SERVICE_EVENT", "검색·추천·상세보기"), ("RECOMMENDATION_SUMMARY", "예산·면적 구간 통계"),
    ]))
    body.append(schema_box(1644, 1240, 702, 410, "OPS", "관리자 변경·장애 추적", [
        ("AUDIT_LOG", "정책·모델 변경이력"), ("SERVICE_INCIDENT", "파이프라인·API 장애"),
    ]))

    # Schema-level relations: clear enough to scan from left to right.
    body += [
        path([(504, 400), (524, 400), (524, 340), (544, 340)]),
        t(524, 324, "수집·정제", 12, "#6A8581", 700, "middle"),
        path([(504, 930), (524, 930), (524, 760), (544, 760)]),
        t(524, 744, "지역 FK", 12, "#6A8581", 700, "middle"),
        path([(1054, 420), (1074, 420), (1074, 388), (1094, 388)]),
        t(1074, 372, "Feature", 12, "#6A8581", 700, "middle"),
        path([(1349, 628), (1349, 674)]),
        t(1364, 655, "학습", 12, "#6A8581", 700),
        path([(1604, 900), (1624, 900), (1624, 750), (1644, 750)]),
        t(1624, 734, "배포", 12, "#6A8581", 700, "middle"),
        path([(1995, 1188), (1995, 1240)]),
        t(2010, 1219, "감사", 12, "#6A8581", 700),
        path([(1644, 1050), (1624, 1050), (1624, 1445), (1604, 1445)]),
        t(1618, 1428, "익명 이벤트", 12, "#6A8581", 700, "end"),
    ]
    svg = base(width, height, "SpotFit PostgreSQL ERD · 전체 구조", "로그인 없이 데이터·모델·예측·추천·운영을 연결하는 논리 구조", "\n".join(body))
    (OUT / "spotfit_erd_full.svg").write_text(svg, encoding="utf-8")


@dataclass
class Table:
    key: str
    schema: str
    title: str
    subtitle: str
    x: int
    y: int
    w: int
    rows: list[tuple[str, str, str]]

    @property
    def h(self):
        return 70 + 31 * len(self.rows) + 12

    def anchor(self, side: str):
        if side == "left":
            return self.x, self.y + self.h / 2
        if side == "right":
            return self.x + self.w, self.y + self.h / 2
        if side == "top":
            return self.x + self.w / 2, self.y
        return self.x + self.w / 2, self.y + self.h

    def draw(self):
        fill, accent = SCHEMA[self.schema]
        out = [rounded(self.x, self.y, self.w, self.h, "#FFFFFF", stroke="#C9DDD7", radius=16, width=1.4, css="card")]
        out.append(rounded(self.x, self.y, self.w, 58, fill, stroke=fill, radius=16, width=0))
        out.append(t(self.x + 18, self.y + 25, self.title, 15, accent, 800))
        out.append(t(self.x + 18, self.y + 46, self.subtitle, 11, "#647F7B", 400))
        for index, (key, name, dtype) in enumerate(self.rows):
            yy = self.y + 80 + index * 31
            if index:
                out.append(f'<line x1="{self.x + 14}" y1="{yy - 20}" x2="{self.x + self.w - 14}" y2="{yy - 20}" stroke="#EDF2F0"/>')
            if key:
                badge_fill = "#DDF4EA" if "PK" in key else "#EAF1F3"
                badge_text = "#007A55" if "PK" in key else "#356B75"
                out.append(rounded(self.x + 16, yy - 17, 38, 20, badge_fill, stroke=badge_fill, radius=6, width=0))
                out.append(t(self.x + 35, yy - 3, key, 10, badge_text, 800, "middle"))
                name_x = self.x + 64
            else:
                name_x = self.x + 18
            out.append(t(name_x, yy, name, 12, "#193D3F", 600))
            out.append(t(self.x + self.w - 16, yy, dtype, 11, "#78908D", 400, "end"))
        return "\n".join(out)


def relation(parent: Table, child: Table, parent_side="right", child_side="left", label="1 : N", offset=0):
    x1, y1 = parent.anchor(parent_side)
    x2, y2 = child.anchor(child_side)
    if parent_side in ("left", "right"):
        mid = (x1 + x2) / 2 + offset
        pts = [(x1, y1), (mid, y1), (mid, y2), (x2, y2)]
        lx, ly = mid + 7, (y1 + y2) / 2
    else:
        mid = (y1 + y2) / 2 + offset
        pts = [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]
        lx, ly = (x1 + x2) / 2, mid - 8
    return path(pts, width=2.4) + t(lx, ly, label, 11, "#5F7D78", 700, "middle")


def build_service_detail():
    width, height = 2500, 1900
    tables = {
        "area": Table("area", "REFERENCE", "REF.ADMIN_AREA", "자치구·행정동 버전", 55, 160, 410, [
            ("PK", "area_id", "bigint"), ("", "area_code", "varchar"), ("FK", "parent_area_id", "bigint"),
            ("", "valid_from / valid_to", "date"), ("", "is_current", "boolean"), ("", "boundary", "geometry"),
        ]),
        "apt": Table("apt", "CORE", "CORE.APARTMENT", "아파트 기본정보", 55, 545, 410, [
            ("PK", "apartment_id", "bigint"), ("FK", "area_id", "bigint"), ("", "apartment_code", "varchar"),
            ("", "apartment_name", "varchar"), ("", "built_year", "integer"), ("", "location", "geometry"),
        ]),
        "feature": Table("feature", "FEATURE", "FEATURE.DATASET_VERSION", "학습 Snapshot·계보", 55, 930, 410, [
            ("PK", "feature_dataset_id", "bigint"), ("FK", "pipeline_run_id", "bigint"), ("", "domain", "varchar"),
            ("", "feature_version", "varchar"), ("", "reference_date", "date"), ("", "storage_uri", "varchar"),
            ("", "schema_hash", "varchar"), ("", "quality_passed", "boolean"),
        ]),
        "train": Table("train", "ML", "ML.TRAINING_RUN", "모델 학습 실행", 540, 160, 420, [
            ("PK", "training_run_id", "bigint"), ("FK", "feature_dataset_id", "bigint"), ("FK", "pipeline_run_id", "bigint"),
            ("", "model_type", "varchar"), ("", "mlflow_run_id", "varchar"), ("", "algorithm", "varchar"), ("", "status", "varchar"),
        ]),
        "model": Table("model", "ML", "ML.MODEL_VERSION", "5개 최종모델 버전", 540, 555, 420, [
            ("PK", "model_version_id", "bigint"), ("FK", "training_run_id", "bigint"), ("", "model_type", "varchar"),
            ("", "version", "varchar"), ("", "prediction_horizon", "integer"), ("", "artifact_uri", "varchar"),
            ("", "status", "varchar"), ("", "is_champion", "boolean"),
        ]),
        "metric": Table("metric", "ML", "ML.MODEL_METRIC", "성능 평가", 540, 1010, 420, [
            ("PK", "metric_id", "bigint"), ("FK", "model_version_id", "bigint"), ("", "dataset_split", "varchar"),
            ("", "metric_name", "varchar"), ("", "metric_value", "numeric"),
        ]),
        "model_feature": Table("model_feature", "ML", "ML.MODEL_FEATURE", "Feature·가중치", 540, 1350, 420, [
            ("PK", "model_feature_id", "bigint"), ("FK", "model_version_id", "bigint"), ("", "feature_name", "varchar"),
            ("", "importance", "numeric"), ("", "weight_pct", "numeric"), ("", "direction", "varchar"),
        ]),
        "property": Table("property", "SERVICE", "SERVICE.PROPERTY_PREDICTION", "매매·전세·전월세", 1040, 160, 470, [
            ("PK", "property_prediction_id", "bigint"), ("FK", "model_version_id", "bigint"), ("FK", "apartment_id", "bigint"),
            ("FK", "area_id", "bigint"), ("", "transaction_type", "varchar"), ("", "prediction_date", "date"),
            ("", "exclusive_area_sqm", "numeric"), ("", "predicted_amount", "numeric"),
        ]),
        "safety": Table("safety", "SERVICE", "SERVICE.SAFETY_PREDICTION", "범죄노출·교통안전", 1040, 650, 470, [
            ("PK", "safety_prediction_id", "bigint"), ("FK", "model_version_id", "bigint"), ("FK", "area_id", "bigint"),
            ("", "safety_domain", "varchar"), ("", "reference_year", "integer"), ("", "prediction_year", "integer"),
            ("", "predicted_value", "numeric"), ("", "safety_score / rank / grade", "mixed"),
        ]),
        "score_policy": Table("score_policy", "SERVICE", "SERVICE.SCORING_POLICY", "범죄·교통 반영비중", 1040, 1140, 470, [
            ("PK", "scoring_policy_id", "bigint"), ("", "policy_name", "varchar"), ("", "crime_weight", "numeric"),
            ("", "traffic_weight", "numeric"), ("", "valid_from / valid_to", "date"), ("", "is_active", "boolean"),
        ]),
        "grade": Table("grade", "SERVICE", "SERVICE.GRADE_POLICY", "A~E 기준·색상", 1040, 1510, 470, [
            ("PK", "grade_policy_id", "bigint"), ("FK", "scoring_policy_id", "bigint"), ("", "grade", "varchar"),
            ("", "minimum_rank / maximum_rank", "integer"), ("", "color", "varchar"),
        ]),
        "neighborhood": Table("neighborhood", "SERVICE", "SERVICE.NEIGHBORHOOD_SCORE", "최종 동네점수", 1590, 160, 480, [
            ("PK", "neighborhood_score_id", "bigint"), ("FK", "area_id", "bigint"), ("FK", "scoring_policy_id", "bigint"),
            ("FK", "crime_prediction_id", "bigint"), ("FK", "traffic_prediction_id", "bigint"), ("", "prediction_year", "integer"),
            ("", "crime_score / traffic_score", "numeric"), ("", "total_score / rank / grade", "mixed"),
        ]),
        "rec_policy": Table("rec_policy", "SERVICE", "SERVICE.RECOMMENDATION_POLICY", "추천지역 5개 규칙", 1590, 650, 480, [
            ("PK", "recommendation_policy_id", "bigint"), ("", "recommendation_count", "integer"),
            ("", "one_dong_per_district", "boolean"), ("", "exclude_missing_price", "boolean"), ("", "sort_rule", "varchar"),
        ]),
        "request": Table("request", "SERVICE", "SERVICE.RECOMMENDATION_REQUEST", "비회원 조건 Snapshot", 1590, 990, 480, [
            ("PK", "request_id", "uuid"), ("FK", "recommendation_policy_id", "bigint"), ("", "transaction_type", "varchar"),
            ("", "minimum / maximum_budget", "numeric"), ("", "preferred_area_sqm", "numeric"),
            ("", "crime_prediction_year", "integer"), ("", "requested_at / expires_at", "timestamptz"),
        ]),
        "result": Table("result", "SERVICE", "SERVICE.RECOMMENDATION_RESULT", "추천 결과 Snapshot", 1590, 1405, 480, [
            ("PK", "request_id + rank", "composite"), ("FK", "area_id", "bigint"), ("FK", "property_prediction_id", "bigint"),
            ("FK", "neighborhood_score_id", "bigint"), ("", "predicted_amount", "numeric"),
            ("", "score / grade", "mixed"), ("", "reason_snapshot", "jsonb"),
        ]),
        "event": Table("event", "ANALYTICS", "ANALYTICS.SERVICE_EVENT", "개인식별 없는 이용통계", 2150, 990, 300, [
            ("PK", "event_id", "bigint"), ("FK", "request_id", "uuid"), ("FK", "area_id", "bigint"),
            ("", "event_name", "varchar"), ("", "budget_band", "varchar"), ("", "occurred_at", "timestamptz"),
        ]),
    }

    body = []
    relations = [
        ("feature", "train", "right", "left", "1 : N", 0),
        ("train", "model", "bottom", "top", "1 : N", 0),
        ("model", "metric", "bottom", "top", "1 : N", -55),
        ("model", "model_feature", "bottom", "top", "1 : N", 90),
        ("model", "property", "right", "left", "1 : N", -20),
        ("model", "safety", "right", "left", "1 : N", 45),
        ("area", "apt", "bottom", "top", "1 : N", 0),
        ("apt", "property", "right", "left", "1 : N", -40),
        ("area", "safety", "right", "left", "1 : N", -250),
        ("safety", "neighborhood", "right", "left", "2 : N", 0),
        ("score_policy", "grade", "bottom", "top", "1 : N", 0),
        ("score_policy", "neighborhood", "right", "left", "1 : N", -40),
        ("rec_policy", "request", "bottom", "top", "1 : N", 0),
        ("request", "result", "bottom", "top", "1 : N", 0),
        ("property", "result", "right", "left", "1 : N", 280),
        ("neighborhood", "result", "bottom", "top", "1 : N", 125),
        ("request", "event", "right", "left", "0..1 : N", 0),
    ]
    for a, b, sa, sb, label, offset in relations:
        body.append(relation(tables[a], tables[b], sa, sb, label, offset))
    for table in tables.values():
        body.append(table.draw())

    body.append(rounded(2115, 160, 335, 680, "#FFFFFF", stroke="#CFE0DB", radius=18, width=1.5, css="card"))
    body.append(t(2140, 202, "모델 5종", 20, "#002D35", 800))
    models = [
        ("CRIME_EXPOSURE", "아동 범죄노출"), ("TRAFFIC_ACCIDENT", "어린이 교통안전"),
        ("APARTMENT_SALE", "아파트 매매"), ("APARTMENT_JEONSE", "아파트 전세"),
        ("APARTMENT_MONTHLY_RENT", "아파트 전월세"),
    ]
    for i, (code, label) in enumerate(models):
        yy = 245 + i * 92
        body.append(rounded(2140, yy, 285, 70, "#E7F8F0", stroke="#C7EBDD", radius=12, width=1))
        body.append(t(2156, yy + 28, code, 12, "#007F5B", 800))
        body.append(t(2156, yy + 51, label, 13, "#315957", 500))
    body.append(t(2140, 750, "원본·모델 Binary", 14, "#002D35", 800))
    body.append(t(2140, 777, "PostgreSQL이 아닌 MinIO 저장", 12, "#6A8581", 500))
    body.append(t(2140, 815, "로그인·회원 테이블 없음", 13, "#B25B3B", 800))

    svg = base(width, height, "SpotFit ERD · 모델·서비스 상세", "PK·FK와 모델 예측 → 동네점수 → 비회원 추천 결과의 관계", "\n".join(body))
    (OUT / "spotfit_erd_service_detail.svg").write_text(svg, encoding="utf-8")


def build_html():
    html = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpotFit ERD</title><style>
body{margin:0;background:#eef4f2;font-family:Pretendard,'Noto Sans KR','Malgun Gothic',sans-serif;color:#16383b}
main{max-width:1500px;margin:0 auto;padding:32px}.head{display:flex;justify-content:space-between;align-items:end;margin-bottom:20px}
h1{margin:0;font-size:34px}p{color:#607d79}.panel{background:white;border:1px solid #d6e4e0;border-radius:18px;padding:16px;margin:20px 0;box-shadow:0 8px 24px #002d3514}
.panel h2{margin:6px 8px 14px}.frame{overflow:auto;border-radius:12px;background:#f5f8f7}.frame img{display:block;width:100%;min-width:1100px;height:auto}
.note{font-size:14px;margin:10px 8px}.badge{display:inline-block;background:#00d084;color:#002d35;border-radius:999px;padding:7px 12px;font-weight:800}
</style></head><body><main><div class="head"><div><h1>SpotFit PostgreSQL ERD</h1><p>비회원 서비스 · Airflow 자동화 · 5개 최종모델 기준</p></div><span class="badge">2026-11-13</span></div>
<section class="panel"><h2>1. 전체 구조</h2><div class="frame"><img src="spotfit_erd_full.svg" alt="SpotFit 전체 ERD"></div><p class="note">데이터 수집부터 서비스·운영까지 Schema 단위 흐름입니다.</p></section>
<section class="panel"><h2>2. 모델·서비스 상세</h2><div class="frame"><img src="spotfit_erd_service_detail.svg" alt="SpotFit 모델 서비스 상세 ERD"></div><p class="note">확대하면 각 테이블의 PK·FK와 주요 컬럼을 확인할 수 있습니다.</p></section>
</main></body></html>"""
    (OUT / "spotfit_erd_visualization.html").write_text(html, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_overview()
    build_service_detail()
    build_html()
    for path_ in sorted(OUT.glob("spotfit_erd_*")):
        print(path_)


if __name__ == "__main__":
    main()
