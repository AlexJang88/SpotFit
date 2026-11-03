"""아동 만족도 모델 결과를 3장짜리 한글 PPTX로 만든다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "artifacts/child_satisfaction_scores/20261028_142519"
NAVY = RGBColor(22, 42, 67)
BLUE = RGBColor(32, 99, 155)
TEAL = RGBColor(20, 132, 123)
PALE_BLUE = RGBColor(228, 239, 249)
PALE_GREEN = RGBColor(225, 243, 238)
PALE_RED = RGBColor(252, 232, 230)
LIGHT = RGBColor(246, 248, 251)
MID = RGBColor(98, 112, 128)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(188, 58, 52)

FEATURE_KO = {
    "bus_stops_per_1000": "주민 1,000명당 버스정류장 수",
    "아파트_평균_면적": "아파트 평균 면적",
    "apartment_households_per_1000": "주민 1,000명당 아파트 가구 수",
    "아파트_평균_시가": "아파트 평균 시가",
    "high_students_per_class": "고등학교 학급당 학생 수",
    "education_spend_per_household": "가구당 교육비 지출액",
}
ALGO_KO = {
    "dummy_mean": "평균 기준모델",
    "ridge_alpha_1": "릿지(규제 1)",
    "ridge_alpha_10": "릿지(규제 10)",
    "elastic_net": "엘라스틱넷",
    "random_forest": "랜덤 포레스트",
    "extra_trees": "엑스트라 트리",
    "hist_gradient_boosting": "히스토그램 그래디언트 부스팅",
}


def set_cell(cell, text: str, size: int = 10, bold: bool = False,
             color: RGBColor = NAVY, fill: RGBColor | None = None,
             align: PP_ALIGN = PP_ALIGN.CENTER) -> None:
    cell.text = str(text)
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = align
        paragraph.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "맑은 고딕"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color


def title(slide, heading: str, subtitle: str, page: int) -> None:
    box = slide.shapes.add_textbox(Inches(.55), Inches(.28), Inches(11.9), Inches(.72))
    p = box.text_frame.paragraphs[0]
    p.text = heading
    p.font.name = "맑은 고딕"
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = NAVY
    sub = slide.shapes.add_textbox(Inches(.58), Inches(.93), Inches(11.6), Inches(.35))
    sp = sub.text_frame.paragraphs[0]
    sp.text = subtitle
    sp.font.name = "맑은 고딕"
    sp.font.size = Pt(10)
    sp.font.color.rgb = MID
    line = slide.shapes.add_shape(1, Inches(.55), Inches(1.30), Inches(12.2), Inches(.04))
    line.fill.solid(); line.fill.fore_color.rgb = BLUE; line.line.fill.background()
    pg = slide.shapes.add_textbox(Inches(12.35), Inches(.32), Inches(.4), Inches(.3))
    pp = pg.text_frame.paragraphs[0]
    pp.text = f"{page}/3"; pp.font.name = "맑은 고딕"; pp.font.size = Pt(9); pp.font.color.rgb = MID


def text_box(slide, x, y, w, h, heading: str, lines: list[str], accent=BLUE) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = RGBColor(220, 226, 233)
    tf = shape.text_frame; tf.margin_left = Inches(.18); tf.margin_right = Inches(.14)
    tf.margin_top = Inches(.12); tf.clear()
    p = tf.paragraphs[0]; p.text = heading; p.font.name = "맑은 고딕"; p.font.size = Pt(14)
    p.font.bold = True; p.font.color.rgb = accent
    for line in lines:
        p = tf.add_paragraph(); p.text = line; p.font.name = "맑은 고딕"; p.font.size = Pt(10.5)
        p.font.color.rgb = NAVY; p.level = 0; p.space_before = Pt(5); p.space_after = Pt(0)


def add_footer(slide, text: str) -> None:
    box = slide.shapes.add_textbox(Inches(.58), Inches(7.08), Inches(12.1), Inches(.25))
    p = box.text_frame.paragraphs[0]; p.text = text; p.font.name = "맑은 고딕"
    p.font.size = Pt(8); p.font.color.rgb = MID


def slide_data(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "학습 데이터와 문제 정의", "서울 25개 자치구 만족도 예측 · 모든 표시 컬럼명은 한글화", 1)
    text_box(slide, .58, 1.55, 3.85, 2.05, "검증 설계", [
        "개발 데이터: 2021~2023년, 집단별 75건",
        "최종 검증: 2024년 25개 자치구",
        "분할: 자치구 기준 5겹 교차검증",
        "선택 기준: 만족도 평균절대오차 최소",
    ])
    text_box(slide, 4.60, 1.55, 4.02, 2.05, "예측값(Y)", [
        "미취학: 실제 이용 가구의 보육시설 만족도",
        "학령기: 서울서베이 교육환경 만족도",
        "척도: 모두 1~5점 연속값",
        "주의: 학령기 Y는 전체 가구 응답 기반 대리변수",
    ], TEAL)
    text_box(slide, 8.80, 1.55, 3.95, 2.05, "평가지표", [
        "주 지표: 평균절대오차(MAE)",
        "보조: 평균제곱근오차, 결정계수",
        "참고: 실제값 ±0.5점 이내 적중률",
        "2024년은 모델 선택에 사용하지 않음",
    ])

    table = slide.shapes.add_table(3, 3, Inches(.58), Inches(3.88), Inches(12.17), Inches(2.75)).table
    table.columns[0].width = Inches(1.55); table.columns[1].width = Inches(5.0); table.columns[2].width = Inches(5.62)
    headers = ["구분", "미취학 후보 컬럼", "학령기 후보 컬럼"]
    for j, value in enumerate(headers): set_cell(table.cell(0, j), value, 11, True, WHITE, NAVY)
    set_cell(table.cell(1, 0), "교육·보육", 10, True, NAVY, PALE_BLUE)
    set_cell(table.cell(1, 1), "어린이집 수, 유치원 수, 유치원 학급당 원아 수, 어린이공원 수", 9, False, NAVY, WHITE, PP_ALIGN.LEFT)
    set_cell(table.cell(1, 2), "학교 수, 학급당 학생 수, 교사 수, 학원 수, 4년제 대학 진학률, 특목·자사고 수", 9, False, NAVY, WHITE, PP_ALIGN.LEFT)
    set_cell(table.cell(2, 0), "공통 생활", 10, True, NAVY, PALE_GREEN)
    common = "아파트 가구 수·평균 면적·평균 시가, 가구당 교육비, 지하철역·버스정류장·병원 수"
    set_cell(table.cell(2, 1), common, 9, False, NAVY, WHITE, PP_ALIGN.LEFT)
    set_cell(table.cell(2, 2), common, 9, False, NAVY, WHITE, PP_ALIGN.LEFT)
    add_footer(slide, "※ 순전입률은 이번 만족도 모델 후보에 아직 포함하지 않음 · 인프라 공통기간 때문에 2010~2020 만족도는 학습에서 제외")


def slide_algorithms(prs: Presentation, algorithms: pd.DataFrame) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "모델 비교와 최종 학습기법 선택", "동일한 최종 feature와 동일한 자치구 분할에서 비교 · 낮은 MAE가 우수", 2)
    preschool = algorithms[
        algorithms["group"].eq("preschool") & algorithms["feature_stage"].eq("checkpoint_2")
    ].set_index("algorithm")
    school = algorithms[
        algorithms["group"].eq("school_age") & algorithms["feature_stage"].eq("checkpoint_3")
    ].set_index("algorithm")
    order = list(ALGO_KO)
    table = slide.shapes.add_table(len(order) + 1, 4, Inches(.58), Inches(1.57), Inches(8.25), Inches(4.86)).table
    widths = [2.75, 1.55, 1.55, 2.40]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for j, value in enumerate(["학습기법", "미취학 MAE", "학령기 MAE", "해석"]):
        set_cell(table.cell(0, j), value, 10, True, WHITE, NAVY)
    best_pre = preschool["cv_mae"].idxmin(); best_school = school["cv_mae"].idxmin()
    for i, algorithm in enumerate(order, start=1):
        pre_value = preschool.loc[algorithm, "cv_mae"]
        school_value = school.loc[algorithm, "cv_mae"]
        selected = algorithm in {best_pre, best_school}
        fill = PALE_GREEN if selected else WHITE
        set_cell(table.cell(i, 0), ALGO_KO[algorithm], 9, selected, NAVY, fill, PP_ALIGN.LEFT)
        set_cell(table.cell(i, 1), f"{pre_value:.4f}", 9, algorithm == best_pre, NAVY, fill)
        set_cell(table.cell(i, 2), f"{school_value:.4f}", 9, algorithm == best_school, NAVY, fill)
        note = "최종 선택" if algorithm == "ridge_alpha_1" else ("비교 기준" if algorithm == "dummy_mean" else "후보")
        set_cell(table.cell(i, 3), note, 9, selected, TEAL if selected else MID, fill)

    text_box(slide, 9.05, 1.57, 3.70, 2.32, "최종 선택: 릿지(규제 1)", [
        "두 집단 모두 개발 CV MAE 최저",
        "작은 표본에서 트리 모델보다 안정적",
        "상관된 주거·교육·교통 변수를 규제로 완화",
        "2024년 결과를 보고 기법을 고르지 않음",
    ], TEAL)
    text_box(slide, 9.05, 4.10, 3.70, 2.33, "성능 해석", [
        "학령기: 평균모델 대비 MAE 13.5% 개선",
        "미취학: 개선 폭 1.4%로 매우 작음",
        "학령기 2024 결정계수 0.094",
        "미취학 2024 결정계수 -0.193",
    ], BLUE)
    add_footer(slide, "선택 원칙: 평균절대오차 우선 · ±0.5점 적중률은 만족도 분포가 좁아 참고지표로만 사용")


def translate_features(raw: str) -> str:
    values = pd.io.json.loads(raw) if False else None
    import json
    return " + ".join(FEATURE_KO.get(x, x) for x in json.loads(raw))


def slide_combinations(prs: Presentation, summary: pd.DataFrame) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title(slide, "Feature 조합 비교와 현재 최종 결론", "개선 조합은 남기고, 변수를 추가해 성능이 떨어진 사례도 함께 검증", 3)
    rows = []
    stage_ko = {"checkpoint_1": "1단계", "checkpoint_2": "2단계", "checkpoint_3": "3단계",
                "rejected_degradation": "하락 사례"}
    group_ko = {"preschool": "미취학", "school_age": "학령기"}
    for _, row in summary.iterrows():
        rows.append([
            group_ko[row["group"]], stage_ko[row["feature_stage"]], translate_features(row["features"]),
            ALGO_KO[row["selected_algorithm"]], f"{row['cv_mae']:.4f}",
            "-" if pd.isna(row["mae_change_vs_previous"]) else f"{row['mae_change_vs_previous']:+.4f}",
        ])
    table = slide.shapes.add_table(len(rows) + 1, 6, Inches(.38), Inches(1.52), Inches(12.57), Inches(3.60)).table
    widths = [1.0, 1.0, 5.10, 2.20, 1.05, 1.22]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for j, value in enumerate(["집단", "단계", "Feature 조합", "조합별 최적 기법", "CV MAE", "직전 대비"]):
        set_cell(table.cell(0, j), value, 9, True, WHITE, NAVY)
    for i, row in enumerate(rows, start=1):
        degraded = row[1] == "하락 사례"
        fill = PALE_RED if degraded else (PALE_GREEN if row[1] in {"2단계", "3단계"} else WHITE)
        for j, value in enumerate(row):
            set_cell(table.cell(i, j), value, 8.3, degraded and j in {1, 4, 5}, RED if degraded else NAVY,
                     fill, PP_ALIGN.LEFT if j == 2 else PP_ALIGN.CENTER)

    text_box(slide, .58, 5.42, 5.95, 1.36, "학령기 결론", [
        "최종: 아파트 평균 시가 + 고등학교 학급당 학생 수 + 버스정류장 수",
        "릿지(규제 1), CV MAE 0.1366 · 평균모델 대비 13.5% 개선",
        "교육비를 추가하면 MAE 0.1397로 악화 → 변수는 많을수록 좋은 것이 아님",
    ], TEAL)
    text_box(slide, 6.75, 5.42, 5.98, 1.36, "미취학 결론", [
        "최종: 버스정류장 수 + 아파트 평균 면적, CV MAE 0.1489",
        "아파트 가구 수 추가 시 MAE 0.1552로 악화",
        "2024 결정계수가 음수 → 현재는 운영 점수 모델로 채택 보류",
    ], BLUE)
    add_footer(slide, "현재 결과는 예측 연관성이지 인과효과가 아님 · 다음 개선: 전년도 연령별 순전입률 및 학령기 자녀가구 전용 만족도 추가")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_dir / "육아_교육_만족도_모델_최종요약_3장.pptx"
    algorithms = pd.read_csv(args.result_dir / "algorithm_comparison.csv")
    summary = pd.read_csv(args.result_dir / "feature_combination_algorithm_summary.csv")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "육아·교육 만족도 모델 최종 요약"
    prs.core_properties.subject = "학습 데이터, 모델 비교, feature 조합 비교 및 결론"
    slide_data(prs)
    slide_algorithms(prs, algorithms)
    slide_combinations(prs, summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    print(output)


if __name__ == "__main__":
    main()
