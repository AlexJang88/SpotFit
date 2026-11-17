"""Update the existing crime-model deck from 1y/2y service to 2y/3y service."""
from __future__ import annotations

from copy import copy
from pathlib import Path

from pptx import Presentation

SOURCE = Path(r"C:\Users\M\Desktop\발표자료\SPOT_FIT_범죄모델_개선과정_5장_20261110.pptx")
OUTPUT = Path(r"C:\Users\M\Desktop\JUS\SpotFit\SPOT_FIT_범죄모델_개선과정_2년3년_20261111.pptx")


def replace_shape_text(shape, text: str) -> None:
    """Replace text while retaining the first run's exact formatting."""
    frame = shape.text_frame
    first_paragraph = frame.paragraphs[0]
    first_run = first_paragraph.runs[0] if first_paragraph.runs else first_paragraph.add_run()
    run_font = copy(first_run.font._element)
    paragraph_props = copy(first_paragraph._p.get_or_add_pPr())
    frame.clear()
    paragraph = frame.paragraphs[0]
    paragraph._p.insert(0, paragraph_props)
    run = paragraph.add_run()
    run.text = text
    run.font._element.getparent().replace(run.font._element, run_font)


def replace_cell_text(cell, text: str) -> None:
    frame = cell.text_frame
    paragraph = frame.paragraphs[0]
    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    font_props = copy(run.font._element)
    paragraph_props = copy(paragraph._p.get_or_add_pPr())
    frame.clear()
    paragraph = frame.paragraphs[0]
    paragraph._p.insert(0, paragraph_props)
    run = paragraph.add_run()
    run.text = text
    run.font._element.getparent().replace(run.font._element, font_props)


def set_table(table, values: list[list[str]]) -> None:
    assert len(values) == len(table.rows)
    assert all(len(row) == len(table.columns) for row in values)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            replace_cell_text(table.cell(row_index, column_index), value)


def main() -> None:
    presentation = Presentation(SOURCE)

    # Slide 2: the selected three-year ensemble is now an operational 2028 model.
    slide = presentation.slides[1]
    replace_shape_text(
        slide.shapes[17],
        "3년 앙상블은 2028년 서비스 모델로 적용합니다. Y는 실제 아동 피해율이 아닌 지역 노출 대리지표입니다.",
    )

    # Slides 3–4: make the target year explicit while retaining the valid 2y experiment story.
    slide = presentation.slides[2]
    replace_shape_text(slide.shapes[4], "범죄모델 개선 ③ · 2027년 2년 모델의 Feature 선택")
    replace_shape_text(
        slide.shapes[6],
        "2027년 2년 실험: 최신 데이터 가용성을 확인하고, 그룹을 추가할 때마다 순위 성능 비교",
    )
    slide = presentation.slides[3]
    replace_shape_text(slide.shapes[4], "범죄모델 개선 ④ · 2027년 2년 모델의 개선 폭")

    # Slide 5: replace the retired 1y service row with the operational 3y ensemble.
    slide = presentation.slides[4]
    set_table(
        slide.shapes[5].table,
        [
            ["현재 서비스", "기법 / 예측 로직", "입력 → 목표연도", "CV 5분위 일치"],
            ["2년 모델", "Ridge α=10 / 미래 Y 직접 예측", "2025 → 2027", "84.0%"],
            ["3년 모델", "구성요소 92% + 직접 Y 8%", "2025 → 2028", "81.3%"],
        ],
    )
    replace_shape_text(
        slide.shapes[6],
        "2년·3년 모델 모두 시간순 CV로 선택하고, 사용하지 않은 2025년을 별도검증",
    )
    set_table(
        slide.shapes[7].table,
        [
            ["지표", "2년 모델 CV", "3년 모델 CV", "2025 별도검증 · 2년 / 3년"],
            ["순위 상관 ↑", "0.9746", "0.9703", "0.9677  /  0.9731"],
            ["평균 순위오차 ↓", "1.07위", "1.17위", "1.36위  /  1.28위"],
            ["5분위 일치율 ↑", "84.0%", "81.3%", "76%  /  84%"],
            ["상위 20% 적중률 ↑", "100%", "93.3%", "80%  /  80%"],
        ],
    )
    replace_shape_text(
        slide.shapes[9],
        "운영 기준: 완결 데이터 2025년 → 2027년은 2년 모델 · 2028년은 3년 앙상블 모델",
    )
    replace_shape_text(
        slide.shapes[10],
        "서비스에서는 2027년·2028년을 선택합니다. 기존 1년 모델은 서비스 대상에서 제외했습니다.",
    )

    presentation.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
