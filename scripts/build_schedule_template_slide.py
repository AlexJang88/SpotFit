"""Create a standalone schedule using the actual working deck's frame."""
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

from build_spotfit_project_deck import change, rect, table, text, PALE, DARK, GREEN, GREY, MEDIUM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SPOT_FIT_범죄모델_개선과정_2년3년_20261111.pptx"
OUTPUT = ROOT / "SPOT_FIT_개인일정_기존틀_1장_20261113부터.pptx"


def main():
    prs = Presentation(SOURCE)
    slide = prs.slides[0]
    # Retain the source's actual frame, decorations, title formatting and master.
    for shape in list(slide.shapes)[5:]:
        element = shape._element
        element.getparent().remove(element)
    change(slide.shapes[3], "08")
    change(slide.shapes[4], "개인 일정 타임라인 · 최종 시연 및 발표 준비")
    rect(slide, 1.25, 3.25, 17.5, .8, PALE)
    text(slide, "2026. 11. 13–11. 20  |  모델 개선 → 시연·리허설 → 프로젝트 수정 → 최종 발표",
         1.5, 3.4, 17, .5, 24, DARK, MEDIUM)

    rows = [
        ["일정", "장의석\nPM · 어린이 건강", "김홍수\n범죄 모델", "남궁다인\n아파트 매매·전월세", "임동현\n교통사고 모델", "오우석\n담당 업무 미기재"],
        ["11/13", "파이프라인 /\n서비스 구현", "범죄 모델 개선", "아파트 매매·전월세\n모델 개선", "교통사고\n모델 개선", "—"],
        ["11/16", "최종 시연·\n발표 준비", "최종 시연", "최종 시연", "최종 시연", "—"],
        ["11/17", "최종 리허설 준비", "최종 리허설 준비", "최종 리허설 준비", "최종 리허설 준비", "—"],
        ["11/18", "최종 프로젝트 수정", "최종 프로젝트 수정", "최종 프로젝트 수정", "최종 프로젝트 수정", "—"],
        ["11/19", "최종 발표 준비", "최종 프로젝트 수정", "최종 프로젝트 수정", "최종 프로젝트 수정", "—"],
        ["11/20", "최종 발표 준비", "최종 발표 준비", "최종 발표 준비", "최종 발표 준비", "—"],
    ]
    schedule = table(slide, rows, 1.25, 4.4, 17.5, 4.7,
                     [1.35, 3.25, 3.25, 3.25, 3.25, 3.15], 19)
    for cell in schedule.rows[0].cells:
        for index, paragraph in enumerate(cell.text_frame.paragraphs):
            for run in paragraph.runs:
                run.font.size = Pt(22 if index == 0 else 16)
    rect(slide, 1.25, 9.4, 17.5, .65, PALE)
    text(slide, "주요 일정   11/16 최종 시연  ·  11/17 리허설  ·  11/18–19 프로젝트 수정  ·  11/20 최종 발표",
         1.45, 9.52, 17.1, .4, 21, GREEN, MEDIUM)
    text(slide, "원본 일정표의 11월 13일 이후 일정만 반영했습니다. ‘—’는 담당 업무 미기재를 뜻합니다.",
         1.25, 10.18, 17.5, .4, 17, GREY)
    slide.notes_slide.notes_text_frame.text = (
        "사용자가 제공한 개인 일정 이미지의 2026년 11월 13일 이후 내용을 옮김. "
        "11월 19일 장의석은 발표 준비, 다른 담당자는 프로젝트 수정으로 원본 차이를 유지함. "
        "11월 20일 담당자별 업무는 발표 준비이며 전체 마일스톤은 최종 발표임."
    )
    for slide_id in list(prs.slides._sldIdLst)[1:]:
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
    prs.save(OUTPUT)
    check = Presentation(OUTPUT)
    assert len(check.slides) == 1
    assert check.slide_width == prs.slide_width and check.slide_height == prs.slide_height
    print(OUTPUT)


if __name__ == "__main__":
    main()
