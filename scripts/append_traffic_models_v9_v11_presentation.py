"""기존 1~8차 PPT를 보존하고 9~11차 결과만 뒤에 추가한다."""
from pathlib import Path

from pptx import Presentation

from create_traffic_models_v1_v5_presentation import (
    BLUE, ORANGE, RED, TEAL, add_table, bullets, card, header, text,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "reports" / "traffic_models_v1_v8" / "교통사고_1차~8차_모델비교.pptx"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "traffic_models_v1_v11"
OUTPUT = OUTPUT_DIR / "교통사고_1차~11차_모델비교.pptx"


def replace_existing_labels(prs: Presentation) -> None:
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = run.text.replace("1차~8차", "1차~11차")


def build() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation(SOURCE)
    replace_existing_labels(prs)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    header(slide, "7차 기반 재확장: 9~11차", "동일 표본의 7차 기준과 내부 시간순 CV 비교", 9)
    rows = [
        ["차수", "추가 변수", "선정 기법", "7차기준 CV MAE", "확장 CV MAE", "개선 폭", "2025 MAE"],
        ["9차", "최근 3년 사고 평균", "Ridge", "126.92", "107.68", "19.24", "105.43"],
        ["10차", "생활인구 월변동성", "Ridge", "97.34", "96.00", "1.34", "103.39"],
        ["11차", "두 변수 모두", "ExtraTrees", "126.92", "124.42", "2.49", "108.97"],
    ]
    add_table(slide, .4, 1.4, 12.55, 3.65, rows, [.8, 2.35, 1.5, 1.75, 1.65, 1.35, 1.45])
    bullets(slide, .75, 5.35, 11.8, 1.35, [
        "9차는 축소된 125행 동일 표본에서 가장 뚜렷한 CV 개선",
        "10·11차도 수치상 개선이나 폭은 작음",
        "세 모델 모두 2025 MAE가 7차 99.19건보다 높아 최우선은 유지",
    ], 15)

    slide = prs.slides.add_slide(blank)
    header(slide, "1~11차 결론 갱신", "기존 1~8차 결론에 9~11차 결과를 추가", 10)
    card(slide, .75, 1.35, 3.7, 1.35, "내부 CV 최우선", "2차 Ridge", "MAE 88.84 · 2025 MAE 99.69", TEAL)
    card(slide, 4.8, 1.35, 3.7, 1.35, "2025 MAE 최저", "7차 Ridge", "MAE 99.19 · CV는 2차보다 열세", BLUE)
    card(slide, 8.85, 1.35, 3.7, 1.35, "7차 확장 후보", "9차 Ridge", "동일표본 CV 19.24건 개선", ORANGE)
    bullets(slide, .85, 3.2, 11.7, 2.8, [
        "사전 내부 CV 기준 최우선은 2차, 2025 최종 MAE 최저는 7차",
        "9차는 7차 기반 확장 중 가장 의미 있는 CV 개선",
        "10·11차는 개선 폭이 작아 추가 연도 확인 전 보류",
        "추가 변수는 예측용이며 안전지수 가중치는 기존 산식 유지",
        "단일 연도 최종검증이므로 새로운 연도에서 우선순위를 재확인",
    ], 18)
    text(slide, .85, 6.55, 11.6, .3, "주의: 테스트 성능을 보고 모델 선택 규칙을 변경하지 않음", 12, True, RED)

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
