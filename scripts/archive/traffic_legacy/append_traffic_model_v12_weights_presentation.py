"""바탕화면의 기존 1~11차 PPT 형식을 유지하고 12차와 1~12차 모델 영향 비율을 추가한다."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from pptx import Presentation

from create_traffic_models_v1_v5_presentation import (
    BLUE, DARK, MID, NAVY, ORANGE, RED, TEAL, add_table, bullets, card, header, text,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"
SOURCE = Path.home() / "Desktop" / "교통사고_1차~11차_모델비교.pptx"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "traffic_models_v1_v12"
OUTPUT = OUTPUT_DIR / "교통사고_1차~12차_모델비교.pptx"

SHORT_NAMES = {
    "인구만명당 교통사고 사망자수": "사망자수/인구1만명",
    "인구만명당 의료보장 사업장수": "의료보장사업장/인구1만명",
    "운전자안전벨트 착용률": "안전벨트 착용률",
    "음주운전 경험률": "음주운전 경험률",
    "현재연도 교통사고 건수": "현재연도 사고건수",
    "사고건수 1년전": "사고건수 1년전",
    "최근 3년 사고건수 평균": "최근 3년 사고평균",
    "사고건수 전년 증감": "사고건수 전년증감",
    "최근 3년 사고건수 변동성": "최근 3년 사고변동성",
    "자동차 등록 대수": "자동차 등록대수",
    "생활인구 연평균": "생활인구 연평균",
    "생활인구 연중시간최대": "생활인구 연중최대",
    "생활인구 월변동성": "생활인구 월변동성",
    "사고당 부상자수": "사고당 부상자수",
    "어린이 사고 비율": "어린이 사고비율",
}


def replace_existing_labels(prs: Presentation) -> None:
    replacements = {
        "1차~11차": "1차~12차",
        "1~11차": "1~12차",
        "11개 모델": "12개 모델",
    }
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    for old, new in replacements.items():
                        run.text = run.text.replace(old, new)


def load_model_weight(version: int) -> dict:
    model_path = next((MODEL_ROOT / f"{version}차").glob("*.joblib"))
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    features = payload["features"]
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        raw = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        values = np.abs(raw)
        directions = ["+" if value > 0 else "−" if value < 0 else "0" for value in raw]
        method = "표준화 계수 절댓값"
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
        directions = [""] * len(values)
        method = "트리 중요도"
    ratios = values / values.sum() * 100
    metadata = json.loads((MODEL_ROOT / f"{version}차" / "모델정보.json").read_text(encoding="utf-8"))
    return {
        "version": version,
        "algorithm": metadata["선정모델"],
        "method": method,
        "items": list(zip(features, ratios, directions)),
    }


def weight_card(slide, x: float, model: dict) -> None:
    title = f"{model['version']}차 · {model['algorithm']}"
    card(slide, x, 1.35, 3.85, .88, "선정 모델", title, model["method"], TEAL)
    lines = []
    for feature, ratio, direction in model["items"]:
        suffix = f" ({direction})" if direction else ""
        lines.append(f"{SHORT_NAMES.get(feature, feature)}  {ratio:.1f}%{suffix}")
    text(slide, x + .10, 2.48, 3.65, 3.85, "\n".join(lines), 12, False, DARK)
    top_feature, top_ratio, _ = max(model["items"], key=lambda item: item[1])
    text(slide, x + .10, 6.45, 3.65, .42,
         f"최대: {SHORT_NAMES.get(top_feature, top_feature)} {top_ratio:.1f}%", 10, True, NAVY)


def build() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation(SOURCE)
    replace_existing_labels(prs)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    header(slide, "2차 기반 확장: 12차", "사고 심각도와 어린이 사고 구성비를 함께 추가", 11)
    rows = [
        ["모델", "추가 변수", "선정 기법", "CV MAE", "2025 MAE", "R²", "WMAPE"],
        ["2차", "없음", "Ridge", "88.84", "99.69", "0.9626", "7.30%"],
        ["12차", "사고당 부상자수 + 어린이 사고비율", "Ridge", "93.50", "100.42", "0.9619", "7.36%"],
    ]
    add_table(slide, .45, 1.45, 12.45, 2.65, rows, [1.0, 3.7, 1.45, 1.55, 1.55, 1.4, 1.5])
    bullets(slide, .8, 4.55, 11.8, 1.7, [
        "내부검증 MAE가 2차보다 4.66건 증가해 개선되지 않음",
        "2025 MAE·RMSE·WMAPE도 모두 소폭 악화",
        "12차는 비교 모델로 보존하고 운영 최우선은 2차 유지",
    ], 17)
    text(slide, .8, 6.55, 11.7, .3, "추가 변수는 예측용이며 공식 안전지수 가중치에는 포함하지 않음", 12, True, RED)

    slide = prs.slides.add_slide(blank)
    header(slide, "모델 컬럼별 영향 비율 산출법", "공식 안전지수 가중치와 예측 모델의 영향 비율은 서로 다른 개념", 12)
    card(slide, .75, 1.4, 3.75, 1.45, "Ridge 모델", "표준화 계수 비율", "|계수| 합계를 100%로 환산", TEAL)
    card(slide, 4.8, 1.4, 3.75, 1.45, "트리 모델", "내장 중요도 비율", "feature_importances_ 합계 100%", BLUE)
    card(slide, 8.85, 1.4, 3.75, 1.45, "방향 표시", "+ / −", "Ridge만 증감 방향 표시", ORANGE)
    bullets(slide, .9, 3.35, 11.6, 2.4, [
        "비율이 클수록 해당 모델의 예측에 미치는 상대적 영향이 큼",
        "Ridge의 비율은 표준화된 입력 기준이므로 서로 단위가 달라도 비교 가능",
        "트리 중요도는 영향 방향을 제공하지 않으므로 +/−를 표시하지 않음",
        "상관된 변수끼리는 중요도가 나뉠 수 있어 인과관계로 해석하지 않음",
        "안전등급 산출에는 기존 안전기준표 가중치를 그대로 사용",
    ], 17)

    weights = {version: load_model_weight(version) for version in range(1, 13)}
    for slide_no, versions in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), start=13):
        slide = prs.slides.add_slide(blank)
        header(slide, f"{versions[0]}~{versions[-1]}차 컬럼별 영향 비율",
               "각 차수에서 최종 선정된 알고리즘 기준 · 합계 100%", slide_no)
        for x, version in zip((.55, 4.75, 8.95), versions):
            weight_card(slide, x, weights[version])
        text(slide, .65, 7.05, 12.0, .2, "+는 예측 증가 방향, −는 예측 감소 방향 · 트리 모델은 방향 미표시", 9, False, MID)

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
