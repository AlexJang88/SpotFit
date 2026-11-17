"""Explain current fitted coefficients separately from the deployed score rule."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from pptx import Presentation

from build_spotfit_project_deck import change, rect, text, table, GREEN, DARK, GREY, PALE, MEDIUM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'SPOT_FIT_범죄모델_개선과정_2년3년_20261111.pptx'
MODEL = ROOT / 'models/recommender/v1/child_crime_exposure/2y/child_crime_exposure_2y.joblib'
OUTPUT = ROOT / 'SPOT_FIT_범죄노출_Feature비중_점수등급_1장.pptx'


def main():
    bundle = joblib.load(MODEL)
    features = bundle['features']
    estimator = bundle['model'].named_steps['model']
    names = bundle['model'].named_steps['prep'].get_feature_names_out()
    assert list(names[:len(features)]) == ['numeric__' + f for f in features]
    coef = np.asarray(estimator.coef_).ravel()[:len(features)]
    weights = np.abs(coef) / np.abs(coef).sum() * 100
    ordered = sorted(zip(features, coef, weights), key=lambda row: -row[2])
    labels = {
        'y': '현재 노출지수 Y',
        'y_lag1': '1년 전 노출지수 Y',
        'log_child_lag2': '2년 전 아동인구 · log1p',
        'log_child': '현재 아동인구 · log1p',
        'log_child_lag1': '1년 전 아동인구 · log1p',
    }
    shown = [round(float(row[2]), 2) for row in ordered[:5]]
    rest = round(100 - sum(shown), 2)
    rows = [['주요 수치 Feature', '설명용 비중']]
    rows.extend([[labels[row[0]], f'{percent:.2f}%'] for row, percent in zip(ordered[:5], shown)])
    rows.extend([['... 나머지 7개 항목', f'{rest:.2f}%'], ['수치 Feature 12개 합계', '100.00%']])

    prs = Presentation(SOURCE)
    slide = prs.slides[0]
    for shape in list(slide.shapes)[5:]:
        shape._element.getparent().remove(shape._element)
    change(slide.shapes[3], '07')
    change(slide.shapes[4], '아동범죄노출 · Feature 비중과 점수·등급 산정')
    rect(slide, 1.25, 3.25, 17.5, .8, PALE)
    text(slide, '노출지수 Y = log(1 + 5대범죄 건수 ÷ 아동인구 × 10,000)  →  예측 Y의 순위를 점수로 환산',
         1.5, 3.4, 17, .5, 23, DARK, MEDIUM)
    text(slide, '① 2027년 2년 모델 · 주요 Feature 비중', 1.25, 4.28, 8.5, .5, 24, GREEN, MEDIUM)
    text(slide, '표준화된 계수의 절댓값 ÷ 12개 절댓값 합 × 100',
         1.25, 4.9, 8.5, .48, 20, DARK)
    table(slide, rows, 1.25, 5.5, 8.5, 3.55, [6.0, 2.5], 20)
    text(slide, 'Ridge가 예측오차와 계수 크기를 함께 줄이도록 학습\n위 비중은 수치 변수끼리 비교한 값이며 자치구 보정은 별도',
         1.25, 9.45, 8.5, .65, 16, GREY)

    text(slide, '② 현재 서비스 · 점수와 등급 만드는 법', 10.2, 4.28, 8.55, .5, 24, GREEN, MEDIUM)
    text(slide, '예측 Y가 낮은 순서로 서울 25개 구를 정렬',
         10.2, 4.9, 8.55, .48, 21, DARK)
    table(slide, [
        ['낮은 노출 순위', '점수', '등급'],
        ['1–5위 · 낮은 위험 20%', '100점', 'A'],
        ['6–10위', '80점', 'B'],
        ['11–15위', '60점', 'C'],
        ['16–20위', '40점', 'D'],
        ['21–25위 · 높은 위험 20%', '20점', 'E'],
    ], 10.2, 5.5, 8.55, 3.55, [5.25, 1.65, 1.65], 20)
    text(slide, '예: 예측 Y가 낮은 순서 7위 → 80점 · B등급\n2년·3년 모델 모두 같은 순위 환산 규칙을 적용',
         10.2, 9.12, 8.55, .75, 18, GREEN)
    text(slide, '설명용 Feature 비중은 점수 가중합이 아닙니다. 3년 모델은 구성요소 예측 92% + 직접 Y 예측 8%를 혼합합니다.',
         1.25, 10.18, 17.5, .45, 17, GREY)
    slide.notes_slide.notes_text_frame.text = (
        '현재 서비스 점수는 항목별 분위 점수를 가중합하는 방식이 아니다. '
        '최종 예측 Y를 오름차순 정렬하여 rank(method=min) 기반 5분위 점수로 변환한다. '
        '현재 두 산출물은 25개 서로 다른 예측값을 가져 각 등급이 5개씩이다. 동점이 발생하면 등급별 개수가 달라질 수 있다. '
        '왼쪽 비중은 이번 PPT 설명을 위해 현행 2년 Ridge(alpha=10) 운영모델의 표준화 수치 계수 절댓값을 정규화한 값이다. '
        '기존 저장 영향력 CSV나 실험 가중합 산식은 현행 앙상블 전체의 Feature 비중으로 제시하지 않았다. '
        '부호를 제거한 비중이므로 인과효과나 위험 방향을 뜻하지 않으며, 상관된 변수 간 계수 분배 영향을 받는다. '
        '자치구 One-hot 25개와 절편을 제외한 수치 Feature 12개 내부 합만 100%이다. '
        '현재 Y와 1년 전 Y의 계수는 양수, 표시한 아동인구 계수는 음수이다. 실제 Ridge 예측은 부호 있는 계수와 절편·구 보정을 사용한다. '
        '표의 나머지 비중은 표시값 합계가 100%가 되도록 소수 둘째자리 반올림 잔여로 산출했다. '
        '3년 92:8 비중은 저장된 실험 메타데이터상 시간순 CV 종합 순위점수를 0.01 간격으로 비교해 선정한 모델 혼합 비율이다. '
        '점수 근거: scripts/child_crime/export_service_crime_forecasts.py 및 backend/app/services/child_crime_exposure.py. '
        '모델 SHA256: ' + hashlib.sha256(MODEL.read_bytes()).hexdigest()
    )
    for sid in list(prs.slides._sldIdLst)[1:]:
        prs.part.drop_rel(sid.rId)
        prs.slides._sldIdLst.remove(sid)
    prs.save(OUTPUT)
    audit = ROOT / 'artifacts/presentations/crime_weight_score_evidence.json'
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        'model': str(MODEL), 'method': 'abs standardized numeric Ridge coefficients / sum * 100',
        'scope': '12 numeric features only; borough one-hot and intercept excluded',
        'not_production_score_weights': True,
        'features': [{'feature':f, 'coefficient':float(c), 'relative_pct':float(w)} for f,c,w in ordered],
        'total_pct': float(weights.sum()),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    assert len(Presentation(OUTPUT).slides) == 1
    print(OUTPUT)
    print(rows)


if __name__ == '__main__':
    main()
