from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
REPORT = ROOT / "models" / "교통사고" / "51차" / "51차_고도화등급예측평가.xlsx"
OUT = ROOT / "artifacts" / "presentations" / "교통사고_51차_고도화모델_발표자료.pptx"
DESKTOP = Path.home() / "Desktop" / OUT.name

NAVY = RGBColor(22, 43, 64); TEAL = RGBColor(22, 139, 133); BLUE = RGBColor(60, 116, 171)
MINT = RGBColor(224, 243, 238); PALE = RGBColor(239, 244, 248); WHITE = RGBColor(255, 255, 255)
INK = RGBColor(38, 49, 60); GRAY = RGBColor(100, 112, 124); ORANGE = RGBColor(221, 132, 62)


def text(slide, value, x, y, w, h, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear(); frame.word_wrap = True; frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]; p.text = value; p.alignment = align
    p.font.name = "맑은 고딕"; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill=WHITE, line=RGBColor(210, 219, 226)):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill; shape.line.color.rgb = line
    return shape


def header(slide, title, subtitle, page):
    text(slide, title, .72, .30, 10.8, .50, 24, NAVY, True)
    text(slide, subtitle, .74, .84, 11.4, .30, 10.5, GRAY)
    text(slide, f"{page:02d}", 12.15, .34, .45, .30, 10, TEAL, True, PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, label, value, note, color=TEAL):
    rect(slide, x, y, w, h, WHITE)
    text(slide, label, x+.18, y+.15, w-.36, .25, 10, GRAY, True)
    text(slide, value, x+.18, y+.50, w-.36, .55, 25, color, True)
    text(slide, note, x+.18, y+1.10, w-.36, .30, 9.5, GRAY)


def bullets(slide, items, x, y, w, h, size=15):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap=True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = f"• {item}"; p.space_after = Pt(10)
        p.font.name="맑은 고딕"; p.font.size=Pt(size); p.font.color.rgb=INK


def table(slide, rows, x, y, w, h, widths=None, font=10):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    t = shape.table
    if widths:
        for i, width in enumerate(widths): t.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell=t.cell(r,c); cell.text=str(value); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if r==0 else (PALE if r%2==0 else WHITE)
            cell.margin_left=Inches(.08); cell.margin_right=Inches(.08)
            for p in cell.text_frame.paragraphs:
                p.font.name="맑은 고딕"; p.font.size=Pt(font); p.font.bold=(r==0); p.font.color.rgb=WHITE if r==0 else INK; p.alignment=PP_ALIGN.CENTER
    return t


def main():
    weights = pd.read_excel(DATA, sheet_name="변수가중치").sort_values("가중치_pct", ascending=False)
    validation = pd.read_excel(DATA, sheet_name="고도화51차_연도별검증")
    forecast = pd.read_excel(DATA, sheet_name="고도화2026등급예측")
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); blank=prs.slide_layouts[6]

    s=prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
    text(s,"SPOTFIT · TRAFFIC SAFETY",.82,.70,8,.35,12,RGBColor(180,215,211),True)
    text(s,"교통사고 51차 고도화 모델",.82,1.45,11.6,.70,34,WHITE,True)
    text(s,"자치구별 다음 연도 위험점수와 5등급 예측",.84,2.35,11,.40,18,RGBColor(205,220,231))
    card(s,.85,3.35,3.55,1.65,"정확한 등급 일치","60.0%","기존 56.0% 대비 +4.0%p",TEAL)
    card(s,4.72,3.35,3.55,1.65,"±1등급 이내","90.4%","한 단계 이내 일치",BLUE)
    card(s,8.59,3.35,3.55,1.65,"평균 등급오차","0.496","0.512 → 0.496등급",ORANGE)
    text(s,"검증기간 2021~2025 · 서울 25개 자치구 · 1등급 고위험 / 5등급 저위험",.85,6.55,11.6,.30,11,WHITE)

    s=prs.slides.add_slide(blank); header(s,"발표 핵심 결론","등급 정확도를 높이되 미래 예측에 사용할 수 있는 정보만 사용",2)
    card(s,.72,1.45,3.75,1.55,"기존 51차","56.0%","Linear SVM + 순위보정",BLUE)
    card(s,4.80,1.45,3.75,1.55,"고도화 51차","60.0%","GBR 이중 예측 앙상블",TEAL)
    card(s,8.88,1.45,3.75,1.55,"개선 폭","+4.0%p","125건 중 5건 추가 적중",ORANGE)
    bullets(s,["단순 분류 대신 다음 연도 위험점수와 점수 변화량을 각각 예측","두 예측의 장점을 결합하고 25개 자치구 위험순위로 등급 결정","정확등급과 평균 등급오차는 개선, ±1등급 정확도는 2.4%p 감소"],.95,3.65,11.3,2.1,17)

    s=prs.slides.add_slide(blank); header(s,"무엇을 예측하는가?","49차 피해자율 모델의 변수 가중치를 안전점수·등급 체계로 확장",3)
    rect(s,.75,1.50,3.55,4.75,MINT); text(s,"입력",1.02,1.78,3,.35,18,TEAL,True)
    bullets(s,["49차 원천변수 13개","변수별 5분위 점수 13개","최근 점수·등급·증감·평균","자치구별 고정 특성"],1.02,2.30,2.95,3.0,14)
    text(s,"→",4.48,3.45,.6,.4,28,TEAL,True,PP_ALIGN.CENTER)
    rect(s,5.12,1.50,3.20,4.75,PALE); text(s,"고도화 모델",5.42,1.78,2.6,.35,18,NAVY,True)
    bullets(s,["위험점수 예측","점수 변화량 예측","75:25 앙상블","위험순위 산출"],5.42,2.30,2.55,3.0,14)
    text(s,"→",8.48,3.45,.6,.4,28,TEAL,True,PP_ALIGN.CENTER)
    rect(s,9.12,1.50,3.45,4.75,MINT); text(s,"출력",9.42,1.78,2.8,.35,18,TEAL,True)
    bullets(s,["다음 연도 위험점수","서울 25개 구 위험순위","5개 구씩 1~5등급","1등급이 상대적 고위험"],9.42,2.30,2.75,3.0,14)

    s=prs.slides.add_slide(blank); header(s,"최종 모델 구조","점수 수준과 변화 방향을 따로 학습한 뒤 결합",4)
    rect(s,.75,1.50,5.75,3.80,PALE); text(s,"구성 A · 75%",1.02,1.78,5.2,.38,18,NAVY,True)
    bullets(s,["Gradient Boosting · depth 1","다음 연도 종합위험점수 예측","예측점수 60% + 현재점수 40%","전체 원천·점수·시계열 변수 사용"],1.02,2.35,5.0,2.5,15)
    rect(s,6.82,1.50,5.75,3.80,MINT); text(s,"구성 B · 25%",7.10,1.78,5.2,.38,18,TEAL,True)
    bullets(s,["Gradient Boosting · depth 3","다음 연도 점수 변화량 예측","자치구 원-핫 변수 추가","지역별 반복 패턴까지 반영"],7.10,2.35,5.0,2.5,15)
    text(s,"최종 위험값 = 구성 A × 0.75 + 구성 B × 0.25",2.30,5.85,8.75,.55,20,WHITE,True,PP_ALIGN.CENTER).fill.solid(); s.shapes[-1].fill.fore_color.rgb=NAVY

    s=prs.slides.add_slide(blank); header(s,"데이터 전처리 구성","225행 · 최종 모델 입력 기준 71개 컬럼",5)
    rows=[["변수군","컬럼 수","역할"],["식별·목표",7,"연도·자치구·다음연도 점수와 등급"],["원천 변수",13,"49차에서 확정한 교통·인구·사고 지표"],["5분위 점수",13,"변수별 상대위험을 20~100점으로 환산"],["시계열 파생",13,"과거 점수·등급·증감·평균·변동성"],["자치구 변수",25,"지역별 고유 패턴을 원-핫으로 반영"]]
    table(s,rows,.82,1.55,11.7,4.25,[2.2,1.6,7.9],12)
    text(s,"미래 정답값이나 다음 연도 자료를 입력변수로 사용하지 않음",1.55,6.25,10.2,.42,14,TEAL,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"13개 원천변수와 가중치","49차 Ridge 표준화계수의 절댓값을 합계 100%로 환산",6)
    rows=[["순위","변수","가중치","방향"]]
    for i,r in weights.iterrows(): rows.append([len(rows),r["변수"],f"{r['가중치_pct']:.2f}%",r["방향"]])
    table(s,rows,.65,1.30,12.0,5.75,[.75,8.15,1.5,1.6],8.5)

    s=prs.slides.add_slide(blank); header(s,"연도별 순차검증 결과","각 검증연도보다 이전 자료만 학습하여 미래 예측 조건 유지",7)
    rows=[["검증연도","정확등급","±1등급","평균오차"]]
    for _,r in validation.iterrows(): rows.append([int(r["검증연도"]),f"{r['정확등급정확도']*100:.0f}%",f"{r['±1등급정확도']*100:.0f}%",f"{r['평균등급오차']:.2f}"])
    rows.append(["평균","60.0%","90.4%","0.496"])
    table(s,rows,.95,1.55,7.1,4.65,[1.7,1.8,1.8,1.8],13)
    card(s,8.45,1.55,3.75,1.55,"최저연도 정확도","52%","특정 연도에도 절반 이상",BLUE)
    card(s,8.45,3.40,3.75,1.55,"최고연도 정확도","68%","2021년·2025년",TEAL)
    text(s,"단일 2025년이 아니라 5개 연도의 평균으로 모델을 선택",8.55,5.45,3.55,.75,12,INK,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"2026년 예측 결과를 읽는 법","예측점수 자체보다 상대순위와 5등급을 함께 확인",8)
    top=forecast.sort_values("2026_고도화_위험순위").head(10)
    rows=[["위험순위","자치구","예측점수","예측등급"]]
    for _,r in top.iterrows(): rows.append([int(r["2026_고도화_위험순위"]),r["자치구"],f"{r['2026_고도화_예측위험점수']:.2f}",int(r["2026_고도화_예측위험등급"])])
    table(s,rows,.75,1.35,7.2,5.55,[1.5,2.0,2.0,1.7],10)
    bullets(s,["1등급: 상대적으로 위험도가 높은 상위 20%","등급은 서울 25개 자치구 사이의 상대평가","정책 우선순위에는 등급과 예상 피해자율을 함께 사용","공식 지역안전지수 등급과는 구분"],8.35,1.75,4.2,3.7,14)

    s=prs.slides.add_slide(blank); header(s,"최종 결론과 활용 원칙","51차 고도화 모델은 등급 의사결정용, 49차는 피해자율 수치 예측용",9)
    card(s,.75,1.45,3.75,1.55,"현재 최종 성능","60.0%","정확한 5등급 일치",TEAL)
    card(s,4.80,1.45,3.75,1.55,"안정성","90.4%","±1등급 이내",BLUE)
    card(s,8.85,1.45,3.75,1.55,"평균 오차","0.496","절반 등급 이하",ORANGE)
    bullets(s,["지도·대시보드에는 위험점수, 순위, 등급을 함께 표시","2027·2028 예측은 미래 입력변수 생성 방식까지 동일하게 유지","새 실제연도 자료가 확보되면 순차검증에 추가하고 재학습","60%는 개선된 결과지만 100% 확정등급이 아닌 예측값으로 안내"],1.05,3.65,11.1,2.3,16)

    OUT.parent.mkdir(parents=True,exist_ok=True); prs.save(OUT)
    try:
        DESKTOP.write_bytes(OUT.read_bytes()); print(DESKTOP)
    except OSError:
        print(f"DESKTOP_COPY_FAILED:{DESKTOP}")
    print(OUT)


if __name__ == "__main__": main()
