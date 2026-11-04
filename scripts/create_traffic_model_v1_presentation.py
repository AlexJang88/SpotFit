from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


MODEL = ROOT / "models" / "안전" / "교통사고" / "1차"
REPORT_DIR = ROOT / "artifacts" / "reports" / "traffic_model_v1"
OUTPUT = REPORT_DIR / "교통사고_1차_초기모델_발표자료.pptx"

NAVY = RGBColor(20, 45, 78)
BLUE = RGBColor(36, 105, 170)
TEAL = RGBColor(29, 142, 134)
ORANGE = RGBColor(232, 126, 61)
RED = RGBColor(196, 65, 65)
LIGHT = RGBColor(242, 246, 250)
MID = RGBColor(110, 125, 140)
DARK = RGBColor(34, 42, 50)
WHITE = RGBColor(255, 255, 255)
FONT = "맑은 고딕"


def set_run(run, size=20, bold=False, color=DARK):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_text(slide, x, y, w, h, text, size=20, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    set_run(r, size, bold, color)
    return box


def add_bullets(slide, x, y, w, h, items, size=19, color=DARK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear(); tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.space_after = Pt(9)
        p.font.name = FONT; p.font.size = Pt(size); p.font.color.rgb = color
    return box


def add_header(slide, title, subtitle=None, number=None):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.16))
    shape.fill.solid(); shape.fill.fore_color.rgb = TEAL; shape.line.fill.background()
    add_text(slide, 0.65, 0.35, 11.8, 0.55, title, 27, True, NAVY)
    if subtitle:
        add_text(slide, 0.67, 0.92, 11.8, 0.35, subtitle, 11, False, MID)
    if number is not None:
        add_text(slide, 12.45, 0.40, 0.35, 0.35, f"{number:02d}", 12, True, TEAL, PP_ALIGN.RIGHT)
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(7.18), Inches(12.0), Inches(0.02))
    footer.fill.solid(); footer.fill.fore_color.rgb = RGBColor(220, 228, 236); footer.line.fill.background()
    add_text(slide, 0.67, 7.22, 8.5, 0.2, "안전 · 교통사고 · 1차 초기모델", 9, False, MID)


def add_card(slide, x, y, w, h, label, value, accent=TEAL, note=None):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.fill.solid(); card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = RGBColor(220, 229, 238)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
    add_text(slide, x+0.25, y+0.18, w-0.4, 0.3, label, 12, True, MID)
    add_text(slide, x+0.25, y+0.52, w-0.4, 0.55, value, 25, True, NAVY)
    if note:
        add_text(slide, x+0.25, y+h-0.40, w-0.4, 0.25, note, 10, False, MID)


def save_charts(tmp: Path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    comp = pd.read_csv(MODEL / "후보모델_비교.csv")
    valid = pd.read_csv(MODEL / "2025년_최종검증_예측결과.csv")
    imp = pd.read_csv(MODEL / "변수중요도.csv")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    comp = comp.sort_values("내부검증_MAE", ascending=True)
    colors = ["#1d8e86" if m == "RandomForest" else "#9eb3c7" for m in comp["모델"]]
    ax.barh(comp["모델"], comp["내부검증_MAE"], color=colors)
    for i, v in enumerate(comp["내부검증_MAE"]): ax.text(v+5, i, f"{v:.1f}", va="center", fontsize=10)
    ax.set_xlabel("내부 순차검증 평균 MAE(건)"); ax.set_title("후보 모델 비교 — 낮을수록 우수", loc="left", fontweight="bold")
    ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); p1=tmp/"model_comparison.png"; fig.savefig(p1,dpi=180,transparent=False); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(valid["다음연도 교통사고 건수"], valid["예측 교통사고 건수"], s=55, color="#1d8e86", alpha=.82)
    lo=min(valid["다음연도 교통사고 건수"].min(),valid["예측 교통사고 건수"].min())
    hi=max(valid["다음연도 교통사고 건수"].max(),valid["예측 교통사고 건수"].max())
    ax.plot([lo,hi],[lo,hi],"--",color="#c44141",lw=1.5,label="완전 일치")
    ax.set_xlabel("2025년 실제 사고 건수"); ax.set_ylabel("2025년 예측 사고 건수")
    ax.set_title("실제값과 예측값", loc="left", fontweight="bold"); ax.legend(frameon=False)
    ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); p2=tmp/"actual_pred.png"; fig.savefig(p2,dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    imp = imp.sort_values("순열중요도_MAE증가", ascending=True)
    vals = imp["순열중요도_MAE증가"]
    ax.barh(imp["변수"], vals, color=["#c44141" if v < 0 else "#2470aa" for v in vals])
    ax.axvline(0,color="#65788a",lw=.8)
    ax.set_xlabel("변수를 섞었을 때 증가한 MAE(건)"); ax.set_title("변수 중요도", loc="left", fontweight="bold")
    ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); p3=tmp/"importance.png"; fig.savefig(p3,dpi=180); plt.close(fig)

    grade = valid["안전등급"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7,4.3))
    ax.bar(grade.index.astype(str), grade.values, color=["#1d8e86","#58a88a","#9eb3c7","#e4a85f","#c44141"])
    for x,v in zip(grade.index.astype(str),grade.values): ax.text(x,v+.15,str(v),ha="center",fontweight="bold")
    ax.set_xlabel("안전등급"); ax.set_ylabel("자치구 수"); ax.set_title("2024년 안전등급 분포",loc="left",fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); ax.set_ylim(0,max(grade.values)+1.5); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); p4=tmp/"grades.png"; fig.savefig(p4,dpi=180); plt.close(fig)
    return p1,p2,p3,p4,valid


def build():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    assets = MODEL / ".ppt_assets"
    assets.mkdir(parents=True, exist_ok=True)
    try:
        p1,p2,p3,p4,valid = save_charts(assets)

        s=prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
        band=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.7),Inches(1.05),Inches(.13),Inches(4.9)); band.fill.solid();band.fill.fore_color.rgb=TEAL;band.line.fill.background()
        add_text(s,1.15,1.25,10.7,.45,"안전 · 교통사고",18,True,RGBColor(106,211,201))
        add_text(s,1.15,1.9,10.7,1.4,"1차 초기 예측모델 결과",36,True,WHITE)
        add_text(s,1.15,3.45,9.8,.8,"안전기준표 항목만으로 다음 연도 자치구별\n교통사고 건수를 예측한 기준 모델",21,False,RGBColor(215,227,238))
        add_text(s,1.15,5.9,8,.35,"검증 대상  2025년  |  서울 25개 자치구",14,True,WHITE)
        add_text(s,11.55,6.85,1,.3,"2026.11",11,False,RGBColor(170,190,210),PP_ALIGN.RIGHT)

        s=prs.slides.add_slide(blank); add_header(s,"한눈에 보는 모델 결과","안전기준표 기반 1년 후 사고 건수 예측",2)
        add_card(s,.7,1.45,2.85,1.45,"선정 모델","Random Forest",TEAL,"내부 순차검증 MAE 최소")
        add_card(s,3.7,1.45,2.85,1.45,"오차 기반 정확도","71.59%",BLUE,"100 - WMAPE")
        add_card(s,6.7,1.45,2.85,1.45,"평균 절대오차","387.86건",ORANGE,"자치구 1곳 기준")
        add_card(s,9.7,1.45,2.85,1.45,"R²","0.3381",RED,"분산설명률 33.81%")
        add_text(s,.75,3.35,11.9,.5,"핵심 결론",20,True,NAVY)
        add_bullets(s,.9,3.95,11.5,2.3,[
            "안전기준표의 정확히 확보된 4개 변수만 사용한 초기 기준 모델",
            "2024년 입력값으로 2025년 사고 건수를 예측한 완전 분리 검증",
            "전체 실제 사고 건수 대비 절대오차는 28.41% — 확장 모델 개선의 기준선으로 활용",
        ],19)

        s=prs.slides.add_slide(blank); add_header(s,"모델의 목적과 예측 구조","X(t) → y(t+1)",3)
        add_text(s,.8,1.45,3.5,.5,"예측 목표",18,True,TEAL)
        add_text(s,.8,2.0,3.7,1.0,"다음 연도\n교통사고 건수",27,True,NAVY,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
        arrow=s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,Inches(4.6),Inches(2.18),Inches(1.6),Inches(.62));arrow.fill.solid();arrow.fill.fore_color.rgb=TEAL;arrow.line.fill.background()
        add_text(s,6.55,1.45,5.6,.45,"시간 분리",18,True,TEAL)
        add_bullets(s,6.55,2.0,5.6,2.5,["학습: 2017~2023년 X → 2018~2024년 y (175행)","최종 검증: 2024년 X → 2025년 y (25행)","2025년 검증값은 모델 선택·학습에 미사용"],18)
        call=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(.8),Inches(4.75),Inches(11.8),Inches(1.35));call.fill.solid();call.fill.fore_color.rgb=LIGHT;call.line.color.rgb=RGBColor(215,226,235)
        add_text(s,1.1,5.02,11.2,.8,"무작위 분할 대신 연도 순서대로 분리하여 미래 정보 누출을 방지",21,True,NAVY,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)

        s=prs.slides.add_slide(blank); add_header(s,"사용 변수와 안전점수 가중치","안전기준표 외 변수와 대리지표는 사용하지 않음",4)
        rows=[("위해","인구만명당 교통사고 사망자수","62.5%",RED),("취약","인구만명당 의료보장 사업장수","12.5%",ORANGE),("의식","운전자안전벨트 착용률","12.70%",TEAL),("의식","음주운전 경험률","12.30%",TEAL)]
        y=1.45
        for cat,var,w,c in rows:
            sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(.8),Inches(y),Inches(11.7),Inches(.85));sh.fill.solid();sh.fill.fore_color.rgb=LIGHT;sh.line.color.rgb=RGBColor(220,229,238)
            add_text(s,1.05,y+.2,1.0,.35,cat,15,True,c); add_text(s,2.15,y+.17,7.8,.4,var,18,True,DARK); add_text(s,10.4,y+.17,1.55,.4,w,18,True,NAVY,PP_ALIGN.RIGHT);y+=1.05
        add_text(s,.85,5.85,11.6,.65,"경감지표는 △ 자료라 제외 → 남은 80%를 100%로 재정규화\n재난약자수·교통법규위반건수도 미확보로 제외",15,False,MID,PP_ALIGN.CENTER)

        s=prs.slides.add_slide(blank); add_header(s,"왜 Random Forest인가?","2025년을 보지 않은 내부 순차검증으로 선정",5)
        s.shapes.add_picture(str(p1),Inches(.65),Inches(1.35),width=Inches(7.3),height=Inches(4.8))
        add_text(s,8.35,1.55,4.2,.45,"선정 근거",19,True,TEAL)
        add_bullets(s,8.35,2.12,4.1,2.9,["4개 후보 중 평균 MAE 최저: 415.66건","비선형 관계와 변수 간 상호작용을 반영","표준화가 필요 없고 이상값 영향이 비교적 작음","작은 표본의 과적합을 줄이기 위해 잎 최소 표본 4 적용"],16)
        add_text(s,8.35,5.45,4.0,.75,"‘가장 복잡한 모델’이 아니라\n시간 검증에서 가장 안정적인 모델을 선택",15,True,NAVY)

        s=prs.slides.add_slide(blank); add_header(s,"2025년 최종 검증 결과","2024년 X로 서울 25개 자치구의 사고 건수 예측",6)
        s.shapes.add_picture(str(p2),Inches(.55),Inches(1.35),width=Inches(6.45),height=Inches(5.45))
        add_card(s,7.35,1.48,2.4,1.35,"MAE","387.86건",ORANGE)
        add_card(s,9.95,1.48,2.4,1.35,"RMSE","556.38건",RED)
        add_card(s,7.35,3.08,2.4,1.35,"WMAPE","28.41%",BLUE)
        add_card(s,9.95,3.08,2.4,1.35,"R²","0.3381",TEAL)
        add_text(s,7.4,5.0,4.85,1.25,"오차 기반 정확도 71.59%\nR² 33.81%는 ‘정확도’가 아니라\n자치구별 차이를 설명한 비율",17,True,NAVY,PP_ALIGN.CENTER)

        s=prs.slides.add_slide(blank); add_header(s,"변수 영향과 해석","순열 중요도: 변수를 섞었을 때 MAE가 얼마나 증가하는가",7)
        s.shapes.add_picture(str(p3),Inches(.65),Inches(1.35),width=Inches(7.6),height=Inches(4.9))
        add_text(s,8.55,1.55,3.7,.4,"해석",19,True,TEAL)
        add_bullets(s,8.55,2.1,3.75,3.35,["의료보장 사업장수 지표의 영향이 가장 큼","사망자수 지표는 연도별 변동성이 커 불확실성도 큼","안전벨트 착용률의 음수 중요도는 ‘효과가 반대’라는 뜻이 아니라, 이 검증 표본에서 추가 예측 기여가 거의 없었다는 뜻","중요도는 인과관계를 의미하지 않음"],15)

        s=prs.slides.add_slide(blank); add_header(s,"사람이 읽는 안전점수와 등급","모델 예측과 별도로 안전기준표 가중치로 산정",8)
        s.shapes.add_picture(str(p4),Inches(.7),Inches(1.55),width=Inches(6.15),height=Inches(4.6))
        add_text(s,7.35,1.6,4.8,.45,"산정 방식",19,True,TEAL)
        add_bullets(s,7.35,2.2,4.75,2.9,["각 연도 25개 자치구 안에서 지표별 0~100점 정규화","위해 62.5% · 취약 12.5% · 의식 25% 적용","연도 내 상대 백분위로 1~5등급 부여","1등급이 상대적으로 안전, 5등급이 상대적으로 취약"],17)
        add_text(s,7.35,5.55,4.8,.65,"공식 지역안전지수 등급을 복제한 것이 아니라\n현재 확보 자료로 만든 초기모델용 상대등급",13,False,MID,PP_ALIGN.CENTER)

        s=prs.slides.add_slide(blank); add_header(s,"한계와 다음 단계","초기 모델은 개선 전 기준선을 만드는 단계",9)
        add_text(s,.8,1.4,5.6,.45,"현재 한계",20,True,RED)
        add_bullets(s,.95,2.0,5.35,3.7,["안전기준표 8개 항목 중 정확히 확보된 4개만 사용","경감지표 전체 제외로 원 기준표 구성과 차이","사고 ‘건수’를 예측하지만 인구·교통량을 별도 X로 사용하지 못함","최종 검증 표본이 25개 자치구로 작음","R² 0.3381로 설명되지 않는 변동이 큼"],17)
        line=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(6.55),Inches(1.45),Inches(.03),Inches(4.8));line.fill.solid();line.fill.fore_color.rgb=RGBColor(220,228,236);line.line.fill.background()
        add_text(s,6.95,1.4,5.2,.45,"다음 단계",20,True,TEAL)
        add_bullets(s,7.1,2.0,5.05,3.7,["1차 초기 모델을 기준 성능으로 고정","교통단속 CCTV·도로면적의 완전한 시계열 확보","재난약자수·교통법규위반건수 추가 수집","기준표 완성 모델과 확장 변수 모델을 분리 비교","연도별 재학습으로 성능 변화 모니터링"],17)
        add_text(s,.8,6.45,11.7,.45,"결론  |  안전기준표만으로도 1년 후 사고 건수의 기준 예측은 가능하지만, 최종 모델로는 추가 데이터 보강이 필요",16,True,NAVY,PP_ALIGN.CENTER)

        prs.save(OUTPUT)
    finally:
        shutil.rmtree(assets, ignore_errors=True)
    print(OUTPUT)


if __name__ == "__main__":
    build()
