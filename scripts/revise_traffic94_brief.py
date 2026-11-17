from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/M/Desktop/교통안전_94차.pptx')
OUT = ROOT / 'artifacts/presentations/교통안전_94차_수정본.pptx'
prs = Presentation(SOURCE)
DARK = RGBColor(95, 123, 122)
INK = RGBColor(48, 62, 62)
PALE = RGBColor(234, 245, 244)

def label(slide, x, y, w, h, value, size=22, bold=False):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = value
    p.font.name = '맑은 고딕'
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = INK

def clear_body(slide):
    for sh in list(slide.shapes):
        if sh.top >= Inches(3):
            el = sh._element
            el.getparent().remove(el)

def card(slide,x,y,w,h,title,body):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=PALE
    sh.line.color.rgb=RGBColor(180,202,200)
    label(slide,x+.25,y+.2,w-.5,.5,title,25,True)
    label(slide,x+.25,y+.9,w-.5,h-1.0,body,20)

def table(slide, x, y, widths, rows, height, size=18):
    t = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(height)).table
    for col,w in zip(t.columns,widths): col.width = Inches(w)
    for i,row in enumerate(rows):
        for j,value in enumerate(row):
            c=t.cell(i,j); c.text=str(value)
            c.margin_top=c.margin_bottom=Inches(.025)
            c.fill.solid(); c.fill.fore_color.rgb = DARK if i==0 else PALE if i%2 else RGBColor(255,255,255)
            for p in c.text_frame.paragraphs:
                p.font.name='맑은 고딕'; p.font.size=Pt(size); p.font.bold=i==0
                p.font.color.rgb=RGBColor(255,255,255) if i==0 else INK
                if j: p.alignment=PP_ALIGN.CENTER
    return t

s=prs.slides[1]; clear_body(s)
label(s,1.2,3.2,8.5,.5,'외부 기준표를 실제 데이터로 구현',26,True)
table(s,1.2,4,[6.8,1.7],[['원지표','가중치'],['교통사고 사망자 / 1만명','50%'],['도로면적당 CCTV (대체)','20%'],['재난약자 / 1만명','10%'],['안전벨트 착용률','8.99%'],['음주운전 경험률','8.70%'],['법규위반 다발지역 (대체)','2.31%']],4.8,21)
label(s,10.4,3.2,8.3,.5,'2025년 실제 검증 성능',26,True)
label(s,10.4,4.2,8.3,.75,'오차율 30.5%',40,True)
label(s,10.4,5.1,8.3,.9,'전체 실제 사고 건수 대비\n절대오차 합계 비율 (WMAPE)',22)
label(s,10.4,6.4,8.3,.7,'평균 약 416건 차이',34,True)
label(s,10.4,7.25,8.3,.9,'자치구별 평균 절대오차 415.8건\n다음 해 전체 교통사고 건수를 예측',22)
label(s,1.2,9.35,17.5,1,'6개 지표 → 100점 가중합 → Gradient Boosting\n외부 참고 실험의 성능이며, 어린이 안전등급을 예측하는 91·94차와는 직접 비교하지 않습니다.',19)

s=prs.slides[3]; clear_body(s)
card(s,1.2,3.2,8.6,2.85,'① 기준 통일 → 같은 구·같은 연도로 묶기',
     '서로 다른 파일의 자치구명·연도·숫자 표기를 통일\n학교·사고다발지역 자료를 구별·연도별로 집계\n사고·인구·학교·도로 자료를 [연도 + 자치구]로 연결')
card(s,10.2,3.2,8.6,2.85,'② 교통량 → 하루 단위 → 연평균',
     '지점번호를 자치구에 연결하고 방향별 차량 수 합산\n일교통량·평일/주말 평균·등하교 시간 비중 계산\n예: 등교 비중 = 7·8·9시 교통량 합 ÷ 일교통량')
card(s,1.2,6.35,8.6,2.85,'③ 규모 차이 보정 → 인구·차량 대비 비율',
     '구마다 인구·차량 수가 달라 단순 건수를 비율로 변환\n예: 사고 10건 ÷ 어린이 20,000명 × 10,000 = 5건\n어린이 1만명당 사고율·자동차 1만대당 사고 수 생성')
card(s,10.2,6.35,8.6,2.85,'④ 과거 흐름 추가 → 미래예측 입력 만들기',
     '현재값·전년값·전년 대비 증감·최근 3년 평균 생성\n예: 80 → 90 → 100이면 현재 100 / 증감 +10 / 평균 90\n결측은 학습 파이프라인에서 중앙값 등으로 처리')
label(s,1.3,9.55,17.4,.65,'결과: 25개 자치구 × 9년 = 225행 · 전체 패널 134컬럼',27,True)
label(s,1.3,10.25,17.4,.5,'숫자 예시는 이해용 가상값 · TOPIS는 24개 구 관측 · 134컬럼 전체가 최종 점수 변수는 아닙니다.',16)

config=pd.read_excel(ROOT/'data/processed/교통사고_94차_데이터.xlsx',sheet_name='94차모델설정')
s=prs.slides[4]; clear_body(s)
for h,x in [(2,1.2),(3,10.15)]:
    df=config[config['예측기간']==f'{h}년 후'].sort_values('순서')
    label(s,x,3.2,8.5,.45,f'{h}년형 · {len(df)}개 변수',26,True)
    rows=[['변수명','가중치']]+[[r['지표'].replace('_',' '),f"{r['가중치']*100:.2f}%"] for _,r in df.iterrows()]
    table(s,x,3.85,[7.15,1.5],rows,len(rows)*.285,14)
label(s,1.2,10.35,17.5,.45,'가중치 = Poisson 계수 절대값의 정규화 비중 · TOPIS 5개는 미래예측 보조변수 · 반올림으로 합계 차이 가능',15)

s=prs.slides[5]; clear_body(s)
card(s,1.2,3.2,5.5,3.6,'91차 | 현재값 그대로',
     '각 지표의 마지막 값을 유지\n\n예: 현재 자동차 수가 100이면\n2년·3년 뒤에도 100으로 예측\n→ 변화는 반영하지 않는 기준선')
card(s,7.25,3.2,5.5,3.6,'92~93차 | 변화 예측',
     '현재·전년·증감·3년평균 입력\n\n이동평균·선형·트리 모델 비교\n지표마다 잘 맞는 방법 선택\n→ 미래값을 각각 예측')
card(s,13.3,3.2,5.5,3.6,'94차 | 교통량 보강',
     '평균·평일·주말 교통량과\n등교·하교 시간 비중 추가\n\n개발기간에서 예측법 재선택\n→ 2년 13개·3년 10개 조정¹')
label(s,6.8,4.55,.45,.6,'→',26,True)
label(s,12.85,4.55,.45,.6,'→',26,True)
label(s,1.3,7.1,17.4,.5,'예측한 여러 지표를 하나의 안전등급으로 바꾸는 방법',25,True)
table(s,1.2,7.85,[5.87,5.87,5.86],[
    ['① 지표별 점수','② 중요도 반영','③ 최종 안전등급'],
    ['예측값을 구별로 비교\n위험 방향에 따라 5분위 점수화','Poisson 계수에서 얻은\n가중치를 곱해 점수 합산','종합점수를 구별로 비교\n1등급(안전)~5등급(위험)'],
],1.65,20)
label(s,1.3,9.8,17.4,.4,'선택: 2021~2024년 개발자료  →  최종 확인: 2025년 25개 자치구',23,True)
label(s,1.3,10.35,17.4,.4,'¹ 조정 수는 93→94차 기준 · 100은 이해용 예시 · 교통량은 미래값 예측을 돕는 보조입력',15)

s=prs.slides[6]; clear_body(s)
for sh in s.shapes:
    if sh.has_text_frame and sh.text=='성능 개선 · 91차 → 94차':
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if '성능 개선' in r.text: r.text='정확도 · 개발 결과와 최종 검증'
table(s,1.2,3.3,[6,3.2,4.2,4.2],[
    ['평가 구간','예측기간','91차 기준','94차 최종'],
    ['개발 · 2021~2024','2년 후','67%','78%'],
    ['개발 · 2021~2024','3년 후','56%','76%'],
    ['최종 검증 · 2025','2년 후','68% (17곳)','68% (17곳)'],
    ['최종 검증 · 2025','3년 후','48% (12곳)','68% (17곳)'],
],3.6,24)
label(s,1.3,7.2,17.4,.6,'68% = 25곳 중 17곳 정답  |  18곳이면 72%, 19곳이면 76%',28,True)
label(s,1.3,8.05,17.4,1.3,'왜 정확도가 그대로일까? 점수 오차가 줄어도 맞힌 등급 수가 같으면 68%로 유지됩니다.\n새로운 2025년에서 8곳의 등급은 아직 틀렸으며, 68%가 성능의 한계로 확정된 것은 아닙니다.',21)
label(s,1.3,9.5,17.4,.55,'3년형 개선: 12곳 → 17곳, 총 5곳 추가 정답 (+20%p)',27,True)
label(s,1.3,10.22,17.4,.5,'개발: 4개 연도 평균(25곳×4년) · 최종: 25곳, 한 곳당 4%p · 정확도 = 정확등급 일치율',16)

# Keep the summary consistent with the simplified performance page.
replacements={
    '정확등급 48 → 68%\n등급 오차 0.56 → 0.40':'정확등급 48 → 68%\n20%p 향상',
    '정확등급 68% 동일\n등급 오차 0.32 동일':'최종 정확등급 68%\n기준모델과 동일',
    '③ 일부 하락':'③ 개발 정확도',
    '3년 ±1등급 이내':'2021~2024년',
    '±1등급 이내 일치 96 → 92%\n모든 지표가 좋아진 건 아님':'2년형 78% · 3년형 76%\n최종 검증은 모두 68%',
    '📉':'📊',
    '13 / 10개':'미래예측',
    '예측법 바뀐 지표':'지표별 예측법 적용',
}
for sh in prs.slides[7].shapes:
    if sh.has_text_frame and sh.text in replacements:
        value=replacements[sh.text]
        p=sh.text_frame.paragraphs[0]
        if p.runs:
            p.runs[0].text=value
            for r in list(p.runs)[1:]: r.text=''
            for rest in list(sh.text_frame.paragraphs)[1:]:
                rest._p.getparent().remove(rest._p)
        else: p.text=value
OUT.parent.mkdir(parents=True,exist_ok=True)
prs.save(OUT)
check=Presentation(OUT)
assert len(check.slides)==8
tables=[sh.table for sh in check.slides[4].shapes if sh.has_table]
assert [len(t.rows)-1 for t in tables]==[21,18]
print(OUT)
