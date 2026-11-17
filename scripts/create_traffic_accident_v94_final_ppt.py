from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from create_traffic_accident_v94_story_ppt import (
    BLUE, GREEN, INK, LIGHT, LINE, MID, NAVY, ORANGE, RED, SKY, TEAL, WHITE,
    add_bar_chart, box, header, metric, rich, text,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/presentations/교통사고_94차_최종모델_개발과정과성능.pptx"


def step(slide, x, y, no, title_, body, color):
    box(slide, x, y, 2.25, 1.75, WHITE, color)
    text(slide, x+.15, y+.15, .42, .35, no, 16, True, WHITE, PP_ALIGN.CENTER)
    circle = slide.shapes[-1]
    circle.fill.solid(); circle.fill.fore_color.rgb = color
    text(slide, x+.65, y+.18, 1.4, .3, title_, 14, True, color)
    text(slide, x+.2, y+.72, 1.85, .7, body, 10, False, INK, PP_ALIGN.CENTER)


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 cover
    s = prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
    text(s,.82,.72,3.0,.3,"TRAFFIC SAFETY · FINAL MODEL",11,True,RGBColor(125,211,252))
    text(s,.82,1.48,11.6,1.25,"교통사고 94차 최종모델\n데이터에서 성능까지",34,True,WHITE)
    text(s,.84,3.1,10.8,.75,"외부 안전기준표 참고 실험과는 분리하고,\n91차 기준모델에서 94차까지 성능을 끌어올린 실제 개발과정을 설명합니다.",17,False,RGBColor(203,213,225))
    for i,(label,c) in enumerate([("외부 참고 실험",SKY),("91차 기준선",MID),("94차 최종",GREEN)]):
        box(s,.84+i*3.1,5.28,2.65,.68,c,c); text(s,.84+i*3.1,5.48,2.65,.25,label,13,True,WHITE,PP_ALIGN.CENTER)
    text(s,.84,6.8,11.6,.25,"2017~2025 · 서울 25개 자치구 · 2년/3년 후 어린이 교통안전등급",10,False,RGBColor(148,163,184))

    # 2 narrative
    s=prs.slides.add_slide(blank); header(s,"발표의 핵심 이야기","외부 기준을 재현해 본 뒤, 내 프로젝트는 별도의 목표와 기준선으로 성능을 개선",2)
    cards=[
        (.7,"01","외부 참고","다른 프로젝트는 안전기준표의\n6개 원지표와 고정 가중치 사용",SKY),
        (3.8,"02","직접 구현","같은 취지의 실제 데이터를 수집해\n1차 모델을 만들고 성능 확인",ORANGE),
        (6.9,"03","내 기준선","내 프로젝트는 91차 Persistence를\n공식 기준모델로 설정",MID),
        (10.0,"04","최종 개선","지표별 미래예측과 교통량을 더해\n94차까지 개선",GREEN),
    ]
    for x,no,t,b,c in cards:
        box(s,x,1.65,2.65,3.65,WHITE,c); text(s,x+.2,1.92,.48,.35,no,15,True,c); text(s,x+.25,2.55,2.15,.35,t,18,True,NAVY,PP_ALIGN.CENTER); text(s,x+.25,3.35,2.15,1.0,b,12,False,INK,PP_ALIGN.CENTER)
    box(s,1.05,5.82,11.2,.68,LIGHT,LIGHT); text(s,1.3,6.02,10.7,.3,"성능 향상률은 동일한 타깃·검증을 쓰는 91차 ↔ 94차 사이에서만 계산합니다.",13,True,NAVY,PP_ALIGN.CENTER)

    # 3 external reference
    s=prs.slides.add_slide(blank); header(s,"외부 안전기준표를 실제로 구현해 본 참고 실험","94차의 전신이 아니라 ‘최소 기준이 어느 정도인지’ 확인한 별도 프로젝트",3)
    weights=[("교통사고 사망자/1만명","50.00",RED),("도로면적당 CCTV(대체)","20.00",GREEN),("재난약자/1만명","10.00",ORANGE),("안전벨트 착용률","8.99",TEAL),("음주운전 경험률","8.70",RED),("법규위반 다발지역(대체)","2.31",ORANGE)]
    for i,(name,w,c) in enumerate(weights):
        x=.7+(i%3)*4.15; y=1.5+(i//3)*1.18
        box(s,x,y,3.75,.92,WHITE,c); text(s,x+.18,y+.19,2.65,.3,name,11,True,INK); text(s,x+2.88,y+.17,.62,.32,w,16,True,c,PP_ALIGN.RIGHT)
    box(s,.7,4.25,5.8,1.55,LIGHT,LIGHT); text(s,.95,4.52,5.3,.3,"구현 방식",14,True,NAVY); text(s,.95,4.98,5.25,.55,"6개 원지표 → 자치구별 정규화 → 방향 변환 → 고정 가중합 100점 → Gradient Boosting",11,False,INK)
    box(s,6.8,4.25,5.8,1.55,RGBColor(255,247,237),RGBColor(253,186,116)); text(s,7.05,4.52,5.3,.3,"실제 성능",14,True,ORANGE); text(s,7.05,4.98,5.25,.55,"2025 전체 교통사고 건수: MAE 415.8 · WMAPE 30.5% · R² 0.016",12,True,INK)
    text(s,.85,6.35,11.65,.42,"이 결과보다 더 실용적인 장기 어린이 안전등급 모델을 만드는 것이 내 프로젝트의 출발점",13,True,BLUE,PP_ALIGN.CENTER)

    # 4 comparison boundary
    s=prs.slides.add_slide(blank); header(s,"안전기준표 1차와 내 모델은 이렇게 다릅니다","참고는 가능하지만 직접적인 ‘몇 % 향상’ 계산은 불가능",4)
    rows=[
        ("역할","외부 방법론 재현 실험","내 프로젝트 기준선","내 프로젝트 최종"),
        ("모델","안전기준표 1차","91차 Persistence","94차 TOPIS 개선"),
        ("예측대상","다음 해 전체 사고 건수","2·3년 후 어린이 안전등급","2·3년 후 어린이 안전등급"),
        ("입력 구조","가중점수 1개","21개/18개 지표의 현재값 유지","지표별 미래예측 + 교통량"),
        ("정량 향상률","94차와 공통척도 없음","94차의 직접 비교 기준","91차 대비 계산 가능"),
    ]
    widths=[2.0,3.1,3.1,3.1]; x0=.65; y0=1.55
    for r,row in enumerate(rows):
        x=x0
        for c,val in enumerate(row):
            fill=NAVY if r==0 else (LIGHT if c==0 else WHITE); col=WHITE if r==0 else (NAVY if c==0 else INK)
            box(s,x,y0+r*.86,widths[c],.78,fill,LINE,False); text(s,x+.12,y0+r*.86+.2,widths[c]-.24,.32,val,11,r==0 or c==0,col,PP_ALIGN.CENTER)
            x+=widths[c]
    box(s,1.0,6.15,11.25,.62,RGBColor(254,242,242),RGBColor(252,165,165)); text(s,1.25,6.32,10.75,.3,"따라서 ‘안전기준표보다 몇 % 좋다’가 아니라, 참고 실험의 한계를 넘어 장기 어린이 안전등급으로 목적을 확장했다고 설명합니다.",11,True,RED,PP_ALIGN.CENTER)

    # 5 goal and baseline
    s=prs.slides.add_slide(blank); header(s,"내 프로젝트의 기준: 91차 Persistence 모델","복잡한 모델이 정말 가치가 있는지 확인하기 위한 정직한 기준선",5)
    box(s,.7,1.55,3.4,3.9,WHITE,MID); text(s,.95,1.88,2.9,.35,"91차의 가정",18,True,MID,PP_ALIGN.CENTER); text(s,1.05,2.72,2.7,1.2,"“미래에도 마지막으로\n관측한 지표 값이\n그대로 유지된다”",18,True,NAVY,PP_ALIGN.CENTER); text(s,1.0,4.45,2.8,.5,"Poisson 가중치와 5분위\n등급 방식은 동일",11,False,INK,PP_ALIGN.CENTER)
    text(s,4.25,3.2,.55,.4,"→",25,True,MID,PP_ALIGN.CENTER)
    box(s,4.9,1.55,3.4,3.9,WHITE,BLUE); text(s,5.15,1.88,2.9,.35,"94차의 질문",18,True,BLUE,PP_ALIGN.CENTER); text(s,5.25,2.72,2.7,1.2,"“각 지표의 미래값을\n별도로 예측하면\n더 정확해지는가?”",18,True,NAVY,PP_ALIGN.CENTER); text(s,5.2,4.45,2.8,.5,"통계·선형·트리 모델과\nTOPIS 교통량 활용",11,False,INK,PP_ALIGN.CENTER)
    text(s,8.45,3.2,.55,.4,"→",25,True,MID,PP_ALIGN.CENTER)
    box(s,9.1,1.55,3.5,3.9,RGBColor(240,253,244),GREEN); text(s,9.35,1.88,3.0,.35,"검증 원칙",18,True,GREEN,PP_ALIGN.CENTER); rich(s,9.48,2.62,2.7,2.1,["같은 2025년", "같은 25개 자치구", "같은 2·3년 예측기간", "같은 등급·점수 지표"],12)
    text(s,.85,6.15,11.65,.4,"91차가 단순 기준선이기 때문에 94차의 개선폭을 과장 없이 설명할 수 있습니다.",13,True,NAVY,PP_ALIGN.CENTER)

    # 6 sources
    s=prs.slides.add_slide(blank); header(s,"94차 데이터는 어디에서 왔나","공공 통계를 자치구×연도 단위로 결합",6)
    sources=[
        ("교통사고", "TAAS·서울시 통계\n전체/어린이/보행/보호구역/음주/자전거", RED),
        ("사고다발지역", "TAAS 다발지역\n전체·법규위반·어린이·자전거", ORANGE),
        ("인구", "주민등록 연령별 인구\n어린이·아동·청소년·노인", BLUE),
        ("학교", "교육통계 학교별 일람표\n유치원·초·중·고", TEAL),
        ("교통환경", "서울시 자동차·도로·CCTV·통행속도", MID),
        ("안전행태", "질병관리청 지역사회건강조사\n안전벨트·음주·사고중독", SKY),
        ("교통량", "서울 TOPIS 월별 조사자료\n2017~2025, 108개 파일", GREEN),
    ]
    for i,(name,body,c) in enumerate(sources):
        x=.65+(i%4)*3.12; y=1.48+(i//4)*2.18
        box(s,x,y,2.82,1.65,WHITE,c); text(s,x+.2,y+.22,2.42,.3,name,15,True,c,PP_ALIGN.CENTER); text(s,x+.2,y+.75,2.42,.55,body,10,False,INK,PP_ALIGN.CENTER)
    box(s,9.98,3.66,2.82,1.65,LIGHT,LIGHT); text(s,10.18,3.93,2.42,.3,"원천 규모",15,True,NAVY,PP_ALIGN.CENTER); text(s,10.18,4.48,2.42,.5,"기반 46개 + TOPIS 108개\n= 총 154개 파일",11,True,INK,PP_ALIGN.CENTER)
    text(s,.8,6.25,11.7,.35,"기간 2017~2025 · 25개 자치구 · 원천별 행정구역명과 연도를 표준화해 결합",12,True,NAVY,PP_ALIGN.CENTER)

    # 7 preprocessing
    s=prs.slides.add_slide(blank); header(s,"Raw 데이터는 이렇게 전처리했습니다","서로 다른 파일을 225행×134컬럼의 일관된 패널로 변환",7)
    steps=[
        ("01","표준화","자치구명·연도·숫자형\n결측/특수기호 정리",MID),
        ("02","집계","학교·사고다발지점·CCTV를\n자치구×연도로 합산",ORANGE),
        ("03","TOPIS","0~23시 합계, 평일/주말,\n등교 07~09·하교 14~18",GREEN),
        ("04","파생변수","어린이 1만명당·자동차 1만대당·\n도로면적당 비율 계산",BLUE),
        ("05","시계열","현재·전년·증감·최근 3년 평균을\n미래예측 입력으로 생성",TEAL),
    ]
    for i,(no,t,b,c) in enumerate(steps):
        x=.52+i*2.55; box(s,x,1.72,2.18,3.38,WHITE,c); text(s,x+.18,1.95,.45,.3,no,14,True,c); text(s,x+.25,2.55,1.68,.35,t,16,True,NAVY,PP_ALIGN.CENTER); text(s,x+.25,3.35,1.68,.85,b,10,False,INK,PP_ALIGN.CENTER)
        if i<4: text(s,x+2.16,3.18,.38,.35,"→",20,True,MID,PP_ALIGN.CENTER)
    metric(s,.8,5.62,"최종 패널","225행","9년 × 25개 자치구",BLUE,2.6); metric(s,3.65,5.62,"전체 컬럼","134개","원천+파생+교통량",TEAL,2.6); metric(s,6.5,5.62,"TOPIS 집계","216행","24개 관측 자치구",GREEN,2.6); metric(s,9.35,5.62,"재현 차이","5.82×10⁻¹¹","Raw↔Processed",ORANGE,2.8)

    # 8 traffic variables
    s=prs.slides.add_slide(blank); header(s,"94차에서 새로 보강한 교통량 5개","사고의 ‘노출량’과 등·하교 시간대 통행 패턴을 반영",8)
    vals=[("평균 일교통량","하루 전체 통행 노출",BLUE),("평일 평균 일교통량","통학·통근 중심 패턴",TEAL),("주말 평균 일교통량","주말 이동 패턴 분리",ORANGE),("등교시간 비중","07~09시 / 일교통량",SKY),("하교시간 비중","14~18시 / 일교통량",GREEN)]
    for i,(name,desc,c) in enumerate(vals):
        x=.75+(i%3)*4.15; y=1.65+(i//3)*1.85
        box(s,x,y,3.7,1.4,WHITE,c); text(s,x+.2,y+.24,3.3,.3,name,15,True,c); text(s,x+.2,y+.76,3.3,.3,desc,11,False,INK)
    box(s,8.95,3.5,3.5,1.4,LIGHT,LIGHT); text(s,9.2,3.75,3.0,.3,"사용 방식",14,True,NAVY); text(s,9.2,4.2,3.0,.42,"최종 점수에 직접 더하지 않고\n핵심 지표의 미래값 예측에 투입",10,False,INK)
    box(s,.9,5.55,11.55,.82,RGBColor(240,253,244),RGBColor(134,239,172)); text(s,1.15,5.78,11.05,.35,"핵심: 94차의 변화는 ‘점수 항목 5개 추가’가 아니라 ‘미래값 예측 정확도를 높이는 보조정보 추가’입니다.",12,True,GREEN,PP_ALIGN.CENTER)

    # 9 final columns
    s=prs.slides.add_slide(blank); header(s,"최종 모델에는 어떤 컬럼이 들어갔나","2년형 21개 · 3년형 18개, 사고·노출·환경·행태를 함께 반영",9)
    groups=[
        ("노출·인구 보정",["어린이1만명당 자동차", "자동차1만대당 전체사고", "자동차 등록대수", "특수차 등록대수"],BLUE),
        ("사고·다발지역",["전체/법규위반 사고다발", "보호구역·보행어린이 사고", "자전거 다발지역", "사망·부상·경상 지표"],RED),
        ("도로·교통환경",["도로연장 또는 도로율", "승용차 평균속도", "전체사고다발 지점수", "TOPIS 교통량 5개(보조)"],ORANGE),
        ("학교·안전행태",["유치원/중학교 사립수", "고등학교 평균학생수", "안전벨트 착용률", "연간 사고·중독 경험률"],TEAL),
    ]
    for i,(name,items,c) in enumerate(groups):
        x=.7+(i%2)*6.1; y=1.5+(i//2)*2.45
        box(s,x,y,5.75,2.05,WHITE,c); text(s,x+.25,y+.25,5.25,.3,name,16,True,c); rich(s,x+.35,y+.75,5.0,1.05,items,10)
    text(s,.85,6.55,11.65,.3,"상세 변수 전체 목록과 가중치는 다음 장에 제시 · 등급 방향: 1=가장 안전, 5=가장 위험",10,False,MID,PP_ALIGN.CENTER)

    # 10 weights
    s=prs.slides.add_slide(blank); header(s,"어떤 변수가 더 크게 반영됐나","Poisson 계수의 절대값을 정규화해 예측기간별 가중치 산정",10)
    top2=[("어린이1만명당 자동차",15.64),("자동차1만대당 전체사고",13.27),("특수차 등록대수",11.80),("유치원 사립수",8.49),("전체사고다발 발생",7.89),("승용차 평균속도",7.55)]
    top3=[("어린이1만명당 자동차",18.03),("자동차1만대당 전체사고",16.07),("특수차 등록대수",13.37),("전체사고다발 발생",9.90),("승용차 평균속도",8.76),("중학교 사립수",6.43)]
    for j,(title_,rows,c,x) in enumerate([("2년형 상위 가중치",top2,BLUE,.7),("3년형 상위 가중치",top3,GREEN,6.75)]):
        box(s,x,1.48,5.85,4.75,WHITE,c); text(s,x+.25,1.75,5.35,.35,title_,17,True,c,PP_ALIGN.CENTER)
        for i,(name,val) in enumerate(rows):
            y=2.42+i*.56; text(s,x+.3,y,3.45,.25,name,10,False,INK); box(s,x+3.45,y+.03,val/20*1.75,.18,c,c,False); text(s,x+5.0,y-.01,.52,.25,f"{val:.2f}%",9,True,c,PP_ALIGN.RIGHT)
    text(s,.9,6.55,11.55,.3,"가중치가 크다고 인과효과가 크다는 뜻은 아니며, 등급 점수 계산에서의 상대적 반영 비중입니다.",10,False,MID,PP_ALIGN.CENTER)

    # 11 model process
    s=prs.slides.add_slide(blank); header(s,"94차 모델은 어떻게 만들어졌나","지표별 최적 미래예측 → 가중점수 → 상대등급 → 독립검증",11)
    flow=[("1","학습 입력","현재·전년·증감·3년평균\n자치구 + 교통량",BLUE),("2","후보 탐색","Persistence·Mean·Ridge\nRF·ExtraTrees·GBR",TEAL),("3","지표별 선택","2년형 13개·3년형 10개\n예측법 변경",ORANGE),("4","점수화","미래값 5분위 변환\nPoisson 가중합",RED),("5","안전등급","연도별 1~5등급\n1=안전, 5=위험",GREEN)]
    for i,(no,t,b,c) in enumerate(flow):
        x=.52+i*2.55; box(s,x,1.55,2.18,3.7,WHITE,c); text(s,x+.2,1.82,.4,.3,no,15,True,c); text(s,x+.25,2.55,1.68,.35,t,16,True,NAVY,PP_ALIGN.CENTER); text(s,x+.25,3.35,1.68,.85,b,10,False,INK,PP_ALIGN.CENTER)
        if i<4:text(s,x+2.16,3.08,.38,.35,"→",20,True,MID,PP_ALIGN.CENTER)
    box(s,1.0,5.72,11.25,.72,LIGHT,LIGHT); text(s,1.25,5.94,10.75,.3,"모델 선택에는 개발기간만 사용하고, 2025년 25개 자치구는 최종 독립검증으로 분리",12,True,NAVY,PP_ALIGN.CENTER)

    # 12 performance
    s=prs.slides.add_slide(blank); header(s,"최종 성능: 91차 기준선에서 얼마나 좋아졌나","동일한 2025 독립검증으로 비교",12)
    add_bar_chart(s,.65,1.45,5.75,3.5,["2년 정확","2년 ±1","3년 정확","3년 ±1"],[("91차",[.68,1,.48,.96]),("94차",[.68,1,.68,.92])],1,True)
    add_bar_chart(s,6.75,1.45,5.85,3.5,["2년 등급오차","2년 점수MAE","3년 등급오차","3년 점수MAE"],[("91차",[.32,2.991,.56,3.754]),("94차",[.32,2.897,.40,3.401])],4.2,False)
    metric(s,.8,5.25,"3년 정확등급","+20%p","48% → 68%",GREEN,2.65); metric(s,3.7,5.25,"상대 정확도","+41.7%","3년형 기준",GREEN,2.65); metric(s,6.6,5.25,"3년 등급오차","−28.6%","0.56 → 0.40",GREEN,2.65); metric(s,9.5,5.25,"3년 점수 MAE","−9.4%","3.754 → 3.401",GREEN,2.65)

    # 13 honest interpretation
    s=prs.slides.add_slide(blank); header(s,"성능을 어떻게 해석해야 하나","좋아진 부분과 좋아지지 않은 부분을 함께 제시",13)
    box(s,.75,1.55,3.75,4.6,RGBColor(240,253,244),RGBColor(134,239,172)); text(s,1.0,1.9,3.25,.35,"분명한 개선",18,True,GREEN,PP_ALIGN.CENTER); rich(s,1.1,2.62,3.05,2.7,["3년 정확등급 +20%p", "3년 등급오차 28.6% 감소", "3년 점수 MAE 9.4% 감소", "2년 점수 MAE 3.1% 감소"],12)
    box(s,4.8,1.55,3.75,4.6,RGBColor(255,247,237),RGBColor(253,186,116)); text(s,5.05,1.9,3.25,.35,"유지·하락",18,True,ORANGE,PP_ALIGN.CENTER); rich(s,5.15,2.62,3.05,2.7,["2년 정확등급 68%로 동일", "2년 평균등급오차 동일", "3년 ±1등급 96→92%", "모든 지표가 개선된 것은 아님"],12)
    box(s,8.85,1.55,3.75,4.6,RGBColor(239,246,255),RGBColor(147,197,253)); text(s,9.1,1.9,3.25,.35,"최종 판단",18,True,BLUE,PP_ALIGN.CENTER); rich(s,9.2,2.62,3.05,2.7,["장기 3년형 개선이 큼", "독립검증에서 일반화 확인", "95차는 3년 정확도 56%로 악화", "따라서 94차를 최종 채택"],12)
    text(s,.85,6.55,11.65,.3,"94차는 완벽한 모델이 아니라, 91차보다 장기 등급 예측을 실질적으로 개선한 의사결정 보조모델입니다.",11,True,NAVY,PP_ALIGN.CENTER)

    # 14 conclusion
    s=prs.slides.add_slide(blank); header(s,"최종 결론","외부 기준을 참고하되, 내 데이터와 내 기준선으로 검증한 94차 최종모델",14)
    metric(s,.75,1.55,"Raw 데이터","154개","기반 46 + TOPIS 108",BLUE,2.75); metric(s,3.72,1.55,"Processed","225 × 134","자치구×연도 패널",TEAL,2.75); metric(s,6.69,1.55,"최종 변수","21 / 18개","2년형 / 3년형",ORANGE,2.75); metric(s,9.66,1.55,"3년 정확등급","68%","91차 대비 +20%p",GREEN,2.75)
    box(s,.75,3.4,11.85,2.45,LIGHT,LIGHT)
    rich(s,1.05,3.72,11.2,1.75,[
        "안전기준표 1차는 다른 프로젝트의 방법을 실제 데이터로 구현해 본 외부 참고 실험입니다.",
        "내 프로젝트는 91차 Persistence를 기준으로, 공공데이터 수집·파생변수·지표별 미래예측·TOPIS 교통량을 단계적으로 추가했습니다.",
        "같은 독립검증에서 특히 3년 후 정확등급을 48%에서 68%로 높였고, 일반화되지 않은 95차는 채택하지 않았습니다."
    ],13)
    box(s,1.45,6.25,10.45,.62,NAVY,NAVY); text(s,1.7,6.43,9.95,.28,"최종 대표모델: 교통사고 94차 TOPIS 교통량 추가모델",14,True,WHITE,PP_ALIGN.CENTER)

    # 15 appendix
    s=prs.slides.add_slide(blank); header(s,"부록: 재현 근거","발표의 수치와 과정은 아래 산출물에서 확인 가능",15)
    refs=[
        "안전기준표 1차: models/교통사고/1차/model_info.json · 모델평가_및_2026예측.xlsx",
        "91차 기준선: models/교통사고/91차.어린이교통사고수_Persistence모델/",
        "94차 데이터: data/raw/94차 모델 데이터/ · data/processed/교통사고_94차_데이터.xlsx",
        "94차 모델: models/교통사고/94차.11.13.개선/model_info.json",
        "재현 검증: models/recommender/v1/traffic_accident_v93_v94_pipeline_report.json",
        "전체 파이프라인: scripts/rebuild_traffic_v93_v94_to_v1.py",
    ]
    box(s,.8,1.55,11.7,4.7,LIGHT,LIGHT); rich(s,1.15,2.0,11.0,3.8,refs,13)
    text(s,.85,6.55,11.65,.3,"작성 기준일 2026-11-16 · 성능 수치는 2025년 25개 자치구 독립검증",10,False,MID,PP_ALIGN.CENTER)

    OUT.parent.mkdir(parents=True,exist_ok=True); prs.save(OUT); print(OUT)


if __name__ == "__main__":
    main()
