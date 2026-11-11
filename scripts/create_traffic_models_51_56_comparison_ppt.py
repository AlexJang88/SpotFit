from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from scripts.create_traffic_v51_advanced_presentation import (
    BLUE, GRAY, INK, MINT, NAVY, ORANGE, PALE, TEAL, WHITE,
    bullets, card, header, rect, table, text,
)


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"presentations"/"교통사고_51차부터_56차_모델개선_비교.pptx"
DESKTOP=Path.home()/"Desktop"/OUT.name

PERFORMANCE=[
    [51,"기준 장기 회귀앙상블",56.0,55.0,55.5,87.2,87.0,"-"],
    [52,"전체 변수군·분류기법·순위앙상블",56.8,57.0,56.9,90.4,93.0,"기법 변경"],
    [53,"후진 제거",55.2,59.0,57.1,90.4,91.0,"보행어린이다발 부상신고 제거"],
    [54,"기간별 최우수 결합",56.8,59.0,57.9,90.4,91.0,"2년=52차·3년=53차"],
    [55,"2년 모델 후진 제거",58.4,59.0,58.7,90.4,91.0,"점수_증감1 제거"],
    [56,"2년 모델 추가 경량화",59.2,59.0,59.1,91.2,91.0,"청소년 인구수 제거"],
]


def main():
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); blank=prs.slide_layouts[6]
    s=prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
    text(s,"SPOTFIT · TRAFFIC MODEL JOURNEY",.82,.72,8,.3,12,RGBColor(180,215,211),True)
    text(s,"교통사고 51차 → 56차 성능개선",.82,1.48,11.8,.70,34,WHITE,True)
    text(s,"전체 변수 추가 탐색 · 기법 전환 · 기간별 후진 제거",.84,2.42,11.2,.40,18,RGBColor(205,220,231))
    card(s,.85,3.42,3.55,1.65,"51차 평균 정확도","55.5%","2년 56.0% · 3년 55.0%",BLUE)
    card(s,4.72,3.42,3.55,1.65,"56차 평균 정확도","59.1%","2년 59.2% · 3년 59.0%",TEAL)
    card(s,8.59,3.42,3.55,1.65,"총 개선 폭","+3.6%p","±1등급 평균 91.1%",ORANGE)
    text(s,"2025년 실제 데이터 → 2027년 2년 후 · 2028년 3년 후 직접예측",.85,6.55,11.6,.30,11,WHITE)

    s=prs.slides.add_slide(blank); header(s,"개선 전략","추가가 막힐 때 기법을 확장하고, 이후 불필요 변수를 제거",2)
    steps=[("51차","기준선","회귀 점수예측 앙상블",BLUE),("추가 탐색","정체","전체 후보 단일·쌍 추가",GRAY),("52차","돌파","분류기 + 순위앙상블",TEAL),("53~56차","정제","후진 제거·기간별 최적화",ORANGE)]
    for i,(v,label,note,color) in enumerate(steps):
        x=.72+i*3.12; rect(s,x,1.65,2.72,3.55,MINT if i in (2,3) else PALE); text(s,v,x+.18,1.92,2.36,.42,20,color,True,PP_ALIGN.CENTER); text(s,label,x+.18,2.52,2.36,.35,14,NAVY,True,PP_ALIGN.CENTER); text(s,note,x+.25,3.15,2.22,1.15,12,INK,False,PP_ALIGN.CENTER)
        if i<3: text(s,"→",x+2.74,3.05,.38,.35,22,TEAL,True,PP_ALIGN.CENTER)
    text(s,"채택 규칙: 정확등급 우선 → ±1등급 → 평균오차 → 최저연도 안정성",1.25,5.88,10.8,.45,15,TEAL,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"51차 기준 모델","변수 추가만으로는 평균 55.5%를 넘지 못함",3)
    card(s,.75,1.45,3.55,1.55,"2년 후","56.0%","±1등급 87.2%",BLUE); card(s,4.55,1.45,3.55,1.55,"3년 후","55.0%","±1등급 87.0%",BLUE); card(s,8.35,1.45,3.55,1.55,"평균","55.5%","평균오차 0.578",ORANGE)
    bullets(s,["확보된 추가 후보를 전부 Ridge 선별에 투입","상위 18개는 51차 실제 앙상블로 재검증","상위 10개 변수의 2개 동시 추가까지 검증","최고 단일 추가 54.8%, 최고 쌍 추가 54.4%로 기준 미달","51차 구조에서 변수만 추가하는 방식의 한계 확인"],1.00,3.52,11.1,2.5,15)

    s=prs.slides.add_slide(blank); header(s,"52차 · 모델 기법 전환","회귀 중심에서 직접 등급분류와 위험순위 앙상블로 확장",4)
    rect(s,.72,1.48,5.88,4.65,PALE); text(s,"2년 후 모델",1.00,1.78,5.3,.35,18,NAVY,True)
    bullets(s,["ExtraTrees + LogisticRegression","상위 25개 추가 후보·자치구 변수 활용","위험순위 앙상블 65:35","정확등급 56.8% · ±1등급 90.4%"],1.02,2.35,5.1,2.8,14)
    rect(s,6.75,1.48,5.88,4.65,MINT); text(s,"3년 후 모델",7.03,1.78,5.3,.35,18,TEAL,True)
    bullets(s,["선형 SVM + ExtraTrees","상위 10개·25개 후보 변수군 결합","위험순위 앙상블 65:35","정확등급 57.0% · ±1등급 93.0%"],7.05,2.35,5.1,2.8,14)
    text(s,"52차 평균 정확등급 56.9% · 51차 대비 +1.4%p",2.15,6.40,9.0,.42,16,TEAL,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"53차·54차 · 예측기간별 최적화","한 모델의 평균만 보지 않고 2년·3년 최우수 조합을 분리",5)
    rows=[["차수","변경","2년 정확도","3년 정확도","평균"],["52차","분류 순위앙상블","56.8%","57.0%","56.9%"],["53차","부상신고 변수 제거","55.2%","59.0%","57.1%"],["54차","2년=52차, 3년=53차","56.8%","59.0%","57.9%"]]
    table(s,rows,.72,1.52,11.9,3.35,[1.4,5.0,1.8,1.8,1.9],12)
    bullets(s,["53차는 평균은 개선됐지만 2년 성능이 하락","2년과 3년은 독립 모델이므로 각 기간의 최우수 모델을 선택","54차에서 두 기간 모두 51차보다 높은 성능 확보"],1.0,5.22,11.0,1.35,14)

    s=prs.slides.add_slide(blank); header(s,"55차·56차 · 2년 모델 후진 제거","중복·불안정 변수를 제거하면서 정확도와 오차를 함께 개선",6)
    rows=[["차수","제거 변수","2년 정확도","±1등급","평균오차","최저연도"],["54차","-","56.8%","90.4%","0.528","48%"],["55차","점수_증감1","58.4%","90.4%","0.512","52%"],["56차","청소년(9~24세) 인구수","59.2%","91.2%","0.496","52%"]]
    table(s,rows,.62,1.50,12.1,3.55,[1.2,4.2,1.7,1.6,1.6,1.8],11.5)
    text(s,"삭제 후 정확도 +2.4%p · 평균오차 0.032 감소 · 최저연도 +4%p",1.18,5.50,10.9,.52,18,TEAL,True,PP_ALIGN.CENTER)
    text(s,"3년 모델은 추가 제거 시 개선되지 않아 59.0% 상태로 유지",2.15,6.25,9.0,.35,13,GRAY,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"51차부터 56차 성능 비교","정확한 등급 일치율 기준",7)
    rows=[["차수","핵심 변경","2년","3년","평균","±1 평균"]]
    for v,method,h2,h3,avg,w2,w3,change in PERFORMANCE: rows.append([f"{v}차",method,f"{h2:.1f}%",f"{h3:.1f}%",f"{avg:.1f}%",f"{(w2+w3)/2:.1f}%"])
    table(s,rows,.42,1.20,12.5,5.75,[1.0,5.0,1.6,1.6,1.6,1.7],10)

    s=prs.slides.add_slide(blank); header(s,"56차 최종 구조","2년·3년 모델을 독립 최적화한 순위 앙상블",8)
    rect(s,.75,1.50,5.75,4.85,MINT); text(s,"2027년 · 2년 후",1.02,1.80,5.2,.40,19,TEAL,True)
    bullets(s,["ExtraTrees + LogisticRegression","위험순위 앙상블","점수_증감1 제거","청소년(9~24세) 인구수 제거","정확등급 59.2%","±1등급 91.2%"],1.02,2.35,5.0,3.4,14)
    rect(s,6.82,1.50,5.75,4.85,PALE); text(s,"2028년 · 3년 후",7.10,1.80,5.2,.40,19,NAVY,True)
    bullets(s,["선형 SVM + ExtraTrees","위험순위 앙상블","53차에서 불필요 변수 제거","이후 추가 제거는 성능 개선 없음","정확등급 59.0%","±1등급 91.0%"],7.10,2.35,5.0,3.4,14)

    s=prs.slides.add_slide(blank); header(s,"개선 과정에서 확인한 사실","변수 수보다 목표와 평가방식에 맞는 구조가 중요",9)
    bullets(s,["51차 회귀 앙상블에는 새 변수를 단순 추가해도 개선되지 않음","직접 등급분류 후 위험순위를 보정하자 장기등급 정확도가 상승","2년·3년 모델은 같은 변수 제거가 서로 다른 영향을 줌","예측기간별로 모델을 독립 선택해야 양쪽 성능을 모두 유지 가능","후진 제거로 정확도·인접등급·평균오차·최저연도가 동시에 개선"],1.0,1.55,11.2,3.5,17)
    text(s,"결론  변수 추가 → 기법 전환 → 기간별 선택 → 후진 제거 순서가 유효",1.15,5.72,11.0,.62,19,WHITE,True,PP_ALIGN.CENTER).fill.solid(); s.shapes[-1].fill.fore_color.rgb=NAVY

    s=prs.slides.add_slide(blank); header(s,"최종 판단과 주의사항","현재 실험상 최고는 56차, 최종 운영모델 확정은 별도",10)
    card(s,.75,1.45,3.75,1.55,"2년 후","59.2%","51차 대비 +3.2%p",TEAL); card(s,4.80,1.45,3.75,1.55,"3년 후","59.0%","51차 대비 +4.0%p",TEAL); card(s,8.85,1.45,3.75,1.55,"평균","59.1%","51차 대비 +3.6%p",ORANGE)
    bullets(s,["현재 확보 데이터와 동일 순차검증에서의 최고 결과","많은 후보를 비교했으므로 새 실제연도 독립검증이 반드시 필요","발표용 51차 API 연결은 그대로 유지하고 56차로 자동 교체하지 않음","최종모델 확정 시 recommender/v1 연결 코드를 별도로 변경"],1.0,3.65,11.2,2.3,15)

    OUT.parent.mkdir(parents=True,exist_ok=True); prs.save(OUT)
    try: DESKTOP.write_bytes(OUT.read_bytes()); print(DESKTOP)
    except OSError: print(f"DESKTOP_COPY_FAILED:{DESKTOP}")
    print(OUT)


if __name__=="__main__": main()
