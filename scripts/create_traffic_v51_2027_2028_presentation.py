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


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
OUT = ROOT / "artifacts" / "presentations" / "교통사고_51차_2027_2028_직접예측_발표자료.pptx"
DESKTOP = Path.home() / "Desktop" / OUT.name


def main():
    performance = pd.read_excel(DATA, sheet_name="51차_장기예측성능")
    forecast = pd.read_excel(DATA, sheet_name="51차_2027_2028예측")
    target_data = pd.read_excel(DATA, sheet_name="51차_2년3년목표데이터")
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); blank=prs.slide_layouts[6]

    s=prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb=NAVY
    text(s,"SPOTFIT · MODEL 51",.82,.72,7,.3,12,RGBColor(180,215,211),True)
    text(s,"2027·2028 교통안전등급 직접예측",.82,1.46,11.8,.72,32,WHITE,True)
    text(s,"2025년 확보 데이터를 기준으로 2년 후와 3년 후를 각각 독립 예측",.84,2.40,11.4,.42,17,RGBColor(205,220,231))
    card(s,.85,3.40,3.55,1.65,"2027년 · 2년 후","56%","±1등급 87.2%",TEAL)
    card(s,4.72,3.40,3.55,1.65,"2028년 · 3년 후","55%","±1등급 87.0%",BLUE)
    card(s,8.59,3.40,3.55,1.65,"예측 방식","직접예측","연쇄예측 오차누적 방지",ORANGE)
    text(s,"1등급 고위험 · 5등급 저위험 · 서울 25개 자치구 상대 5분위",.85,6.55,11.6,.30,11,WHITE)

    s=prs.slides.add_slide(blank); header(s,"왜 모델을 바꿨나?","기존 51차는 다음 연도용이므로 2027·2028에 반복 적용하면 오차가 누적",2)
    rect(s,.75,1.48,5.75,4.65,PALE); text(s,"사용하지 않는 방식",1.03,1.80,5.2,.35,18,ORANGE,True)
    bullets(s,["2025 → 2026 예측","2026 예측값 → 2027 재예측","2027 예측값 → 2028 재예측","예측값이 다시 입력되어 오차가 연속 누적"],1.05,2.38,5.0,2.9,15)
    rect(s,6.82,1.48,5.75,4.65,MINT); text(s,"채택한 직접예측 방식",7.10,1.80,5.2,.35,18,TEAL,True)
    bullets(s,["2017 자료 → 2019 등급처럼 2년 간격 학습","2017 자료 → 2020 등급처럼 3년 간격 학습","2025 실제 자료 → 2027 직접예측","2025 실제 자료 → 2028 직접예측"],7.12,2.38,5.0,2.9,15)

    s=prs.slides.add_slide(blank); header(s,"두 모델의 학습 구조","목표연도 간격에 맞춰 Y를 완전히 다시 생성",3)
    rows=[["구분","2027 모델","2028 모델"],["예측 간격","2년 후","3년 후"],["학습쌍","2017→2019 … 2023→2025","2017→2020 … 2022→2025"],["운영 입력","2025년 실제 데이터","2025년 실제 데이터"],["운영 출력","2027 위험점수·등급","2028 위험점수·등급"],["검증 목표연도","2021~2025","2022~2025"]]
    table(s,rows,.72,1.55,11.9,4.80,[2.2,4.85,4.85],12)
    text(s,"Y = 목표연도의 종합위험점수와 상대 위험등급",2.0,6.50,9.2,.38,15,TEAL,True,PP_ALIGN.CENTER)

    s=prs.slides.add_slide(blank); header(s,"2027년 2년 후 직접예측 모델","점수 수준 예측과 변화량 예측을 결합",4)
    card(s,.78,1.45,3.55,1.55,"정확등급","56.0%","5개 검증연도 평균",TEAL)
    card(s,4.55,1.45,3.55,1.55,"±1등급","87.2%","한 등급 이내",BLUE)
    card(s,8.32,1.45,3.55,1.55,"평균오차","0.576","등급 단위",ORANGE)
    bullets(s,["구성 A: GBR depth 3으로 2년 후 위험점수 예측","구성 B: Ridge로 2년 후 점수 변화량 예측","최종 앙상블: 구성 A 75% + 구성 B 25%","2025년 실제 입력에서 2027년 순위와 등급 산출"],1.05,3.65,11.0,2.4,16)

    s=prs.slides.add_slide(blank); header(s,"2028년 3년 후 직접예측 모델","장기 변화에 대응하도록 서로 다른 점수예측기를 결합",5)
    card(s,.78,1.45,3.55,1.55,"정확등급","55.0%","4개 검증연도 평균",TEAL)
    card(s,4.55,1.45,3.55,1.55,"±1등급","87.0%","한 등급 이내",BLUE)
    card(s,8.32,1.45,3.55,1.55,"평균오차","0.580","등급 단위",ORANGE)
    bullets(s,["구성 A: GBR depth 3 위험점수 예측","구성 B: Ridge 위험점수 예측","최종 앙상블: 두 예측기 50%씩 결합","2025년 실제 입력에서 2028년 순위와 등급 산출"],1.05,3.65,11.0,2.4,16)

    s=prs.slides.add_slide(blank); header(s,"전처리 데이터 변경사항","1년 후 목표값 대신 2년 후·3년 후 목표값을 자치구별로 생성",6)
    rows=[["항목","내용"],["기준 데이터","2017~2025년 · 25개 자치구 · 225행"],["2년 후 Y","2년후_종합위험점수 · 2년후_위험등급 · 목표연도"],["3년 후 Y","3년후_종합위험점수 · 3년후_위험등급 · 목표연도"],["입력변수","원천 13개 + 5분위 점수 + 과거 추세 + 자치구 변수"],["미래 누수 방지","검증 목표연도보다 앞선 자료만 학습"],["운영 기준","2025년 실제 행 25개로 2027·2028 각각 예측"]]
    table(s,rows,.72,1.42,11.9,5.30,[2.4,9.5],12)

    s=prs.slides.add_slide(blank); header(s,"2027·2028 고위험 상위 10개 자치구","등급은 서울 25개 구 안에서 5개씩 배정되는 상대등급",7)
    rows=[["연도","순위","자치구","예측점수","등급"]]
    for year in (2027,2028):
        for _,r in forecast[forecast["예측연도"]==year].sort_values("예측위험순위").head(5).iterrows():
            rows.append([year,int(r["예측위험순위"]),r["자치구"],f"{r['예측위험점수']:.2f}",int(r["예측위험등급"])])
    table(s,rows,.72,1.30,7.55,5.72,[1.2,1.0,2.0,1.8,1.3],10.5)
    bullets(s,["순위가 높을수록 상대적 위험도가 높음","1등급은 매년 상위 위험 5개 구","2027과 2028 결과는 서로 다른 모델","정책 판단 시 49차 피해자율 전망도 함께 확인"],8.62,1.85,3.9,3.6,14)

    s=prs.slides.add_slide(blank); header(s,"성능 해석과 한계","예측기간이 길어질수록 정확도보다 등급 범위와 순위 안정성을 함께 확인",8)
    rows=[["모델","정확등급","±1등급","평균오차"],["기존 1년 후","60.0%","90.4%","0.496"],["2027 · 2년 후","56.0%","87.2%","0.576"],["2028 · 3년 후","55.0%","87.0%","0.580"]]
    table(s,rows,.82,1.48,7.2,3.60,[2.5,1.6,1.6,1.5],12)
    bullets(s,["장기예측 정확도 하락은 자연스러운 결과","2028 모델은 학습 가능한 연도쌍이 더 적음","공식 확정등급이 아닌 상대 위험 예측등급","새 실제연도 확보 시 반드시 재학습"],8.42,1.65,4.0,3.8,13.5)
    text(s,"결론: 1년 후 모델을 반복하지 않고 목표연도별 직접예측 모델을 사용",1.20,5.80,10.9,.55,18,WHITE,True,PP_ALIGN.CENTER).fill.solid(); s.shapes[-1].fill.fore_color.rgb=NAVY

    s=prs.slides.add_slide(blank); header(s,"최종 운영안","2027과 2028을 분리해 제공하고 매년 실제 데이터로 갱신",9)
    card(s,.75,1.45,3.75,1.55,"2027 제공값","2년 후 모델","점수·순위·5등급",TEAL)
    card(s,4.80,1.45,3.75,1.55,"2028 제공값","3년 후 모델","점수·순위·5등급",BLUE)
    card(s,8.85,1.45,3.75,1.55,"공통 입력","2025 실제값","예측값 재사용 없음",ORANGE)
    bullets(s,["대시보드에는 예측연도와 예측간격을 반드시 표시","등급만 단독 표시하지 않고 점수·순위·불확실성을 함께 제공","2026 실제 자료 확보 시 2027은 1년 후 모델로 재산출","2027 실제 자료 확보 시 2028도 1년 후 모델로 재산출"],1.05,3.65,11.0,2.35,16)

    OUT.parent.mkdir(parents=True,exist_ok=True); prs.save(OUT)
    try: DESKTOP.write_bytes(OUT.read_bytes()); print(DESKTOP)
    except OSError: print(f"DESKTOP_COPY_FAILED:{DESKTOP}")
    print(OUT)


if __name__=="__main__": main()
