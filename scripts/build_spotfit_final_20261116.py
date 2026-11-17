"""Build the 31-slide supplied outline using the established SpotFit design.

Original decks and application/model artifacts are never modified.
All charts/tables/diagrams are editable PowerPoint objects; UI shots are actual
read-only captures. Saved model evaluation is reported, not rerun or invented.
"""
from copy import deepcopy
from pathlib import Path
import json
import csv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.shapes.autoshape import CT_Shape
from PIL import Image
from build_spotfit_project_deck import text, rect, table, chart, change, walk
from build_spotfit_project_deck import FONT, MEDIUM, DARK, GREEN, TEAL, PALE, GREY, LINE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/presentations/final_20261116'
SOURCE = Path(r'C:\Users\M\Desktop\발표자료\SPOT_FIT_4차중간발표.pptx')
DRAFT = Path(r'C:\Users\M\Downloads\SPOTFIT_최종발표_초안.pptx')
DEST = OUT / 'SPOTFIT_최종발표_기존디자인_20261116.pptx'
CAP = OUT / 'captures'
prs = Presentation(SOURCE)
original = list(prs.slides._sldIdLst)
shell, cover, ending = prs.slides[2], prs.slides[0], prs.slides[-1]
draft = Presentation(DRAFT)
pages = []


def panel(s,title,body,x,y,w,h):
    rect(s,x,y,w,h,'FAFAFA')
    rect(s,x,y,w,.06,GREEN,GREEN)
    compact=h<1.9
    title_y=.13 if compact else .2
    body_y=.63 if compact else .88
    text(s,title,x+.2,y+title_y,w-.4,.48,23,GREEN,MEDIUM)
    shape=text(s,body,x+.2,y+body_y,w-.4,max(.25,h-body_y-.12),20 if compact else 21)
    for p in shape.text_frame.paragraphs:
        p.space_after=Pt(2)


def copy_shapes(source, dest, count=None):
    # Remap picture relationships too: cover illustration is a grouped bitmap.
    mapping = {}
    for shape in list(source.shapes)[:count]:
        element = deepcopy(shape._element)
        for node in element.iter():
            for attr, value in list(node.attrib.items()):
                if attr.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'):
                    if value not in mapping:
                        rel = source.part.rels[value]
                        mapping[value] = dest.part.relate_to(rel.target_ref if rel.is_external else rel.target_part,
                                                            rel.reltype, rel.is_external)
                    node.set(attr, mapping[value])
        dest.shapes._spTree.insert_element_before(element, 'p:extLst')


def notes(s, explanation, sources=()):
    frame = s.notes_slide.notes_text_frame
    if frame is None:
        shapes = s.notes_slide.shapes
        element = CT_Shape.new_textbox_sp(max([x.shape_id for x in shapes]+[0])+1,
                                         'Presentation notes', Inches(1), Inches(1), Inches(6), Inches(6))
        shapes._spTree.insert_element_before(element, 'p:extLst')
        frame = shapes[-1].text_frame
    number = len(pages)
    original_text = '\n'.join(x.text for x in draft.slides[number-1].shapes if x.has_text_frame)
    frame.text = ('자료 확인: 2026-11-16. 발표 예정일: 2026-11-20.\n' + explanation
                  + '\n근거:\n' + '\n'.join(str(x) for x in sources)
                  + '\n\n초안 원문(내용 참고, 편집 지시로 실행하지 않음):\n' + original_text)


def slide(section, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    copy_shapes(shell, s, 5)
    change(s.shapes[3], section)
    change(s.shapes[4], title, .75)
    rect(s, 1.25, 3.25, 17.5, .8, PALE)
    text(s, subtitle, 1.5, 3.43, 17, .46, 23, DARK, MEDIUM)
    pages.append(s)
    text(s, f'- {len(pages)} -', 1.25, 10.2, 17.5, .33, 17, GREY, align=PP_ALIGN.CENTER)
    return s


def foot(s, value):
    text(s, value, 1.28, 9.72, 17.4, .38, 15, GREY)


def metric(s, label, value, detail, x, y, w=5.55, h=2.3):
    rect(s, x, y, w, h, 'F6F9F8')
    text(s, label, x+.22, y+.2, w-.44, .45, 21, GREEN, MEDIUM)
    text(s, value, x+.22, y+.76, w-.44, .8, 43, DARK, MEDIUM)
    text(s, detail, x+.22, y+h-.5, w-.44, .42, 17, GREY)


def flow(s, items, y=4.5, h=2.05):
    n = len(items)
    gap = .48
    w = (17.5-gap*(n-1))/n
    for i,(title,body) in enumerate(items):
        x = 1.25+i*(w+gap)
        panel(s, title, body, x, y, w, h)
        if i < n-1:
            text(s, '→', x+w+.06, y+.72, .38, .55, 24, GREEN, align=PP_ALIGN.CENTER)


def picture(s, name, x, y, w, h, crop=None):
    path = CAP / name
    if crop:
        im = Image.open(path)
        im = im.crop(crop)
        path = CAP / (Path(name).stem+'_crop_'+str(crop[0])+'.png')
        im.save(path)
    iw, ih = Image.open(path).size
    scale = min(w/iw,h/ih)
    dw,dh = iw*scale,ih*scale
    rect(s,x,y,w,h,'F6F9F8')
    s.shapes.add_picture(str(path), Inches(x+(w-dw)/2), Inches(y+(h-dh)/2), Inches(dw), Inches(dh))


def link(s,x1,y1,x2,y2,label='1 : N'):
    line=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    line.line.color.rgb=RGBColor.from_string(GREEN)
    line.line.width=Pt(1.5)
    if label:
        text(s,label,(x1+x2)/2-.4,(y1+y2)/2-.25,.8,.34,15,GREEN,align=PP_ALIGN.CENTER)


def modelmeta(h):
    return json.loads((ROOT/f'models/recommender/v1/child_crime_exposure/{h}y/model_metadata.json').read_text(encoding='utf-8'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m2,m3=modelmeta(2),modelmeta(3)
    source_model=['models/recommender/v1/child_crime_exposure/2y/model_metadata.json',
                  'models/recommender/v1/child_crime_exposure/3y/model_metadata.json']
    source_service=['backend/app/services/neighborhood_recommendations.py',
                    'backend/app/services/recommender_v1.py','frontend/src/views/pages/FinalServicePage.tsx']
    # 01 Cover: exact frame, illustration and headline typography from user's deck.
    s=prs.slides.add_slide(prs.slide_layouts[6]);copy_shapes(cover,s);pages.append(s)
    for sh in walk(s.shapes):
        if sh.has_text_frame and '아이와 살기좋은' in sh.text:
            sh.text_frame.clear()
    text(s,'서울, 아이와 함께 더 안심되는 동네',2.2,5.85,15.6,.7,32,GREEN,align=PP_ALIGN.CENTER)
    text(s,'예산과 생활환경을 함께 비교하는 데이터 기반 동네 추천 서비스',2.2,6.73,15.6,.55,23,DARK,align=PP_ALIGN.CENTER)
    text(s,'1팀  |  장의석 · 김홍수 · 임동현 · 남궁다인',2.2,8.35,15.6,.5,23,GREEN,align=PP_ALIGN.CENTER)
    text(s,'최종 발표  2026. 11. 20',2.2,9.04,15.6,.4,20,GREY,align=PP_ALIGN.CENTER)
    notes(s,'기존 4차 발표 표지의 원본 장식·폰트·비율을 유지했다. 팀 구성은 사용자 제공 역할분담을 따른다.',[SOURCE])
    # 02 Contents
    s=slide('00','발표 순서','문제 정의에서 모델 검증, 실제 서비스 구현까지')
    entries=[('01  프로젝트 개요','누구를 위한 서비스인가'),('02  데이터 · 구조','수집·가공·배포 흐름'),('03  분석 · 모델링','범죄·교통·부동산 예측'),
             ('04  서비스 설계 · 구현','모델 결과를 사용자 경험으로'),('05  테스트 · 시연','추천부터 상세 조회까지'),('06  성과 · 협업 · 결론','성능 개선과 다음 단계')]
    for i,(a,b) in enumerate(entries):
        panel(s,a,b,1.25+(i%2)*8.95,4.4+(i//2)*1.67,8.55,1.5)
    notes(s,'제공된 초안의 6개 섹션과 31장 흐름을 유지한다.',[DRAFT])
    # 03 Problem
    s=slide('01','안전과 예산을 한 번에 비교하기 어렵다','문제 정의  |  가격·범죄·교통 정보가 서로 다른 기준과 화면에 흩어져 있다')
    flow(s,[('주거비 확인','매매·전세 가격\n예산과 면적 비교'),('생활환경 확인','범죄·교통 자료\n연도·지역 기준 확인'),('후보지 결정','여러 지역을 직접 비교\n판단 근거 정리')],h=2.4)
    panel(s,'우리가 줄이려는 비교 비용','여러 자료를 찾아보는 과정을 하나의 지도와 추천 흐름으로 연결합니다.\n“얼마인가?”와 “어떤 생활환경인가?”를 같은 지역 기준으로 보여줍니다.',1.25,7.25,17.5,2.1)
    foot(s,'문제 가설을 설명하는 개념도입니다. 별도 이용자 설문이나 시장 규모 통계를 주장하지 않습니다.')
    notes(s,'초안의 근거 통계 작성 예정 부분은 미확인 수치로 채우지 않았다. 공공데이터 결합에서의 정보 비교 과제를 제시한다.',[DRAFT])
    # 04 Target
    s=slide('01','예산이 정해진 아이 있는 가구를 위한 서비스','타겟  |  서울에서 이사 후보지를 찾는 아동 동반 가구')
    panel(s,'사용 상황','예산은 정해져 있지만\n어느 동네부터 볼지 모를 때\n\n관심 행정동의 가격과\n소속 구의 생활환경을 비교할 때',1.25,4.45,5.55,4.6)
    panel(s,'서비스가 제공하는 답','① 예산 범위에 맞는 동네\n② 구별 생활환경 점수\n③ 지도에서 확인하는 위치\n④ 가격·점수 산출 기준',7.2,4.45,5.55,4.6)
    panel(s,'기대효과','정량: 추천 후보를 최대 5곳으로 축소\n정성: 비교 기준을 한 화면에 정리\n\n확장: 월세·데이터 갱신 자동화\n검증: 향후 사용자 평가로 확인',13.15,4.45,5.6,4.6)
    foot(s,'실제 거주 적합성이나 안전을 보장하는 서비스가 아닌, 후보 지역 탐색을 돕는 참고 도구입니다.')
    notes(s,'효과는 설계상 기대효과로 표현한다. 시간 절감률·만족도 등 미측정 결과는 제시하지 않는다.',source_service)
    # 05 Scope
    s=slide('01','구현 범위와 연구 범위를 분리했다','현재 서비스  |  매매·전세 예산 기반 추천 + 행정동 검색 + 구 단위 점수')
    table(s,[['구분','현재 상태','사용자에게 보이는 기준'],
             ['매매·전세','서비스 연결','예산 범위·면적 조건에 맞는 대표 추정가격'],
             ['아동범죄노출','2027·2028 선택 가능','2025 기준 2년·3년 모델 / Feature 가중합 점수'],
             ['교통안전','참고 결과 연결','현재 화면은 2025년 결과 / 2027·2028 연구 모델은 별도'],
             ['월세','모델 연구·저장 / 화면 미연동','보증금·건물 정보를 받는 별도 입력 흐름 필요'],
             ['범위 밖','구 단위 점수만 제공','행정동별 실제 범죄율·실제 아동 피해율은 아님']],1.25,4.4,17.5,4.6,[2.8,4.5,10.2],21)
    foot(s,'미래가격 연구 모델의 성능과 현재 서비스의 대표가격 추정값은 구분해서 해석합니다.')
    notes(s,'코드의 최종 서비스는 범죄2y/3y, legacy 교통2025, property representative model adapter를 호출한다. 월세 endpoint 또는 final property bundle 호출로 오인하지 않도록 범위를 수정했다.',source_service)
    # 06 Functions
    s=slide('01','조건 입력부터 지역 상세까지 네 가지 핵심 기능','기능 요약  |  로그인 없이 지도에서 바로 시작')
    table(s,[['핵심 기능','입력 · 동작','출력 · 보호 조건'],
             ['예산 기반 추천','매매/전세 + 하한·상한 + 면적','서로 다른 구의 행정동을 최대 5곳 추천'],
             ['행정동 검색','자치구·행정동 이름 검색','동 가격 + 소속 구 점수 표시'],
             ['점수 확인','추천 마커·지역 선택','범죄·교통 점수와 기준연도 안내'],
             ['예측연도 선택','2027 / 2028 선택','범죄 2년 / 3년 결과 전환']],1.25,4.4,17.5,3.65,[3.4,6.2,7.9],22)
    rect(s,1.25,8.4,17.5,.85,PALE)
    text(s,'공통 기준   지역 단위 명시  ·  예측연도 명시  ·  오류 메시지 표시  ·  서버 예외 로그 기록',1.5,8.62,17,.42,22,GREEN,MEDIUM)
    notes(s,'공식 요구사항 정의서 번호나 완료율을 새로 꾸며내지 않고 구현 기능 요약으로 대체한다. 지연시간 SLA는 측정하지 않아 기재하지 않는다.',source_service)
    # 07 Concept
    s=slide('01','예산 안에서 점수가 높은 동네를 최대 5곳 추천','추천 방식  |  예산을 만족하는 후보 중 구별 최저가 동을 하나씩 선택')
    flow(s,[('조건 입력','거래유형·예산 범위'),('가격 필터','범위 안에 드는 동'),('구별 후보','구마다 최저가 동 1곳'),('점수 정렬','높은 구 순으로 최대 5곳')],4.45,2.0)
    metric(s,'범죄 · 2년 모델','84.0%','CV 5분위 일치율',1.25,6.95)
    metric(s,'교통 · 2년 모델','56.0%','CV 정확등급 일치율',7.2,6.95)
    metric(s,'전세 · 2년 모델','0.796','연구 모델 R²',13.15,6.95,5.6)
    foot(s,'영역마다 평가 대상과 검증 구간이 다릅니다. 세 지표를 같은 “정답률”로 비교하지 않습니다.')
    notes(s,'각 성능 지표는 이후 상세 페이지에서 검증 범위와 한계를 설명한다. 서비스 추천은 결합 점수 기준이며 모델 CV를 추천 정확도로 간주하지 않는다.',source_model+[SOURCE])
    # 08 Data
    s=slide('02','공공·실거래 데이터를 지역과 시점 기준으로 결합','전처리 원칙  |  지역코드 정합성 · 공개 시점 · 결측값 · 예측 시점 누수 점검')
    table(s,[['데이터 영역','자료·출처 계열','분석 단위 / 가공'],
             ['범죄 · 인구','서울시 5대범죄 / 행정안전부 연령별 주민등록인구','구·연도 / 0~17세 합계, log·시차·변화량'],
             ['범죄 보조 후보','서울시 인구·가구·사업체·이동·학생·스트레스','동 자료는 구 합산 / 비율 재계산 / 공표 지연 반영'],
             ['교통안전','교통사고 및 지역 인구·시설 자료','구·연도 / 미래 위험등급과 설명변수'],
             ['매매 · 전세 · 월세','아파트 실거래 / 지역·건물 특성 / 대출금리','동·월 또는 계약 / 평당가·상대가격·기간 통계']],1.25,4.4,17.5,3.65,[2.8,7.0,7.7],20)
    panel(s,'갱신과 저장','공공통계는 원천별 주기가 달라 모든 지표를 일괄 최신값으로 취급하지 않습니다.\n원본 CSV·API 응답 → 정제 산출물 → 모델·예측 CSV/JSON 순으로 버전을 구분합니다.',1.25,8.3,17.5,1.55)
    notes(s,'단일 고정 갱신주기 또는 모든 지표 2025 확보를 주장하지 않는다. 원천별 출처·기간 상세는 raw inventory와 학습 스크립트를 따른다.',[
        'docs/child-crime-raw-data-inventory.md','ml/training/AptJeonseForecastV33.py','ml/training/RentContractFinal.py'])
    # 09 Pipeline
    s=slide('02','수집부터 서비스까지 산출물 중심으로 연결','현재 흐름  |  원본을 보존하고, 학습 결과와 서비스 배포 결과를 분리')
    flow(s,[('수집','공개 파일·API\n출처·기준일'),('전처리','지역·시점 결합\nFeature 생성'),('학습·검증','시간순 검증\n모델 선택'),('배포','joblib·CSV·JSON\nAPI 조회')],4.4,2.4)
    panel(s,'현재 구현','파일 기반 수집·가공·학습 스크립트\n서비스 산출물 유효성 검사 및 누락 복구\nFastAPI → React 지도 조회',1.25,7.25,8.55,2.1)
    panel(s,'자동화 확장 설계','Airflow: 수집 → 품질검사 → 학습 → 검증 → 배포\nPostgreSQL: 지역·모델·예측 이력 구조화\nDAG·서비스 업무 테이블은 후속 구현 단계',10.2,7.25,8.55,2.1)
    notes(s,'Compose에 Airflow/PostgreSQL/MLflow/MinIO 구성은 있으나 infra/airflow/dags는 .gitkeep만 확인되었고 업무 DB 테이블 구축을 완료했다고 주장하지 않는다.',[
        'compose.yaml','infra/airflow/dags','infra/postgres/init','backend/app/services/service_model_bootstrap.py'])
    # 10 Modeling Overview
    s=slide('03','예측 모델과 서비스 점수는 역할이 다르다','모델은 미래값·순위를 예측하고, 서비스는 설명 가능한 기준으로 점수화')
    table(s,[['영역','예측 대상 · 기법','공간 단위','평가'],
             ['범죄노출','2년: Ridge 직접 Y / 3년: 분리예측+직접 Y 결합','자치구','Spearman · 순위오차 · 5분위'],
             ['교통안전','2·3년 직접 예측 / GBR + Ridge','자치구','정확등급 · 등급오차'],
             ['매매·전세','1·2·3년 평당 가격 / XGBoost','행정동','R² · 가격오차'],
             ['월세','보증금·건물 특성 조건부 / HistGradientBoosting','계약','MAE · R²']],1.25,4.4,17.5,3.55,[2.5,8.5,2.3,4.2],20)
    rect(s,1.25,8.3,17.5,1.05,PALE)
    text(s,'동네점수 = 범죄 Feature 가중합 점수 × 50% + 교통안전 점수 × 50%',1.5,8.56,17,.55,28,GREEN,MEDIUM)
    foot(s,'2025 → 2027·2028은 범죄·교통 연구 모델의 기준입니다. 가격 모델의 기준월·기간은 별도로 관리합니다.')
    notes(s,'영역 결합 50:50은 서비스 정책 가중치이며 학습으로 자동 추정한 비율이 아니다. 범죄 Feature 영향력 가중치와 구별해야 한다.',source_model+source_service+[SOURCE])
    # 11 Crime target + score
    s=slide('03','아동범죄노출: 실제 피해율이 아닌 지역 대리지표','아동 피해건수의 장기간 구별 자료가 부족해 지역 범죄 수준을 아동인구와 함께 해석')
    rect(s,1.25,4.45,17.5,1.03,PALE)
    text(s,'모델용 Y = log(1 + 5대범죄 발생건수 ÷ 0~17세 인구 × 10,000)',1.5,4.73,17,.55,28,DARK,MEDIUM)
    panel(s,'왜 로그를 쓰는가?','큰 값의 차이를 압축해 규모 편차를 완화\n상대적인 수준·변화 학습에 활용\n아동인구가 적은 구의 높은 비율에 유의',1.25,5.85,8.55,2.8)
    panel(s,'화면의 점수는 별도 계산','Feature별 유리한 순위 → 100·80·60·40·20점\n각 점수 × 영향력 가중치의 합 (총 가중치 100%)\n최종 점수순 25개 구를 5개씩 A~E로 구분',10.2,5.85,8.55,2.8)
    text(s,'높은 Y = 노출 수준이 높은 방향     /     높은 서비스 점수 = 낮은 위험 방향',1.5,9.0,17,.5,24,GREEN,MEDIUM)
    notes(s,'5대범죄는 전체 연령 대상 지역 발생건수다. 이 지표는 아동에게 발생한 건수나 피해확률을 추정한 것이 아니다. log는 통계적 변환이지 데이터 부족을 보완하는 장치가 아니다. 초안의 Y 순위만으로 100~20점을 준다는 표현을 실제 Feature 가중합 로직으로 교정했다.',source_model+source_service)
    # 12 Crime methods
    s=slide('03','예측기간에 맞춰 로직과 Feature 조합을 선택','2년은 단순한 추세 직접예측, 3년은 분리예측의 장점을 유지한 결합')
    table(s,[['구분','2027 · 2년 모델','2028 · 3년 모델'],
             ['예측 방법','Ridge(alpha=10)로 미래 Y 직접 예측','범죄·인구 분리예측 92% + 직접 Y 8%'],
             ['입력 특징','Y·범죄·아동인구의 현재값/시차/변화량','인구·사업체·가구·이동·출생사망·추세 + 학생이동'],
             ['선택 원칙','시간순 CV에서 순위 성능 우선','직접 Y 단독 성능 부족 → 소폭 결합 채택'],
             ['서비스 점수 항목','수치 Feature 12개 / 표준화 계수 영향력','통합 수치 항목 43개 / 최종 예측 민감도']],1.25,4.4,17.5,3.5,[2.6,7.1,7.8],20)
    text(s,'2년 CV 5분위 일치율 개선',1.25,8.2,6.1,.5,23,GREEN,MEDIUM)
    text(s,'65.3%  →  81.3%  →  84.0%',7.3,8.1,11.4,.75,37,DARK,MEDIUM)
    text(s,'현재값 유지 기준   →   기본 8개 항목   →   범죄·인구 시차 4개 추가',7.3,8.96,11.4,.45,18,GREY)
    notes(s,'2년 baseline_cv 65.333; strict_forward_history 단계0 81.333(8개), 단계1 84(12개). 3년 blend baseline_weight .92, direct .08. 43은 서비스 점수 산출의 통합 수치 항목 수이며 one-hot 등 전체 모델 입력 열수와 같지 않다.',source_model)
    # 13 Crime metrics
    s=slide('03','순위 성능을 선택 기준으로, 절대값 오차는 함께 확인','검증  |  CV: 2022~2024년 목표연도 / 최종 확인: 2025년 홀드아웃')
    chart(s,['2027 · 2년','2028 · 3년'],[('CV 5분위 일치(%)',[84,81.3333]),('2025 별도검증(%)',[76,84])],1.25,4.5,8.5,4.5,100,20,'0.0',True)
    table(s,[['지표','2년','3년'],['CV Spearman','0.9746','0.9703'],['CV 평균 순위오차','1.07위','1.17위'],['2025 Spearman','0.9677','0.9731'],['2025 Y RMSE','0.1015','0.1764']],10.25,4.55,8.5,3.4,[4.6,1.95,1.95],20)
    text(s,'주의  |  2년 모델은 기준모델 대비\n2025 Y RMSE가 0.0928 → 0.1015로 악화',10.3,8.35,8.3,.9,21,GREEN,MEDIUM)
    foot(s,'위 일치율은 예측 Y의 상대순위 검증값입니다. Feature 가중합 서비스 점수의 정확도를 의미하지 않습니다.')
    notes(s,'3년 baseline 대비 CV exact는81.333으로 동일; Spearman .969487→.970256, holdout RMSE .185077→.176441. 2년 exact는 CV+18.67%p,holdout+8%p지만 Spearman/절대값 모두 개선된 것은 아니다. 2027/28 정답은 아직 없어 실제 미래 정확도는 측정할 수 없다.',source_model)
    # 14 Traffic logic
    s=slide('03','교통안전은 2027·2028 위험등급을 직접 예측','2025 입력 → 목표연도 직접 예측  |  1년 모델을 반복 적용하지 않음')
    panel(s,'2027 · 2년 모델','GBR 75% + Ridge 25%\n\nGBR: 39개 Feature, depth 3\nRidge: 26개 Feature, alpha 0.1\n현재 등급 정보를 함께 반영',1.25,4.45,8.55,3.8)
    panel(s,'2028 · 3년 모델','GBR 50% + Ridge 50%\n\n각 64개 Feature, 구 one-hot 포함\nGBR: depth 3 / Ridge: alpha 30\n미래 3년 후 위험등급 직접 결합',10.2,4.45,8.55,3.8)
    rect(s,1.25,8.62,17.5,.72,PALE)
    text(s,'위험등급 1 = 상대적으로 안전  →  5 = 상대적으로 위험  |  서비스 안전점수는 반대 방향',1.5,8.77,17,.42,22,GREEN,MEDIUM)
    notes(s,'2년 A 예측60%+현재40%, B 현재+예측변화량*.6; A/B최종75/25. 3년64열에는25구 one-hot포함,50/50. 연구 모델 구성은 4차 발표의 교통 담당 자료를 계승. 운영 adapter는 여전히2025참고파일.',[SOURCE,'backend/app/services/recommender_v1.py'])
    # 15 Traffic results
    s=slide('03','등급 유지 기준보다 정확등급 일치율을 개선','평가 지표  |  “정확히 같은 등급”을 맞힌 비율과 평균 등급오차')
    chart(s,['2년 예측','3년 예측'],[('등급 유지 기준(%)',[48.8,34]),('개선 모델(%)',[56,55])],1.25,4.5,9.1,4.6,70,10,'0.0',True)
    metric(s,'2년 개선폭','+7.2%p','48.8% → 56.0%',10.8,4.5,7.95,2.05)
    metric(s,'3년 개선폭','+21.0%p','34.0% → 55.0%',10.8,6.9,7.95,2.05)
    foot(s,'평균 등급오차: 2년 0.576 / 3년 0.580. ±1등급 일치: 87.2% / 87.0% (정확 일치와 별개).')
    notes(s,'기존 교통 발표의5년(2y)·4년(3y)검증 결과다. 독립 신규 실험을 수행하지 않았다. 표본과 기간이 한정되고 지표 정의에 따라 수치가 달라지므로 서비스 사고감소 효과로 해석하지 않는다.',[SOURCE])
    # 16 Sale
    s=slide('03','매매: 금리·상대가격으로 장기 예측을 보완','연구 결과  |  절대 수준보다 지역의 상대 위치와 시장 변화를 함께 학습')
    chart(s,['v52','v54','v55','v64'],[('3년 R²',[.518,.534,.538,.579])],1.25,4.5,8.8,4.45,.7,.1,'0.000',True)
    table(s,[['버전','변경 사항','3년 R²'],['v52','금리 반영 / 13개 항목','0.518'],['v54','현재가격 → 서울 대비 상대가격','0.534'],['v55','가구수·지역난방 제외 / 11개','0.538'],['v64','전세·주변·구 특성 보강 / 15개','0.579']],10.45,4.55,8.3,3.55,[1.2,5.6,1.5],19)
    text(s,'v64 연구 성능: 1년 0.890 / 2년 0.754\n저장 번들의 MODEL_INFO는 v55 → 배포 버전 구분',10.5,8.43,8.2,.88,20,GREEN,MEDIUM)
    foot(s,'v52의 금리 추가 시 +0.147, v54의 상대가격 전환 시 +0.016은 각 직전 비교 모델 대비 R² 증가분입니다.')
    notes(s,'초안의v46→v64 .39→.58는 기준v46원본 수치의 재검증 없이 추가하지 않고 기존4차발표의명시적버전표v52~64를 사용했다. v64 연구 최고 결과와 현재apart_sale_final/MODEL_INFO v55는 다르다. 서비스는 legacy대표가격 adapter로 미래가격 R²를 서비스 성능으로 주장하지 않는다.',[SOURCE,'models/recommender/v1/apart_sale_final/MODEL_INFO.txt'])
    # 17 Jeonse
    s=slide('03','전세: 다섯 가지 핵심 Feature로 단순화','최종 선택 v33  |  2년 성능과 모델 단순성을 함께 고려')
    table(s,[['모델','Feature 수','1년 R²','2년 R²','3년 R²'],['v28','10개','0.836','0.745','0.621'],['v33 · 선택','5개','0.825','0.796','0.633'],['v39 · 후보','7개','0.828','0.788','0.641']],1.25,4.4,17.5,2.55,[3.5,3,3.65,3.65,3.7],23)
    panel(s,'선택된 다섯 가지','6개월 상승률 · 서울 대비 상대가격 · 금리 12개월 변화\n전세가율 · 신축 건물 거래 비율',1.25,7.3,10.7,2.0)
    panel(s,'선택의 의미','v28 대비 2년 R² +0.051\nv39보다 3년 R²는 낮지만,\n적은 변수로 2년 성능을 유지',12.35,7.3,6.4,2.25)
    notes(s,'AptJeonseForecastV33.py상 최종5개변수 확인. 3년 최고후보v39보다 낮은지표를 숨기지 않는다. 현재 저장번들은v33. 연구검증을 새로 재실행하지 않았다.',[SOURCE,'ml/training/AptJeonseForecastV33.py','models/recommender/v1/apart_jeonse_final/MODEL_INFO.txt'])
    # 18 Monthly
    s=slide('03','월세: 계약 조건을 넣어 가격 오차를 줄였다','보증금에 따라 달라지는 월세  |  행정동 평균에서 계약 단위 예측으로')
    chart(s,['동 단위 비교','계약 단위'],[('기존 발표 비교 MAE(만원)',[56.7,22.2])],1.25,4.55,8.8,4.4,70,10,'0.0',True)
    panel(s,'HistGradientBoosting','면적·층·연식·보증금\n동·구·서울 수준 및 건물 이력\n건물 특성 기반 월세 조건부 추정',10.5,4.5,8.25,2.25)
    panel(s,'저장 모델과 서비스 상태','저장 메타데이터: MAE 20.7만원 / R² 0.898\n재학습 전 평가값이며 위 비교 실험과 구분\n월세 화면은 미연동: 입력·API 계약 추가 필요',10.5,7.05,8.25,2.25)
    foot(s,'56.7 → 22.2만원은 기존 발표의 실험 비교입니다. 저장 번들 평가값과 동일 실험으로 섞어 쓰지 않습니다.')
    notes(s,'기존발표 계약단위test2025.01~2026.09 MAE22.2R².885. 저장summary test_before_refit MAE20.65699R².89785; 서로다른평가맥락. trained_until2026-09-30. 모델은미래27/28월세모델이아니라보증금조건부계약예측.',[SOURCE,'models/recommender/v1/apart_wolse_final/summary.json','ml/training/RentContractFinal.py'])
    # 19 Score and deployment
    s=slide('03','Feature 영향력을 점수로 환산해 서비스에 연결','영향력 가중치와 영역 결합 비중을 분리해 설명')
    table(s,[['2027 범죄 점수 주요 항목','가중치','안전한 방향'],['현재 노출 Y','17.7%','낮을수록'],['전년도 노출 Y','17.7%','낮을수록'],['2년 전 아동인구(log)','12.6%','높을수록'],['나머지 9개 항목 …','52.0%','영향력 부호별'],['전체 12개 합계','100%','정규화 합계']],1.25,4.4,8.2,4.0,[4.3,1.5,2.4],19)
    panel(s,'01  범죄 안전점수','Σ (Feature별 5분위 점수 × 가중치/100)\n항목별 점수: 100·80·60·40·20\n따라서 총점은 92.6처럼 소수점도 가능',9.9,4.4,8.85,2.25)
    panel(s,'02  동네점수 · 등급','범죄 점수 50% + 교통안전 점수 50%\n25개 구를 총점순으로 5개씩 A~E\n동일 구의 행정동은 동일 점수를 표시',9.9,7.0,8.85,2.25)
    foot(s,'2028년은 통합 민감도 기반 43개 항목이며 가중치 합계도 100%입니다. 영향력은 인과관계를 뜻하지 않습니다.')
    notes(s,'2y 표준화Ridge수치계수절댓값 정규화,3y최종통합예측민감도정규화. onehot·절편 점수항목제외. 상위3개합48.0327907로 나머지51.9672093. 표는소수점한자리반올림이며전체합100%. 계수부호는모델조건부관련성. 최종등급동점은범죄점수/구명규칙.',source_service+['models/recommender/v1/child_crime_exposure/2y/feature_score_formula.csv'])
    # 20 Architecture
    s=slide('04','역할을 나눠 예측 결과를 안정적으로 제공','서비스 경로  |  브라우저 → Nginx → FastAPI → 검증된 모델 산출물')
    flow(s,[('화면','React · TypeScript\nLeaflet 지도'),('라우팅','Nginx\n정적 파일·API 연결'),('API','FastAPI · Python\n추천·검색·오류 처리'),('모델 산출물','joblib / CSV / JSON\n로딩·검사·복구')],4.4,2.4)
    table(s,[['운영 구성','선택 이유 · 역할'],['Docker Compose · Ubuntu','개발/서버 실행환경을 명시적으로 구성'],['MinIO · MLflow','파일·실험 아티팩트 관리용 인프라'],['PostgreSQL · Airflow','지역·이력 DB와 수집/학습 자동화 확장 기반'],['예외 응답 · 파일 로그','사용자에게 안내 / 관리자는 시각·원인·요청 추적']],1.25,7.1,17.5,2.3,[5,12.5],19)
    notes(s,'기술스택은package.json,requirements-server,compose기준. 타기술보다 성능우수하다는 미측정벤치마크는 제외. DB·Airflow 인프라구성과 업무DAG구현 완료는 다르다.',source_service+['compose.yaml','frontend/package.json','requirements-server.txt'])
    # 21 ERD conceptual native, simple noncrossing relationships
    s=slide('04','지역·예측·모델 버전을 연결하는 ERD 설계','확장 설계안  |  로그인 테이블 없이 지역 조회와 모델 운영 이력 중심')
    def entity(x,y,w,title,lines):
        rect(s,x,y,w,1.7,'FFFFFF')
        rect(s,x,y,w,.48,PALE)
        text(s,title,x+.15,y+.08,w-.3,.36,21,GREEN,MEDIUM)
        text(s,lines,x+.15,y+.6,w-.3,1.0,17,DARK)
    entity(1.25,4.5,4.6,'region · 지역','PK region_id   |   FK parent_id\n구·동 코드 / 유효시작·종료일')
    entity(7.7,4.5,4.6,'prediction · 예측','PK prediction_id\nFK region_id / model_version_id\n기준시점 · 목표시점 · 예측값')
    entity(14.15,4.5,4.6,'model_version · 모델','PK model_version_id\n영역 · horizon · 아티팩트 URI\n검증지표 · 배포상태')
    entity(1.25,7.5,4.6,'boundary_notice · 공고','PK notice_id   |   FK region_id\n공고 URL · 변경예정일 · 확인일')
    entity(7.7,7.5,4.6,'score · 서비스 점수','PK score_id\nFK prediction_id / release_id\n영역점수 · 종합점수 · 등급')
    entity(14.15,7.5,4.6,'pipeline_run · 실행','PK run_id   |   FK model_version_id\n작업 · 시작/종료시각 · 상태\n데이터 버전 · 오류 로그 경로')
    link(s,5.85,5.35,7.7,5.35,'1 : N');link(s,12.3,5.35,14.15,5.35,'N : 1')
    link(s,3.55,6.2,3.55,7.5,'1 : N');link(s,10,6.2,10,7.5,'1 : N');link(s,16.45,6.2,16.45,7.5,'1 : N')
    foot(s,'발표용 핵심 관계만 표현한 개념 ERD입니다. 원천 데이터·배포묶음 등의 상세 테이블과 DDL은 별도 설계 범위입니다.')
    notes(s,'이 장은 제안 스키마로 현재 운영DB에 적용되었다고 주장하지 않는다. region은parent자기참조로 구/동 계층표현. prediction grain은지역·모델버전·기준시점·목표시점·대상항목, score는예측·배포버전·지역단위로유일성필요. release참조테이블은요약도에서생략. 로그인기능없음. 자세한제안은기존erd-v2문서참고.',[
        'docs/architecture/spotfit-service-erd-v2.md','infra/postgres/init'])
    # 22 UI
    s=slide('04','검색·조건·추천을 하나의 지도에 모았다','화면 구성  |  간결한 아이콘 툴바와 선택 시 열리는 조건 패널')
    picture(s,'01_home.png',1.25,4.4,10.8,5.15)
    panel(s,'간결한 탐색','검색 / 거래유형 / 예산 / 면적 / 예측연도\n필요한 기능만 펼치는 상단 툴바\n하단 목록 대신 지도에 추천 마커 표시',12.45,4.4,6.3,2.35)
    panel(s,'가격과 점수를 함께','기본 지도는 아파트 가격\n지역 선택 → 가격·구 점수 카드\n세로 A~E 범례로 상대등급 확인',12.45,7.05,6.3,2.4)
    notes(s,'2026-11-16 실행중 localhost8080 실제캡처. 배경지도키가없어행정경계fallback표시. 화면을목업으로위장하지않았으며네트워크연결상태의영향은있다.',[CAP/'01_home.png'])
    # 23 Testing
    s=slide('05','추천·검색·연도 전환을 실제 화면과 API로 확인','확인일 2026.11.16  |  읽기 전용 시연 동선 점검 결과')
    table(s,[['검증 시나리오','확인 내용','결과'],['기본 지도 진입','추천 마커 5개 표시','통과'],['예산 3~6억원 추천','5개 구가 서로 다르고 모든 후보가 예산 범위 내','통과'],['추천 → 상세 조회','마커 클릭 후 요약 카드·상세 창 표시','통과'],['역삼1동 검색','검색 결과 선택 후 해당 행정동 카드 표시','통과'],['2028년 전환','horizon=3 카탈로그 API HTTP 200','통과']],1.25,4.4,17.5,3.85,[4.4,10.7,2.4],22)
    panel(s,'품질 관리 범위','기존 테스트 코드: Feature 가중치·추천·예외 응답·모델 누락 복구 등\n이번 확인은 핵심 동선이며, 전체 회귀·부하·장애 복구 시험을 모두 재실행한 것은 아닙니다.',1.25,8.4,17.5,1.55)
    notes(s,'capture_final_presentation.py실행결과5개체크모두통과. 서버부하테스트나의도적모델삭제시험은수행하지않았다. 보유테스트와이번실행테스트를구별한다.',[
        'scripts/capture_final_presentation.py',OUT/'capture_checks.json','backend/tests/integration','backend/tests/unit/test_child_crime_feature_score.py'])
    # 24 Demo input
    s=slide('05','시연: 3~6억원 안에서 이사 후보지를 찾는다','예시 사용자  |  아이와 함께 서울 내 매매 이사를 검토하는 가구')
    picture(s,'02_budget.png',1.25,4.4,10.8,5.15)
    table(s,[['입력 항목','시연값'],['거래유형','매매'],['예산 범위','3억원 ~ 6억원'],['전용면적','미입력 · 지역 대표값'],['범죄 예측연도','2027년']],12.45,4.45,6.3,3.35,[2.8,3.5],21)
    text(s,'직접 입력 또는 슬라이더로 범위를 조정\n조건 변경 → 추천 결과 자동 갱신',12.5,8.2,6.2,1.05,22,GREEN,MEDIUM)
    notes(s,'예시시나리오는검증에실제로사용한3~6억원. 실제매물이나개인맞춤보장아님. 캡처는현재서비스의대표값산출결과.',[CAP/'02_budget.png',OUT/'capture_checks.json'])
    # 25 Demo result
    s=slide('05','시연: 추천 마커에서 가격·점수 상세로 이동','한 곳을 선택하면 해당 동의 대표가격과 소속 구의 점수 근거를 확인')
    picture(s,'03_recommend.png',1.25,4.4,10.6,5.1)
    picture(s,'04_detail.png',12.2,4.4,6.55,5.1,crop=(510,105,1095,795))
    notes(s,'실제캡처에서신월1동대표가격41180만원,양천구동네점수96.3,범죄92.6,교통100. 점수는기준일에따라변할수있으므로고정성과치로표현하지않았다.',[CAP/'03_recommend.png',CAP/'04_detail.png'])
    # 26 Backup
    s=slide('05','시연 백업: 행정동 검색 화면과 결과 캡처','네트워크 상황과 무관하게 서비스 흐름을 설명할 수 있도록 캡처를 보관')
    picture(s,'05_search.png',1.25,4.4,10.8,5.15)
    panel(s,'검색에서 확인하는 기준','역삼1동 → 강남구 연결\n동 단위 대표가격 + 구 단위 점수\n같은 구의 동은 같은 점수를 사용',12.45,4.4,6.3,2.3)
    panel(s,'시연 장애 대응','화면 캡처로 추천·상세 흐름 설명\n연결 상태·모델 준비 상태 확인\n서버 로그로 원인과 시각 추적',12.45,7.05,6.3,2.35)
    notes(s,'실제녹화영상은없으므로영상제작완료표현은사용하지않는다. 별도동영상대신현재PPT내장캡처가백업이다.',[CAP/'05_search.png'])
    # 27 Outcomes
    s=slide('06','모델 개선을 서비스 비교 경험으로 연결했다','성과와 한계  |  개선된 지표를 제시하되 과도한 안전 보장은 하지 않음')
    metric(s,'범죄 · 2년 CV','+18.7%p','5분위 일치율 65.3 → 84.0%',1.25,4.45)
    metric(s,'교통 · 3년 CV','+21.0%p','정확등급 일치율 34.0 → 55.0%',7.2,4.45)
    metric(s,'전세 · 2년 R²','+0.051','v28 0.745 → v33 0.796',13.15,4.45,5.6)
    panel(s,'현재 한계','구 단위 대리지표·적은 지역 표본\n범죄·교통·가격의 기준시점 차이\n미래가격·월세 연구 모델과 화면 연결 차이',1.25,7.15,8.55,2.2)
    panel(s,'다음 단계','모델별 API 계약과 예측연도 통일\nAirflow·DB로 데이터와 모델 이력 자동화\n신규연도 검증·사용자 평가·세부 지역자료 확대',10.2,7.15,8.55,2.2)
    notes(s,'성과지표는영역별평가기준이다. 미래시점의실제정확도·사고감소·만족도등미측정효과를주장하지않는다.',source_model+[SOURCE])
    # 28 Team
    s=slide('06','역할별 모델 개발과 서비스 통합을 함께 진행','1팀  |  데이터 수집·정제 → 모델 개선 → API·화면·배포 연결')
    table(s,[['팀원','담당 · 기여'],['장의석 · 팀장','프로젝트 총괄·기획 / 아동범죄노출 모델 / 백엔드·프론트엔드·서버'],['김홍수','데이터 수집·정제 / 아파트 가격 예측모델'],['임동현','데이터 수집·정제 / 어린이 교통사고 모델'],['남궁다인','데이터 수집·정제 / 아파트 가격 예측모델']],1.25,4.4,17.5,3.35,[3.5,14],22)
    panel(s,'협업 방식','Git 브랜치 기반 변경 관리 · 모델과 메타데이터 공유\n공통 서비스 입력·출력 기준 정리 · 실행환경 Compose 구성',1.25,8.1,10.7,1.45)
    panel(s,'발표 준비','11/16 기준 동선 점검·자료 정리\n11/20 최종 발표 예정',12.35,8.1,6.4,1.45)
    notes(s,'사용자가직접제공한역할분담을사용했다. 개인별매매/전세분담·프로젝트착수일·미래리허설완료여부는추정하지않았다. 일정완료율과개인구글시트는미확인으로추가하지않았다.',[DRAFT,'사용자 제공 팀 역할분담','compose.yaml'])
    # 29 Conclusion
    s=slide('06','예산 안에서 더 안심되는 동네를 데이터로 찾는다','SPOT FIT  |  생활환경과 주거비를 같은 지도에서 비교')
    items=[('01','분산된 정보를 연결','범죄·교통·가격을 지역 기준으로 묶었습니다.'),('02','예측을 설명 가능한 점수로','Feature 영향력·점수·등급의 계산 근거를 구분했습니다.'),('03','모델을 서비스로 연결','예산 안의 후보를 찾고 가격과 점수를 함께 확인합니다.')]
    for i,(num,title,body) in enumerate(items):
        y=4.5+i*1.55
        text(s,num,1.4,y,.9,.75,43,TEAL,MEDIUM)
        text(s,title,2.65,y+.02,15.6,.55,29,GREEN,MEDIUM)
        text(s,body,2.65,y+.68,15.6,.52,24,DARK)
    foot(s,'안전을 단정하는 결론이 아니라, 비교 가능한 근거를 제공하는 의사결정 지원 서비스입니다.')
    notes(s,'발표핵심메시지3개정리. 범죄노출대리지표·지역단위·가격해석의제한은앞선슬라이드와일관되게유지한다.',source_service)
    # 30 Q&A keeps original closing design
    s=prs.slides.add_slide(prs.slide_layouts[6]);copy_shapes(ending,s);pages.append(s)
    change(s.shapes[4],'Q & A')
    change(s.shapes[5],'감사합니다')
    text(s,'SPOT FIT  |  아이와 함께, 더 안심되는 동네',3.0,7.3,14,.55,25,GREEN,align=PP_ALIGN.CENTER)
    notes(s,'기존종료페이지디자인유지. 부록은발표질의응답때참고.',[SOURCE])
    # 31 appendix
    s=slide('A','부록 · 주요 수치와 해석 기준','발표 시 제외 가능  |  지표의 대상·시점·서비스 연결 범위를 확인')
    table(s,[['질문','답변 요약'],['범죄 84%는 무엇인가?','2년 모델 CV의 Y 순위 5분위 일치율. 실제 아동 피해 예측 정확도가 아님.'],['2027·2028을 지금 검증했나?','아니오. 과거 목표연도 CV·2025 홀드아웃으로 평가한 뒤 미래값을 예측.'],['왜 점수가 소수점인가?','각 Feature의 5분위 점수에 가중치를 곱해 합산. 등급은 최종 순위로 구분.'],['행정동별 안전점수가 다른가?','아니오. 안전점수는 구 단위이며, 동 단위로 달라지는 것은 가격.'],['미래가격·교통 연구 모델도 연결됐나?','현재 화면은 대표가격·2025 교통 참고값. 연구 모델과 배포 범위 구분.'],['Airflow·DB가 모두 자동 운영 중인가?','인프라 구성·ERD 설계 단계. 업무 DAG·테이블·자동 배포는 후속 구현.']],1.25,4.4,17.5,4.65,[4.7,12.8],20)
    foot(s,'세부 수치·근거 파일·초안 대비 보정 내용은 각 슬라이드 발표자 노트와 함께 제공됩니다.')
    notes(s,'부록원문요청의모델지표·ERD·테스트는각본문페이지로분산했고이장에서는예상질의응답요약. 상세수치근거는notes에경로수록.',source_model+source_service+[OUT/'capture_checks.json'])

    # Remove original deck pages, retaining referenced theme, masters, artwork.
    for sid in original:
        prs.part.drop_rel(sid.rId)
        prs.slides._sldIdLst.remove(sid)
    # Explicit East Asian typefaces preserve the user's typography on every new run.
    for s in prs.slides:
        for sh in walk(s.shapes):
            frames=[]
            if sh.has_text_frame: frames.append(sh.text_frame)
            if sh.has_table: frames.extend(c.text_frame for row in sh.table.rows for c in row.cells)
            for tf in frames:
                for p in tf.paragraphs:
                    for run in p.runs:
                        name=run.font.name or FONT
                        rp=run._r.get_or_add_rPr()
                        ea=rp.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
                        if ea is None: ea=OxmlElement('a:ea');rp.append(ea)
                        ea.set('typeface',name)
    assert len(prs.slides)==31
    prs.save(DEST)
    loaded=Presentation(DEST)
    assert len(loaded.slides)==31
    manifest={'output':str(DEST),'slide_count':31,'draft':str(DRAFT),'design_reference':str(SOURCE),
              'checked_at':'2026-11-16','screenshots':'Live localhost:8080 read-only walkthrough',
              'changes':['범죄점수: Y순위 단순배점 → Feature 영향력 가중합 실제로직',
                         '2027/2028 범죄 서비스연동 확인',
                         '연구용 교통·미래가격 모델과 현재 서비스출력 분리',
                         'Airflow·DB·ERD는 설계/확장 단계로 명시',
                         '매매 v64 연구기록과 저장 v55 번들 구분',
                         '월세 기존 실험과 저장 번들 평가값 구분',
                         '실제 UI/API 시연동선 5건 확인'],
              'model_training_performed':False}
    (OUT/'presentation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(DEST)


if __name__=='__main__':
    main()
