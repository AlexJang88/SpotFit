"""Fill the supplied SpotFit template without replacing its theme or fonts.

Existing decoration, master, theme and typography are retained. All added tables,
diagrams and charts use the template's NEMO Gothic fonts and green-grey palette.
Model numbers are read from saved metadata; no models are retrained here.
"""
from copy import deepcopy
from pathlib import Path
import json

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/M/Desktop/SPOT_FIT_발표자료.pptx')
OUT = ROOT / 'artifacts/presentations'
REVIEW = OUT / 'template_review'
DEST = OUT / 'SPOT_FIT_프로젝트발표_템플릿유지_20261110.pptx'
FONT = '네모고딕'
LIGHT = '네모고딕 Light'
MEDIUM = '네모고딕 Medium'
DARK = '354241'
GREEN = '617B78'
TEAL = '8EB3AD'
PALE = 'E9F5F1'
GREY = '717272'
LINE = 'C9D3D0'
WHITE = 'FFFFFF'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def walk(shapes):
    for shape in shapes:
        yield shape
        if shape.shape_type == 6:
            yield from walk(shape.shapes)


def change(shape, value, height=None):
    """Keep the original run/paragraph formatting, including East Asian fonts."""
    frame = shape.text_frame
    proto_p = deepcopy(frame.paragraphs[0]._p.pPr)
    runs = [r for p in frame.paragraphs for r in p.runs]
    proto_r = deepcopy(runs[0]._r.rPr) if runs and runs[0]._r.rPr is not None else None
    frame.clear()
    for index, text in enumerate(str(value).split('\n')):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        if proto_p is not None:
            old = para._p.pPr
            if old is not None:
                para._p.remove(old)
            para._p.insert(0, deepcopy(proto_p))
        run = para.add_run()
        run.text = text
        if proto_r is not None:
            run._r.insert(0, deepcopy(proto_r))
    frame.word_wrap = True
    if height is not None:
        shape.height = Inches(height)


def put(slide, mapping):
    for idx, value in mapping.items():
        change(slide.shapes[idx], value)


def named(slide, name, value):
    shape = next(s for s in walk(slide.shapes) if s.name == name)
    change(shape, value)


def remove(slide, indices):
    shapes = list(slide.shapes)
    for index in indices:
        element = shapes[index]._element
        element.getparent().remove(element)


def text(slide, value, x, y, w, h, size=22, color=DARK, font=FONT, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.03)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for index, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.space_after = Pt(5)
        if align is not None:
            p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor.from_string(color)
    return shape


def rect(slide, x, y, w, h, fill=PALE, border=LINE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.color.rgb = RGBColor.from_string(border)
    shape.line.width = Pt(.6)
    return shape


def panel(slide, title, body, x, y, w, h):
    rect(slide, x, y, w, h, 'FAFAFA')
    rect(slide, x, y, w, .07, GREEN, GREEN)
    text(slide, title, x+.2, y+.22, w-.4, .5, 24, GREEN, MEDIUM)
    text(slide, body, x+.2, y+.96, w-.4, h-1.1, 21)


def table(slide, rows, x, y, w, h, widths=None, size=20):
    result = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for col, weight in zip(result.columns, widths):
            col.width = Inches(w * weight / sum(widths))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = result.cell(i, j)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(.12)
            cell.margin_top = cell.margin_bottom = Inches(.07)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(PALE if i == 0 else ('FFFFFF' if i % 2 else 'F5F7F6'))
            for p in cell.text_frame.paragraphs:
                p.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = MEDIUM if i == 0 else FONT
                    r.font.size = Pt(size)
                    r.font.color.rgb = RGBColor.from_string(DARK)
            # Thin template-coloured grid, not the default PowerPoint blue style.
            from pptx.oxml.xmlchemy import OxmlElement
            tcpr = cell._tc.get_or_add_tcPr()
            for edge in ('lnL', 'lnR', 'lnT', 'lnB'):
                line = OxmlElement('a:' + edge)
                line.set('w', '6350')
                fill = OxmlElement('a:solidFill')
                rgb = OxmlElement('a:srgbClr')
                rgb.set('val', LINE)
                fill.append(rgb)
                line.append(fill)
                tcpr.append(line)
    return result


def chart(slide, categories, series, x, y, w, h, maximum, major, fmt='0', label=False):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    c = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    c.has_title = False
    c.has_legend = len(series) > 1
    if c.has_legend:
        c.legend.position = XL_LEGEND_POSITION.BOTTOM
        c.legend.font.name = FONT
        c.legend.font.size = Pt(16)
    c.font.name = FONT
    c.font.size = Pt(17)
    for a in (c.category_axis, c.value_axis):
        a.tick_labels.font.name = FONT
        a.tick_labels.font.size = Pt(16)
        a.tick_labels.font.color.rgb = RGBColor.from_string(GREY)
        a.format.line.color.rgb = RGBColor.from_string(LINE)
    c.value_axis.minimum_scale = 0
    c.value_axis.maximum_scale = maximum
    c.value_axis.major_unit = major
    c.value_axis.tick_labels.number_format = fmt
    c.value_axis.has_major_gridlines = True
    c.value_axis.major_gridlines.format.line.color.rgb = RGBColor.from_string('E5EBE8')
    c.plots[0].gap_width = 75
    for index, ser in enumerate(c.series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = RGBColor.from_string([GREEN, TEAL, 'CAD8D4'][index % 3])
        ser.format.line.fill.background()
    if label:
        plot = c.plots[0]
        plot.has_data_labels = True
        plot.data_labels.position = XL_DATA_LABEL_POSITION.OUTSIDE_END
        plot.data_labels.font.name = FONT
        plot.data_labels.font.size = Pt(21)
        plot.data_labels.font.color.rgb = RGBColor.from_string(DARK)
        plot.data_labels.number_format = fmt
    return c


def note(slide, body, sources):
    slide.notes_slide.notes_text_frame.text = body + '\n\n근거 파일 (프로젝트 루트 기준):\n' + '\n'.join(sources) + '\n자료 확인일: 2026-11-10'


def fresh(prs, title, subtitle):
    """Reuse only the template's exact frame and section typography."""
    source = prs.slides[8]
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for idx in (0, 1, 2, 4, 5, 6):
        el = deepcopy(source.shapes[idx]._element)
        slide.shapes._spTree.insert_element_before(el, 'p:extLst')
    change(slide.shapes[4], title)
    slide.shapes[4].width = Inches(15.5)
    change(slide.shapes[5], subtitle)
    slide.shapes[5].width = Inches(15.5)
    return slide


def main():
    prs = Presentation(SOURCE)
    one = read('models/recommender/v1/child_crime_exposure/1y/model_metadata.json')['winner']
    two = read('models/recommender/v1/child_crime_exposure/2y/model_metadata.json')
    env = read('models/recommender/v1/child_environment/model_metadata.json')['models'][1]['holdout_metrics']
    sale = read('models/recommender/v1/apart_sale/model_contract.json')['validation']
    rent = read('models/recommender/v1/apart_rent/model_contract.json')['validation']
    traffic = read('models/recommender/v1/traffic_accident/모델정보.json')['최종검증지표']

    # 01. Cover. Preserve the distinctive original display title/font.
    s = prs.slides[0]
    named(s, 'TextBox 10', 'SPOTFIT  ·  아이와 함께, 더 안전한 동네\n프로젝트 개발 결과 발표  |  2026.11.10')
    note(s, 'SpotFit 프로젝트의 실제 코드, 모델 산출물과 현재 로컬 서비스 상태를 정리한 발표자료. 원본 템플릿의 디자인과 글꼴을 유지했다.', ['frontend/src/views/pages/HomePage.tsx', 'models/recommender/v1/'])

    # 02. Correct the original contents numbering while retaining all shapes.
    s = prs.slides[1]
    put(s, {2:'01',3:'프로젝트 개요',4:'Project Overview',14:'02',15:'시스템 아키텍처',16:'System Architecture',8:'03',9:'담당 영역 및 개발 단계',10:'Roles & Milestones',20:'04',21:'요구사항 정의서',22:'Requirements',5:'05',6:'ERD 데이터 모델',7:'Logical Data Model',17:'06',18:'UI 설계 및 실행 화면',19:'UI Design',11:'07',12:'데이터 · 모델 · 성능',13:'Data & Model Evaluation',23:'08',24:'테스트 내역서',25:'Test Report'})
    note(s, '8개 섹션. 07 섹션은 데이터 처리, 5종 모델, 범죄노출 로직, 성능 개선, 운영 한계의 5장으로 구성된다.', [])

    # 03. Overview.
    s = prs.slides[2]
    put(s, {17:'아이와 함께 살 동네를, 가격과 지역환경 데이터로 비교합니다.\n서울 자치구의 범죄노출·육아환경·교통사고를 주거 정보와 함께 지도에서 확인하는 서비스',19:'주제 선정 이유',20:'가격만으로 판단하기 어려운\n아이의 생활환경을 함께 비교',21:'핵심 과제',22:'서로 다른 지역·연도 데이터를\n통합하고 미래 순위를 예측',23:'프로젝트 목표',24:'지역점수와 예측 근거를\n지도에서 쉽게 이해하도록 제공'})
    for idx in (20,22,24):
        s.shapes[idx].top = Inches(8.9)
        s.shapes[idx].height = Inches(.85)
    note(s, '지역 비교를 지원하는 프로토타입이다. 범죄노출은 실제 아동 피해 범죄율이 아니며, 가격 모델은 미래 시장가격 예측 모델이 아니라 거래 특성 기반 가격 추정 모델이다.', ['docs/crime-exposure-service.md','frontend/src/views/pages/HomePage.tsx'])

    # 04. Architecture: replace unrelated sample chart/table with real flow.
    s = prs.slides[3]
    put(s,{7:'서비스 데이터 흐름',8:'수집 → 전처리 → 학습 → 예측 산출물 → API → 지도',13:'서비스',14:'학습',15:'수집',16:'API · 지도 UI',17:'CSV · joblib',18:'통계 · 인구 · 시설'})
    s.shapes[7].width=Inches(5)
    s.shapes[8].left=Inches(1.5); s.shapes[8].top=Inches(3.83); s.shapes[8].width=Inches(8.8)
    for idx in (16,17,18):
        s.shapes[idx].left=Inches(14.0); s.shapes[idx].width=Inches(4.4)
    remove(s,[9,19])
    for i,(title,body) in enumerate([('원천 데이터','서울 공공통계\n인구 · 실거래'),('전처리 · 학습','자치구·연도 패널\n시차·변화량 생성'),('서비스 산출물','모델·메타데이터\n자치구 예측 CSV')]):
        x=1.25+i*3.17
        panel(s,title,body,x,4.6,2.9,2.52)
        if i<2: text(s,'→',x+2.92,5.45,.28,.5,16,GREEN)
    text(s,'현재 범죄 API: 미리 생성된 예측 CSV 조회 → JSON 응답',1.3,7.35,9.2,.6,19,GREEN)
    table(s,[['구분','현재 적용','연결 방식','운영 참고'],['웹 서비스','Nginx · React · FastAPI','HTTP / JSON','Docker Compose'],['범죄 모델','models/recommender/v1','1년·2년 예측 CSV 조회','학습과 조회 분리'],['저장 인프라','MinIO · PostgreSQL/PostGIS','모델 저장소 / DB 인프라','상세 ORM·스케줄 자동화는 확장 과제']],1.25,8.35,17.5,1.95,[1.6,3.2,3.2,4],17)
    note(s,'실행 중인 Compose 서비스는 nginx/frontend/backend/minio/postgres다. 범죄 요청마다 joblib.predict를 수행하지 않고, 배포된 CSV를 읽고 캐시한다. Airflow/MLflow 구성의 존재를 전체 자동화 완성으로 표현하지 않았다.', ['compose.yaml','backend/app/services/child_crime_exposure.py','backend/app/services/crime_model_training.py'])

    # 05. Responsibility areas, not invented teammates or dates.
    s=prs.slides[4]
    put(s,{14:'담당 영역 및 개발 단계',16:'실명·개별 일정 대신, 현재 프로젝트 코드의 구현 영역을 기준으로 정리했습니다.',17:'진행 흐름: 원천 확보 → 전처리·모델 비교 → 모델 패키징 → API 연동 → 지도 UI·검증',18:'',21:'',24:'',27:'DATA & ML',28:'데이터 · 모델링',20:'수집 → 전처리 → 검증',19:'지역·연도별 패널 구축\n변수 조합·모델 기법 비교\n최종 모델·평가 결과 저장',29:'BACKEND',30:'API · 배포 환경',23:'산출물 → API → 배포',22:'예측값·지역점수 조회\n모델 누락 상태 확인·재생성\nDocker·MinIO 환경 구성',31:'FRONTEND',32:'지도 · 사용자 경험',26:'조회 → 비교 → 상세 확인',25:'시·구·행정동 지도 탐색\n지역 검색·등급 범례·순위\n상세 패널·가독성 개선'})
    s.shapes[14].width=Inches(12);s.shapes[14].height=Inches(.6)
    for idx in (19,22,25):
        s.shapes[idx].top=Inches(8.4);s.shapes[idx].height=Inches(1.2)
    note(s,'실제 팀원 이름이나 담당자 확정 자료가 없으므로 역할별 기능을 요약했다. 일정은 작업 순서이며 특정 개인의 작업 기간을 의미하지 않는다.', ['ml/training/','backend/app/services/','frontend/src/views/pages/'])

    # 06. Functional/nonfunctional requirements.
    s=prs.slides[5]
    put(s,{14:'NFR-01 데이터 정합성',15:'NFR-02 서비스 안정성',16:'NFR-03 학습 API 보호',17:'NFR-04 결과 해석',18:'지역·연도·단위·산식 일치',19:'누락 산출물 확인 및 상태 안내',20:'관리키 헤더로 학습 요청 보호',21:'예측연도·지표 한계를 함께 고지',22:'품질',23:'운영',24:'보안',25:'설명',28:'구현 기준: 범죄노출은 자치구 단위로 제공합니다. 행정동에는 자치구 예측을 개별 동의 추정값처럼 표시하지 않습니다.'})
    s.shapes[9].width=Inches(6.5)
    remove(s,[26])
    table(s,[['ID','기능 요구사항','현재 범위'],['FR-01','서울 지역 검색·지도 탐색','시 / 구 / 행정동'],['FR-02','지역점수 비교·범례','선택 영역 동일비중'],['FR-03','범죄노출 예측 조회','2026년 / 2027년'],['FR-04','순위·등급·상세 확인','자치구 25개'],['FR-05','모델 상태·재생성 API','관리자 요청 방식']],1.25,4.0,8.5,5.9,[1.4,4.1,3.1],20)
    note(s,'요구사항은 코드에서 확인되는 구현 범위로 작성했다. 모델 재생성은 준비된 전처리 패널과 의존성이 필요하며 수집부터 완전 자동 복원하는 API가 아니다. force 옵션의 동작까지 검증한 자료는 아니다.', ['frontend/src/views/pages/HomePage.tsx','backend/app/services/crime_model_training.py','backend/app/services/child_crime_exposure.py'])

    # 07. Clearly mark ERD as a proposed logical schema.
    s=prs.slides[6]
    put(s,{16:'ERD 데이터 모델 · 설계안',17:'Logical Data Model · Not Yet Implemented',18:'',20:'',22:'',23:'',26:'REGION',27:'지역 · 경계 버전',19:'지역코드 · 지역명\n상위 지역 · 유효기간',28:'DATA_SOURCE',29:'원천 · 배포 이력',21:'출처 · 기준연도\n배포일 · 데이터 버전',30:'MODEL_VERSION',32:'모델 · 검증 이력',24:'모델명 · 예측기간\nFeature · 지표 · 파일 경로',31:'FORECAST',33:'지역별 예측 결과',25:'기준연도 · 목표연도\n예측 Y · 순위 · 등급',35:'관계 및 저장 원칙 · 현재 서비스는 CSV / JSON / joblib 산출물을 사용',41:'1 : N',42:'1 : N',43:'1 : N',44:'지역 계층',45:'예측 식별',46:'지역 → 예측',47:'모델 → 예측',48:'출처 → 데이터 버전',49:'시 → 구 → 행정동',50:'지역 × 기준연도\n× 예측기간 × 모델'})
    s.shapes[16].width=Inches(14);s.shapes[17].width=Inches(14)
    for idx in (19,21,24,25):
        s.shapes[idx].height=Inches(1.0)
    s.shapes[35].width=Inches(16)
    s.shapes[50].height=Inches(.75)
    note(s,'기존 ERD 초안을 범죄 예측 서비스 중심의 4개 핵심 개념으로 요약한 논리 설계다. 현재 backend/app/models에 서비스 ORM 구현이 없으며 이 페이지는 실제 생성된 테이블을 의미하지 않는다.', ['docs/architecture/spotfit-erd-draft.md','backend/app/models/__init__.py'])

    # 08. Real service capture; retain the template's right-hand four-step cards.
    s=prs.slides[7]
    put(s,{13:'UI 설계 및 실행 화면',15:'현재 로컬 실행 화면',23:'사용자 탐색 흐름',28:'시 · 구 · 행정동 이동',29:'관심 영역 복수 선택',30:'자치구 순위 · 등급 확인',31:'연도와 결과 상세 확인',32:'01  지역 탐색',33:'02  비교 기준 설정',34:'03  지역점수 비교',35:'04  상세 정보',36:'검색과 지도 확대 수준으로 지역 선택',37:'교통사고 · 아동 범죄노출 · 육아환경',38:'범례와 순위 목록으로 상대 위치 파악',39:'행정동에는 자치구 전용 평가를 숨김'})
    s.shapes[13].width=Inches(12);s.shapes[15].width=Inches(8)
    s.shapes[23].width=Inches(6.8)
    remove(s,[4,9,10,11,16,17,18,19,20,21])
    s.shapes.add_picture(str(REVIEW/'service-ui.png'), Inches(1.25),Inches(4.1),width=Inches(9.0),height=Inches(5.625))
    text(s,'2026.11.10 실행 화면 · 배경지도 키 미설정 / 일부 가격값 미연결\n행정동 기준월 표시는 레지스트리 값이며, 공식 최신 여부는 별도 확인 필요',1.25,9.82,9.1,.6,15,GREY,LIGHT)
    note(s,'실제 로컬 캡처. VWorld 키가 없어 배경 타일은 표시되지 않지만 행정경계와 지역점수는 보인다. 캡처의 행정동 2026.10은 현재 레지스트리 표시값이며 공식 최신 경계 확정 근거가 아니다. 주거 가격은 일부 원자료 미탑재 상태이다.', ['frontend/src/views/pages/HomePage.tsx','frontend/src/views/pages/readability-refresh.css','artifacts/presentations/template_review/service-ui.png'])

    # 09. Data coverage and preprocessing.
    s=prs.slides[8]
    put(s,{5:'데이터 통합 및 전처리',6:'Data Preparation & Coverage',7:'범죄노출 Y가 있는 자치구 수',8:'연도별 결합 결과 · 25개 자치구 기준',9:'단위: 자치구 수',12:'전처리 원칙',16:'지역 단위 통일',17:'연도 · 시차 정합성',18:'학습 데이터 구성',19:'행정동 Feature는 자치구로 집계',20:'발표 지연과 과거 시차를 구분',21:'현재값 + 과거값 + 변화량',22:'25개 구',23:'연간 패널',24:'시간순 검증',29:'현재 범죄노출 모델의 핵심 데이터 범위'})
    s.shapes[5].width=Inches(14);s.shapes[7].width=Inches(9)
    change(s.shapes[8],'2010–2025 · 연도별 유효 Y 관측 구 수')
    s.shapes[8].left=Inches(1.53);s.shapes[8].top=Inches(3.88);s.shapes[8].width=Inches(8.5)
    s.shapes[9].left=Inches(7.6);s.shapes[9].top=Inches(7.8);s.shapes[9].width=Inches(2.8)
    s.shapes[29].width=Inches(16)
    remove(s,[10,30])
    panel_data=pd.read_csv(ROOT/'artifacts/child_crime_borough_v2/panel.csv')
    counts=panel_data.groupby('year').y.count()
    chart(s,[str(v)[2:] for v in counts.index],[('Y 관측 자치구',counts.tolist())],1.25,4.45,9.1,3.25,25,5)
    table(s,[['자료 / 산출물','기간 · 범위','적용 내용'],['연령별 인구 패널','2010–2025 / 서울 25구','0–17세 아동인구 산출'],['범죄건수와 Y 유효값','2014–2025 / 연 25구','5대범죄 ÷ 아동인구 × 10,000'],['2년 모델 최종 학습쌍','입력 2015–2023 → 정답 2017–2025','225개 자치구·연도 표본']],1.25,8.7,17.5,1.7,[3.4,5,6],17)
    note(s,'기본 패널은 2010–2025, 400행이다. Y와 범죄 건수 유효값은 2014년부터다. 그래프는 핵심 Y의 유효 구 수이며 모든 보조 Feature가 이 전체 기간을 충족한다는 뜻이 아니다. 1인가구로 받은 파일은 실제 가구형태별 자료여서 가구구조로 명명했다. 인허가 이력은 현재 주소와 개폐업일로 재구성하므로 과거 이전·휴업 정보 한계가 있다.', ['artifacts/child_crime_borough_v2/panel.csv','artifacts/child_crime_borough_v2/source_audit.json','models/recommender/v1/child_crime_exposure/2y/model_metadata.json'])

    # Supplemental slides reuse the exact original section header/frame.
    additions=[]
    s=fresh(prs,'프로젝트 모델 구성','Model Portfolio · Saved Evaluation Results')
    additions.append(s)
    text(s,'5개 도메인 모델을 하나의 지역 비교 서비스로 연결',1.25,3.25,17.5,.6,27,GREEN,MEDIUM)
    table(s,[['도메인','기법 / 출력','저장된 주요 평가 지표','검증 범위'],['아파트 매매','Random Forest / 만원·평 추정',f'MAE {sale["mae"]:,.1f} · R² {sale["r2"]:.3f}','2025 시간 홀드아웃'],['아파트 전세','Random Forest / 만원·평 추정',f'MAE {rent["mae"]:,.1f} · R² {rent["r2"]:.3f}','2024 시간 홀드아웃'],['아동 물리환경','Extra Trees / 3년 후 환경점수',f'MAE {env["mae"]:.2f}점 · R² {env["r2"]:.3f}','2024 홀드아웃'],['아동 지역범죄 노출','Ridge / 1년·2년 후 Y와 순위','2년: 순위 상관 0.968 · 5분위 76%','2025 홀드아웃'],['교통사고','Ridge / 다음 해 사고 건수',f'MAE {traffic["MAE"]:.1f}건 · R² {traffic["R2"]:.3f}','2025 최종 검증']],1.25,4.2,17.5,4.6,[2.8,5.3,5.5,3.9],21)
    rect(s,1.25,9.2,17.5,.9,'FAFAFA')
    text(s,'해석 주의  |  모델마다 정답·단위·검증연도가 달라 성능 수치를 직접 비교할 수 없습니다.\n가격 모델은 거래 특성 기반 추정 모델이며, 미래 시장가격 예측 성능을 검증한 모델은 아닙니다.',1.5,9.32,17,.8,19,GREY)
    note(s,'메타데이터에 저장된 성능을 그대로 요약했다. 이번 발표 작업에서 모델을 다시 학습하거나 모든 도메인의 추론을 재검증하지 않았다. 아동환경 모델은 1년 Ridge도 보유하나 표에서는 사용자 관심인 3년 모델만 제시했다.', ['models/recommender/v1/apart_sale/model_contract.json','models/recommender/v1/apart_rent/model_contract.json','models/recommender/v1/child_environment/model_metadata.json','models/recommender/v1/traffic_accident/모델정보.json','models/recommender/v1/child_crime_exposure/2y/model_metadata.json'])

    s=fresh(prs,'아동 지역범죄 노출 · 모델과 점수','Crime Exposure · Target, Features & Scoring')
    additions.append(s)
    rect(s,1.25,3.3,17.5,1.0,PALE)
    text(s,'Y = log(1 + 5대범죄 발생건수 ÷ 0–17세 아동인구 × 10,000)',1.55,3.52,17,.55,28,DARK,MEDIUM)
    table(s,[['서비스 모델','예측 방식','입력 → 목표연도','사용자 관점: 2026년 현재'],['1년 모델','Ridge α=1 · Y 변화량 예측','2025 → 2026','올해 수준 추정'],['2년 모델','Ridge α=10 · 미래 Y 직접 예측','2025 → 2027','내년 수준 예측']],1.25,4.65,17.5,1.9,[2.4,5.2,3.6,5.0],21)
    panel(s,'공통 Feature · 수치 12개 + 자치구','현재: Y · log 범죄건수 · log 아동인구\n과거: 각 변수의 1년 전 · 2년 전 값\n변화: 각 변수의 전년 대비 변화량\n지역: 자치구 One-hot 인코딩',1.25,6.9,8.5,3.25)
    panel(s,'서비스 점수 · 낮은 노출이 높은 점수','예측 Y가 낮은 순서로 25개 구 정렬\n5개 구씩 A / B / C / D / E 부여\n100 / 80 / 60 / 40 / 20점으로 환산\n현재 점수는 Feature 가중합 방식이 아님',10.05,6.9,8.7,3.25)
    note(s,'Y는 지역의 전체 5대범죄를 아동인구로 표준화한 상대 노출지수다. 실제 아동 피해 확률이나 아동 대상 범죄율로 해석할 수 없다. 1년 모델은 최신 Y에 예측한 변화량을 더한다. 2년 모델은 Y를 직접 예측한다. 수치 Feature 전체: '+', '.join(one['features'])+'. 자치구 one-hot을 추가한다. 3년 모델은 별도 실험 산출물이 있지만 현재 서비스는 1·2년을 사용한다.', ['ml/training/train_service_crime_exposure_1y.py','ml/training/optimize_exposure_2y_2027.py','scripts/child_crime/export_service_crime_forecasts.py','backend/app/services/child_crime_exposure.py'])

    s=fresh(prs,'범죄노출 모델 · 성능 개선 결과','Model Improvement · Time-based Validation')
    additions.append(s)
    text(s,'2년 모델: 5분위 일치율 개선 과정',1.25,3.3,10,.55,26,GREEN,MEDIUM)
    text(s,'교차검증 2022–2024 평균 · 단위 %',1.25,3.9,10,.4,18,GREY)
    base=two['baseline_cv']; cv=two['strict_winner_cv']; hold=two['strict_holdout']; bh=two['baseline_holdout']
    chart(s,['기준값 유지','Ridge · 핵심 8개','Ridge · 추세 12개'],[('5분위 일치율',[base['exact_quintile_pct'],two['strict_forward_history'][0]['exact_quintile_pct'],cv['exact_quintile_pct']])],1.25,4.45,9.3,3.45,100,20,'0.0',True)
    panel(s,'개선된 Feature와 로직','기준: 현재 Y가 미래에도 유지\n1단계: 현재값·Y 시차·변화량 학습\n2단계: 범죄·인구의 1·2년 시차 추가\n기법: Ridge 정규화 · 미래 Y 직접 예측',11.0,3.35,7.75,3.25)
    text(s,f'CV 5분위 일치율 +{cv["exact_quintile_pct"]-base["exact_quintile_pct"]:.1f}%p\n평균 순위오차 {base["rank_MAE"]:.2f} → {cv["rank_MAE"]:.2f}위',11.2,6.9,7.2,1.0,26,GREEN,MEDIUM)
    table(s,[['평가 대상','순위 상관 ↑','순위오차 ↓','5분위 일치 ↑','상위 20% 적중 ↑'],['1년 모델 / CV',f'{one["Spearman"]:.3f}',f'{one["rank_MAE"]:.2f}위',f'{one["exact_quintile_pct"]:.1f}%',f'{one["top20_overlap_pct"]:.0f}%'],['2년 기준 / 2025 별도검증',f'{bh["Spearman"]:.3f}',f'{bh["rank_MAE"]:.2f}위',f'{bh["exact_quintile_pct"]:.0f}%',f'{bh["top20_overlap_pct"]:.0f}%'],['2년 최종 / 2025 별도검증',f'{hold["Spearman"]:.3f}',f'{hold["rank_MAE"]:.2f}위',f'{hold["exact_quintile_pct"]:.0f}%',f'{hold["top20_overlap_pct"]:.0f}%']],1.25,8.3,17.5,1.65,[5.3,3,3,3,3.4],18)
    text(s,'별도검증: 등급 일치는 +8%p, 순위오차·Y 오차는 악화 → 모든 지표의 개선을 의미하지 않음',1.25,10.12,17.5,.38,17,GREY,LIGHT)
    note(s,'기준모델은 입력시점 Y 유지(Persistence). 2년 CV에서 5분위 일치율 65.333→81.333→84%, 최종 순위 상관 .974615, 순위오차1.066667, 상위20%100%. 2025 홀드아웃에서는 5분위68→76%지만 Spearman .972308→.967692, 순위오차1.2→1.36, Y RMSE .092828→.101463으로 악화했다. 상위20%는 범죄노출이 높은 실제 상위5개 구와 예측 상위5개 구의 겹침이다. CV는 모델 선택에 사용되었으며 독립 정답률이 아니다. 1년과2년 검증 학습창도 동일하지 않아 수평 비교는 제한된다. 1년 모델은 현재 메타데이터에 별도 홀드아웃 성능이 없다.', ['models/recommender/v1/child_crime_exposure/1y/model_metadata.json','models/recommender/v1/child_crime_exposure/2y/model_metadata.json','ml/training/optimize_exposure_2y_2027.py'])

    s=fresh(prs,'모델 해석과 다음 개선 과제','Limitations & Next Steps')
    additions.append(s)
    text(s,'현재 결과를 설명할 때 구분해야 하는 세 가지',1.25,3.3,17.5,.6,27,GREEN,MEDIUM)
    panel(s,'01  지표의 의미','전체 5대범죄 ÷ 아동인구의 상대지수\n실제 아동 피해율·개인 위험도 아님\n범죄가 많아도 아동인구 규모에 영향\n\n다음: 유형별 범죄·선행변수 검토',1.25,4.3,5.6,4.6)
    panel(s,'02  예측 성능의 한계','자치구 25개 · 연간 표본이 적음\n공통 연도 충격·순위 변동에 민감\nCV 개선이 별도검증 개선은 아님\n\n다음: 연도별 재검증·오차 추적',7.17,4.3,5.6,4.6)
    panel(s,'03  서비스 운영의 한계','범죄 모델은 자치구 단위로 제공\n재생성에는 전처리 패널이 필요\n행정동 공지 자동수집은 미구현\n\n다음: 데이터·경계 버전 자동 관리',13.1,4.3,5.65,4.6)
    rect(s,1.25,9.35,17.5,.75,PALE)
    text(s,'설명 가능한 지역 비교를 목표로, 모델의 순위 성능과 데이터 최신성을 함께 검증합니다.',1.5,9.5,17,.5,23,DARK,MEDIUM)
    note(s,'추가 Feature나 범죄·인구의 개별 모델 오차 개선이 Y의 순위 개선을 보장하지 않는다. 인구 분모가 작아지면 노출지수가 상승할 수 있다. 현재 사용자 점수는 예측 Y의 5분위이고, 향후 통합 Feature 영향력 기반 가중합 점수는 별도 정의·검증이 필요하다. 행정동 공지 자동수집 및 신뢰 가능한 공식 변경 예정 알림은 미완성 과제다.', ['backend/app/services/child_crime_exposure.py','backend/app/services/crime_model_training.py','docs/crime-exposure-service.md','artifacts/child_crime_exposure_2y_2027_v1/result.json'])

    # 14. Test report: only checks executed for this presentation.
    s=prs.slides[9]
    put(s,{14:'발표자료 작성 시 수행한 점검',15:'2026.11.10 로컬 검증 요약',16:'선택한 점검 항목',17:'선택 항목 통과',18:'검증 범위',19:'5개',20:'5 / 5',21:'핵심 경로',37:'TC-01  1년 범죄노출 API',38:'TC-02  2년 범죄노출 API',39:'TC-03  지역 필터 · 예외 처리',40:'TC-04  점수 · 등급 · 산식 정합성',41:'TC-05  프론트엔드 타입 검사',42:'25개 구 · 2025년 입력 → 2026년 결과',43:'25개 구 · 2025년 입력 → 2027년 결과',44:'강남구 결과 동일 · 없는 지역은 404',45:'두 기간 50행 · 순위와 A–E 매핑 일치',46:'npm run typecheck 정상 종료',47:'API 4종 + 타입 검사',48:'전체 회귀·부하 검사는 아님',49:'외부 연동 별도 점검'})
    for idx in (19,20,21):
        s.shapes[idx].width=Inches(5.8);s.shapes[idx].height=Inches(1.0)
    for idx in (47,48,49):
        s.shapes[idx].left=Inches(14.2);s.shapes[idx].width=Inches(4.1)
    for i in range(5):
        group=s.shapes[32+i]
        for sh in walk(group.shapes):
            if sh.has_text_frame and sh.text:
                change(sh,'통과')
            if sh.shape_type == 5:
                try:
                    sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(TEAL)
                except (AttributeError,TypeError):pass
    note(s,'이 발표자료를 만드는 동안 실제로 실행한 읽기 전용 점검이다. scripts/verify_spotfit_deck_evidence.py가 API 4개 그룹을 검증했고, docker compose exec -T frontend npm run typecheck가 exit0으로 종료했다. 전체 테스트 스위트 통과율이나 알려진 결함0건을 뜻하지 않는다. Docker backend/frontend/minio/postgres healthy 상태와 nginx 실행을 확인했다.', ['scripts/verify_spotfit_deck_evidence.py','artifacts/presentations/template_review/verification.json','frontend/package.json'])

    # 15. Closing: no invented email or telephone details.
    s=prs.slides[10]
    named(s,'TextBox 10','SpotFit  ·  아이와 함께 살 동네를 데이터로 비교합니다.\n지역의 차이를 보여주고, 예측의 한계까지 설명하는 서비스')
    named(s,'TextBox 13','프로젝트 개발 결과 발표  |  2026.11.10\n질의응답')
    note(s,'발표자 연락처는 제공되지 않아 임의 연락처를 기재하지 않았다. 질문 시 모델 성능, 데이터 최신성, 서비스 구현 범위를 구분해 설명한다.',[])

    # Insert supplements after the original data slide, before tests and ending.
    ids=prs.slides._sldIdLst
    for offset,slide in enumerate(additions):
        item=next(e for e in ids if int(e.id)==slide.slide_id)
        ids.remove(item)
        ids.insert(9+offset,item)

    # Keep slide-level notes as the audit trail; no template placeholders remain.
    for number,slide in enumerate(prs.slides,1):
        for shape in walk(slide.shapes):
            if shape.has_text_frame:
                value=shape.text
                if any(token in value for token in ('[프로젝트','[입력','[화면','[담당','[모델','[수집','[전처리','[이메일','[연락처','[N]')):
                    raise ValueError(f'Placeholder in slide {number}: {value}')
    prs.save(DEST)
    print(DEST)
    print(f'Slides: {len(prs.slides)}; template dimensions: {prs.slide_width/914400} × {prs.slide_height/914400} in')


if __name__=='__main__':
    main()
