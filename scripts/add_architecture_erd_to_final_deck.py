"""Replace the two summary slides with five editable architecture/ERD slides.

Read the user's latest edited presentation, preserving all other slide content.
The logical schema is an extract of docs/architecture/spotfit-service-erd-v2.md;
it is not represented as an already-deployed database.
"""
from pathlib import Path
from copy import deepcopy
import re, json, hashlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.shapes.autoshape import CT_Shape
from build_spotfit_project_deck import text, rect, change, walk, FONT, MEDIUM, DARK, GREEN, TEAL, PALE, GREY, LINE

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/presentations/final_20261116/SPOTFIT_최종발표_기존디자인_20261116.pptx'
OUT=ROOT/'artifacts/presentations/final_architecture_erd_20261116'
DEST=OUT/'SPOTFIT_최종발표_아키텍처_ERD포함_20261116.pptx'
prs=Presentation(SOURCE)
old_ids=list(prs.slides._sldIdLst)
shell=prs.slides[19]
assert '역할을 나눠' in shell.shapes[4].text
assert len(prs.slides)==31
new=[]


def box(s,x,y,w,h,title,body,planned=False):
    b=rect(s,x,y,w,h,'F6F9F8')
    if planned:b.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    rect(s,x,y,w,.06,GREEN,GREEN)
    text(s,title,x+.16,y+.16,w-.32,.5,24,GREEN,MEDIUM)
    compact=h<1.7
    body_y=.74 if compact else .83
    tf=text(s,body,x+.16,y+body_y,w-.32,h-body_y-.06,19 if compact else 20,DARK).text_frame
    for p in tf.paragraphs:p.space_after=Pt(2)


def entity(s,x,y,w,h,title,name,fields):
    rect(s,x,y,w,h,'FFFFFF')
    rect(s,x,y,w,.65,PALE)
    text(s,title,x+.13,y+.08,w-.26,.35,21,GREEN,MEDIUM)
    text(s,name,x+.13,y+.40,w-.26,.24,12.5,GREY)
    sh=text(s,fields,x+.13,y+.76,w-.26,h-.80,15.5,DARK)
    for p in sh.text_frame.paragraphs:p.space_after=Pt(0)


def route(s,points,label=None,labelbox=None,arrow=False,double=False,dashed=False):
    for i,((x1,y1),(x2,y2)) in enumerate(zip(points,points[1:])):
        ln=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
        ln.line.color.rgb=RGBColor.from_string(GREEN)
        ln.line.width=Pt(1.5)
        if dashed:ln.line.dash_style=MSO_LINE_DASH_STYLE.DASH
        el=ln._element.spPr.get_or_add_ln()
        if i==len(points)-2 and arrow:
            tip=OxmlElement('a:tailEnd');tip.set('type','triangle');el.append(tip)
        if i==0 and double:
            tip=OxmlElement('a:headEnd');tip.set('type','triangle');el.append(tip)
    if label:
        x,y,w,h=labelbox
        text(s,label,x,y,w,h,15,GREEN,align=PP_ALIGN.CENTER)


def fresh(title,subtitle,notes):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    for shape in list(shell.shapes)[:5]:
        s.shapes._spTree.insert_element_before(deepcopy(shape._element),'p:extLst')
    change(s.shapes[3],'04')
    change(s.shapes[4],title,.75)
    rect(s,1.25,3.25,17.5,.8,PALE)
    text(s,subtitle,1.5,3.43,17,.46,23,DARK,MEDIUM)
    text(s,'- 0 -',1.25,10.2,17.5,.33,17,GREY,align=PP_ALIGN.CENTER)
    frame=s.notes_slide.notes_text_frame
    if frame is None:
        shapes=s.notes_slide.shapes
        el=CT_Shape.new_textbox_sp(max([x.shape_id for x in shapes]+[0])+1,'Architecture notes',Inches(1),Inches(1),Inches(6),Inches(6))
        shapes._spTree.insert_element_before(el,'p:extLst');frame=shapes[-1].text_frame
    frame.text=notes+'\n\n확인일: 2026-11-16\n근거: compose.yaml; backend/app/services/neighborhood_recommendations.py; backend/app/services/service_model_bootstrap.py; docs/architecture/spotfit-service-erd-v2.md\nERD는 논리 설계안이며 Migration 또는 업무 DAG가 구현 완료되었다는 의미가 아닙니다.'
    new.append(s)
    return s


def foot(s,t):text(s,t,1.28,9.74,17.4,.33,15,GREY)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    # 20 Runtime architecture. Every line corresponds to a real implemented path.
    s=fresh('서비스 아키텍처 · 화면에서 모델 결과까지',
            '현재 구현  |  Ubuntu · Docker Compose 기반 서비스 / 브라우저의 요청과 응답 흐름',
            '브라우저 React·Leaflet → Nginx /api 프록시 → FastAPI. Nginx는 화면 요청을 frontend 서비스로 프록시한다. '
            '백엔드는 models/recommender/v1 아래 모델과 예측 CSV/JSON을 검증하여 읽는다. '
            '로그는 호스트 logs/backend에 보존. 누락 복구는 입력 원본과 학습 의존성이 있어야 가능하다. '
            'PostgreSQL을 추천 서비스의 현재 주 저장소로 그리지 않았다. 범죄는2y/3y, 교통2025참고, 가격은현재대표추정값이다.')
    bg=rect(s,5.55,4.32,13.4,5.15,'FFFFFF');bg.line.color.rgb=RGBColor.from_string(LINE)
    text(s,'Ubuntu 서버  /  Docker Compose',5.75,4.42,12.8,.4,19,GREEN,MEDIUM)
    box(s,1.25,5.05,3.6,2.0,'사용자 브라우저','React · TypeScript\nLeaflet 지도\n예산·지역·연도 선택')
    box(s,5.9,5.05,3.2,1.75,'Nginx','화면 요청 / API 분기\n8080 → 내부 서비스')
    box(s,10.15,5.05,3.65,1.75,'FastAPI','추천·검색·점수 API\n검증·예외 응답')
    box(s,15.1,5.05,3.55,1.75,'모델·예측 파일','joblib / CSV / JSON\nrecommender/v1')
    route(s,[(4.85,5.95),(5.9,5.95)],'HTTP',(4.85,5.57,1.05,.3),True,True)
    route(s,[(9.1,5.95),(10.15,5.95)],'/api',(9.1,5.57,1.05,.3),True,True)
    route(s,[(15.1,5.95),(13.8,5.95)],'검증·조회',(13.8,5.57,1.3,.3),True)
    box(s,5.9,7.6,3.2,1.55,'Frontend','React 화면 제공\n지도·조건·상세 UI')
    box(s,10.15,7.6,3.65,1.55,'운영 로그','시각·요청·예외 기록\n호스트 logs/backend')
    box(s,15.1,7.6,3.55,1.55,'시작 시 누락 복구','누락 검사 → 재생성\n원본·학습환경 필요')
    route(s,[(7.5,6.8),(7.5,7.6)],'화면',(7.58,6.99,1,.3),True)
    route(s,[(11.97,6.8),(11.97,7.6)],'기록',(12.06,6.99,1,.3),True)
    route(s,[(16.88,7.6),(16.88,6.8)],'생성',(16.97,6.99,1,.3),True)
    foot(s,'조회는 파일 기반으로 동작합니다. Airflow 자동 학습·업무 DB 적재는 다음 장의 확장 설계입니다.')
    # 21 Target data/ML architecture. Whole flow explicitly a proposal.
    s=fresh('데이터·MLOps 아키텍처 · 자동화 확장 설계',
            '설계 목표  |  수집 → 전처리 → 5개 모델 학습 → 품질·성능 검증 → 서비스 배포',
            'Compose에는 Airflow·PostgreSQL·MinIO·MLflow가 구성되어 있으나 업무 DAG와 DB migration은 설계 단계다. '
            '최종모델5종은범죄노출/교통안전/매매/전세/월세. 원본과binary는객체저장소, 정형정보·버전·이력은DB, 실험은MLflow역할이다. '
            '검증 실패시 기존 운영모델 유지, 중복원본검사와 공표시점검증·시간순평가를 배포 전 조건으로 제안한다.')
    text(s,'Airflow 오케스트레이션  ·  아래 단계 연결은 구축 예정',1.3,4.35,17.3,.45,22,GREEN,MEDIUM)
    stages=[('수집','공공 파일·API\n출처·기준일 기록'),('전처리','지역·기간 정합성\nFeature 생성'),('모델 학습','범죄·교통\n매매·전세·월세'),('배포 전 검증','데이터 품질검사\n시간순 성능평가'),('서비스 배포','통과 모델·예측값\nAPI → 화면 반영')]
    xs=[1.25,4.85,8.45,12.05,15.65]
    for i,((t,b),x) in enumerate(zip(stages,xs)):
        box(s,x,5.0,3.1,1.85,t,b,True)
        if i<4:route(s,[(x+3.1,5.9),(xs[i+1],5.9)],arrow=True,dashed=True)
    box(s,1.25,7.8,5.3,1.6,'MinIO · 파일 저장','원본 / 정제 파일 / 모델 아티팩트',True)
    box(s,7.35,7.8,5.3,1.6,'MLflow · 실험 관리','파라미터 / 성능지표 / 모델 버전',True)
    box(s,13.45,7.8,5.3,1.6,'PostgreSQL · PostGIS','지역·이력 / 예측·점수 / 실행 상태',True)
    route(s,[(2.8,6.85),(2.8,7.8)],'원본·가공본',(3,7.15,2,.32),True,dashed=True)
    route(s,[(10,6.85),(10,7.8)],'학습 기록',(10.15,7.15,1.7,.32),True,dashed=True)
    route(s,[(17.2,6.85),(17.2,7.8)],'조회용 결과',(14.9,7.15,2.15,.32),True,dashed=True)
    foot(s,'점선 = 확장 설계 경로. 검증 실패 시 기존 운영 모델을 유지하며, 원본·모델 Binary는 DB가 아닌 파일 저장소에 보관합니다.')
    # 22 ERD data lineage and authoritative administrative-area relationships.
    s=fresh('ERD ① · 원본 수집과 행정구역 변경 이력',
            '논리 설계안  |  PK: 기본키 · FK: 외래키 · 1:N: 하나의 부모에 여러 하위 기록',
            '기존ERD v2의핵심관계를추출. META_DATA_SOURCE1:N META_INGESTION_RUN1:N META_DATA_ASSET_VERSION. '
            'REF_BOUNDARY_NOTICE와REF_ADMIN_AREA는연결테이블REF_BOUNDARY_NOTICE_AREA로N:M구현. '
            '연결테이블복합PK는notice_id,area_id,relation_type. REF_ADMIN_AREA.parent_area_id는자기참조FK. '
            '원본체크섬중복검사, 지역valid_from/valid_to이력보존. CORE_*통계테이블의area_id/asset_version_idFK는발표축약도에서생략.')
    y=4.45;h=1.95;w=5.0;xx=[1.25,7.5,13.75]
    entity(s,xx[0],y,w,h,'데이터 출처','META_DATA_SOURCE','PK source_id\n기관 · 원천 URL · 갱신주기')
    entity(s,xx[1],y,w,h,'수집 실행','META_INGESTION_RUN','PK ingestion_run_id  |  FK source_id\n수집 기준기간 · 시작/종료 · 상태')
    entity(s,xx[2],y,w,h,'원본 파일 버전','META_DATA_ASSET_VERSION','PK asset_version_id\nFK ingestion_run_id\n파일 URI · 체크섬 · 수집시각')
    route(s,[(6.25,5.5),(7.5,5.5)],'1 : N',(6.25,5.13,1.25,.3))
    route(s,[(12.5,5.5),(13.75,5.5)],'1 : N',(12.5,5.13,1.25,.3))
    y=7.4
    entity(s,xx[0],y,w,h,'행정구역 변경 공고','REF_BOUNDARY_NOTICE','PK notice_id\n공고일 · 시행일 · 상태 · 원문 URL')
    entity(s,xx[1],y,w,h,'공고 ↔ 지역 연결','REF_BOUNDARY_NOTICE_AREA','PK/FK notice_id, area_id\nPK relation_type: OLD / NEW / AFFECTED')
    entity(s,xx[2],y,w,h,'자치구·행정동','REF_ADMIN_AREA','PK area_id  |  FK parent_area_id\n지역코드 · 구/동 구분 · 경계\nvalid_from / valid_to 이력 보존')
    route(s,[(6.25,8.45),(7.5,8.45)],'1 : N',(6.25,8.08,1.25,.3))
    route(s,[(13.75,8.45),(12.5,8.45)],'N : 1',(12.5,8.08,1.25,.3))
    text(s,'수집 계보: 어디서 어떤 원본을 받았는가',1.3,6.65,8,.35,17,GREY)
    text(s,'지역 이력: 공고 한 건이 여러 동의 분동·통합과 연결',9.4,6.65,9.2,.35,17,GREY)
    foot(s,'핵심 관계 발췌. 인구·범죄·교통·거래 CORE 테이블은 area_id와 asset_version_id로 지역·원본 버전을 함께 참조합니다.')
    # 23 ERD model lineage with simple branching below the selected model.
    s=fresh('ERD ② · 학습 데이터에서 모델·성능·배포까지',
            '논리 설계안  |  어떤 데이터와 학습 실행으로 만들어진 모델인지 추적',
            'FEATURE_DATASET_VERSION1:N ML_TRAINING_RUN1:N ML_MODEL_VERSION. 모델버전은성능지표·Feature영향력·배포기록을1:N보유. '
            '모델type은5영역을구별하고prediction_horizon으로기간구분. 실제표시필드는기존v2논리ERD에서추출. '
            'pipeline_run_id참조대상META_PIPELINE_RUN과품질검사테이블은이장생략. 동일type·예측기간별활성champion1개제약을도입하는설계.')
    y=4.45;h=1.95
    entity(s,1.25,y,5,h,'Feature 데이터셋 버전','FEATURE_DATASET_VERSION','PK feature_dataset_id\nFK pipeline_run_id\n가공버전 · 기준일 · 품질 통과 여부')
    entity(s,7.5,y,5,h,'모델 학습 실행','ML_TRAINING_RUN','PK training_run_id\nFK feature_dataset_id, pipeline_run_id\nMLflow Run · 알고리즘 · 실행상태')
    entity(s,13.75,y,5,h,'모델 버전','ML_MODEL_VERSION','PK model_version_id  |  FK training_run_id\n모델유형 · 예측기간 · 아티팩트 URI\n버전 · 상태 · Champion 여부')
    route(s,[(6.25,5.5),(7.5,5.5)],'1 : N',(6.25,5.13,1.25,.3))
    route(s,[(12.5,5.5),(13.75,5.5)],'1 : N',(12.5,5.13,1.25,.3))
    # Shared parent bus routed through the empty center gutter.
    route(s,[(16.25,6.4),(16.25,7.08),(3.75,7.08)])
    for x in (3.75,10,16.25):route(s,[(x,7.08),(x,7.65)],'1 : N',(x+.12,7.22,1,.3))
    entity(s,1.25,7.65,5,1.78,'성능지표','ML_MODEL_METRIC','PK metric_id  |  FK model_version_id\n검증구간 · 지표명 · 지표값')
    entity(s,7.5,7.65,5,1.78,'Feature·영향력','ML_MODEL_FEATURE','PK model_feature_id  |  FK model_version_id\n변수명 · 가중치% · 방향 · 표시순서')
    entity(s,13.75,7.65,5,1.78,'서비스 배포 이력','ML_MODEL_DEPLOYMENT','PK deployment_id  |  FK model_version_id\n환경 · 상태 · 배포/종료시각')
    foot(s,'모델 파일은 URI로 참조하고 DB에는 메타데이터를 보관합니다. 동일 모델유형·예측기간의 활성 Champion은 하나로 제한합니다.')
    # 24 ERD service graph; orthogonal edges avoid text boxes and crossings.
    s=fresh('ERD ③ · 예측·동네점수·비회원 추천 결과',
            '논리 설계안  |  예측값 + 점수 정책 + 예산 조건을 연결해 추천 근거를 재현',
            'SERVICE_SAFETY_PREDICTION은crime_prediction_id와traffic_prediction_id의서로다른역할로SCORE에연결된다. '
            'SERVICE_SCORING_POLICY1:N SCORE. SCORE와PROPERTY_PREDICTION은각각1:N RESULT,REQUEST1:N RESULT. '
            '실제PK는neighborhood_score_id,property_prediction_id,safety_prediction_id;결과복합PKrequest_id/recommendation_rank. '
            '공통REF_ADMIN_AREA/ML_MODEL_VERSION관계와추천정책·등급정책테이블은지면상생략. '
            '추천요청저장은선택기능. 로그인·회원테이블없음, expires_at보존정책으로이력자동삭제설계.')
    w=5.3;y=4.45;h=1.95
    entity(s,1.25,y,w,h,'범죄·교통 예측','SERVICE_SAFETY_PREDICTION','PK safety_prediction_id\nFK area_id, model_version_id\n영역 · 기준/목표연도 · 예측값·점수')
    entity(s,7.35,y,w,h,'동네점수','SERVICE_NEIGHBORHOOD_SCORE','PK neighborhood_score_id\nFK area_id, scoring_policy_id\nFK crime_prediction_id, traffic_prediction_id')
    entity(s,13.45,y,w,h,'아파트 가격 예측','SERVICE_PROPERTY_PREDICTION','PK property_prediction_id\nFK area_id, model_version_id, apartment_id\n거래유형 · 기준/목표일 · 면적·금액')
    route(s,[(6.55,5.5),(7.35,5.5)],'1:N',(6.55,5.12,.8,.3))
    text(s,'범죄·교통 2개 FK',7.45,6.55,4.6,.32,15,GREY)
    entity(s,1.25,7.65,w,1.8,'점수 산출 정책','SERVICE_SCORING_POLICY','PK scoring_policy_id\n범죄·교통 가중치 · 정책 유효기간')
    entity(s,7.35,7.65,w,1.95,'추천 결과','SERVICE_RECOMMENDATION_RESULT','PK/FK request_id  +  PK recommendation_rank\nFK area_id, property_prediction_id\nFK neighborhood_score_id  |  결과 스냅샷')
    entity(s,13.45,7.65,w,1.95,'추천 요청 · 선택 저장','SERVICE_RECOMMENDATION_REQUEST','PK request_id  |  FK recommendation_policy_id\n거래유형 · 예산범위 · 면적 · 예측연도\n요청시각 · expires_at')
    route(s,[(6.55,8.55),(6.95,8.55),(6.95,7.05),(8.05,7.05),(8.05,6.4)],'1:N',(6.98,7.12,.8,.3))
    route(s,[(10,6.4),(10,7.65)],'1:N',(10.1,7.12,.8,.3))
    route(s,[(16.1,6.4),(16.1,7.05),(12.0,7.05),(12.0,7.65)],'1:N',(12.12,7.15,.8,.3))
    route(s,[(13.45,8.55),(12.65,8.55)],'N:1',(12.65,8.17,.8,.3))
    foot(s,'로그인·개인정보 테이블 없음. 요청·결과 저장은 선택이며 보존기간을 둡니다. 공통 지역·모델 버전 FK와 일부 정책 관계는 축약했습니다.')

    # Replace only source pages 20/21; untouched user edits remain intact.
    added_ids=list(prs.slides._sldIdLst)[len(old_ids):]
    for sid in added_ids:prs.slides._sldIdLst.remove(sid)
    for sid in old_ids[19:21]:
        prs.part.drop_rel(sid.rId);prs.slides._sldIdLst.remove(sid)
    for offset,sid in enumerate(added_ids):prs.slides._sldIdLst.insert(19+offset,sid)
    assert len(prs.slides)==34
    for n,s in enumerate(prs.slides,1):
        for sh in s.shapes:
            if sh.has_text_frame and re.fullmatch(r'-\s*\d+\s*-',sh.text.strip()):change(sh,f'- {n} -')
    # Only new text needs East Asian font assignment; preserve original XML elsewhere.
    for s in new:
        for sh in walk(s.shapes):
            if not sh.has_text_frame:continue
            for p in sh.text_frame.paragraphs:
                for run in p.runs:
                    rp=run._r.get_or_add_rPr()
                    ea=rp.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
                    if ea is None:ea=OxmlElement('a:ea');rp.append(ea)
                    ea.set('typeface',run.font.name or FONT)
    prs.save(DEST)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before
    check=Presentation(DEST);assert len(check.slides)==34
    (OUT/'manifest.json').write_text(json.dumps({'source':str(SOURCE),'source_sha256':before,'output':str(DEST),
        'slides':34,'replacement':'Original slides 20–21 replaced with five native diagram slides 20–24',
        'new_pages':['20 서비스 아키텍처','21 데이터·MLOps 아키텍처','22 수집·지역 ERD','23 학습·모델 ERD','24 예측·추천 ERD'],
        'schema_status':'Logical proposal, no migration applied','other_user_edits':'Preserved; only footer page numbers updated'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(DEST)


if __name__=='__main__':main()
