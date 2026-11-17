"""Replace three ERD pages with a single high-resolution snapshot, preserving the deck."""
from copy import deepcopy
from pathlib import Path
from html import escape
import hashlib
import json
import re

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN
from pptx.oxml.shapes.autoshape import CT_Shape
from build_spotfit_project_deck import text, change, GREY, GREEN, MEDIUM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/presentations/final_architecture_erd_20261116/SPOTFIT_최종발표_아키텍처_ERD포함_20261116.pptx'
OUT = ROOT / 'artifacts/presentations/final_erd_snapshot_20261117'
DEST = OUT / 'SPOTFIT_최종발표_ERD한장통합_20261117.pptx'
W, H = 2520, 860
CW, CH = 370, 178
XS = [25 + 420*i for i in range(6)]
YS = [60, 340, 620]
svg = []


def label(x, y, value, size=24, color='#354241', weight='normal', anchor='start'):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def line(points, caption=None, at=None):
    svg.append('<polyline points="' + ' '.join(f'{x},{y}' for x,y in points) + '" fill="none" stroke="#617B78" stroke-width="2.5"/>')
    if caption:
        x,y = at
        width = len(caption)*12+14
        svg.append(f'<rect x="{x-width/2}" y="{y-19}" width="{width}" height="26" fill="white"/>')
        label(x,y,caption,20,'#617B78',anchor='middle')


def card(col,row,title,name,fields):
    x,y = XS[col], YS[row]
    svg.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="6" fill="white" stroke="#C9D3D0" stroke-width="2"/>')
    svg.append(f'<path d="M{x+6},{y} H{x+CW-6} Q{x+CW},{y} {x+CW},{y+6} V{y+52} H{x} V{y+6} Q{x},{y} {x+6},{y}" fill="#E9F5F1"/>')
    label(x+15,y+35,title,29,'#617B78','bold')
    label(x+15,y+77,name,16,'#717272')
    for i,f in enumerate(fields):
        label(x+15,y+107+27*i,f,21)


def snapshot():
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg.append('<style>text{font-family:"네모고딕","Malgun Gothic",sans-serif}</style><rect width="100%" height="100%" fill="white"/>')
    label(25,35,'01  원본 수집 · 행정구역 이력',28,'#617B78','bold')
    label(25,292,'02  학습 데이터 · 모델 버전',28,'#617B78','bold')
    label(25,592,'03  예측 · 점수 · 비회원 추천',28,'#617B78','bold')
    card(0,0,'데이터 출처','META_DATA_SOURCE',['PK source_id','제공기관 · 원천 URL','갱신주기'])
    card(1,0,'수집 실행','META_INGESTION_RUN',['PK ingestion_run_id','FK source_id','수집기간 · 실행상태'])
    card(2,0,'원본 파일 버전','META_DATA_ASSET_VERSION',['PK asset_version_id','FK ingestion_run_id','파일 URI · 체크섬'])
    card(3,0,'행정구역 변경 공고','REF_BOUNDARY_NOTICE',['PK notice_id','공고일 · 시행일','원문 URL · 상태'])
    card(4,0,'공고–지역 연결','REF_BOUNDARY_NOTICE_AREA',['PK/FK notice_id, area_id','PK relation_type','이전 · 신규 · 영향 지역'])
    card(5,0,'자치구 · 행정동','REF_ADMIN_AREA',['PK area_id','FK parent_area_id (자기참조)','지역코드 · 유효기간 · 경계'])
    card(0,1,'Feature 데이터셋','FEATURE_DATASET_VERSION',['PK feature_dataset_id','FK pipeline_run_id','가공버전 · 기준일'])
    card(1,1,'모델 학습 실행','ML_TRAINING_RUN',['PK training_run_id','FK feature_dataset_id','FK pipeline_run_id'])
    card(2,1,'모델 버전','ML_MODEL_VERSION',['PK model_version_id','FK training_run_id','모델유형 · 예측기간 · URI'])
    card(3,1,'모델 성능지표','ML_MODEL_METRIC',['PK metric_id','FK model_version_id','검증구간 · 지표값'])
    card(4,1,'Feature · 영향력','ML_MODEL_FEATURE',['PK model_feature_id','FK model_version_id','변수 · 가중치 · 방향'])
    card(5,1,'서비스 배포 이력','ML_MODEL_DEPLOYMENT',['PK deployment_id','FK model_version_id','환경 · 상태 · 배포시각'])
    card(0,2,'추천 요청','SERVICE_RECOMMENDATION_REQUEST',['PK request_id','FK recommendation_policy_id','예산 · 면적 · 예측연도'])
    card(1,2,'추천 결과','SERVICE_RECOMMENDATION_RESULT',['PK/FK request_id + PK 순위¹','FK area_id, 가격 예측 ID²','FK 동네점수 ID³'])
    card(2,2,'아파트 가격 예측','SERVICE_PROPERTY_PREDICTION',['PK property_prediction_id','FK area_id, model_version_id','FK apartment_id'])
    card(3,2,'점수 산출 정책','SERVICE_SCORING_POLICY',['PK scoring_policy_id','범죄 · 교통 가중치','정책 유효기간'])
    card(4,2,'동네점수','SERVICE_NEIGHBORHOOD_SCORE',['PK neighborhood_score_id','FK area_id, scoring_policy_id','FK 범죄 · 교통 예측 ID⁴'])
    card(5,2,'범죄 · 교통 예측','SERVICE_SAFETY_PREDICTION',['PK safety_prediction_id','FK area_id, model_version_id','영역 · 예측연도 · 예측값'])
    # Horizontal relationships: caption is left-to-right cardinality, not direction.
    for row,pairs in [(0,[(0,'1:N'),(1,'1:N'),(3,'1:N'),(4,'N:1')]),
                      (1,[(0,'1:N'),(1,'1:N')]),
                      (2,[(0,'1:N'),(1,'N:1'),(3,'1:N'),(4,'N:1')])]:
        for col,cap in pairs:
            x=XS[col]+CW; end=XS[col+1]; y=YS[row]+120
            line([(x,y),(end,y)],cap,((x+end)/2,y-12))
    # Shared model-version parent -> metric / feature / deployment children.
    line([(XS[2]+CW/2,340),(XS[2]+CW/2,310),(XS[5]+CW/2,310)])
    for col in (3,4,5):
        x=XS[col]+CW/2
        line([(x,310),(x,340)],'1:N',(x+46,329))
    # Model-version FK references for both prediction tables.
    line([(XS[2]+CW/2,518),(XS[2]+CW/2,548),(XS[5]+CW/2,548)])
    for col in (2,5):
        x=XS[col]+CW/2
        line([(x,548),(x,620)],'1:N',(x+44,609))
    # Score referenced by the recommendation result; route below cards.
    line([(XS[4]+CW/2,798),(XS[4]+CW/2,830),(XS[1]+CW/2,830),(XS[1]+CW/2,798)],'1:N',(XS[3]+CW/2,830))
    svg.append('</svg>')
    path=OUT/'SPOTFIT_ERD_통합스냅샷.svg'
    path.write_text(''.join(svg),encoding='utf-8')
    png=path.with_suffix('.png')
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':W,'height':H},device_scale_factor=2)
        page.goto(path.as_uri())
        page.evaluate('document.fonts.ready')
        page.screenshot(path=str(png))
        browser.close()
    return png


def content(slide):
    return [sh.text for sh in slide.shapes if sh.has_text_frame and not re.fullmatch(r'-\s*\d+\s*-',sh.text.strip())]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    png=snapshot()
    prs=Presentation(SOURCE)
    assert len(prs.slides)==34
    old_ids=list(prs.slides._sldIdLst)
    shell=prs.slides[21]
    preserved=[content(s) for i,s in enumerate(prs.slides) if i not in (21,22,23)]
    assert all('ERD' in prs.slides[i].shapes[4].text for i in (21,22,23))
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    for sh in list(shell.shapes)[:5]:
        slide.shapes._spTree.insert_element_before(deepcopy(sh._element),'p:extLst')
    change(slide.shapes[4],'통합 ERD · 수집부터 예측·추천까지',.75)
    text(slide,'핵심 테이블 18개  |  PK 기본키 · FK 외래키 · 1:N 일대다 관계',1.28,3.17,17.4,.44,22,GREEN,MEDIUM)
    picture=slide.shapes.add_picture(str(png),Inches(1.25),Inches(3.73),width=Inches(17.5))
    picture.name='통합 ERD 고해상도 스냅샷 (18개 핵심 테이블)'
    picture._element.nvPicPr.cNvPr.set('descr','수집·지역, 학습·모델, 예측·추천 ERD를 하나의 이미지로 통합. 상세 필드는 슬라이드 노트와 ERD 문서 참조.')
    text(slide,'공통 지역 FK·보조 테이블은 생략  |  ¹ recommendation_rank  ² property_prediction_id  ³ neighborhood_score_id',1.28,9.74,17.4,.27,13,GREY)
    text(slide,'⁴ crime_prediction_id / traffic_prediction_id → safety_prediction_id  |  추천 요청·결과 저장은 선택 · 로그인 없음',1.28,10.01,17.4,.27,13,GREY)
    text(slide,'- 22 -',1.25,10.39,17.5,.33,17,GREY,align=PP_ALIGN.CENTER)
    frame=slide.notes_slide.notes_text_frame
    if frame is None:
        shapes=slide.notes_slide.shapes
        node=CT_Shape.new_textbox_sp(max([x.shape_id for x in shapes]+[0])+1,'ERD notes',Inches(1),Inches(1),Inches(6),Inches(6))
        shapes._spTree.insert_element_before(node,'p:extLst')
        frame=shapes[-1].text_frame
    frame.text=('기존 ERD ①~③의 핵심 18개 테이블을 단일 스냅샷으로 통합. 전체 물리 스키마의 모든 테이블을 표시한 그림은 아님.\n'
        '원본 문서: docs/architecture/spotfit-service-erd-v2.md\n'
        '기존 각 페이지 설명:\n'+'\n\n'.join(sh.text for i in (21,22,23)
            for sh in prs.slides[i].notes_slide.shapes if sh.has_text_frame))
    new_id=prs.slides._sldIdLst[-1]
    prs.slides._sldIdLst.remove(new_id)
    for sid in old_ids[21:24]:
        prs.part.drop_rel(sid.rId)
        prs.slides._sldIdLst.remove(sid)
    prs.slides._sldIdLst.insert(21,new_id)
    for n,s in enumerate(prs.slides,1):
        for sh in s.shapes:
            if sh.has_text_frame and re.fullmatch(r'-\s*\d+\s*-',sh.text.strip()):
                change(sh,f'- {n} -')
    prs.save(DEST)
    check=Presentation(DEST)
    assert len(check.slides)==32
    assert preserved==[content(s) for i,s in enumerate(check.slides) if i!=21]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest
    (OUT/'manifest.json').write_text(json.dumps({'source':str(SOURCE),'source_sha256':digest,'output':str(DEST),
        'slides':32,'erd_page':22,'snapshot':str(png),'entities':18,'other_slides_preserved':31},ensure_ascii=False,indent=2),encoding='utf-8')
    print(DEST)


if __name__=='__main__':
    main()
