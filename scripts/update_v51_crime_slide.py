from copy import deepcopy
from pathlib import Path
import shutil
import pandas as pd
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
PATH = next((ROOT / 'docs').glob('*51*템플릿*.pptx'))
BACKUP = ROOT / 'artifacts/reports/v51_ppt_review/before_crime_slide_update.pptx'

def put(frame, value):
    templates = [deepcopy(p._p) for p in frame.paragraphs]
    for p in list(frame.paragraphs):
        frame._txBody.remove(p._p)
    for i, line in enumerate(value.split('\n')):
        frame._txBody.append(deepcopy(templates[min(i, len(templates)-1)]))
        p = frame.paragraphs[-1]
        if p.runs:
            p.runs[0].text = line
            for run in list(p.runs)[1:]:
                p._p.remove(run._r)
        else:
            p.add_run().text = line

def txt(slide, index, value):
    put(slide.shapes[index].text_frame, value)

def table(slide, index, rows):
    target = slide.shapes[index].table
    if index == 9 and len(rows) == 7:
        rows = rows + [['두 예측 간격 단순평균 개선', '+14.1%p']]
    assert len(rows) == len(target.rows)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            put(target.cell(r,c).text_frame, value)

def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(PATH, BACKUP)
    prs = Presentation(PATH)
    slide = prs.slides[5]
    weights = pd.read_excel(ROOT/'data/processed/교통사고_51차_최종모델데이터.xlsx', sheet_name='변수가중치')
    txt(slide,4,'교통안전모델 51차 · 목표점수 가중치와 안전등급 산정')
    txt(slide,6,'13개 지표의 5분위 점수 가중합 → 미래 위험점수 예측 → 낮은 위험부터 안전 1~5등급')
    txt(slide,7,'① 학습 목표점수 · 주요 지표 가중치')
    txt(slide,8,'49차 표준화계수 절댓값 ÷ 13개 계수 절댓값 합 × 100')
    labels=['어린이 1만 명당 자동차 등록대수','자전거 사고다발 지점 수','자전거 사고다발 경상자 수','고등학교 평균 학생 수','보호구역 보행어린이 사고 부상자 수']
    rows=[['목표점수 구성 지표','가중치']]+[[name,f'{weights.iloc[i]["가중치_pct"]:.2f}%'] for i,name in enumerate(labels)]
    rows += [['나머지 8개 지표',f'{weights.iloc[5:]["가중치_pct"].sum():.2f}%'],['13개 지표 합계','100.00%']]
    table(slide,9,rows)
    txt(slide,10,'지표별 방향을 반영한 5분위 점수(20~100점)를 가중합\n이 가중치는 목표점수 산식이며 51차 모델의 변수 중요도는 아님')
    txt(slide,11,'② 2027·2028년 예측 · 안전등급 만드는 법')
    txt(slide,12,'예측 위험점수가 낮은 순서로 서울 25개 자치구 정렬')
    table(slide,13,[['안전순위','비율','안전등급'],['1~5위 · 가장 안전','20%','1등급'],['6~10위','20%','2등급'],['11~15위','20%','3등급'],['16~20위','20%','4등급'],['21~25위 · 가장 위험','20%','5등급']])
    txt(slide,14,'예: 안전순위 7위 → 안전 2등급\n안전점수 = 120 − 예측 위험점수 (순위별 고정 점수 아님)')
    txt(slide,15,'51차 입력: 2년 39개 · 3년 64개 | 13개 지표 가중치는 목표점수 구성용 | 안전 1등급이 가장 안전')
    # Keep the existing six slides and append a comparison using the same layout.
    if len(prs.slides)==6:
        last=prs.slides.add_slide(slide.slide_layout)
        for shape in list(last.shapes):
            shape._element.getparent().remove(shape._element)
        for shape in slide.shapes:
            last.shapes._spTree.insert_element_before(deepcopy(shape.element),'p:extLst')
    else:
        assert len(prs.slides)==7
        last=prs.slides[6]
    txt(last,4,'교통안전모델 51차 · Persistence 기준모델과 성능 비교')
    txt(last,6,'Persistence: 현재 안전등급 유지 | 51차: 현재 입력으로 2년 후·3년 후 안전등급 예측')
    txt(last,7,'① 동일 검증기간 · 정확한 안전등급 일치율')
    txt(last,8,'개선 폭은 정확도 차이(%p) · 2년과 3년을 각각 비교')
    table(last,9,[['비교 항목','정확도 / 개선 폭'],['2년 · Persistence','48.8%'],['2년 · 51차','56.0%'],['2년 · 기준 대비 개선','+7.2%p'],['3년 · Persistence','34.0%'],['3년 · 51차','55.0%'],['3년 · 기준 대비 개선','+21.0%p']])
    txt(last,10,'2025년 기준 Persistence는 2027·2028년 모두 2025년 등급 유지\n51차는 두 예측 간격에 서로 다른 모델을 적용')
    txt(last,11,'② 등급 오차 · 보조 성능지표')
    txt(last,12,'±1등급 이내 일치율은 높을수록, 평균 등급오차는 낮을수록 좋음')
    table(last,13,[['지표','Persistence','51차'],['2년 · ±1등급 이내','84.8%','87.2%'],['2년 · 평균 등급오차','0.672','0.576'],['3년 · ±1등급 이내','85.0%','87.0%'],['3년 · 평균 등급오차','0.820','0.580'],['등급 방향','1 가장 안전','5 가장 위험']])
    txt(last,14,'2년: 2021~2025년 125건 · 3년: 2022~2025년 100건\n동일 목표등급·검증기간의 연도별 지표를 평균')
    txt(last,15,'모델 선택에 사용한 과거 검증 결과입니다. 2027·2028년 실제 정확도를 뜻하지 않습니다.')
    # Correct the remaining reversed grade definition without rebuilding user edits.
    for sl in prs.slides:
        for shape in sl.shapes:
            if shape.has_text_frame and '1등급 고위험' in shape.text:
                put(shape.text_frame,shape.text.replace('1등급 고위험 · 5등급 저위험','안전 1등급 가장 안전 · 안전 5등급 가장 위험'))
    prs.save(PATH)
    checked=Presentation(PATH)
    assert len(checked.slides)==7
    assert not any('범죄' in s.text for sl in checked.slides for s in sl.shapes if s.has_text_frame)
    print(str(PATH))
    print('Validated: 7 slides; crime content replaced; comparison appended.')

if __name__=='__main__':
    main()
