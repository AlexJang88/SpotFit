from copy import deepcopy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'docs/SPOT_FIT_범죄_교통사고51차_Feature비중_통합.pptx'
output = ROOT / 'docs/SPOT_FIT_범죄_교통안전_주요Feature비교_1장.pptx'
prs = Presentation(source)
crime, traffic = prs.slides[0], prs.slides[1]
traffic_shapes = [deepcopy(traffic.shapes[i].element) for i in (7, 8, 9)]

def text(shape, value):
    frame = shape.text_frame
    paragraph = frame.paragraphs[0]
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in list(paragraph.runs)[1:]:
            paragraph._p.remove(run._r)
    else:
        paragraph.add_run().text = value
    for p in list(frame.paragraphs)[1:]:
        frame._txBody.remove(p._p)

# Retain the template and the crime feature section only.
for i in sorted((5, 6, 10, 11, 12, 13, 14, 15), reverse=True):
    shape = crime.shapes[i]
    shape.element.getparent().remove(shape.element)
text(crime.shapes[4], '아동범죄 · 교통안전 51차 주요 Feature 비교')
text(crime.shapes[5], '① 아동범죄 · 주요 수치 Feature')
text(crime.shapes[6], '2년 모델 · 표준화계수 절댓값 기준 설명용 비중')
crime.shapes[5].top = Inches(3.45)
crime.shapes[6].top = Inches(4.10)
crime.shapes[7].top = Inches(4.80)

for element in traffic_shapes:
    crime.shapes._spTree.insert_element_before(element, 'p:extLst')
right_title, right_note, right_table = list(crime.shapes)[-3:]
for shape in (right_title, right_note, right_table):
    shape.left = Inches(10.2)
text(right_title, '② 교통안전 51차 · 주요 구성 지표')
text(right_note, '목표점수 산정 가중치 · 51차의 변수 중요도와는 구분')
right_title.top = Inches(3.45)
right_note.top = Inches(4.10)
right_table.top = Inches(4.80)

item = prs.slides._sldIdLst[1]
prs.part.drop_rel(item.rId)
prs.slides._sldIdLst.remove(item)
prs.save(output)
result = Presentation(output)
assert len(result.slides) == 1
tables = [s.table for s in result.slides[0].shapes if s.has_table]
assert len(tables) == 2
original = Presentation(source)
for actual, expected in zip(tables, [original.slides[0].shapes[9].table, original.slides[1].shapes[9].table]):
    assert [[c.text for c in r.cells] for r in actual.rows] == [[c.text for c in r.cells] for r in expected.rows]
print(output)

