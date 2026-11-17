from copy import deepcopy
from pathlib import Path
from pptx import Presentation

root = Path(__file__).resolve().parents[1]
crime = root / 'docs/SPOT_FIT_범죄노출_Feature비중.pptx'
traffic = root / 'docs/SPOT_FIT_교통사고_51차_직접예측_템플릿유지_20261111.pptx'
output = root / 'docs/SPOT_FIT_범죄_교통사고51차_Feature비중_통합.pptx'
prs = Presentation(crime)
assert len(prs.slides) == 1
source = Presentation(traffic).slides[5]
slide = prs.slides.add_slide(prs.slides[0].slide_layout)
for shape in list(slide.shapes):
    slide.shapes._spTree.remove(shape.element)
for shape in source.shapes:
    # These slides contain only native shapes, groups and tables.
    assert not shape.element.xpath('.//@r:embed | .//@r:link | .//@r:id')
    slide.shapes._spTree.insert_element_before(deepcopy(shape.element), 'p:extLst')
prs.save(output)
check = Presentation(output)
assert len(check.slides) == 2
for actual, expected in zip(check.slides, [Presentation(crime).slides[0], source]):
    assert [s.element.xml for s in actual.shapes] == [s.element.xml for s in expected.shapes]
print(output)
