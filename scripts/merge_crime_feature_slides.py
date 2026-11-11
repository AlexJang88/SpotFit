from pathlib import Path
import shutil
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'docs/SPOT_FIT_범죄노출_Feature비중.pptx'
backup = ROOT / 'artifacts/reports/crime_feature_slide/원본_2장.pptx'
prs = Presentation(path)
assert len(prs.slides) == 2

def content(slide):
    return [(s.text if s.has_text_frame else
             tuple(tuple(c.text for c in row.cells) for row in s.table.rows)
             if s.has_table else str(s.shape_type),
             s.left, s.top, s.width, s.height) for s in slide.shapes]

assert content(prs.slides[0]) == content(prs.slides[1])
backup.parent.mkdir(parents=True, exist_ok=True)
if not backup.exists():
    shutil.copy2(path, backup)
original = content(prs.slides[0])
item = prs.slides._sldIdLst[1]
prs.part.drop_rel(item.rId)
prs.slides._sldIdLst.remove(item)
prs.save(path)
result = Presentation(path)
assert len(result.slides) == 1
assert content(result.slides[0]) == original
print('Verified: one slide, all original content and positions preserved.')
