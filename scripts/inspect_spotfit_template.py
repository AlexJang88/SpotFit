from pathlib import Path
from pptx import Presentation
import json
from zipfile import ZipFile

SOURCE = Path('C:/Users/M/Desktop/SPOT_FIT_발표자료.pptx')
OUT = Path('artifacts/presentations/template_review')
OUT.mkdir(parents=True, exist_ok=True)
p = Presentation(SOURCE)
slides = []
for i, slide in enumerate(p.slides, 1):
    shapes = []
    for n, s in enumerate(slide.shapes):
        d = dict(index=n, name=s.name, type=str(s.shape_type), box=[round(v/914400, 3) for v in (s.left,s.top,s.width,s.height)])
        if s.has_text_frame:
            d['text'] = s.text
            d['paragraphs'] = [dict(text=para.text, level=para.level, runs=[dict(text=r.text, font=r.font.name, size=r.font.size.pt if r.font.size else None, bold=r.font.bold) for r in para.runs]) for para in s.text_frame.paragraphs]
        if s.has_table:
            d['table'] = [[c.text for c in row.cells] for row in s.table.rows]
        shapes.append(d)
    slides.append(dict(number=i,layout=slide.slide_layout.name,shapes=shapes))
(OUT/'structure.json').write_text(json.dumps(dict(size=[p.slide_width/914400,p.slide_height/914400],slides=slides),ensure_ascii=False,indent=2),encoding='utf-8')
for slide in slides:
    print('\nSLIDE', slide['number'], slide['layout'])
    for s in slide['shapes']:
        print(s['index'],s['type'],s['box'],repr(s.get('text',''))[:1400])
with ZipFile(SOURCE) as z:
    for name in z.namelist():
        if name in ('docProps/thumbnail.jpeg','docProps/thumbnail.png'):
            (OUT/Path(name).name).write_bytes(z.read(name))
