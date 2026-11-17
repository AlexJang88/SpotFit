"""Build contact sheets and validate the final PPTX package."""
from pathlib import Path
import re,json,zipfile
from PIL import Image,ImageDraw,ImageFont
from pptx import Presentation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/presentations/final_20261116'
paths=sorted((OUT/'render').glob('*.PNG'),key=lambda p:int(re.search(r'\d+',p.stem).group()))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
for start in range(0,len(paths),12):
    group=paths[start:start+12]
    sheet=Image.new('RGB',(1600,3*255),'#d8dedb')
    draw=ImageDraw.Draw(sheet)
    for n,path in enumerate(group):
        x=(n%4)*400;y=(n//4)*255
        im=Image.open(path);im.thumbnail((392,220))
        sheet.paste(im,(x+4,y+27))
        draw.text((x+10,y+2),str(start+n+1),font=font,fill='#354241')
    sheet.save(OUT/f'contact_{start+1:02}.jpg',quality=90)
pptx=next(OUT.glob('*.pptx'))
p=Presentation(pptx)
assert len(p.slides)==31
with zipfile.ZipFile(pptx) as z:
    assert z.testzip() is None
texts='\n'.join(sh.text for s in p.slides for sh in s.shapes if sh.has_text_frame)
for unwanted in ['[작성 예정]','[확정 필요]','#########']:
    assert unwanted not in texts
print(json.dumps({'slides':len(p.slides),'rendered':len(paths),'pptx_bytes':pptx.stat().st_size},ensure_ascii=False))
