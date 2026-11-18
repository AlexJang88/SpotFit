"""Render every SQL column and FK into a single, zoomable physical ERD sheet."""
from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/erd/physical_schema'
VIZ_VERSION='3.31.0'
INK='#073C38'
TEAL='#087F72'


def label(table):
    rows=[]
    pk={x for c in table['constraints'] if c['kind']=='p' for x in c['columns']}
    fk={x for c in table['constraints'] if c['kind']=='f' for x in c['columns']}
    uq={x for c in table['constraints'] if c['kind']=='u' for x in c['columns']}
    for col in table['columns']:
        name=col['name']
        keys='/'.join(k for k,s in [('PK',pk),('FK',fk),('UQ',uq)] if name in s)
        fill='#ECF7F3' if name in pk else '#FFFFFF'
        key_color=TEAL if name in pk or name in fk else '#697C78'
        rows.append(f'<TR><TD PORT="{name}_in" BGCOLOR="{fill}" ALIGN="LEFT"><FONT COLOR="{key_color}" POINT-SIZE="10">{keys or " "}</FONT></TD>'
                    f'<TD BGCOLOR="{fill}" ALIGN="LEFT"><FONT FACE="Consolas" POINT-SIZE="13">{html.escape(name)}</FONT></TD>'
                    f'<TD BGCOLOR="{fill}" ALIGN="LEFT"><FONT FACE="Consolas" COLOR="#526761" POINT-SIZE="11">{html.escape(col["type"])}</FONT></TD>'
                    f'<TD PORT="{name}_out" BGCOLOR="{fill}"><FONT COLOR="#526761" POINT-SIZE="10">{"NULL" if col["nullable"] else "NN"}</FONT></TD></TR>')
    return (f'<TABLE BORDER="1" COLOR="#ACCBC3" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">'
            f'<TR><TD COLSPAN="4" BGCOLOR="{INK}" ALIGN="LEFT" CELLPADDING="10">'
            f'<FONT COLOR="white" FACE="Consolas" POINT-SIZE="16"><B>{table["name"]}</B></FONT></TD></TR>'
            +''.join(rows)+'</TABLE>')


def dot(snapshot):
    tables=snapshot['tables']
    groups={
        'REFERENCE': ['ref_'],
        'CORE DATA': ['core_'],
        'INGESTION / PIPELINE': ['meta_'],
        'FEATURES / MODELS': ['feature_','ml_'],
        'SERVICE / SCORING': ['service_'],
        'OPERATIONS': ['ops_','analytics_','schema_'],
    }
    lines=['digraph G {',
           'graph [bgcolor="white", pad="0.6", overlap=true, splines=polyline, outputorder=edgesfirst, fontname="Arial", fontsize=20];',
           'node [shape=plain, fontname="Arial"];',
           'edge [color="#67988CCC", penwidth=1.2, arrowsize=0.9, arrowhead=crowodot, arrowtail=tee, dir=both];']
    xbase=0
    for i,(group,prefixes) in enumerate(groups.items()):
        lines.append(f'subgraph group_{i} {{')
        y=[0,0];count=0
        for table in tables:
            if any(table['name'].startswith(p) for p in prefixes):
                lane=count//4 if group=='SERVICE / SCORING' else 0
                height=42+26*len(table['columns'])
                x=(xbase+lane*640+300)/72
                center_y=-(y[lane]+height/2)/72
                lines.append(f'"{table["name"]}" [pin=true,pos="{x},{center_y}!",label=<{label(table)}>];')
                y[lane]+=height+88;count+=1
        lines.append('}')
        xbase+=1280 if group=='SERVICE / SCORING' else 640
    relationships=[]
    membership={t['name']:group for group,prefixes in groups.items() for t in tables if any(t['name'].startswith(p) for p in prefixes)}
    for table in tables:
        for c in table['constraints']:
            if c['kind']!='f':continue
            relationships.append({'child':table['name'],**c})
            parent=c['target_table'];child=table['name']
            # Composite keys retain one relation, with every participating column marked FK.
            tail=c['target_columns'][0]+'_out';head=c['columns'][0]+'_in'
            optional=any(col['nullable'] for col in table['columns'] if col['name'] in c['columns'])
            arrowtail='teeodot' if optional else 'teetee'
            tooltip=f'{child}({", ".join(c["columns"])}) -> {parent}({", ".join(c["target_columns"])})'
            constraint='true' if membership[parent]==membership[child] else 'false'
            lines.append(f'"{parent}":"{tail}":e -> "{child}":"{head}":w [constraint={constraint}, arrowtail={arrowtail}, tooltip={json.dumps(tooltip)}];')
    lines.append('}')
    return '\n'.join(lines),relationships


def main():
    snapshot=json.loads((OUT/'schema_snapshot.json').read_text(encoding='utf-8'))
    graph,relations=dot(snapshot)
    (OUT/'SPOTFIT_physical_erd.dot').write_text(graph,encoding='utf-8')
    library=OUT/f'viz-{VIZ_VERSION}.js'
    if not library.exists():
        with urllib.request.urlopen(f'https://cdn.jsdelivr.net/npm/@viz-js/viz@{VIZ_VERSION}/dist/viz-global.js',timeout=60) as response:
            library.write_bytes(response.read())
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1600,'height':900},device_scale_factor=1)
        page.goto((OUT/'schema_snapshot.json').as_uri())
        page.set_content('<html><body style="margin:0;background:white"></body></html>')
        page.add_script_tag(path=str(library))
        raw=page.evaluate('async dot => { const viz=await Viz.instance(); return viz.renderString(dot,{format:"svg",engine:"neato"}); }',graph)
        page.set_content(raw)
        view=page.locator('svg').get_attribute('viewBox').split()
        width,height=float(view[2]),float(view[3])
        print('Graph dimensions',width,height)
        # Add a restrained title band and exact coverage, without Korean narrative.
        title_height=116
        inner=re.sub(r'^.*?<svg[^>]*>|</svg>\s*$','',raw,flags=re.S)
        ncols=sum(len(t['columns']) for t in snapshot['tables'])
        header=(f'<rect width="100%" height="100%" fill="white"/>'
                f'<text x="48" y="43" font-family="Arial" font-size="30" font-weight="700" fill="{INK}">SPOTFIT / PHYSICAL ERD</text>'
                f'<text x="48" y="72" font-family="Arial" font-size="15" fill="#526761">PostgreSQL + PostGIS | spotfit schema | {len(snapshot["tables"])} tables / {ncols} columns / {len(relations)} foreign keys</text>'
                f'<text x="48" y="98" font-family="Arial" font-size="12" fill="#526761">PK primary key   FK foreign key   UQ unique-key member   NN not null   NULL nullable | All persisted columns shown</text>')
        footer=(f'<text x="48" y="{height+title_height+26}" font-family="Arial" font-size="13" fill="#526761">'
                'SOURCE: 000_install.sql + 001_schema.sql | SQL-defined physical schema, not a deployed DB snapshot | 2026-11-18</text>'
                f'<text x="48" y="{height+title_height+47}" font-family="Consolas" font-size="12" fill="#526761">'
                'COMPOSITE FK: service_neighborhood_score(scoring_policy_id, total_grade) -&gt; service_grade_policy(scoring_policy_id, grade)</text>')
        totalheight=height+title_height+70
        svg=(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{width}" height="{totalheight}" viewBox="0 0 {width} {totalheight}">'
             +header+f'<g transform="translate(0,{title_height})">'+inner+'</g>'+footer+'</svg>')
        svgpath=OUT/'SPOTFIT_physical_erd.svg'
        svgpath.write_text(svg,encoding='utf-8')
        # Coverage is checked against SVG text, never a hand-written column list.
        page.set_content(svg)
        texts=page.locator('text').all_text_contents()
        missing=[t['name']+'.'+c['name'] for t in snapshot['tables'] for c in t['columns'] if c['name'] not in texts]
        assert not missing,missing
        assert sum(x in {t['name'] for t in snapshot['tables']} for x in texts)==len(snapshot['tables'])
        nodes=page.locator('g.node').evaluate_all('(nodes)=>Object.fromEntries(nodes.map(n=>[n.querySelector("title").textContent,[...n.querySelectorAll("text")].map(t=>t.textContent)]))')
        for table in snapshot['tables']:
            assert all(c['name'] in nodes[table['name']] for c in table['columns']),table['name']
        assert page.locator('g.edge').count()==len(relations)
        scale=min(2.0,12000/width)
        pixel_width=int(width*scale);pixel_height=int(totalheight*scale)
        page.set_viewport_size({'width':pixel_width,'height':pixel_height})
        page.locator('svg').evaluate('(el,size)=>{el.setAttribute("width",size[0]);el.setAttribute("height",size[1]);}',[pixel_width,pixel_height])
        page.locator('svg').screenshot(path=str(OUT/'SPOTFIT_physical_erd.png'),timeout=60000)
        page.pdf(path=str(OUT/'SPOTFIT_physical_erd.pdf'),width=f'{pixel_width}px',height=f'{pixel_height}px',print_background=True,margin={'top':'0','bottom':'0','left':'0','right':'0'})
        # Preview only; full-size PNG and SVG above preserve small column text.
        from PIL import Image
        with Image.open(OUT/'SPOTFIT_physical_erd.png') as preview:
            preview.thumbnail((2200,1800));preview.save(OUT/'preview.png')
        browser.close()
    report={'source':snapshot['source'],'tables':len(snapshot['tables']),'columns':ncols,
            'foreign_keys':len(relations),'png_pixels':[pixel_width,pixel_height],
            'missing_column_labels':missing,'relationships':relations}
    (OUT/'coverage_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='relationships'},indent=2))


if __name__=='__main__':main()
