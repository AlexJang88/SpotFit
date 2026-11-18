"""Read-only PostgreSQL catalog snapshot for the SpotFit physical ERD."""
from __future__ import annotations

import json
import argparse
import re
from datetime import datetime, timezone
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/erd/physical_schema'

QUERY = r"""
BEGIN READ ONLY;
WITH tables AS (
 SELECT c.oid, n.nspname AS schema_name, c.relname AS table_name
 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
 WHERE c.relkind IN ('r','p') AND n.nspname NOT IN ('information_schema','pg_catalog')
 AND n.nspname NOT LIKE 'pg_%'
 AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_class'::regclass
                 AND d.objid=c.oid AND d.deptype='e')
), cols AS (
 SELECT t.oid, jsonb_agg(jsonb_build_object(
  'name',a.attname, 'position',a.attnum, 'type',format_type(a.atttypid,a.atttypmod),
  'nullable',NOT a.attnotnull,'identity',a.attidentity,
  'default',pg_get_expr(ad.adbin,ad.adrelid)
 ) ORDER BY a.attnum) AS columns
 FROM tables t JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum>0 AND NOT a.attisdropped
 LEFT JOIN pg_attrdef ad ON ad.adrelid=a.attrelid AND ad.adnum=a.attnum GROUP BY t.oid
), constraints AS (
 SELECT t.oid, jsonb_agg(jsonb_build_object(
  'name',c.conname,'kind',c.contype,'definition',pg_get_constraintdef(c.oid),
  'columns',ARRAY(SELECT a.attname FROM unnest(c.conkey) WITH ORDINALITY k(num,seq)
                  JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.num ORDER BY k.seq),
  'target_schema',n.nspname,'target_table',r.relname,
  'target_columns',ARRAY(SELECT a.attname FROM unnest(c.confkey) WITH ORDINALITY k(num,seq)
                  JOIN pg_attribute a ON a.attrelid=c.confrelid AND a.attnum=k.num ORDER BY k.seq)
 ) ORDER BY c.contype,c.conname) AS constraints
 FROM tables t JOIN pg_constraint c ON c.conrelid=t.oid
 LEFT JOIN pg_class r ON r.oid=c.confrelid LEFT JOIN pg_namespace n ON n.oid=r.relnamespace
 GROUP BY t.oid
)
SELECT jsonb_build_object('database',current_database(),'server_version',current_setting('server_version'),
 'captured_at',current_timestamp,'source','Live PostgreSQL catalog (read-only)',
 'tables',COALESCE(jsonb_agg(jsonb_build_object('schema',t.schema_name,'name',t.table_name,
 'columns',cols.columns,'constraints',COALESCE(constraints.constraints,'[]'::jsonb))
 ORDER BY t.schema_name,t.table_name),'[]'::jsonb))
FROM tables t JOIN cols ON cols.oid=t.oid LEFT JOIN constraints ON constraints.oid=t.oid;
ROLLBACK;
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('docker','dev','ddl'),default='docker')
    args=parser.parse_args()
    if args.source == 'ddl':
        snapshot=read_ddl()
    elif args.source == 'dev':
        import psycopg2
        from dotenv import dotenv_values
        env=dotenv_values(ROOT/'.env.dev')
        if not env.get('POSTGRES_HOST'):
            raise ValueError('.env.dev does not configure a remote PostgreSQL host')
        with psycopg2.connect(host=env['POSTGRES_HOST'],port=env.get('POSTGRES_PORT','5432'),
                             dbname=env['POSTGRES_DB'],user=env['POSTGRES_USER'],password=env['POSTGRES_PASSWORD'],
                             connect_timeout=8) as connection:
            with connection.cursor() as cursor:
                cursor.execute(QUERY.replace('ROLLBACK;',''))
                snapshot=cursor.fetchone()[0]
            connection.rollback()
    else:
        snapshot=read_docker()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'schema_snapshot_{args.source}.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    if not snapshot['tables']:
        print('No application tables found in '+args.source+' database: '+snapshot['database'])
        return
    (OUT/'schema_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'database':snapshot['database'],'tables':len(snapshot['tables']),
                     'columns':sum(len(t['columns']) for t in snapshot['tables']),
                     'foreign_keys':sum(c['kind']=='f' for t in snapshot['tables'] for c in t['constraints']),
                     'names':[t['schema']+'.'+t['name'] for t in snapshot['tables']]},ensure_ascii=False,indent=2))


def read_docker():
    command = ['docker','compose','exec','-T','postgres','sh','-c',
               'exec psql -X -q -A -t -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"']
    result = subprocess.run(command,input=QUERY,encoding='utf-8',capture_output=True,cwd=ROOT,check=True)
    return json.loads(result.stdout.strip())


def split_top_level(text):
    """Split SQL lists without splitting numeric typmods, checks or quoted commas."""
    depth=0; quoted=False; start=0; parts=[]; index=0
    while index<len(text):
        char=text[index]
        if char=="'":
            if quoted and index+1<len(text) and text[index+1]=="'":
                index+=2;continue
            quoted=not quoted
        elif not quoted:
            if char=='(':depth+=1
            elif char==')':depth-=1
            elif char==',' and depth==0:parts.append(text[start:index].strip());start=index+1
        index+=1
    return [*parts,text[start:].strip()]


def read_ddl():
    files=[ROOT/'infra/postgres/migrations'/name for name in ('000_install.sql','001_schema.sql')]
    tables=[]
    for path in files:
        sql=re.sub(r'--[^\n]*','',path.read_text(encoding='utf-8'))
        for match in re.finditer(r'CREATE TABLE (?:IF NOT EXISTS )?(\w+)\.(\w+)\s*\(',sql):
            index=match.end();depth=1;quoted=False;end=index
            while depth:
                char=sql[end]
                if char=="'":
                    if quoted and sql[end+1:end+2]=="'":end+=2;continue
                    quoted=not quoted
                elif not quoted:
                    depth+= (char=='(')-(char==')')
                end+=1
            table={'schema':match[1],'name':match[2],'columns':[],'constraints':[]}
            for part in split_top_level(sql[index:end-1]):
                table_level=re.match(r'(PRIMARY KEY|FOREIGN KEY|UNIQUE(?: NULLS NOT DISTINCT)?|CHECK|EXCLUDE)\b',part)
                if table_level:
                    kind={'PRIMARY KEY':'p','FOREIGN KEY':'f','CHECK':'c','EXCLUDE':'x'}.get(table_level[1],'u')
                    cols=re.search(r'\(([^()]*)\)',part)
                    constraint={'kind':kind,'columns':[c.strip() for c in cols[1].split(',')] if kind in 'pfu' else [],'definition':part}
                    table['constraints'].append(constraint)
                    if kind=='f':fill_reference(constraint,part)
                    continue
                column_match=re.match(r'(\w+)\s+(.+)',part,re.S)
                if not column_match:raise ValueError(part)
                name,rest=column_match.groups()
                type_name=re.split(r'\s+(?:GENERATED|PRIMARY|NOT|NULL|DEFAULT|REFERENCES|UNIQUE|CHECK)\b',rest,maxsplit=1)[0]
                table['columns'].append({'name':name,'position':len(table['columns'])+1,'type':type_name,
                                         'nullable':not ('NOT NULL' in rest or 'PRIMARY KEY' in rest),'identity':'GENERATED' in rest})
                for keyword,kind in [('PRIMARY KEY','p'),('UNIQUE','u'),('REFERENCES','f')]:
                    if re.search(r'\b'+keyword+r'\b',rest):
                        constraint={'kind':kind,'columns':[name],'definition':part}
                        if kind=='f':fill_reference(constraint,rest)
                        table['constraints'].append(constraint)
            tables.append(table)
    by_name={(t['schema'],t['name']):t for t in tables}
    for t in tables:
        for c in t['constraints']:
            if c['kind']=='p':
                for column in t['columns']:
                    if column['name'] in c['columns']:column['nullable']=False
            if c['kind']=='f' and not c['target_columns']:
                parent=by_name[(c['target_schema'],c['target_table'])]
                c['target_columns']=next(k['columns'] for k in parent['constraints'] if k['kind']=='p')
            if c['kind']=='f':
                assert len(c['columns'])==len(c['target_columns']),c
    return {'database':'SpotFit','source':'Repository PostgreSQL DDL; not a deployed database snapshot',
            'source_files':[str(p.relative_to(ROOT)) for p in files],
            'captured_at':datetime.now(timezone.utc).isoformat(),'tables':tables}


def fill_reference(constraint,text):
    match=re.search(r'REFERENCES\s+(\w+)\.(\w+)(?:\s*\(([^)]+)\))?',text)
    if not match:raise ValueError(text)
    constraint.update(target_schema=match[1],target_table=match[2],
                      target_columns=[s.strip() for s in match[3].split(',')] if match[3] else [])


if __name__ == '__main__':
    main()
