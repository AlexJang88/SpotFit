"""Run the checked-in SQL on the Compose postgres service (Windows / Ubuntu).

Standard library only. Defaults to a read-only target check; --apply is explicit.
Reads POSTGRES_USER/POSTGRES_DB inside the container; never displays credentials.
"""
from pathlib import Path
import argparse
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f','--compose-file',action='append',default=[])
    parser.add_argument('-p','--project-name')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--apply',action='store_true',help='Create schema and insert initial reference data')
    group.add_argument('--verify',action='store_true',help='Read-only schema/seed validation')
    args=parser.parse_args()
    command=['docker','compose']
    for file in args.compose_file:command+=['-f',file]
    if args.project_name:command+=['-p',args.project_name]
    def run(parts,capture=False):
        return subprocess.run(command+parts,cwd=ROOT,check=True,text=True,encoding='utf-8',
                              errors='replace',capture_output=capture)
    target=run(['exec','-T','postgres','sh','-c','printf "%s" "$POSTGRES_DB"'],True).stdout.strip()
    if not target or target in {'airflow','mlflow','postgres','template0','template1'}:
        raise SystemExit('Refusing empty/system/metadata database target: '+target)
    print('Compose service: postgres | database: '+target+' | schema: spotfit',flush=True)
    if not (args.apply or args.verify):
        print('Read-only check complete. Use --apply to install, or --verify to check an existing installation.')
        return
    dest='/tmp/spotfit-sql-'+uuid.uuid4().hex[:12]
    run(['cp',str(ROOT/'infra/postgres/migrations'),'postgres:'+dest])
    filename='000_install.sql' if args.apply else '090_verify.sql'
    # Positional argument avoids interpolating container env vars or file paths.
    run(['exec','-T','postgres','sh','-c',
         'exec psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$1"',
         'sh',dest+'/'+filename])
    print('Completed. Application reads are unchanged; no backend storage adapter was switched.')

if __name__=='__main__':main()
