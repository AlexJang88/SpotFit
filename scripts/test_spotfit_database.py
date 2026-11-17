"""Validate SQL using a disposable, network-isolated PostGIS container.

No published ports, no production Compose DB, no persistent volume. Requires
Docker and the already pulled postgis/postgis:17-3.5-alpine image.
"""
from pathlib import Path
import json
import re
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
SQL=ROOT/'infra/postgres/migrations'
OUT=ROOT/'artifacts/database_sql_validation'

def run(*args,check=True):
    p=subprocess.run(['docker',*args],cwd=ROOT,encoding='utf-8',errors='replace',capture_output=True)
    if check and p.returncode:
        raise RuntimeError('docker '+args[0]+' failed:\n'+p.stdout[-6000:]+'\n'+p.stderr[-6000:])
    return p

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex[:12]
    name='spotfit-sql-test-'+token
    started=False
    try:
        run('run','--rm','-d','--name',name,'--label','spotfit.sql-test='+token,
            '--network','none','--tmpfs','/var/lib/postgresql/data:rw',
            '-e','POSTGRES_HOST_AUTH_METHOD=trust','-e','POSTGRES_USER=spotfit_test','-e','POSTGRES_DB=spotfit_test',
            '--mount',f'type=bind,source={SQL},target=/sql,readonly','postgis/postgis:17-3.5-alpine')
        started=True
        for _ in range(60):
            # Initialization temporarily accepts connections; wait for the final server.
            p=run('exec',name,'sh','-c','test -f "$PGDATA/postmaster.pid" && pg_isready -U spotfit_test -d spotfit_test',check=False)
            if p.returncode==0 and 'PostgreSQL init process complete' in run('logs',name).stdout:
                break
            time.sleep(1)
        else:raise RuntimeError('PostGIS test container did not initialize')
        base=['exec',name,'psql','-X','-v','ON_ERROR_STOP=1','-U','spotfit_test','-d','spotfit_test']
        for i in (1,2):
            p=run(*base,'-f','/sql/000_install.sql')
            (OUT/f'install_{i}.log').write_text(p.stdout+p.stderr,encoding='utf-8')
            print(f'PASS: install {i} (atomic + repeatable)',flush=True)
        # Compare table names to all 33 ERD entities, not just an expected total.
        tables=set(run(*base,'-At','-c',"SELECT tablename FROM pg_tables WHERE schemaname='spotfit'").stdout.split())
        erd=(ROOT/'docs/architecture/spotfit-service-erd-v2.md').read_text(encoding='utf-8')
        expected={x.lower() for x in re.findall(r'^    ([A-Z][A-Z_]+) \{',erd,re.M)}
        assert len(expected)==33 and expected <= tables,expected-tables
        assert tables-expected=={'schema_migration','ml_model_type'},tables-expected
        print('PASS: all 33 ERD entities + 2 support tables',flush=True)
        p=run(*base,'-f','/sql/tests/constraints.sql')
        (OUT/'constraints.log').write_text(p.stdout+p.stderr,encoding='utf-8')
        print(p.stdout[-4000:],flush=True)
        # Test fixture transaction rolled back; install contains NO invented model runs.
        result=run(*base,'-At','-c',"SELECT (SELECT count(*) FROM spotfit.ml_training_run), (SELECT count(*) FROM spotfit.service_safety_prediction), (SELECT count(*) FROM spotfit.ref_admin_area)").stdout.strip()
        assert result=='0|0|453',result
        (OUT/'result.json').write_text(json.dumps({'status':'PASS','erd_tables':33,'support_tables':2,
            'seeded_areas':453,'schema_applied_twice':True,'fixtures_rolled_back':True,
            'database':'disposable isolated PostGIS 17/3.5, production untouched'},indent=2),encoding='utf-8')
        print('PASS: fixture rollback, 453 real seed areas; production DB untouched',flush=True)
    finally:
        if started:
            owner=run('inspect','--format','{{ index .Config.Labels "spotfit.sql-test" }}',name,check=False)
            if owner.returncode==0 and owner.stdout.strip()==token:
                run('stop',name)
                print('Removed only this run\'s disposable test container.',flush=True)

if __name__=='__main__':main()
