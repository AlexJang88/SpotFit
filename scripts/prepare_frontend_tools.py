"""Use Playwright's bundled Node and a workspace-local npm for validation."""
import io
import os
from pathlib import Path
import subprocess
import tarfile

import playwright
import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'tmp' / 'npm-tool'
NODE = Path(playwright.__file__).parent / 'driver' / 'node.exe'

if not (TARGET / 'package/bin/npm-cli.js').exists():
    response = requests.get('https://registry.npmjs.org/npm/latest', timeout=30)
    response.raise_for_status()
    response = requests.get(response.json()['dist']['tarball'], timeout=60)
    response.raise_for_status()
    TARGET.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as archive:
        archive.extractall(TARGET, filter='data')
env = dict(os.environ)
env['PATH'] = str(NODE.parent) + os.pathsep + env['PATH']
subprocess.run([str(NODE), str(TARGET / 'package/bin/npm-cli.js'), 'ci', '--cache', str(ROOT / 'tmp/npm-cache')],
               cwd=ROOT / 'frontend', env=env, check=True)
