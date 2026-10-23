"""데이터와 ``models/recommender`` 모델만 MinIO에 업로드한다.

다른 팀원이 ``models`` 아래에서 실험 중인 모델을 실수로 공유 저장소에
올리지 않도록 모델 탐색 범위를 recommender 하위로 강제한다.
"""

import argparse
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def files_under(source):
    """Exclude hidden files/directories and symlinks from uploads."""
    for directory, folders, files in os.walk(source, followlinks=False):
        folders[:] = sorted(
            name for name in folders
            if not name.startswith('.') and not Path(directory, name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory, name)
            if not name.startswith('.') and not path.is_symlink() and path.is_file():
                yield path


def upload_tree(client, bucket, source, prefix, missing_error):
    uploaded = skipped = 0
    for path in files_under(source):
        key = '/'.join(filter(None, [prefix, path.relative_to(source).as_posix()]))
        try:
            client.stat_object(bucket, key)
        except missing_error as error:
            if error.code != 'NoSuchKey':
                raise
        else:
            print(f'SKIP s3://{bucket}/{key}')
            skipped += 1
            continue
        client.fput_object(
            bucket, key, str(path),
            content_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
        )
        print(f'UPLOAD s3://{bucket}/{key}')
        uploaded += 1
    return uploaded, skipped


def recommender_sources(models_root):
    """``models/recommender/<version>/<file>``만 업로드 계획으로 만든다.

    모델 버전 하나에는 파일 하나 또는 하나로 묶은 아카이브만 허용한다.
    recommender 폴더가 없거나 숨김 파일만 있으면 모델 업로드를 건너뛴다.
    ``models``의 다른 하위 폴더는 탐색조차 하지 않는 것이 충돌 방지 의도다.
    """
    root = models_root / 'recommender'
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'Recommender path must be a real directory: {root}')
    groups = {}
    for source in files_under(root):
        parts = source.relative_to(root).parts
        if len(parts) != 2:
            raise ValueError(f'Expected models/recommender/<version>/<file>: {source}')
        version, _ = parts
        groups.setdefault(version, []).append(source)
    plans = []
    for version, sources in sorted(groups.items()):
        if len(sources) != 1:
            raise ValueError(f'recommender/{version}: use one model file or one archive per version')
        plans.append((sources[0], 'recommender', version))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env.dev')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--bucket', help='Default: S3_DATA_BUCKET or S3_MODEL_BUCKET')
    # 이 인수는 models 루트만 바꾼다. 어떤 경로를 주더라도 그 아래의
    # recommender 폴더만 탐색하므로 다른 모델이 함께 업로드되지 않는다.
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models',
                        help='Models root; only its recommender child is uploaded')
    parser.add_argument('--registry-dir', type=Path, default=ROOT / 'model_registry')
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f'Data directory does not exist: {args.data_dir}')
    if not args.env_file.is_file():
        parser.error(f'Environment file does not exist: {args.env_file}')
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from app.repositories.model_store import ModelStore
        from app.services.model_artifacts import upload_model
        from dotenv import dotenv_values
        from minio import Minio
        from minio.error import S3Error
        from urllib3 import PoolManager, Timeout
        from urllib3.exceptions import HTTPError
    except ImportError:
        print('Install dependencies: python -m pip install -r requirements-dev.txt', file=sys.stderr)
        return 1
    # Read only this file, so unrelated shell AWS credentials cannot override it.
    env = dotenv_values(args.env_file)
    endpoint = urlsplit(env.get('S3_ENDPOINT_URL') or '')
    access = env.get('S3_ACCESS_KEY_ID')
    secret = env.get('S3_SECRET_ACCESS_KEY')
    bucket = args.bucket or env.get('S3_DATA_BUCKET') or env.get('S3_MODEL_BUCKET')
    if (endpoint.scheme not in {'http', 'https'} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query
            or endpoint.fragment or endpoint.path not in {'', '/'}):
        parser.error('S3_ENDPOINT_URL must be an http(s) server URL without a path')
    if not all([access, secret, bucket]):
        parser.error('Set S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY and S3_MODEL_BUCKET (or S3_DATA_BUCKET)')
    try:
        client = Minio(
            endpoint.netloc, access_key=access, secret_key=secret,
            secure=endpoint.scheme == 'https', region=env.get('S3_REGION') or 'us-east-1',
            http_client=PoolManager(timeout=Timeout(connect=5, read=120), retries=False),
        )
        if not client.bucket_exists(bucket):
            print(f'Bucket {bucket!r} does not exist. Create it in MinIO first.', file=sys.stderr)
            return 1
        uploaded, skipped = upload_tree(client, bucket, args.data_dir, '', S3Error)
        model_bucket = env.get('S3_MODEL_BUCKET') or bucket
        store = ModelStore(client, model_bucket)
        plans = recommender_sources(args.models_dir)
        if plans:
            store.require_versioning()
        for source, name, version in plans:
            manifest_path = args.registry_dir / name / f'{version}.json'
            existed = manifest_path.exists()
            manifest = upload_model(store, source, name, version, args.registry_dir,
                                    skip_existing=True)
            print(f"{'SKIP' if existed else 'REGISTER'} model: {manifest}")
        print(f'Completed: uploaded={uploaded}, skipped={skipped}')
    except (S3Error, HTTPError, OSError, ValueError) as error:
        print(f'Upload failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
