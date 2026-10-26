"""MinIO의 데이터 객체를 로컬 ``data`` 디렉터리로 동기화한다.

로컬 파일과 원격 객체의 내용이 같으면 건너뛴다. 내용이 다르거나 로컬에 없는
파일은 임시 파일로 다운로드한 뒤 교체하므로 중단된 다운로드가 기존 파일을
훼손하지 않는다. ``upload_minio.py``가 사용하는 모델 영역 ``recommender/``는
기본적으로 제외한다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
MODEL_PREFIX = "recommender/"
CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(data_dir: Path, object_name: str, prefix: str) -> Path:
    """객체 키를 안전한 로컬 경로로 변환하고 data 디렉터리 이탈을 막는다."""
    relative_name = object_name[len(prefix):] if prefix else object_name
    relative = PurePosixPath(relative_name)
    if (not relative_name or relative.is_absolute() or ".." in relative.parts
            or "\\" in relative_name):
        raise ValueError(f"Unsafe object name: {object_name!r}")
    destination = data_dir.joinpath(*relative.parts)
    root = data_dir.resolve()
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Object escapes data directory: {object_name!r}")
    return destination


def download_tree(client, bucket: str, data_dir: Path, prefix: str = "") -> tuple[int, int]:
    downloaded = skipped = 0
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in client.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(item, "is_dir", False):
            continue
        object_name = item.object_name
        if not prefix and object_name.startswith(MODEL_PREFIX):
            continue
        destination = destination_for(data_dir, object_name, prefix)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # 먼저 별도 파일로 받은 뒤 내용 비교 및 교체한다. MinIO ETag는 multipart
        # 객체에서 파일 해시가 아니므로 동일성 판정에 사용하지 않는다.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            client.fget_object(bucket, object_name, str(temporary))
            if (destination.is_file() and destination.stat().st_size == temporary.stat().st_size
                    and file_sha256(destination) == file_sha256(temporary)):
                print(f"SKIP s3://{bucket}/{object_name}")
                skipped += 1
                continue
            if destination.exists() and not destination.is_file():
                raise ValueError(f"Download target is not a regular file: {destination}")
            os.replace(temporary, destination)
            print(f"DOWNLOAD s3://{bucket}/{object_name} -> {destination}")
            downloaded += 1
        finally:
            temporary.unlink(missing_ok=True)
    return downloaded, skipped


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.dev")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--bucket", help="Default: S3_DATA_BUCKET or S3_MODEL_BUCKET")
    parser.add_argument(
        "--prefix", default="",
        help="다운로드할 객체 prefix. 지정하면 로컬 data 경로에서는 prefix를 제거합니다.",
    )
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error(f"Environment file does not exist: {args.env_file}")
    if args.data_dir.exists() and (not args.data_dir.is_dir() or args.data_dir.is_symlink()):
        parser.error(f"Data path must be a real directory: {args.data_dir}")
    if args.prefix.startswith("/") or ".." in PurePosixPath(args.prefix).parts or "\\" in args.prefix:
        parser.error("Prefix must be a safe relative object prefix")

    try:
        from dotenv import dotenv_values
        from minio import Minio
        from minio.error import S3Error
        from urllib3 import PoolManager, Timeout
        from urllib3.exceptions import HTTPError
    except ImportError:
        print("Install dependencies: python -m pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    # 셸의 다른 AWS 자격증명과 섞지 않고 지정된 파일만 사용한다.
    env = dotenv_values(args.env_file)
    endpoint = urlsplit(env.get("S3_ENDPOINT_URL") or "")
    access = env.get("S3_ACCESS_KEY_ID")
    secret = env.get("S3_SECRET_ACCESS_KEY")
    bucket = args.bucket or env.get("S3_DATA_BUCKET") or env.get("S3_MODEL_BUCKET")
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query
            or endpoint.fragment or endpoint.path not in {"", "/"}):
        parser.error("S3_ENDPOINT_URL must be an http(s) server URL without a path")
    if not all([access, secret, bucket]):
        parser.error("Set S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY and S3_MODEL_BUCKET (or S3_DATA_BUCKET)")

    try:
        client = Minio(
            endpoint.netloc,
            access_key=access,
            secret_key=secret,
            secure=endpoint.scheme == "https",
            region=env.get("S3_REGION") or "us-east-1",
            http_client=PoolManager(timeout=Timeout(connect=5, read=120), retries=False),
        )
        if not client.bucket_exists(bucket):
            print(f"Bucket {bucket!r} does not exist.", file=sys.stderr)
            return 1
        downloaded, skipped = download_tree(client, bucket, args.data_dir, args.prefix)
        print(f"Completed: downloaded={downloaded}, skipped={skipped}")
    except (S3Error, HTTPError, OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
