"""MinIO 데이터 버킷의 객체를 프로젝트 ``data`` 디렉터리로 안전하게 동기화한다.

원격 객체를 기준으로 파일을 내려받되, 원격에 없는 로컬 파일은 삭제하지 않는다. 다운로드는
같은 디렉터리의 임시파일에 먼저 기록하고 성공한 뒤 교체하여 중간에 끊긴 파일이 학습에
사용되는 일을 막는다. 인증정보는 지정한 dotenv 파일에서만 읽는다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


def _safe_destination(data_dir: Path, object_name: str, prefix: str) -> Path:
    """객체 키를 로컬 경로로 바꾸고 절대경로·상위경로 이동을 거부한다."""
    relative_name = object_name[len(prefix):].lstrip("/") if prefix else object_name
    relative = PurePosixPath(relative_name)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe MinIO object key: {object_name!r}")
    destination = data_dir.joinpath(*relative.parts).resolve()
    root = data_dir.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Object escapes data directory: {object_name!r}")
    return destination


def resolve_env_file(root: Path, requested: Path | None = None) -> Path:
    """명시 경로 또는 개발용 우선순위(.env.dev → .env)로 설정 파일을 찾는다."""
    if requested is not None:
        path = requested if requested.is_absolute() else root / requested
        if not path.is_file():
            raise FileNotFoundError(f"MinIO environment file does not exist: {path}")
        return path
    for name in (".env.dev", ".env"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("MinIO settings require .env.dev or .env in the project root")


def sync_minio_data(
    root: Path,
    data_dir: Path,
    env_file: Path | None = None,
    prefix: str = "",
    bucket_override: str | None = None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    """MinIO 데이터 객체 전체를 내려받고 다운로드·건너뜀 개수를 반환한다."""
    try:
        from dotenv import dotenv_values
        from minio import Minio
        from urllib3 import PoolManager, Timeout
    except ImportError as error:
        raise RuntimeError("Install MinIO dependencies from requirements-dev.txt") from error

    selected_env = resolve_env_file(root, env_file)
    values = dotenv_values(selected_env)
    endpoint = urlsplit(values.get("S3_ENDPOINT_URL") or "")
    access = values.get("S3_ACCESS_KEY_ID")
    secret = values.get("S3_SECRET_ACCESS_KEY")
    bucket = bucket_override or values.get("S3_DATA_BUCKET") or values.get("S3_MODEL_BUCKET")
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in {"", "/"}
    ):
        raise ValueError("S3_ENDPOINT_URL must be an http(s) server URL without a path")
    if not all((access, secret, bucket)):
        raise ValueError(
            "Set S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY and S3_DATA_BUCKET "
            "(or S3_MODEL_BUCKET) in the selected environment file"
        )

    client = Minio(
        endpoint.netloc,
        access_key=access,
        secret_key=secret,
        secure=endpoint.scheme == "https",
        region=values.get("S3_REGION") or "us-east-1",
        http_client=PoolManager(timeout=Timeout(connect=5, read=300), retries=False),
    )
    if not client.bucket_exists(bucket):
        raise FileNotFoundError(f"MinIO bucket does not exist: {bucket}")

    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    for item in objects:
        # 디렉터리 표시 객체는 실제 파일이 아니므로 제외한다.
        if item.is_dir or item.object_name.endswith("/"):
            continue
        # upload_minio.py가 같은 버킷에 저장하는 서비스 모델은 data 입력이 아니다.
        # 학습 시작 동기화가 모델을 data/recommender 아래로 잘못 복사하지 않도록 제외한다.
        relative_object = item.object_name[len(prefix):].lstrip("/") if prefix else item.object_name
        if relative_object == "recommender" or relative_object.startswith("recommender/"):
            skipped += 1
            continue
        destination = _safe_destination(data_dir, item.object_name, prefix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and not overwrite:
            print(f"SKIP existing: {destination}")
            skipped += 1
            continue
        # 덮어쓰기는 임시 파일 다운로드가 완료된 뒤 원자적으로 수행한다.
        temporary = destination.with_name(destination.name + ".minio-download.tmp")
        try:
            print(f"DOWNLOAD s3://{bucket}/{item.object_name} -> {destination}")
            client.fget_object(bucket, item.object_name, str(temporary))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded += 1

    if downloaded + skipped == 0:
        raise FileNotFoundError(f"No data objects found in s3://{bucket}/{prefix}")
    return {
        "bucket": str(bucket),
        "prefix": prefix,
        "downloaded": downloaded,
        "skipped": skipped,
        "env_file": str(selected_env),
    }


def main() -> int:
    """Download MinIO data objects into the project's data directory."""
    parser = argparse.ArgumentParser(
        description="MinIO 버킷의 파일을 프로젝트 data 디렉터리로 다운로드합니다."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="MinIO 설정 파일. 기본값: 프로젝트의 .env.dev, 없으면 .env",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="다운로드 대상 디렉터리. 기본값: 프로젝트/data",
    )
    parser.add_argument(
        "--bucket",
        help="버킷 이름. 기본값: S3_DATA_BUCKET, 없으면 S3_MODEL_BUCKET",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="다운로드할 MinIO 객체 경로 접두사",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="로컬에 같은 경로의 파일이 있어도 다시 다운로드하여 덮어씁니다.",
    )
    args = parser.parse_args()

    try:
        result = sync_minio_data(
            root=ROOT,
            data_dir=args.data_dir,
            env_file=args.env_file,
            prefix=args.prefix,
            bucket_override=args.bucket,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"MinIO download failed: {error}", file=sys.stderr)
        return 1

    print(
        "MinIO download completed: "
        f"bucket={result['bucket']}, prefix={result['prefix']!r}, "
        f"downloaded={result['downloaded']}, skipped={result['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

