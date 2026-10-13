"""Upload local CSV files to a MinIO bucket, preserving relative paths."""

import argparse
import sys
from pathlib import Path

from minio.error import S3Error
from urllib3.exceptions import HTTPError

from app.repositories.model_store import ModelStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/raw"))
    parser.add_argument("--bucket", help="Defaults to S3_MODEL_BUCKET")
    parser.add_argument("--prefix", default="rawdata")
    parser.add_argument("--dry-run", action="store_true", help="List uploads without contacting MinIO")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"Source directory does not exist: {args.source}")
    prefix = args.prefix.strip("/")
    if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")):
        parser.error("Prefix must be a nonempty object path without '.' or '..'")
    files = sorted(
        path for path in args.source.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv" and not path.is_symlink()
    )
    if not files:
        print(f"No CSV files found in {args.source}")
        return 0
    try:
        store = ModelStore.from_env()
        bucket = args.bucket or store.bucket
        if not args.dry_run and not store.client.bucket_exists(bucket):
            print(f"Bucket {bucket!r} does not exist. Create it first or select --bucket.", file=sys.stderr)
            return 1
        for path in files:
            key = f"{prefix}/{path.relative_to(args.source).as_posix()}"
            if not args.dry_run:
                store.client.fput_object(bucket, key, str(path), content_type="text/csv")
            print(f"{'Would upload' if args.dry_run else 'Uploaded'}: {path} -> s3://{bucket}/{key}")
    except (S3Error, HTTPError, OSError, ValueError, KeyError) as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        return 1
    print(f"{'Planned' if args.dry_run else 'Completed'} {len(files)} CSV upload(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
