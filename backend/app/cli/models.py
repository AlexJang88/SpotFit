import argparse
import os
import sys
from pathlib import Path

from minio.error import S3Error
from urllib3.exceptions import HTTPError

from app.repositories.model_store import ModelStore
from app.services.model_artifacts import download_model, upload_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Store versioned spot_fit models in MinIO")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the model bucket and enable object versioning")
    upload = commands.add_parser("upload", help="Upload a file and write a Git-trackable manifest")
    upload.add_argument("--file", type=Path, required=True)
    upload.add_argument("--name", required=True)
    upload.add_argument("--version", required=True)
    download = commands.add_parser("download", help="Download the exact version from a manifest")
    download.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        store = ModelStore.from_env()
        if arguments.command == "init":
            store.initialize()
            print(f"Model bucket ready (versioning enabled): {store.bucket}")
        elif arguments.command == "upload":
            manifest = upload_model(
                store, arguments.file, arguments.name, arguments.version,
                Path(os.environ.get("MODEL_REGISTRY_DIR", "/model_registry")),
            )
            print(f"Uploaded. Commit this manifest to Git: {manifest}")
        else:
            path = download_model(
                store, arguments.manifest,
                Path(os.environ.get("MODEL_CACHE_DIR", "/artifacts/models")),
            )
            print(f"Verified model: {path}")
    except (OSError, ValueError, KeyError, S3Error, HTTPError) as error:
        print(f"Model storage error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
