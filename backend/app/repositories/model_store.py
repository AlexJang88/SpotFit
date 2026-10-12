"""MinIO access; credentials and server addresses never enter model manifests."""

import os
from urllib.parse import urlsplit

import urllib3
from minio import Minio
from minio.error import S3Error
from minio.versioningconfig import ENABLED, VersioningConfig


class ModelStore:
    def __init__(self, client: Minio, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> "ModelStore":
        url = urlsplit(os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"))
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("S3_ENDPOINT_URL must be an http(s) server URL without a path")
        client = Minio(
            url.netloc,
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            secure=url.scheme == "https",
            region=os.environ.get("S3_REGION", "us-east-1"),
            http_client=urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=5, read=120),
                retries=urllib3.Retry(total=2, backoff_factor=0.5),
            ),
        )
        return cls(client, os.environ.get("S3_MODEL_BUCKET", "spot-fit-models"))

    def initialize(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as error:
                if error.code != "BucketAlreadyOwnedByYou":
                    raise
        self.client.set_bucket_versioning(self.bucket, VersioningConfig(ENABLED))

    def require_versioning(self) -> None:
        if self.client.get_bucket_versioning(self.bucket).status != ENABLED:
            raise ValueError("Run the model-tools init command to enable bucket versioning")
