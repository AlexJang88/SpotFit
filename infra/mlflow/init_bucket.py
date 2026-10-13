"""Create the MLflow artifact bucket in the Compose MinIO service."""

import os
import time

from minio import Minio
from urllib3.exceptions import HTTPError


client = Minio(
    os.environ["MINIO_ENDPOINT"],
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False,
)
bucket = os.environ["MLFLOW_ARTIFACT_BUCKET"]

for attempt in range(30):
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        print(f"MLflow artifact bucket '{bucket}' is ready.")
        break
    except (HTTPError, OSError) as exc:
        if attempt == 29:
            raise
        print(f"Waiting for MinIO ({attempt + 1}/30): {exc}")
        time.sleep(2)
