"""Create a restricted developer service identity in the MinIO admin container."""

import os

from minio import Minio
from minio.credentials import StaticProvider
from minio.error import MinioAdminException
from minio.minioadmin import MinioAdmin
from minio.versioningconfig import ENABLED, VersioningConfig


client = Minio(
    "minio:9000",
    access_key=os.environ["MINIO_ROOT_USER"],
    secret_key=os.environ["MINIO_ROOT_PASSWORD"],
    secure=False,
)
bucket = os.environ.get("DEV_S3_MODEL_BUCKET", "spot-fit-models")
access_key = os.environ["DEV_S3_ACCESS_KEY_ID"]
secret_key = os.environ["DEV_S3_SECRET_ACCESS_KEY"]

if not client.bucket_exists(bucket):
    client.make_bucket(bucket)
client.set_bucket_versioning(bucket, VersioningConfig(ENABLED))

policy_name = "spot-fit-dev-models-rw"
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:ListBucketVersions",
                "s3:GetBucketVersioning",
                "s3:PutBucketVersioning",
            ],
            "Resource": [f"arn:aws:s3:::{bucket}"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:PutObject",
                "s3:DeleteObject",
            ],
            "Resource": [f"arn:aws:s3:::{bucket}/*"],
        },
    ],
}

admin = MinioAdmin(
    endpoint="minio:9000",
    credentials=StaticProvider(os.environ["MINIO_ROOT_USER"], os.environ["MINIO_ROOT_PASSWORD"]),
    secure=False,
)
try:
    admin.user_remove(access_key)
except MinioAdminException as error:
    if "Status: 404" not in str(error):
        raise
try:
    admin.policy_remove(policy_name)
except MinioAdminException as error:
    if "Status: 404" not in str(error):
        raise
admin.policy_add(policy_name, policy=policy)
admin.user_add(access_key, secret_key)
admin.policy_set(policy_name, user=access_key)
print(f"Restricted developer identity is ready for bucket {bucket!r}.")
