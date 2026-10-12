"""Uses a unique temporary bucket; never reads or deletes application models."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from minio.error import S3Error

from app.repositories.model_store import ModelStore
from app.services.model_artifacts import download_model, upload_model


class ModelArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ModelStore.from_env()
        cls.store.bucket = f"spot-fit-test-{uuid.uuid4().hex}"
        cls.store.initialize()

    @classmethod
    def tearDownClass(cls):
        # Delete only the uniquely named bucket created by this test run.
        for obj in cls.store.client.list_objects(cls.store.bucket, recursive=True, include_version=True):
            cls.store.client.remove_object(cls.store.bucket, obj.object_name, version_id=obj.version_id)
        cls.store.client.remove_bucket(cls.store.bucket)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "model.bin"
        self.source.write_bytes(b"spot_fit model artifact fixture\x00\x01" * 1024)
        self.registry = self.root / "registry"
        self.cache = self.root / "cache"
        self.name = "test_" + uuid.uuid4().hex

    def upload(self):
        return upload_model(self.store, self.source, self.name, "v1", self.registry)

    def test_round_trip_pins_object_version_and_repairs_corrupt_cache(self):
        manifest = self.upload()
        metadata = json.loads(manifest.read_text())
        # A later replacement at the same key must not affect the pinned manifest.
        self.store.client.put_object(
            self.store.bucket, metadata["object_key"], io.BytesIO(b"changed"), 7,
        )
        result = download_model(self.store, manifest, self.cache)
        self.assertEqual(result.read_bytes(), self.source.read_bytes())
        timestamp = result.stat().st_mtime_ns
        self.assertEqual(download_model(self.store, manifest, self.cache).stat().st_mtime_ns, timestamp)
        result.write_bytes(b"corrupted cache")
        download_model(self.store, manifest, self.cache)
        self.assertEqual(result.read_bytes(), self.source.read_bytes())

    def test_duplicate_manifest_is_not_overwritten(self):
        manifest = self.upload()
        previous = manifest.read_bytes()
        self.source.write_bytes(b"new data")
        with self.assertRaises(FileExistsError):
            self.upload()
        self.assertEqual(manifest.read_bytes(), previous)

    def test_bad_checksum_preserves_existing_file_and_cleans_temporary_file(self):
        manifest = self.upload()
        cached = download_model(self.store, manifest, self.cache)
        original = cached.read_bytes()
        metadata = json.loads(manifest.read_text())
        metadata["sha256"] = "0" * 64
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "checksum"):
            download_model(self.store, manifest, self.cache)
        self.assertEqual(cached.read_bytes(), original)
        self.assertEqual(list(cached.parent.iterdir()), [cached])

    def test_missing_object_version_fails_without_cache_file(self):
        manifest = self.upload()
        metadata = json.loads(manifest.read_text())
        metadata["object_version_id"] = str(uuid.uuid4())
        manifest.write_text(json.dumps(metadata))
        with self.assertRaises(S3Error):
            download_model(self.store, manifest, self.cache)
        self.assertEqual(list(self.cache.rglob("model.bin")), [])

    def test_invalid_paths_and_empty_models_are_rejected(self):
        with self.assertRaises(ValueError):
            upload_model(self.store, self.source, "../outside", "v1", self.registry)
        self.source.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.upload()
        self.assertFalse(self.registry.exists())

    def test_cli_upload_download_and_duplicate_exit_code(self):
        environment = {
            **os.environ,
            "S3_MODEL_BUCKET": self.store.bucket,
            "MODEL_REGISTRY_DIR": str(self.registry),
            "MODEL_CACHE_DIR": str(self.cache),
        }
        command = [sys.executable, "-m", "app.cli.models"]
        upload_args = ["upload", "--file", str(self.source), "--name", self.name, "--version", "v1"]
        uploaded = subprocess.run(command + upload_args, env=environment, capture_output=True, text=True)
        self.assertEqual(uploaded.returncode, 0, uploaded.stderr)
        manifest = self.registry / self.name / "v1.json"
        downloaded = subprocess.run(
            command + ["download", "--manifest", str(manifest)],
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(downloaded.returncode, 0, downloaded.stderr)
        self.assertEqual((self.cache / self.name / "v1" / self.source.name).read_bytes(), self.source.read_bytes())
        duplicate = subprocess.run(command + upload_args, env=environment, capture_output=True, text=True)
        self.assertEqual(duplicate.returncode, 1)
        self.assertIn("already exists", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
