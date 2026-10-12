"""Upload model bytes and fetch an exact version with checksum verification."""

import hashlib
import os
import tempfile
from pathlib import Path

from app.repositories.model_store import ModelStore
from app.schemas.model_manifest import ModelManifest

CHUNK_SIZE = 1024 * 1024


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def cache_path(root: Path, manifest: ModelManifest) -> Path:
    root = root.resolve()
    path = (root / manifest.name / manifest.version / manifest.filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Model destination must stay inside the cache directory")
    return path


def upload_model(store: ModelStore, source: Path, name: str, version: str, registry: Path) -> Path:
    # Validate identifiers before deriving paths or talking to storage.
    draft = ModelManifest(
        name=name, version=version, filename=source.name,
        bucket=store.bucket, object_key="pending", object_version_id="pending",
        sha256="0" * 64, size_bytes=1,
    )
    registry = registry.resolve()
    destination = (registry / draft.name / f"{draft.version}.json").resolve()
    if not destination.is_relative_to(registry):
        raise ValueError("Manifest must stay inside the registry directory")
    if destination.exists():
        raise FileExistsError(f"Manifest already exists: {destination}. Use a new model version.")
    store.require_versioning()
    # Snapshot the source so training cannot change bytes between hashing and upload.
    with tempfile.TemporaryFile() as snapshot, source.open("rb") as stream:
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            snapshot.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        if not size:
            raise ValueError("Cannot upload an empty model file")
        sha256 = digest.hexdigest()
        key = f"{draft.name}/{draft.version}/{sha256}/{draft.filename}"
        snapshot.seek(0)
        result = store.client.put_object(
            store.bucket, key, snapshot, size,
            content_type="application/octet-stream", metadata={"sha256": sha256},
        )
    manifest = ModelManifest(
        name=draft.name, version=draft.version, filename=draft.filename,
        bucket=store.bucket, object_key=key, object_version_id=result.version_id or "null",
        sha256=sha256, size_bytes=size,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a version manifest, including competing uploads.
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return destination


def download_model(store: ModelStore, manifest_file: Path, cache: Path) -> Path:
    manifest = ModelManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    destination = cache_path(cache, manifest)
    expected = (manifest.sha256, manifest.size_bytes)
    if destination.is_file() and file_digest(destination) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    response = store.client.get_object(
        manifest.bucket, manifest.object_key, version_id=manifest.object_version_id,
    )
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            size = 0
            for chunk in response.stream(CHUNK_SIZE):
                size += len(chunk)
                if size > manifest.size_bytes:
                    raise ValueError("Downloaded model exceeds the manifest size")
                digest.update(chunk)
                stream.write(chunk)
        if (digest.hexdigest(), size) != expected:
            raise ValueError("Model checksum or size does not match the manifest")
        # A failed download never replaces a previously cached model.
        os.replace(temporary, destination)
    finally:
        response.close()
        response.release_conn()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
