"""Small, Git-tracked references to versioned model objects."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    version: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    filename: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    bucket: str = Field(min_length=3)
    object_key: str = Field(min_length=1)
    object_version_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("name", "version", "filename")
    @classmethod
    def portable_component(cls, value: str) -> str:
        reserved = {"con", "prn", "aux", "nul"}
        reserved.update(f"{prefix}{n}" for prefix in ("com", "lpt") for n in range(1, 10))
        if value.endswith(".") or value.split(".")[0].lower() in reserved:
            raise ValueError("Use a filename/version that works on Windows and macOS")
        return value

    @field_validator("object_version_id")
    @classmethod
    def version_required(cls, value: str) -> str:
        if value == "null":
            raise ValueError("MinIO bucket versioning must be enabled")
        return value
