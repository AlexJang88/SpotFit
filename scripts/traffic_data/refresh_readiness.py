"""Report whether all annual raw inputs exist for a new traffic-model release.

Only TOPIS has an automated collector today. Missing source manifests are
reported explicitly and must keep training and deployment disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.traffic_data.collect_topis_refresh import RAW_ROOT


SOURCES = ("topis", "accidents", "hotspots", "population", "schools", "health")
AUTOMATED_SOURCES = frozenset({"topis"})
COLLECTION_ROOT = RAW_ROOT.parent


def readiness(year: int, root: Path = COLLECTION_ROOT) -> dict:
    status = {}
    for source in SOURCES:
        directory = root / source / str(year)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            status[source] = {"ready": False, "reason": "manifest missing"}
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest["records"]
            if manifest["year"] != year or not manifest["complete_year"] or not records:
                raise ValueError("year incomplete or manifest mismatched")
            for record in records:
                path = (directory / record["file"]).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file():
                    raise ValueError("source file missing or outside its raw directory")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != record["sha256"]:
                    raise ValueError(f"checksum mismatch: {record['file']}")
            if source == "topis" and {int(row["month"]) for row in records} != set(range(1, 13)):
                raise ValueError("TOPIS does not have all 12 months")
            status[source] = {"ready": True, "files": len(records)}
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            status[source] = {"ready": False, "reason": str(error)}
    missing_collectors = sorted(set(SOURCES) - AUTOMATED_SOURCES)
    return {
        "year": year,
        "ready_for_preprocessing": not missing_collectors and all(
            item["ready"] for item in status.values()),
        "collectors_not_implemented": missing_collectors,
        "sources": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(readiness(args.year), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
