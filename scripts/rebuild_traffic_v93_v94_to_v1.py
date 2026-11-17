"""Validate Raw data, rebuild v93/v94, and publish both into recommender/v1.

Pipeline:
    Raw inventory/hash validation
    -> v94 TOPIS Raw aggregation and Processed cross-check
    -> v93/v94 model rebuild from canonical Processed workbooks
    -> recommender/v1 service publication
    -> artifact load/hash/forecast validation

The canonical Processed workbooks are leakage-reviewed training matrices. Raw
validation is mandatory, so missing or changed source files stop publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.traffic_accident_pipeline_paths import (
    INTERIM_V94_TRAFFIC,
    MODEL_V93_REBUILT,
    MODEL_V94_REBUILT,
    PROCESSED_V93,
    PROCESSED_V94,
    RAW_V93,
    RAW_V94,
    RAW_V94_TRAFFIC,
    RECOMMENDER_V1,
)


REPORT = ROOT / "models/recommender/v1/traffic_accident_v93_v94_pipeline_report.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".gitkeep"
    }


def validate_raw() -> dict:
    if not RAW_V93.is_dir() or not RAW_V94.is_dir():
        raise FileNotFoundError(f"Raw 폴더가 없습니다: {RAW_V93}, {RAW_V94}")
    v93 = relative_files(RAW_V93)
    v94 = relative_files(RAW_V94)
    if len(v93) != 46:
        raise ValueError(f"93차 Raw는 46개여야 합니다: {len(v93)}")
    common_missing = sorted(set(v93).difference(v94))
    common_changed = sorted(
        name for name in set(v93).intersection(v94) if hash_file(v93[name]) != hash_file(v94[name])
    )
    if common_missing or common_changed:
        raise ValueError(
            f"94차의 93차 기반 Raw 불일치: missing={common_missing}, changed={common_changed}"
        )

    traffic_counts = {}
    for year in range(2017, 2026):
        files = sorted((RAW_V94_TRAFFIC / str(year)).glob("*.xlsx"))
        traffic_counts[str(year)] = len(files)
        if len(files) != 12:
            raise ValueError(f"TOPIS {year}년 파일은 12개여야 합니다: {len(files)}")
    traffic_files = sorted(RAW_V94_TRAFFIC.rglob("*.xlsx"))
    if len(traffic_files) != 108:
        raise ValueError(f"TOPIS Raw는 108개여야 합니다: {len(traffic_files)}")
    if len(v94) != 154:
        raise ValueError(f"94차 Raw는 기반 46개+TOPIS 108개여야 합니다: {len(v94)}")
    return {
        "v93_files": len(v93),
        "v94_files": len(v94),
        "common_files_hash_matched": len(v93),
        "topis_files": len(traffic_files),
        "topis_yearly_counts": traffic_counts,
    }


def run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def validate_topis_processed() -> dict:
    generated = pd.read_csv(INTERIM_V94_TRAFFIC, encoding="utf-8-sig")
    recorded = pd.read_excel(PROCESSED_V94, sheet_name="추가교통량지표")
    generated = generated.sort_values(["연도", "자치구"]).reset_index(drop=True)
    recorded = recorded.sort_values(["연도", "자치구"]).reset_index(drop=True)
    if not generated[["연도", "자치구"]].equals(recorded[["연도", "자치구"]]):
        raise ValueError("TOPIS Raw 재집계와 94차 Processed의 연도·자치구 키가 다릅니다.")
    value_columns = [column for column in recorded if column not in {"연도", "자치구"}]
    differences = {
        column: float(
            np.nanmax(
                np.abs(
                    generated[column].to_numpy(dtype=float)
                    - recorded[column].to_numpy(dtype=float)
                )
            )
        )
        for column in value_columns
    }
    if not all(
        np.allclose(generated[column], recorded[column], rtol=1e-12, atol=1e-10, equal_nan=True)
        for column in value_columns
    ):
        raise ValueError(f"TOPIS Raw 재집계 값이 94차 Processed와 다릅니다: {differences}")
    return {
        "rows": len(generated),
        "districts": int(generated["자치구"].nunique()),
        "max_abs_diff": max(differences.values(), default=0.0),
    }


def validate_processed() -> dict:
    required = {
        PROCESSED_V93: {"전체정제패널", "지표가중치", "지표별예측법"},
        PROCESSED_V94: {"94차전체입력패널", "94차모델설정", "추가교통량지표"},
    }
    result = {}
    for path, sheets in required.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        available = set(pd.ExcelFile(path).sheet_names)
        missing = sorted(sheets.difference(available))
        if missing:
            raise ValueError(f"{path.name} 필수 시트 누락: {missing}")
        result[path.name] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    return result


def validate_service(version: int) -> dict:
    directory = RECOMMENDER_V1 / f"traffic_accident_v{version}"
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    forecast = pd.read_csv(directory / "service_forecast.csv", encoding="utf-8-sig")
    year_counts = forecast.groupby("prediction_year")["district"].nunique().to_dict()
    if len(forecast) != 50 or year_counts != {2027: 25, 2028: 25}:
        raise ValueError(f"v{version} 서비스 예측 구조 오류: rows={len(forecast)}, {year_counts}")
    models = {}
    for horizon in (2, 3):
        path = directory / f"traffic_accident_{horizon}y.joblib"
        model = joblib.load(path)
        if model.get("version") != version or model.get("horizon") != horizon:
            raise ValueError(f"v{version} {horizon}년 모델 메타데이터가 다릅니다.")
        digest = hash_file(path)
        if digest != metadata["artifacts"][str(horizon)]["sha256"]:
            raise ValueError(f"v{version} {horizon}년 모델 해시가 다릅니다.")
        models[str(horizon)] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "features": len(model["features"]),
        }
    return {"rows": len(forecast), "year_district_counts": year_counts, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="기존 재현 모델을 사용하고 Raw/Processed/v1 검증과 배포만 수행",
    )
    args = parser.parse_args()

    report = {"raw": validate_raw(), "processed": validate_processed()}
    run_module("scripts.traffic_data.build_v94_traffic_volume_features")
    report["topis_raw_to_processed"] = validate_topis_processed()

    if not args.skip_training:
        run_module("ml.rebuild_traffic_accident_v93_from_processed")
        run_module("ml.train_traffic_accident_v94_from_processed")
    for directory in (MODEL_V93_REBUILT, MODEL_V94_REBUILT):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    run_module("scripts.publish_traffic_accident_v93_v94")
    report["recommender_v1"] = {
        "93": validate_service(93),
        "94": validate_service(94),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
