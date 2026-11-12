"""Restore every model artifact required by the public recommendation UI."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "recommender" / "v1"
LOG_ROOT = ROOT / "artifacts" / "model_bootstrap"
logger = logging.getLogger(__name__)

GROUPS = {
    "property": {
        "script": "scripts/rebuild_property_service_models.py",
        "required": (
            "apart_sale/model_bundle.joblib", "apart_sale/model_contract.json",
            "apart_rent/model_bundle.joblib", "apart_rent/model_contract.json",
            "apart_sale_final/model_1y.joblib", "apart_sale_final/model_2y.joblib",
            "apart_sale_final/model_3y.joblib", "apart_sale_final/neighborhood_forecasts.csv",
            "apart_jeonse_final/model_1y.joblib", "apart_jeonse_final/model_2y.joblib",
            "apart_jeonse_final/model_3y.joblib", "apart_jeonse_final/neighborhood_forecasts.csv",
            "apart_wolse_final/model_bundle.joblib", "apart_wolse_final/summary.json",
            "property_service_forecasts.json",
        ),
    },
    "traffic": {
        "script": "scripts/rebuild_traffic_service_model.py",
        "required": (
            "traffic_accident/교통사고_2차_모델.joblib",
            "traffic_accident/2025년_최종검증_예측결과.csv",
            "traffic_accident/모델정보.json",
        ),
    },
    "child_crime": {
        "script": "scripts/child_crime/rebuild_service_crime_models.py",
        "required": tuple(
            f"child_crime_exposure/{h}y/{name}"
            for h in (2, 3)
            for name in (
                f"child_crime_exposure_{h}y.joblib", "service_forecast.csv", "model_metadata.json",
                "feature_score_formula.csv", "feature_score_breakdown.csv",
            )
        ),
    },
}

_LOCK = Lock()
_STATE = {"state": "idle", "started_at": None, "finished_at": None, "groups": {}}


def missing(group: str, root: Path = ROOT) -> list[str]:
    model_root = root / "models" / "recommender" / "v1"
    return [name for name in GROUPS[group]["required"]
            if not (model_root / name).is_file() or (model_root / name).stat().st_size == 0]


def bootstrap_status(root: Path = ROOT) -> dict:
    readiness = {name: {"ready": not missing(name, root), "missing": missing(name, root)} for name in GROUPS}
    return {**_STATE, "ready": all(item["ready"] for item in readiness.values()), "readiness": readiness}


def _run_script(group: str, root: Path) -> int:
    script = root / str(GROUPS[group]["script"])
    if not script.is_file():
        raise FileNotFoundError(f"복구 스크립트가 없습니다: {script}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{group}.log"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(root)}
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(script)], cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def ensure_all_models(root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in GROUPS:
        before = missing(group, root)
        if not before:
            result[group] = "exists"
            _STATE["groups"][group] = "exists"
            continue
        _STATE["groups"][group] = "building"
        try:
            code = _run_script(group, root)
            after = missing(group, root)
            if code == 0 and not after:
                result[group] = "restored"
            else:
                result[group] = f"failed: exit={code}, missing={after}"
        except Exception as error:
            result[group] = f"failed: {error}"
        _STATE["groups"][group] = result[group]
        logger.info("model bootstrap %s: %s", group, result[group])
    return result


def _background(root: Path) -> None:
    try:
        results = ensure_all_models(root)
        _STATE["state"] = "failed" if any(v.startswith("failed") for v in results.values()) else "done"
        if _STATE["state"] == "done":
            from app.services.recommender_v1 import clear_recommender_cache
            from app.services.neighborhood_recommendations import clear_neighborhood_cache
            clear_recommender_cache()
            clear_neighborhood_cache()
    except Exception as error:
        _STATE["state"] = "failed"
        logger.exception("model bootstrap crashed: %s", error)
    finally:
        _STATE["finished_at"] = datetime.now(timezone.utc).isoformat()
        _LOCK.release()


def start_model_bootstrap(root: Path = ROOT, force: bool = False) -> bool:
    if os.getenv("MODEL_AUTO_BOOTSTRAP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not force and all(not missing(group, root) for group in GROUPS):
        _STATE.update(state="done", groups={group: "exists" for group in GROUPS})
        return False
    if not _LOCK.acquire(blocking=False):
        return False
    _STATE.update(state="running", started_at=datetime.now(timezone.utc).isoformat(),
                  finished_at=None, groups={})
    Thread(target=_background, args=(root,), name="service-model-bootstrap", daemon=True).start()
    return True
