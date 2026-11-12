"""Create the final apartment sale / jeonse / wolse models in models/recommender/v1 when they are missing.

Runs once in the background when the API server starts, so the server never waits for training.
  1. Every target file already exists                -> nothing to do.
  2. The training output already exists (ml/training) -> copy it.
  3. Otherwise                                        -> run the training script, then copy its output.
The copy is atomic per file, so a half-written model is never mistaken for a finished one.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

logger = logging.getLogger("property_model_bootstrap")

ROOT = Path(__file__).resolve().parents[3]
LOG_NAME = "property_model_bootstrap.log"


@dataclass(frozen=True)
class ModelSpec:
    name: str                    # directory under models/recommender/v1
    script: str                  # file in ml/training
    outputs: tuple[tuple[str, str], ...]   # (file written by the script in data/processed/output, file name in the model directory)
    description: str


SPECS = (
    ModelSpec(
        name="apart_sale_final",
        script="AptSaleForecastV55.py",
        outputs=(
            ("아파트매매_v55_1년뒤모델.joblib", "model_1y.joblib"),
            ("아파트매매_v55_2년뒤모델.joblib", "model_2y.joblib"),
            ("아파트매매_v55_3년뒤모델.joblib", "model_3y.joblib"),
            ("아파트매매_v55_동네별예측_1년2년3년뒤.csv", "neighborhood_forecasts.csv"),
            ("아파트매매_v55_동네별예측_1년2년3년뒤.json", "neighborhood_forecasts.json"),
        ),
        description="아파트 매매 v55: 행정동별 1·2·3년 뒤 평당 매매가",
    ),
    ModelSpec(
        name="apart_jeonse_final",
        script="AptJeonseForecastV33.py",
        outputs=(
            ("아파트전세_v33_1년뒤모델.joblib", "model_1y.joblib"),
            ("아파트전세_v33_2년뒤모델.joblib", "model_2y.joblib"),
            ("아파트전세_v33_3년뒤모델.joblib", "model_3y.joblib"),
            ("아파트전세_v33_동네별예측_1년2년3년뒤.csv", "neighborhood_forecasts.csv"),
            ("아파트전세_v33_동네별예측_1년2년3년뒤.json", "neighborhood_forecasts.json"),
        ),
        description="아파트 전세 v33: 행정동별 1·2·3년 뒤 평당 전세 보증금",
    ),
    ModelSpec(
        name="apart_wolse_final",
        script="RentContractFinal.py",
        outputs=(
            ("월세계약모델_최종.joblib", "model_bundle.joblib"),
            ("월세계약모델_최종요약.json", "summary.json"),
        ),
        description="아파트 월세 계약 단위 모델: 보증금을 입력하면 월세를 예측 (같은 동 계약 10건 미만이면 예측 불가)",
    ),
)

_LOCK = Lock()
_STATE: dict = {"state": "idle", "started_at": None, "finished_at": None, "models": {}}


def paths(root: Path = ROOT) -> dict[str, Path]:
    return {
        "model_root": root / "models" / "recommender" / "v1",
        "training": root / "ml" / "training",
        "output": root / "data" / "processed" / "output",
        "log": root / "artifacts" / LOG_NAME,
    }


def is_ready(spec: ModelSpec, root: Path = ROOT) -> bool:
    target = paths(root)["model_root"] / spec.name
    return all((target / dest).is_file() and (target / dest).stat().st_size > 0 for _, dest in spec.outputs)


def status(root: Path = ROOT) -> dict:
    return {**_STATE, "ready": {spec.name: is_ready(spec, root) for spec in SPECS}}


def _copy_outputs(spec: ModelSpec, root: Path) -> bool:
    p = paths(root)
    sources = [p["output"] / src for src, _ in spec.outputs]
    if not all(s.is_file() and s.stat().st_size > 0 for s in sources):
        return False
    target = p["model_root"] / spec.name
    target.mkdir(parents=True, exist_ok=True)
    for source, (_, dest) in zip(sources, spec.outputs):
        temporary = target / f".{dest}.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, target / dest)
    (target / "MODEL_INFO.txt").write_text(
        f"{spec.description}\nsource script: ml/training/{spec.script}\n"
        f"created: {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    return True


def _train(spec: ModelSpec, root: Path) -> int:
    p = paths(root)
    script = p["training"] / spec.script
    if not script.is_file():
        raise FileNotFoundError(f"학습 스크립트가 없습니다: {script}")
    if not (root / "data" / "processed").is_dir():
        raise FileNotFoundError(f"학습 데이터 폴더가 없습니다: {root / 'data' / 'processed'}")
    p["output"].mkdir(parents=True, exist_ok=True)
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    with p["log"].open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} train {spec.name} ({spec.script}) =====\n")
        log.flush()
        result = subprocess.run([sys.executable, "-W", "ignore", str(script)], cwd=p["training"], env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def ensure_models(root: Path = ROOT) -> dict:
    """Make sure every final model exists. Returns {model name: 'exists' | 'copied' | 'trained' | 'failed: ...'}."""
    result: dict[str, str] = {}
    for spec in SPECS:
        _STATE["models"][spec.name] = "checking"
        try:
            if is_ready(spec, root):
                result[spec.name] = "exists"
            elif _copy_outputs(spec, root):
                result[spec.name] = "copied"
            else:
                _STATE["models"][spec.name] = "training"
                code = _train(spec, root)
                if code != 0:
                    result[spec.name] = f"failed: 학습 스크립트 종료 코드 {code} (로그: {paths(root)['log']})"
                elif _copy_outputs(spec, root):
                    result[spec.name] = "trained"
                else:
                    result[spec.name] = "failed: 학습은 끝났지만 결과 파일이 없습니다"
        except Exception as error:  # one failing model must not stop the others or the server
            result[spec.name] = f"failed: {error}"
            logger.exception("property model bootstrap failed for %s", spec.name)
        _STATE["models"][spec.name] = result[spec.name]
        level = logging.ERROR if result[spec.name].startswith("failed") else logging.INFO
        logger.log(level, "property model %s: %s", spec.name, result[spec.name])
    return result


def _run(root: Path) -> None:
    try:
        ensure_models(root)
        failed = [name for name, value in _STATE["models"].items() if str(value).startswith("failed")]
        _STATE["state"] = "failed" if failed else "done"
    except Exception as error:
        _STATE["state"] = "failed"
        logger.exception("property model bootstrap crashed: %s", error)
    finally:
        _STATE["finished_at"] = datetime.now(timezone.utc).isoformat()
        _LOCK.release()


def start_background_bootstrap(root: Path = ROOT) -> bool:
    """Start the check in a daemon thread. Returns False when disabled or already running."""
    if os.getenv("PROPERTY_MODEL_BOOTSTRAP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if all(is_ready(spec, root) for spec in SPECS):
        _STATE.update(state="done", models={spec.name: "exists" for spec in SPECS})
        return False
    if not _LOCK.acquire(blocking=False):
        return False
    _STATE.update(state="running", started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    Thread(target=_run, args=(root,), name="property-model-bootstrap", daemon=True).start()
    return True
