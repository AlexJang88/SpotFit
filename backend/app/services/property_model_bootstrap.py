"""Create the final apartment sale / jeonse / wolse models in models/recommender/v1 when they are missing.

Runs once in the background when the API server starts, so the server never waits for training.
  1. Every target file already exists                -> nothing to do.
  2. The training output already exists (ml/training) -> copy it.
  3. Otherwise                                        -> build the model's input file (ml/etl) if it is missing,
                                                         run the training script, then copy its output.
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
    input_file: str = ""         # file in data/processed that the script reads (built by `build` when missing)
    build: str = ""              # file in ml/etl that creates input_file


SPECS = (
    ModelSpec(
        name="apart_sale_final",
        script="AptSaleForecastV64Input.py",
        outputs=(
            ("아파트매매_v64_입력_1년뒤모델.joblib", "model_1y.joblib"),
            ("아파트매매_v64_입력_2년뒤모델.joblib", "model_2y.joblib"),
            ("아파트매매_v64_입력_3년뒤모델.joblib", "model_3y.joblib"),
            ("아파트매매_v64_입력_동네별예측_1년2년3년뒤.csv", "neighborhood_forecasts.csv"),
            ("아파트매매_v64_입력_동네별예측_1년2년3년뒤.json", "neighborhood_forecasts.json"),
        ),
        description="아파트 매매 v64: 행정동별 1·2·3년 뒤 평당 매매가",
        input_file="매매_V64.csv",
        build="BuildSaleInput.py",
    ),
    ModelSpec(
        name="apart_jeonse_final",
        script="AptJeonseForecastV33Input.py",
        outputs=(
            ("아파트전세_v33_입력_1년뒤모델.joblib", "model_1y.joblib"),
            ("아파트전세_v33_입력_2년뒤모델.joblib", "model_2y.joblib"),
            ("아파트전세_v33_입력_3년뒤모델.joblib", "model_3y.joblib"),
            ("아파트전세_v33_입력_동네별예측_1년2년3년뒤.csv", "neighborhood_forecasts.csv"),
            ("아파트전세_v33_입력_동네별예측_1년2년3년뒤.json", "neighborhood_forecasts.json"),
        ),
        description="아파트 전세 v33: 행정동별 1·2·3년 뒤 평당 전세 보증금",
        input_file="전세_V33.csv",
        build="BuildJeonseInput.py",
    ),
    ModelSpec(
        name="apart_wolse_final",
        script="RentConversionFinalInput.py",
        outputs=(
            ("월세계약_전환율모델_최종_입력.joblib", "model_bundle.joblib"),
            ("월세계약_전환율모델_최종_입력요약.json", "summary.json"),
        ),
        description="아파트 월세 계약 단위 전환율 모델: 보증금이나 월세 중 하나를 입력하면 나머지를 예측 (같은 건물·면적대 직전 12개월 전세 2건 미만이면 예측 불가)",
        input_file="월세_전환율.csv",
        build="BuildWolseInput.py",
    ),
)

_LOCK = Lock()
_STATE: dict = {"state": "idle", "started_at": None, "finished_at": None, "models": {}}


def paths(root: Path = ROOT) -> dict[str, Path]:
    return {
        "model_root": root / "models" / "recommender" / "v1",
        "training": root / "ml" / "training",
        "etl": root / "ml" / "etl",
        "processed": root / "data" / "processed",
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


def _run_script(script: Path, cwd: Path, log_path: Path, label: str) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} {label} ({script.name}) =====\n")
        log.flush()
        result = subprocess.run([sys.executable, "-W", "ignore", str(script)], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def _build_input(spec: ModelSpec, root: Path) -> None:
    """Create the model's input file in data/processed when it is missing."""
    p = paths(root)
    if not spec.build or (p["processed"] / spec.input_file).is_file():
        return
    script = p["etl"] / spec.build
    if not script.is_file():
        raise FileNotFoundError(f"입력 파일 생성 스크립트가 없습니다: {script}")
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    code = _run_script(script, p["etl"], p["log"], f"build input {spec.name}")
    if code != 0 or not (p["processed"] / spec.input_file).is_file():
        raise RuntimeError(f"입력 파일 생성 실패: {spec.input_file} (종료 코드 {code}, 로그: {p['log']})")


def _train(spec: ModelSpec, root: Path) -> int:
    p = paths(root)
    script = p["training"] / spec.script
    if not script.is_file():
        raise FileNotFoundError(f"학습 스크립트가 없습니다: {script}")
    if not (root / "data" / "processed").is_dir():
        raise FileNotFoundError(f"학습 데이터 폴더가 없습니다: {root / 'data' / 'processed'}")
    p["output"].mkdir(parents=True, exist_ok=True)
    _build_input(spec, root)
    return _run_script(script, p["training"], p["log"], f"train {spec.name}")


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
        _STATE["models"][spec.name] = result[spec.name]
        logger.info("property model %s: %s", spec.name, result[spec.name])
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
