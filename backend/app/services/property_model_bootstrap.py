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
    kind: str = "forecast"       # "forecast" (sale / jeonse: per-dong forecast table) or "bundle" (wolse model bundle)


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
        input_file="부동산/매매_V64.csv",
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
        input_file="부동산/전세_V33.csv",
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
        input_file="부동산/월세_전환율.csv",
        build="BuildWolseInput.py",
        kind="bundle",
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


def _run_script(script: Path, cwd: Path, log_path: Path, label: str, extra_env: dict | None = None) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1", **(extra_env or {})}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} {label} ({script.name}) =====\n")
        log.flush()
        result = subprocess.run([sys.executable, "-W", "ignore", str(script)], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def _build_input(spec: ModelSpec, root: Path, force: bool = False) -> None:
    """Create the model's input file in data/processed when it is missing (or always, with force)."""
    p = paths(root)
    if not spec.build or ((p["processed"] / spec.input_file).is_file() and not force):
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


# ---------------------------------------------------------------------------------------------
# Monthly refresh: rebuild every model from the newest raw data and replace the served copy.
# The served copy is only touched after the new model passed validation, so a failed refresh keeps the old model.
# ---------------------------------------------------------------------------------------------
META_NAME = "pipeline_meta.json"
MIN_ROW_RATIO = 0.9          # a refreshed forecast table may not lose more than 10% of its dongs
PRICE_RATIO = (0.7, 1.5)     # median of (new current price / old current price) outside this range means broken input


def validate_candidate(spec: ModelSpec, root: Path = ROOT) -> list[str]:
    """Return the reasons the freshly trained model must not replace the served one (empty list = fine)."""
    import math

    import pandas as pd

    p = paths(root)
    served = p["model_root"] / spec.name
    problems: list[str] = []
    if spec.kind == "bundle":
        import joblib
        new = joblib.load(p["output"] / spec.outputs[0][0])
        for key in ("model", "features", "tables", "trained_until", "train_n"):
            if key not in new:
                problems.append(f"번들에 {key}가 없습니다")
        old_path = served / spec.outputs[0][1]
        if not problems and old_path.is_file():
            old = joblib.load(old_path)
            if new["trained_until"] < old["trained_until"]:
                problems.append(f"학습 마지막 날이 과거로 갔습니다 ({old['trained_until']} → {new['trained_until']})")
            if new["train_n"] < old["train_n"] * MIN_ROW_RATIO:
                problems.append(f"학습 건수가 줄었습니다 ({old['train_n']:,} → {new['train_n']:,})")
        return problems
    csv_name = next(src for src, dest in spec.outputs if dest == "neighborhood_forecasts.csv")
    new = pd.read_csv(p["output"] / csv_name, encoding="utf-8-sig")
    forecasts = [c for c in new.columns if c.endswith("뒤 예측(만원/평)")]
    if len(forecasts) != 3:
        return ["예측 열이 1·2·3년 3개가 아닙니다"]
    for column in forecasts:
        values = new[column].dropna()
        if values.empty or not all(math.isfinite(v) and v > 0 for v in values):
            problems.append(f"{column}에 비어 있거나 0 이하·무한대인 값이 있습니다")
    now = next(c for c in new.columns if c.startswith("현재 평당"))
    old_csv = served / "neighborhood_forecasts.csv"
    if old_csv.is_file():
        old = pd.read_csv(old_csv, encoding="utf-8-sig")
        if new[forecasts[0]].notna().sum() < old[forecasts[0]].notna().sum() * MIN_ROW_RATIO:
            problems.append("예측이 있는 동네 수가 10% 넘게 줄었습니다")
        both = new.merge(old, on="행정동_코드", suffixes=("", "_old"))
        ratio = (both[now] / both[now + "_old"]).dropna().median()
        if not (PRICE_RATIO[0] <= ratio <= PRICE_RATIO[1]):
            problems.append(f"현재 평당 가격 중앙값이 이전 대비 {ratio:.2f}배로 달라졌습니다")
    return problems


def _write_meta(spec: ModelSpec, root: Path, meta: dict) -> None:
    import json
    target = paths(root)["model_root"] / spec.name / META_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def refresh_models(root: Path = ROOT, data_end: str | None = None, base_month: str | None = None) -> dict:
    """Rebuild input files, retrain, validate, then replace the served models.

    data_end    : last contract day to use, YYYY-MM-DD (the wolse script reads it from PROPERTY_DATA_END)
    base_month  : YYYY-MM the forecasts start from; stored in pipeline_meta.json for the service
    Returns {model name: 'refreshed' | 'rejected: ...' | 'failed: ...'}.
    """
    result: dict[str, str] = {}
    p = paths(root)
    extra = {"PROPERTY_DATA_END": data_end} if data_end else {}
    for spec in SPECS:
        _STATE["models"][spec.name] = "refreshing"
        try:
            script = p["training"] / spec.script
            if not script.is_file():
                raise FileNotFoundError(f"학습 스크립트가 없습니다: {script}")
            p["output"].mkdir(parents=True, exist_ok=True)
            for source, _ in spec.outputs:                         # a stale output must never pass for a new one
                (p["output"] / source).unlink(missing_ok=True)
            if spec.input_file:
                (p["processed"] / spec.input_file).unlink(missing_ok=True)
            _build_input(spec, root)
            code = _run_script(script, p["training"], p["log"], f"refresh {spec.name}", extra)
            if code != 0:
                result[spec.name] = f"failed: 학습 스크립트 종료 코드 {code} (로그: {p['log']})"
            else:
                problems = validate_candidate(spec, root)
                if problems:
                    result[spec.name] = "rejected: " + "; ".join(problems)
                elif not _copy_outputs(spec, root):
                    result[spec.name] = "failed: 학습은 끝났지만 결과 파일이 없습니다"
                else:
                    _write_meta(spec, root, {"base_year_month": base_month, "data_end": data_end,
                                             "refreshed_at": datetime.now(timezone.utc).isoformat()})
                    result[spec.name] = "refreshed"
        except Exception as error:
            result[spec.name] = f"failed: {error}"
            logger.exception("property model refresh failed for %s", spec.name)
        _STATE["models"][spec.name] = result[spec.name]
        logger.log(logging.INFO if result[spec.name] == "refreshed" else logging.ERROR,
                   "property model refresh %s: %s", spec.name, result[spec.name])
    return result
