"""예외 승인된 범죄안전 직접점수 ML 모델을 MLflow/Registry에 저장한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
BASE_SCRIPT = ROOT / "ml" / "training" / "train_crime_safety_1y.py"
EVAL_DIR = ROOT / "artifacts" / "crime_safety_forecast" / "grade_transition_rolling_2021_2023"
LOCAL_MODEL = EVAL_DIR / "crime_safety_direct_best_ml_exception.joblib"

spec = importlib.util.spec_from_file_location("crime_component", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.dev", override=True)


def score_pairs(data: pd.DataFrame) -> pd.DataFrame:
    pairs = base.make_pairs(data)
    future = data[["연도", "자치구", "범죄안전_대체점수"]].copy()
    future["기준연도"] = future["연도"] - 1
    future = future.drop(columns="연도").rename(columns={"범죄안전_대체점수": "target_score"})
    return pairs.merge(future, on=["기준연도", "자치구"], validate="one_to_one")


def main() -> int:
    load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment", default="crime-safety-forecast")
    parser.add_argument("--registered-name", default="crime_safety_direct_score_1y")
    args = parser.parse_args()
    if not args.tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI 또는 --tracking-uri가 필요합니다.")

    data = pd.read_csv(base.INPUT, encoding="utf-8-sig")
    pairs = score_pairs(data)
    train = pairs[pairs["목표연도"] <= 2021]
    validation = pairs[pairs["목표연도"] == 2022]

    comparisons = []
    for name, estimator in base.candidates().items():
        candidate = base.pipeline(estimator)
        candidate.fit(train[base.FEATURES], train["target_score"])
        rmse = mean_squared_error(
            validation["target_score"], candidate.predict(validation[base.FEATURES])
        ) ** 0.5
        comparisons.append({"모델": name, "검증_RMSE": float(rmse)})
    comparison = pd.DataFrame(comparisons).sort_values("검증_RMSE")
    best_name = str(comparison.iloc[0]["모델"])

    model = base.pipeline(base.candidates()[best_name])
    model.fit(pairs[base.FEATURES], pairs["target_score"])
    example = pairs[base.FEATURES].head(5)
    signature = infer_signature(example, model.predict(example))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "algorithm": best_name,
        "features": base.FEATURES,
        "target": "다음연도_범죄안전_대체점수",
        "training_target_years": sorted(pairs["목표연도"].unique().tolist()),
        "exception_approval": True,
    }
    joblib.dump(bundle, LOCAL_MODEL)
    model_sha256 = hashlib.sha256(LOCAL_MODEL.read_bytes()).hexdigest()
    data_sha256 = hashlib.sha256(base.INPUT.read_bytes()).hexdigest()
    metrics = json.loads((EVAL_DIR / "등급전환_평가지표.json").read_text(encoding="utf-8"))[
        "DirectBestML"
    ]
    comparison_path = EVAL_DIR / "예외저장_직접점수_ML비교.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = f"direct_best_ml_exception_{datetime.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "algorithm": best_name,
            "target": "next_year_crime_safety_proxy_score",
            "feature_count": len(base.FEATURES),
            "features": json.dumps(base.FEATURES, ensure_ascii=False),
            "training_target_years": ",".join(map(str, sorted(pairs["목표연도"].unique()))),
            "dataset_sha256": data_sha256,
            "model_sha256": model_sha256,
            "selection_rule": "best_ml_validation_rmse_2022_excluding_persistence",
        })
        mlflow.log_metrics({
            "rolling_rmse": metrics["RMSE"],
            "rolling_mae": metrics["MAE"],
            "rolling_r2": metrics["R2"],
            "grade_accuracy": metrics["등급_정확도"],
            "transition_recall": metrics["등급전환_탐지율_재현율"],
            "transition_precision": metrics["등급전환_정밀도"],
            "transition_f1": metrics["등급전환_F1"],
            "changed_district_direction_accuracy": metrics["실제전환구_방향정확도"],
        })
        mlflow.set_tags({
            "owner": "SpotFit",
            "task_id": "crime_safety_transition_exception_save",
            "selection_status": "exception_saved_not_champion",
            "review_status": "requires_review",
            "model_role": "grade_transition_warning_challenger",
            "standard_model": "false",
            "exception_approval": "user_requested_2026-11-03",
            "mlflow.note.content": (
                "RMSE 우승 모델이 아니지만 등급전환 탐지용 도전자로 사용자 예외 승인에 따라 저장. "
                "75개 롤링 표본에서 전환 재현율 61.1%, 정밀도 42.3%로 오탐 위험이 있어 champion 지정 금지."
            ),
        })
        mlflow.log_dict({
            "features": base.FEATURES,
            "target": "다음 연도 범죄안전 대체점수",
            "weights": base.WEIGHTS,
            "limitations": ["공식 지역안전등급 아님", "실제 전환 18건", "오탐 15건"],
        }, "model_definition.json")
        mlflow.log_artifact(str(LOCAL_MODEL), artifact_path="joblib")
        mlflow.log_artifact(str(comparison_path), artifact_path="evidence")
        mlflow.log_artifact(str(EVAL_DIR / "등급전환_평가지표.json"), artifact_path="evidence")
        mlflow.log_artifact(str(EVAL_DIR / "report.md"), artifact_path="evidence")
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example,
            serialization_format="cloudpickle",
        )
        version = mlflow.register_model(model_info.model_uri, args.registered_name)
        client = mlflow.MlflowClient()
        client.set_model_version_tag(args.registered_name, version.version, "selection_status", "exception_saved_not_champion")
        client.set_model_version_tag(args.registered_name, version.version, "review_status", "requires_review")
        client.set_model_version_tag(args.registered_name, version.version, "model_sha256", model_sha256)

        record = {
            "tracking_uri": args.tracking_uri,
            "experiment": args.experiment,
            "run_id": run.info.run_id,
            "registered_model": args.registered_name,
            "model_version": str(version.version),
            "algorithm": best_name,
            "features": base.FEATURES,
            "local_model": str(LOCAL_MODEL),
            "model_sha256": model_sha256,
        }
        (EVAL_DIR / "예외저장_MLflow기록.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mlflow.log_dict(record, "registration_record.json")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
