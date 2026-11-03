"""어린이보호구역 특징으로 다음 연도 연령대별 아동 인구 증가율을 예측한다.

모델 선정은 2023년 자료 내부의 5-fold 교차검증 RMSE로 수행하고, 2024년 특징→2025년
증가율은 완전히 분리된 시간 외 테스트로 한 번만 평가한다. 테스트 성능을 보고 모델을 고르면
테스트셋에 과적합되므로 선정 기준과 최종 평가 기준을 분리한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/interim/child_safety_zone/safety_zone_analysis_panel.csv"
AGE_GROUPS = ["infant", "elementary", "middle", "high"]
FEATURES = [
    "safety_zone_total", "safety_zone_elementary", "safety_zone_kindergarten",
    "safety_zone_daycare", "safety_zone_special_school", "safety_zone_academy",
    "safety_zone_foreign_school", "safety_zone_alternative_school",
    "log_total_population", "log_current_children",
]


def candidates() -> dict[str, object]:
    """선형 기준모델과 비선형 앙상블을 함께 비교해 선정 근거를 남긴다."""
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    tree = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    return {
        "dummy_median": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))]),
        "linear": scaled(LinearRegression()),
        "ridge": scaled(Ridge(alpha=10.0)),
        "random_forest": tree(RandomForestRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=1,
        )),
        "extra_trees": tree(ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=1,
        )),
    }


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
    }


def configure_plot() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/child_safety_zone")
    parser.add_argument(
        "--tracking-uri",
        help="MLflow 서버 URI. 생략하면 프로젝트 artifacts 아래의 로컬 SQLite를 사용합니다.",
    )
    parser.add_argument(
        "--experiment-name",
        default="child-safety-zone-next-year-growth",
        help="후보 모델 Run을 기록할 MLflow Experiment 이름입니다.",
    )
    parser.add_argument(
        "--register-best",
        action="store_true",
        help="연령군별 CV 우승 모델만 MLflow Model Registry에 등록합니다.",
    )
    parser.add_argument(
        "--minio-env-file",
        type=Path,
        help="MinIO 설정 파일. 생략하면 프로젝트의 .env.dev, .env 순서로 찾습니다.",
    )
    parser.add_argument(
        "--data-prefix",
        default="",
        help="MinIO 데이터 버킷의 객체 접두사. 기존 upload_minio.py 구조는 빈 값입니다.",
    )
    parser.add_argument(
        "--skip-data-sync",
        action="store_true",
        help="오프라인 재현 등 명시적인 경우에만 시작 전 MinIO 데이터 동기화를 건너뜁니다.",
    )
    args = parser.parse_args()

    # 어느 PC에서 실행해도 동일한 공유 입력을 먼저 확보한다. 동기화 실패 시 기존 로컬 자료로
    # 계속하지 않고 예외를 발생시켜 데이터 버전이 다른 상태에서 모델이 등록되는 것을 막는다.
    sync_result: dict[str, object] = {"status": "skipped_by_user"}
    if not args.skip_data_sync:
        sys.path.insert(0, str(ROOT))
        from scripts.minio_data_sync import sync_minio_data

        sync_result = sync_minio_data(
            root=ROOT,
            data_dir=ROOT / "data",
            env_file=args.minio_env_file,
            prefix=args.data_prefix,
        )
        print(
            "MinIO data sync completed: "
            f"downloaded={sync_result['downloaded']}, skipped={sync_result['skipped']}"
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / stamp; out.mkdir(parents=True, exist_ok=False)
    model_dir = out / "models"; model_dir.mkdir()

    # 원격 URI가 주어지면 서버 MLflow를 그대로 사용한다. 연결 실패 시 로컬로 자동
    # 우회하지 않으므로 PostgreSQL에 기록했다고 오인하는 일을 막을 수 있다.
    local_tracking = ROOT / "artifacts/mlflow_child_safety_zone"
    local_tracking.mkdir(parents=True, exist_ok=True)
    tracking_uri = args.tracking_uri or f"sqlite:///{(local_tracking / 'tracking.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(
            args.experiment_name,
            artifact_location=None if args.tracking_uri else (local_tracking / "runs").as_uri(),
        )
    mlflow.set_experiment(args.experiment_name)

    data = pd.read_csv(args.data, dtype={"dong_code": str})
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    selected: dict[str, str] = {}
    candidate_run_ids: dict[tuple[str, str], str] = {}

    # 부모 Run은 한 번의 전체 실행을 나타내고, 4개 연령군 × 5개 알고리즘은 자식 Run으로
    # 기록한다. 따라서 팀원이 동시에 실행해도 각 실행과 후보가 서로 섞이지 않는다.
    parent_run = mlflow.start_run(run_name=f"safety_zone_comparison_{stamp}")
    mlflow.log_params({
        "data_path": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_feature_year": 2023,
        "test_feature_year": 2024,
        "cv_folds": 5,
        "minimum_base_children": 30,
        "candidate_count_per_age": len(candidates()),
        "minio_data_bucket": sync_result.get("bucket", "sync_skipped"),
        "minio_data_prefix": sync_result.get("prefix", ""),
        "minio_downloaded_files": sync_result.get("downloaded", 0),
        "minio_skipped_files": sync_result.get("skipped", 0),
    })
    mlflow.set_tags({
        "task": "next_year_child_population_log_growth",
        "dataset_unit": "administrative_dong_year",
        "selection_rule": "lowest_2023_5fold_cv_rmse",
        "deployment_status": "exploratory_not_production",
        "mlflow.note.content": "실제 어린이보호구역 연도별 자료 기반 탐색 실험. 시간 전환이 두 개뿐이므로 자동 배포 금지.",
    })
    mlflow.log_artifact(str(Path(__file__)), artifact_path="source")

    for group in AGE_GROUPS:
        target = f"target_next_year_log_growth_{group}"
        work = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
        work["log_total_population"] = np.log1p(work["total_population"])
        work["log_current_children"] = np.log1p(work[f"children_{group}"])
        train = work[work["year"].eq(2023)].copy()
        test = work[work["year"].eq(2024)].copy()
        if train.empty or test.empty:
            raise ValueError(f"{group}: 2023 학습 또는 2024 시간 테스트 자료가 없습니다.")

        # 한 연도 안에서 모델 선택의 분산을 줄이기 위한 CV다. 시간 일반화 성능은 아래의
        # 2024→2025 홀드아웃에서 별도로 판단해야 한다.
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        fitted: dict[str, object] = {}
        for name, model in candidates().items():
            scores = -cross_val_score(model, train[FEATURES], train[target], cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
            model.fit(train[FEATURES], train[target])
            prediction = model.predict(test[FEATURES])
            test_metrics = metrics(test[target], prediction)
            rows.append({
                "age_group": group, "model": name,
                "cv_rmse_mean": float(scores.mean()), "cv_rmse_std": float(scores.std()),
                "train_rows": len(train), "test_rows": len(test), **{f"test_{k}": v for k, v in test_metrics.items()},
            })
            fitted[name] = model

            # 후보마다 별도 자식 Run을 만들어 알고리즘과 성능을 비교할 근거를 남긴다.
            # 테스트 지표도 기록하지만 모델 선정에는 CV 지표만 사용한다.
            with mlflow.start_run(run_name=f"{group}_{name}", nested=True) as candidate_run:
                candidate_run_ids[(group, name)] = candidate_run.info.run_id
                mlflow.log_params({
                    "age_group": group,
                    "algorithm": name,
                    "features": json.dumps(FEATURES, ensure_ascii=False),
                    "feature_count": len(FEATURES),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "random_seed": 42,
                })
                mlflow.log_metrics({
                    "cv_rmse_mean": float(scores.mean()),
                    "cv_rmse_std": float(scores.std()),
                    **{f"holdout_{key}": value for key, value in test_metrics.items()},
                })
                mlflow.set_tags({
                    "selection_status": "candidate_not_yet_selected",
                    "holdout_usage": "evaluation_only_not_selection",
                })

        group_result = pd.DataFrame([row for row in rows if row["age_group"] == group])
        winner = group_result.sort_values(["cv_rmse_mean", "cv_rmse_std"]).iloc[0]["model"]
        selected[group] = str(winner)
        winner_model = fitted[str(winner)]
        winner_prediction = winner_model.predict(test[FEATURES])
        joblib.dump(winner_model, model_dir / f"child_safety_zone_{group}.joblib")

        # 우승 후보의 기존 자식 Run을 다시 열어 모델과 선정 근거를 연결한다. Registry 등록은
        # --register-best를 명시한 경우에만 수행하며, 등록되어도 운영 배포 상태는 아니다.
        with mlflow.start_run(run_id=candidate_run_ids[(group, str(winner))], nested=True):
            mlflow.set_tags({
                "selection_status": "cv_winner",
                "validation_status": "exploratory_not_production",
            })
            model_info = mlflow.sklearn.log_model(
                winner_model,
                name="model",
                serialization_format="cloudpickle",
                # 추론 시 결측값이 들어오면 정수 컬럼도 실수형으로 변환될 수 있다. MLflow
                # signature가 이를 거부하지 않도록 현실적인 입력 스키마를 float로 기록한다.
                input_example=train[FEATURES].head(3).astype(float),
            )
            mlflow.log_artifact(
                str(model_dir / f"child_safety_zone_{group}.joblib"),
                artifact_path="joblib",
            )
            if args.register_best:
                registered_name = f"spotfit-child-safety-zone-{group}"
                version = mlflow.register_model(model_info.model_uri, registered_name).version
                client = mlflow.MlflowClient()
                client.set_model_version_tag(
                    registered_name, version, "validation_status", "exploratory_not_production"
                )
                client.set_model_version_tag(registered_name, version, "source_run_id", candidate_run_ids[(group, str(winner))])

        prediction_frame = test[["year", "dong_code", "gu", "dong", target]].copy()
        prediction_frame["age_group"] = group
        prediction_frame["selected_model"] = winner
        prediction_frame["prediction"] = winner_prediction
        prediction_frame["absolute_error"] = (prediction_frame[target] - prediction_frame["prediction"]).abs()
        predictions.append(prediction_frame)

        # 순열 중요도는 테스트 자료에서 한 특징을 섞었을 때 RMSE가 얼마나 나빠지는지 측정한다.
        # 음수 중요도는 해당 특징이 시간 외 예측에 도움을 주지 못했을 가능성을 뜻한다.
        permutation = permutation_importance(
            winner_model, test[FEATURES], test[target], scoring="neg_root_mean_squared_error",
            n_repeats=20, random_state=42, n_jobs=1,
        )
        for feature, mean, std in zip(FEATURES, permutation.importances_mean, permutation.importances_std):
            importance_rows.append({"age_group": group, "model": winner, "feature": feature, "importance": mean, "importance_std": std})

    comparison = pd.DataFrame(rows).sort_values(["age_group", "cv_rmse_mean"])
    prediction_data = pd.concat(predictions, ignore_index=True)
    importance = pd.DataFrame(importance_rows).sort_values(["age_group", "importance"], ascending=[True, False])
    comparison.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")
    prediction_data.to_csv(out / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(out / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    (out / "model_metadata.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(), "target_horizon": "next_year",
        "train_feature_year": 2023, "test_feature_year": 2024,
        "selected_by": "lowest_2023_5fold_cv_rmse", "selected_models": selected,
        "features": FEATURES,
        "warning": "Only two annual transitions are available; results are experimental, not production-ready.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_plot()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison, x="age_group", y="test_rmse", hue="model")
    plt.title("2024년 특징 → 2025년 증가율 시간 외 RMSE 비교")
    plt.tight_layout(); plt.savefig(out / "01_model_comparison.png", dpi=160); plt.close()
    top_importance = importance.groupby("age_group", group_keys=False).head(7)
    grid = sns.catplot(data=top_importance, x="importance", y="feature", col="age_group", kind="bar", col_wrap=2, height=4, sharex=False, sharey=False)
    grid.fig.suptitle("선정 모델의 시간 외 순열 중요도", y=1.02)
    grid.savefig(out / "02_feature_importance.png", dpi=160); plt.close("all")

    winners = comparison[comparison.apply(lambda row: row["model"] == selected[row["age_group"]], axis=1)]
    report = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>어린이보호구역 예측모델</title>
<style>body{{font-family:Malgun Gothic,sans-serif;max-width:1150px;margin:36px auto;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}.warn{{background:#fff3cd;padding:14px;border-left:5px solid #d39e00}}</style>
<h1>어린이보호구역 기반 다음 연도 아동 인구 로그 증감률 모델</h1>
<h2>검증 설계</h2><p>2023년 보호구역 특징으로 모델을 학습·교차검증하고, 2024년 특징으로 2025년 증가율을 예측했습니다. 우승 모델은 테스트 성능이 아니라 2023년 내부 5-fold CV RMSE로 선택했습니다.</p>
<p>목표는 <code>log((다음 연도 아동 수+1)/(현재 아동 수+1))</code>입니다. 일반 증감률은 시작 인구가 매우 작은 행정동에서 수십 배가 되어 모델 손실을 왜곡하므로 로그 비율로 안정화했습니다. 또한 현재 해당 연령군 인구가 30명 미만인 경우 행 자체는 보존하되 그 연령군의 학습 목표에서만 제외했습니다. 양수는 증가, 0은 유지, 음수는 감소입니다.</p>
<div class='warn'><b>중요:</b> 연간 전환이 두 개뿐이어서 시간 변화 패턴을 충분히 학습하지 못합니다. 이 모델은 방법 검증용이며 2026년 인구 정답과 추가 연도 자료가 쌓이기 전에는 운영 배포 모델로 사용하면 안 됩니다.</div>
<h2>선정 모델</h2>{winners.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}
<h2>전체 모델 비교</h2>{comparison.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}
<img src='01_model_comparison.png' alt='모델 성능 비교'>
<h2>특징 중요도</h2><p>순열 중요도는 예측 기여도이며 인과 가중치가 아닙니다. 점수 가중치로 사용하려면 추가 연도에서 방향·순위가 반복되는지 검증해야 합니다.</p><img src='02_feature_importance.png' alt='특징 중요도'>
<h2>다음 단계</h2><ol><li>2026년 연말 보호구역과 아동 인구를 추가해 2025→2026 전환을 확보합니다.</li><li>주택공급·가격·출생·전입전출 특징을 추가해 혼란요인을 통제합니다.</li><li>연도 롤링 검증이 최소 3회 이상 가능해진 뒤 가중치 후보를 선정합니다.</li></ol></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")

    # 비교 근거와 보고서도 부모 Run에 함께 저장한다. 원격 MLflow 서버가 PostgreSQL과
    # MinIO로 구성돼 있다면 메타데이터는 PostgreSQL, 이 파일들은 서버 artifact 저장소에 간다.
    mlflow.log_dict(selected, "selected_models.json")
    for result_path in out.iterdir():
        if result_path.is_file():
            mlflow.log_artifact(str(result_path), artifact_path="evidence")
    mlflow.set_tag("registry_requested", str(args.register_best).lower())
    mlflow.end_run()
    print(comparison.to_string(index=False))
    print(f"선정 모델: {selected}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow Registry 등록: {'완료 요청' if args.register_best else '미실행'}")
    print(f"결과: {out.resolve()}")


if __name__ == "__main__":
    main()
