"""Train honest 1-year and 3-year Seoul child-environment score forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "child_environment_scores"
INPUT_CANDIDATES = [
    DATA_DIR / "scored_training_panel.csv",
    DATA_DIR / "scored_training_panel_updated.csv",
]
OUT = ROOT / "models" / "child_environment_forecast" / "v1"
TARGET = "child_environment_score_100"
HORIZONS = (1, 3)
GRADE_LABELS = np.array(["A", "B", "C", "D", "E"])


def add_exact_quintile_grade(frame: pd.DataFrame, score_col: str, group_col: str) -> pd.DataFrame:
    result = frame.sort_values([group_col, score_col, "gu"], ascending=[True, False, True]).copy()
    result["forecast_rank"] = result.groupby(group_col).cumcount() + 1
    result["grade"] = GRADE_LABELS[((result["forecast_rank"] - 1) // 5).clip(0, 4).to_numpy()]
    return result


def load_panel() -> tuple[pd.DataFrame, Path]:
    path = next(p for p in INPUT_CANDIDATES if p.exists())
    frame = pd.read_csv(path, encoding="utf-8-sig").sort_values(["gu", "year"]).reset_index(drop=True)
    if frame[["year", "gu"]].duplicated().any():
        raise ValueError("Duplicate gu-year rows")
    return frame, path


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    indicator_cols = [
        c for c in result.columns
        if ("_per_" in c or c.endswith("_per_capita")) and not c.startswith("score_")
    ]
    score_cols = [c for c in result.columns if c.startswith("score_")]
    base = indicator_cols + score_cols + [TARGET]
    grouped = result.groupby("gu", sort=False)
    for col in base:
        result[f"lag1_change__{col}"] = grouped[col].diff(1)
        result[f"recent3_mean__{col}"] = grouped[col].transform(
            lambda s: s.shift(0).rolling(3, min_periods=1).mean()
        )
    numeric = base + [f"lag1_change__{c}" for c in base] + [f"recent3_mean__{c}" for c in base]
    return result, numeric


def make_supervised(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = frame[["gu", "year", TARGET]].rename(
        columns={"year": "target_year", TARGET: "target_score"}
    )
    source = frame.copy()
    source["target_year"] = source["year"] + horizon
    return source.merge(future, on=["gu", "target_year"], how="inner", validate="many_to_one")


def pipeline(model, numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("gu", OneHotEncoder(handle_unknown="ignore"), ["gu"]),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def candidates(numeric: list[str]) -> dict[str, Pipeline]:
    return {
        "ridge_1": pipeline(Ridge(alpha=1.0), numeric),
        "ridge_10": pipeline(Ridge(alpha=10.0), numeric),
        "random_forest": pipeline(RandomForestRegressor(
            n_estimators=600, min_samples_leaf=4, max_features=0.7,
            random_state=42, n_jobs=-1,
        ), numeric),
        "extra_trees": pipeline(ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=-1,
        ), numeric),
    }


def metrics(y, pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def train_horizon(features: pd.DataFrame, numeric: list[str], horizon: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    supervised = make_supervised(features, horizon)
    holdout_year = int(supervised["target_year"].max())
    cv_years = [y for y in sorted(supervised["target_year"].unique()) if 2020 <= y < holdout_year]
    model_rows, residual_rows = [], []
    model_set = candidates(numeric)

    # Expanding-window, target-year-separated validation prevents future rows
    # from entering preprocessing or fitting.
    for name, model in model_set.items():
        fold_metrics = []
        for val_year in cv_years:
            train = supervised[supervised["target_year"] < val_year]
            val = supervised[supervised["target_year"] == val_year]
            if len(train) < 50 or val.empty:
                continue
            model.fit(train[numeric + ["gu"]], train["target_score"])
            pred = np.clip(model.predict(val[numeric + ["gu"]]), 25, 100)
            fold_metrics.append(metrics(val["target_score"], pred))
            residual_rows.extend({
                "horizon": horizon, "model": name, "target_year": val_year,
                "gu": gu, "actual": actual, "prediction": estimate,
                "residual": actual - estimate,
            } for gu, actual, estimate in zip(val["gu"], val["target_score"], pred))
        model_rows.append({
            "horizon": horizon, "model": name, "validation_years": ",".join(map(str, cv_years)),
            "cv_mae": np.mean([m["mae"] for m in fold_metrics]),
            "cv_rmse": np.mean([m["rmse"] for m in fold_metrics]),
            "cv_r2": np.mean([m["r2"] for m in fold_metrics]),
        })

    # Persistence is a non-fitted benchmark, not eligible as the saved ML model.
    baseline_folds = []
    for val_year in cv_years:
        val = supervised[supervised["target_year"] == val_year]
        baseline_folds.append(metrics(val["target_score"], val[TARGET]))
    model_rows.append({
        "horizon": horizon, "model": "baseline_persistence",
        "validation_years": ",".join(map(str, cv_years)),
        "cv_mae": np.mean([m["mae"] for m in baseline_folds]),
        "cv_rmse": np.mean([m["rmse"] for m in baseline_folds]),
        "cv_r2": np.mean([m["r2"] for m in baseline_folds]),
    })

    comparison = pd.DataFrame(model_rows)
    best_name = comparison[comparison["model"] != "baseline_persistence"].sort_values("cv_rmse").iloc[0]["model"]
    train = supervised[supervised["target_year"] < holdout_year]
    holdout = supervised[supervised["target_year"] == holdout_year].copy()
    selected = candidates(numeric)[best_name]
    selected.fit(train[numeric + ["gu"]], train["target_score"])
    holdout_pred = np.clip(selected.predict(holdout[numeric + ["gu"]]), 25, 100)
    holdout_baseline = holdout[TARGET].to_numpy()
    holdout_ml_metrics = metrics(holdout["target_score"], holdout_pred)
    holdout_base_metrics = metrics(holdout["target_score"], holdout_baseline)
    for key, value in holdout_ml_metrics.items():
        comparison.loc[comparison["model"] == best_name, f"holdout_{key}"] = value
    for key, value in holdout_base_metrics.items():
        comparison.loc[comparison["model"] == "baseline_persistence", f"holdout_{key}"] = value

    holdout_result = pd.DataFrame({
        "horizon": horizon, "source_year": holdout["year"], "target_year": holdout_year,
        "gu": holdout["gu"], "actual": holdout["target_score"],
        "ml_prediction": holdout_pred, "persistence_prediction": holdout_baseline,
    })
    actual_order = holdout_result.sort_values(["actual", "gu"], ascending=[False, True]).copy()
    actual_order["actual_rank"] = np.arange(1, len(actual_order) + 1)
    actual_order["actual_grade"] = GRADE_LABELS[((actual_order["actual_rank"] - 1) // 5).clip(0, 4)]
    pred_order = holdout_result.sort_values(["ml_prediction", "gu"], ascending=[False, True]).copy()
    pred_order["predicted_rank"] = np.arange(1, len(pred_order) + 1)
    pred_order["predicted_grade"] = GRADE_LABELS[((pred_order["predicted_rank"] - 1) // 5).clip(0, 4)]
    holdout_result = holdout_result.merge(
        actual_order[["gu", "actual_rank", "actual_grade"]], on="gu"
    ).merge(pred_order[["gu", "predicted_rank", "predicted_grade"]], on="gu")

    # Refit on every observed target pair for deployment.
    final_model = candidates(numeric)[best_name]
    final_model.fit(supervised[numeric + ["gu"]], supervised["target_score"])
    cv_residuals = pd.DataFrame(residual_rows)
    best_residuals = cv_residuals[cv_residuals["model"] == best_name]
    interval_half_width = float(np.quantile(np.abs(best_residuals["residual"]), 0.90))
    bundle = {
        "pipeline": final_model,
        "model_name": best_name,
        "horizon_years": horizon,
        "target": TARGET,
        "numeric_features": numeric,
        "categorical_features": ["gu"],
        "training_source_years": [int(supervised.year.min()), int(supervised.year.max())],
        "training_target_years": [int(supervised.target_year.min()), int(supervised.target_year.max())],
        "holdout_target_year": holdout_year,
        "holdout_metrics": holdout_ml_metrics,
        "persistence_holdout_metrics": holdout_base_metrics,
        "prediction_interval": "empirical symmetric 90% absolute-CV-residual range",
        "interval_half_width": interval_half_width,
        "score_bounds": [25.0, 100.0],
    }
    return bundle, comparison, holdout_result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, input_path = load_panel()
    features, numeric = engineer_features(panel)
    latest = features[features["year"] == features["year"].max()].copy()
    all_metrics, all_holdouts, forecasts, summary = [], [], [], []

    for horizon in HORIZONS:
        bundle, comparison, holdout = train_horizon(features, numeric, horizon)
        model_path = OUT / f"child_environment_{horizon}y.joblib"
        joblib.dump(bundle, model_path, compress=3)
        pred = np.clip(bundle["pipeline"].predict(latest[numeric + ["gu"]]), 25, 100)
        half = bundle["interval_half_width"]
        forecasts.append(pd.DataFrame({
            "base_year": int(latest["year"].iloc[0]),
            "forecast_year": int(latest["year"].iloc[0]) + horizon,
            "horizon_years": horizon, "gu": latest["gu"],
            "predicted_score_100": pred,
            "lower_90": np.clip(pred - half, 25, 100),
            "upper_90": np.clip(pred + half, 25, 100),
            "model": bundle["model_name"],
        }))
        all_metrics.append(comparison)
        all_holdouts.append(holdout)
        summary.append({k: v for k, v in bundle.items() if k != "pipeline" and k != "numeric_features"})

    forecast = pd.concat(forecasts, ignore_index=True)
    forecast = add_exact_quintile_grade(forecast, "predicted_score_100", "horizon_years")
    forecast.to_csv(OUT / "forecasts_2025_2027.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_metrics, ignore_index=True).to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    holdout_df = pd.concat(all_holdouts, ignore_index=True)
    holdout_df.to_csv(OUT / "holdout_2024_predictions.csv", index=False, encoding="utf-8-sig")
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    grade_summary = []
    grade_confusion = []
    for horizon, group in holdout_df.groupby("horizon"):
        distance = (group["actual_grade"].map(grade_order) - group["predicted_grade"].map(grade_order)).abs()
        grade_summary.append({
            "horizon_years": horizon,
            "exact_grade_accuracy": float((distance == 0).mean()),
            "within_one_grade_accuracy": float((distance <= 1).mean()),
            "mean_absolute_grade_error": float(distance.mean()),
        })
        table = pd.crosstab(group["actual_grade"], group["predicted_grade"])
        for actual in GRADE_LABELS:
            for predicted in GRADE_LABELS:
                grade_confusion.append({
                    "horizon_years": horizon, "actual_grade": actual,
                    "predicted_grade": predicted,
                    "count": int(table.get(predicted, pd.Series()).get(actual, 0)),
                })
    pd.DataFrame(grade_summary).to_csv(OUT / "grade_accuracy_2024.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(grade_confusion).to_csv(OUT / "grade_confusion_2024.csv", index=False, encoding="utf-8-sig")
    (OUT / "model_metadata.json").write_text(json.dumps({
        "input": str(input_path.relative_to(ROOT)),
        "latest_observed_year": int(panel.year.max()),
        "models": summary,
        "limitations": [
            "Only 10 annual observations per gu are available.",
            "Hospital histories are reconstructed from currently operating providers and establishment dates.",
            "The target is a relative Seoul-within-year score; forecasts are not facility demand or causal effects.",
            "The 90% ranges are empirical CV error ranges, not guaranteed probabilistic coverage.",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    lines = [
        "# 아동 물리환경 점수 1년·3년 예측모델", "",
        f"입력: `{input_path.relative_to(ROOT)}`", "",
        "예측 대상은 자치구별 `child_environment_score_100`이다. 연도 순서를 보존한 확대창 검증으로 "
        "후보 모델을 선택하고, 2024년 목표값은 최종 시간외 평가에만 사용했다. 이후 배포 모델은 "
        "관측 가능한 전체 목표쌍으로 다시 학습했다.", "", "## 성능", "",
        "|기간|선택 모델|CV RMSE|2024 holdout RMSE|지속성 기준 RMSE|", "|---:|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        sub = metrics_df[metrics_df.horizon == h]
        selected = sub[(sub.model != "baseline_persistence")].sort_values("cv_rmse").iloc[0]
        baseline = sub[sub.model == "baseline_persistence"].iloc[0]
        lines.append(
            f"|{h}년|{selected.model}|{selected.cv_rmse:.3f}|{selected.holdout_rmse:.3f}|{baseline.holdout_rmse:.3f}|"
        )
    lines += ["", "## 해석 주의", "",
              "이 예측은 상대점수의 통계적 외삽이다. 3년 예측은 관측기간 대비 먼 외삽이고, 병원 과거치는 "
              "현재 생존기관의 설립연도 역산치라 과거 폐업기관이 누락된다. 정책효과나 인과관계로 해석하지 않는다."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote models and {len(forecast)} forecasts to {OUT}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
