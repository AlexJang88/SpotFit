"""Export reproducible 1y/2y child crime-exposure forecasts for the API."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "models" / "recommender" / "v1" / "child_crime_exposure"
TWO_YEAR_EXPERIMENT = ROOT / "models" / "experiments" / "child_crime_exposure_2y_2027_v1"
TWO_YEAR_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_2y_2027_v1"
THREE_YEAR_EXPERIMENT = ROOT / "models" / "experiments" / "child_crime_exposure_integrated_v2"
THREE_YEAR_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_integrated_v2"
PANEL = ROOT / "artifacts" / "child_crime_borough_v2" / "panel.csv"


def ridge_numeric_formula(bundle: dict) -> pd.DataFrame:
    features = bundle["features"]
    coefficients = bundle["model"].named_steps["model"].coef_[:len(features)]
    weights = np.abs(coefficients)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("2년 모델의 Feature 가중치를 계산할 수 없습니다.")
    return pd.DataFrame({
        "feature": features,
        "direction": np.where(coefficients >= 0, "higher_risk", "lower_risk"),
        "signed_effect": coefficients,
        "weight_pct": weights / weights.sum() * 100,
        "weight_method": "abs standardized Ridge coefficient / numeric coefficient sum",
    }).sort_values("weight_pct", ascending=False)


def integrated_formula(bundle: dict) -> pd.DataFrame:
    weights = bundle["baseline_bundle"]["integrated_weights"].copy()
    weights = weights.loc[weights["score_eligible"].astype(bool) & weights["score_weight_pct"].gt(0)].copy()
    result = weights[["feature", "direction", "mean_signed_y_change", "score_weight_pct"]].rename(
        columns={"mean_signed_y_change": "signed_effect", "score_weight_pct": "weight_pct"})
    result["weight_pct"] = result["weight_pct"] / result["weight_pct"].sum() * 100
    result["weight_method"] = "integrated prediction sensitivity / eligible sensitivity sum"
    return result.sort_values("weight_pct", ascending=False)


def add_weighted_feature_score(frame: pd.DataFrame, prediction: str, formula: pd.DataFrame,
                               horizon: int, target: Path) -> pd.DataFrame:
    """Build the requested Feature weight × borough quintile-point score."""
    if not PANEL.is_file():
        raise FileNotFoundError(f"점수 입력 패널이 없습니다: {PANEL}")
    panel = pd.read_csv(PANEL, encoding="utf-8-sig")
    base_year = int(frame["base_year"].max())
    latest = panel.loc[panel["year"].eq(base_year)].sort_values("gu").copy()
    if len(latest) != 25 or latest.gu.nunique() != 25:
        raise ValueError(f"{base_year}년 점수 입력이 서울 25개 자치구 구조가 아닙니다.")
    if not np.isclose(formula["weight_pct"].sum(), 100, atol=1e-8):
        raise ValueError("Feature 가중치 합계가 100%가 아닙니다.")
    missing = sorted(set(formula.feature) - set(latest.columns))
    if missing:
        raise ValueError(f"점수 Feature 입력이 없습니다: {missing}")

    contributions = []
    district_scores = pd.DataFrame({"district": latest.gu.to_numpy()})
    for item in formula.itertuples(index=False):
        values = pd.to_numeric(latest[item.feature], errors="coerce")
        if values.isna().any() or values.nunique() <= 1:
            raise ValueError(f"점수 Feature를 25개 구에서 순위화할 수 없습니다: {item.feature}")
        # Safety direction: a risk-increasing Feature is safer when its value is lower.
        favorable_rank = values.rank(
            method="first", ascending=item.direction == "higher_risk").astype(int)
        points = 100 - ((favorable_rank - 1) // 5) * 20
        weighted = points * float(item.weight_pct) / 100
        district_scores[item.feature] = weighted.to_numpy()
        for district, value, rank, point, contribution in zip(
                latest.gu, values, favorable_rank, points, weighted):
            contributions.append({
                "district": district, "base_year": base_year, "prediction_year": base_year + horizon,
                "feature": item.feature, "direction": item.direction,
                "raw_value": float(value), "favorable_rank": int(rank),
                "quintile_points": int(point), "weight_pct": float(item.weight_pct),
                "weighted_contribution": float(contribution),
            })
    district_scores["safety_score"] = district_scores.drop(columns="district").sum(axis=1)
    district_scores = district_scores[["district", "safety_score"]]
    result = frame.merge(district_scores, on="district", validate="one_to_one")
    result = result.sort_values(["safety_score", "district"], ascending=[False, True])
    result["safety_rank"] = np.arange(1, len(result) + 1)
    result["safety_grade"] = result["safety_rank"].map(lambda rank: "ABCDE"[(rank - 1) // 5])
    result["score_feature_count"] = len(formula)
    result["score_method"] = "sum(feature quintile safety points * normalized feature weight)"
    formula.to_csv(target / "feature_score_formula.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contributions).to_csv(
        target / "feature_score_breakdown.csv", index=False, encoding="utf-8-sig")
    return result


def add_service_columns(frame: pd.DataFrame, prediction: str, formula: pd.DataFrame,
                        horizon: int, target: Path) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_exposure_rate_per_10k"] = np.expm1(result[prediction])
    result["risk_rank"] = result[prediction].rank(ascending=False, method="min").astype(int)
    return add_weighted_feature_score(result, prediction, formula, horizon, target).sort_values("safety_rank")


def write_scoring_metadata(path: Path, formula: pd.DataFrame) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["service_score"] = {
        "method": "Feature별 유리한 방향 순위의 5분위 점수 × 정규화 가중치 합",
        "feature_count": len(formula),
        "weight_sum_pct": float(formula.weight_pct.sum()),
        "score_range": [20, 100],
        "grade_rule": "최종 가중합 점수 내림차순 5개 구씩 A/B/C/D/E",
        "formula_file": "feature_score_formula.csv",
        "breakdown_file": "feature_score_breakdown.csv",
        "target_prediction_is_separate": True,
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def rescore_existing(horizon: int) -> None:
    """Rebuild only score artifacts from an already deployed model and forecast."""
    target = SERVICE / f"{horizon}y"
    model_path = target / f"child_crime_exposure_{horizon}y.joblib"
    forecast_path = target / "service_forecast.csv"
    metadata_path = target / "model_metadata.json"
    missing = [path for path in (model_path, forecast_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"재점수화에 필요한 배포 산출물이 없습니다: {missing}")
    bundle = joblib.load(model_path)
    formula = ridge_numeric_formula(bundle) if horizon in (1, 2) else integrated_formula(bundle)
    existing = pd.read_csv(forecast_path, encoding="utf-8-sig")
    required = ["district", "base_year", "prediction_year", "predicted_y"]
    if not set(required).issubset(existing.columns):
        raise ValueError(f"배포 예측 CSV에 필요한 열이 없습니다: {required}")
    frame = existing[required].copy()
    add_service_columns(frame, "predicted_y", formula, horizon, target).to_csv(
        forecast_path, index=False, encoding="utf-8-sig")
    write_scoring_metadata(metadata_path, formula)


def export_one_year() -> None:
    target = SERVICE / "1y"
    bundle_path = target / "child_crime_exposure_1y.joblib"
    bundle = joblib.load(bundle_path)
    formula = ridge_numeric_formula(bundle)
    raw_path = target / "raw_forecast.csv"
    if not raw_path.is_file():
        raise FileNotFoundError("1년 서비스 예측 산출물이 없습니다. train_service_crime_exposure_1y.py를 먼저 실행하세요.")
    frame = pd.read_csv(raw_path, encoding="utf-8-sig")
    add_service_columns(frame, "predicted_y", formula, 1, target).to_csv(
        target / "service_forecast.csv", index=False, encoding="utf-8-sig")
    metadata = target / "model_metadata.json"
    if metadata.is_file():
        write_scoring_metadata(metadata, formula)


def export_two_year() -> None:
    target = SERVICE / "2y"
    target.mkdir(parents=True, exist_ok=True)
    source_model = TWO_YEAR_EXPERIMENT / "operational_exposure_2y_2027_strict.joblib"
    source_forecast = TWO_YEAR_ARTIFACT / "forecast_2027.csv"
    source_metadata = TWO_YEAR_ARTIFACT / "result.json"
    if not source_model.is_file() or not source_forecast.is_file() or not source_metadata.is_file():
        raise FileNotFoundError("2년 범죄노출 모델 실험 산출물이 없습니다. optimize_exposure_2y_2027.py를 먼저 실행하세요.")
    shutil.copy2(source_model, target / "child_crime_exposure_2y.joblib")
    shutil.copy2(source_metadata, target / "model_metadata.json")
    bundle = joblib.load(source_model)
    formula = ridge_numeric_formula(bundle)
    raw = pd.read_csv(source_forecast, encoding="utf-8-sig")
    frame = raw.rename(columns={"gu": "district", "year": "base_year", "predicted_y_2027": "predicted_y"})
    frame = frame[["district", "base_year", "target_year", "predicted_y"]].rename(columns={"target_year": "prediction_year"})
    add_service_columns(frame, "predicted_y", formula, 2, target).to_csv(
        target / "service_forecast.csv", index=False, encoding="utf-8-sig")
    write_scoring_metadata(target / "model_metadata.json", formula)


def export_three_year() -> None:
    target = SERVICE / "3y"
    target.mkdir(parents=True, exist_ok=True)
    source_model = THREE_YEAR_EXPERIMENT / "operational_exposure_integrated_blend_3y.joblib"
    source_forecast = THREE_YEAR_ARTIFACT / "forecast_2028_blended.csv"
    source_metadata = THREE_YEAR_ARTIFACT / "result.json"
    if not all(path.is_file() for path in (source_model, source_forecast, source_metadata)):
        raise FileNotFoundError("3년 범죄노출 앙상블 산출물이 없습니다.")
    shutil.copy2(source_model, target / "child_crime_exposure_3y.joblib")
    shutil.copy2(source_metadata, target / "model_metadata.json")
    bundle = joblib.load(source_model)
    formula = integrated_formula(bundle)
    raw = pd.read_csv(source_forecast, encoding="utf-8-sig")
    frame = raw.rename(columns={"gu": "district", "blended_y_2028": "predicted_y"})
    frame["base_year"], frame["prediction_year"] = 2025, 2028
    frame = frame[["district", "base_year", "prediction_year", "predicted_y"]]
    add_service_columns(frame, "predicted_y", formula, 3, target).to_csv(
        target / "service_forecast.csv", index=False, encoding="utf-8-sig")
    write_scoring_metadata(target / "model_metadata.json", formula)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore-only", action="store_true")
    args = parser.parse_args()
    if args.rescore_only:
        for horizon in (2, 3):
            rescore_existing(horizon)
        print(json.dumps({"rescored_horizons": [2, 3]}, ensure_ascii=False))
        return
    export_one_year()
    export_two_year()
    export_three_year()
    manifest = {
        "service": "child_regional_crime_exposure",
        "target_formula": "log(1 + 5대범죄 발생건수 / 0~17세 아동인구 * 10,000)",
        "horizons": [1, 2, 3],
        "score_rule": "Feature별 5분위 안전점수 × 관련도 가중치 합; 가중치 합 100%; 최종 점수 순위로 A~E",
    }
    (SERVICE / "service_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
