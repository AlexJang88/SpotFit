"""Train the apartment sale and jeonse price models used by SpotFit.

The models estimate price per pyeong for a supplied transaction profile. They
are not one-year/three-year market-forecast models: Random Forest cannot safely
extrapolate future market trends. Validation is time based (train through 2024,
hold out 2025), and the deployment model is then refit on all complete years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "recommender" / "v1"
MAX_COMPLETE_YEAR = 2025
MIN_YEAR = 2021

COMMON_FEATURES = [
    "거리_to_GBD(km)",
    "거리_to_CBD(km)",
    "거리_to_YBD(km)",
    "접근성_score_GBD",
    "접근성_score_CBD",
    "접근성_score_YBD",
]
SALE_FEATURES = [
    "계약연도",
    "전용면적(㎡)",
    "전용면적(평)",
    "층",
    "건축년도",
    "건물 나이(년)",
] + COMMON_FEATURES
JEONSE_FEATURES = [
    "계약연도",
    "임대면적(㎡)",
    "임대면적(평)",
    "층",
    "건축년도",
    "건물 나이(년)",
] + COMMON_FEATURES


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def prepare_sale(raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = read_csv(raw_dir / "아파트_매매.csv")
    numeric(df, SALE_FEATURES + ["거래금액(만원)", "평당 금액(만원)"])

    calculated = df["거래금액(만원)"] / df["전용면적(평)"]
    target = df["평당 금액(만원)"].where(df["평당 금액(만원)"] > 0, calculated)
    valid = (
        df["계약연도"].between(MIN_YEAR, MAX_COMPLETE_YEAR)
        & (df["전용면적(평)"] > 0)
        & target.notna()
        & (target > 0)
    )
    return df.loc[valid, SALE_FEATURES].copy(), target.loc[valid].copy()


def prepare_jeonse(raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = read_csv(raw_dir / "아파트_전월세.csv")
    if "전월세구분" not in df.columns:
        raise ValueError("Required column is missing: 전월세구분")
    df = df.loc[df["전월세구분"].astype(str).str.strip().eq("전세")].copy()
    numeric(df, JEONSE_FEATURES + ["보증금(만원)", "평당 보증금(만원)"])

    calculated = df["보증금(만원)"] / df["임대면적(평)"]
    target = df["평당 보증금(만원)"].where(df["평당 보증금(만원)"] > 0, calculated)
    valid = (
        df["계약연도"].between(MIN_YEAR, MAX_COMPLETE_YEAR)
        & (df["임대면적(평)"] > 0)
        & target.notna()
        & (target > 0)
    )
    return df.loc[valid, JEONSE_FEATURES].copy(), target.loc[valid].copy()


def build_model() -> Pipeline:
    regressor = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    error_ratio = np.abs((y_true.to_numpy() - prediction) / y_true.to_numpy())
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "mape_percent": float(np.mean(error_ratio) * 100),
    }


def train_one(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    target_name: str,
    unit: str,
) -> dict[str, Any]:
    year_counts = X["계약연도"].value_counts().sort_index()
    eligible_years = year_counts[year_counts >= 100].index
    if len(eligible_years) < 2:
        raise ValueError(f"{name}: at least two years with 100+ rows are required")
    holdout_year = int(eligible_years.max())
    train_mask = X["계약연도"] < holdout_year
    test_mask = X["계약연도"] == holdout_year
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"{name}: time split cannot be formed for {holdout_year}")

    validation_model = build_model()
    validation_model.fit(X.loc[train_mask], y.loc[train_mask])
    score = metrics(y.loc[test_mask], validation_model.predict(X.loc[test_mask]))

    deployment_model = build_model()
    deployment_model.fit(X, y)
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "model_name": name,
        "task": "transaction_profile_price_estimation",
        "is_future_forecast": False,
        "target": target_name,
        "unit": unit,
        "features": list(X.columns),
        "training_years": sorted(int(year) for year in X["계약연도"].unique()),
        "validation": {
            "method": "time_holdout",
            "train_years": sorted(int(year) for year in X.loc[train_mask, "계약연도"].unique()),
            "holdout_year": holdout_year,
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
            **score,
        },
        "deployment_rows": int(len(X)),
        "algorithm": "RandomForestRegressor with median imputation and log1p target",
        "limitation": (
            "This estimates a transaction profile's price per pyeong. It is not a "
            "one-year or three-year future market forecast and does not extrapolate trends."
        ),
    }
    bundle = {"pipeline": deployment_model, "metadata": metadata}
    joblib.dump(bundle, model_dir / "model_bundle.joblib", compress=3)
    (model_dir / "model_contract.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sale_X, sale_y = prepare_sale(args.raw_dir)
    jeonse_X, jeonse_y = prepare_jeonse(args.raw_dir)
    results = [
        train_one(
            "apart_sale",
            sale_X,
            sale_y,
            args.output_dir,
            "평당 금액(만원)",
            "만원/평",
        ),
        train_one(
            "apart_rent",
            jeonse_X,
            jeonse_y,
            args.output_dir,
            "평당 보증금(만원)",
            "만원/평",
        ),
    ]
    summary = pd.DataFrame(
        [
            {
                "model": item["model_name"],
                **item["validation"],
                "deployment_rows": item["deployment_rows"],
            }
            for item in results
        ]
    )
    summary.to_csv(args.output_dir / "property_model_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
