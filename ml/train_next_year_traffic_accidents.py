"""Train and evaluate a district-level next-year traffic accident model.

The feature row for year t only uses information available in year t or earlier.
The target is the accident count in year t+1 for the same Seoul district.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SAFETY_DIR = ROOT / "data" / "processed" / "안전"
OUTPUT_DIR = ROOT / "artifacts" / "traffic_accident_next_year"
RANDOM_STATE = 42

KEYS = ["연도", "자치구"]
TARGET = "다음연도_사고건수"
TARGET_YEAR = "예측대상연도"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SAFETY_DIR / name, encoding="utf-8-sig")


def build_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    accidents = pd.read_excel(
        SAFETY_DIR / "자치구별_교통사고_1차.xlsx", sheet_name="자치구별"
    ).sort_values(KEYS)
    drivers = read_csv("자치구별_운전자안전_자동차등록대수_2006_2025.csv")
    nightlife = read_csv("자치구별_유흥단란주점_1차.csv")
    noise = read_csv("자치구별_소음진동_1차.csv")

    nightlife_column = "유흥주점및단란주점_연말기준_영업중업소수"
    noise_columns = [
        "소음진동_전체민원건수",
        "교통소음진동_민원건수",
    ]

    frame = accidents[
        KEYS + ["사고건수", "사망자수", "부상자수"]
    ].copy()
    frame = frame.merge(drivers, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(
        nightlife[KEYS + [nightlife_column]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        noise[KEYS + noise_columns], on=KEYS, how="left", validate="one_to_one"
    )
    frame = frame.sort_values(KEYS).reset_index(drop=True)

    grouped = frame.groupby("자치구", sort=False)
    frame["사고건수_1년전"] = grouped["사고건수"].shift(1)
    frame["사고건수_최근3년평균"] = grouped["사고건수"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    frame["사고건수_전년증감"] = frame["사고건수"] - frame["사고건수_1년전"]
    frame["자동차등록대수_전년증감률"] = grouped["자동차등록대수"].pct_change(
        fill_method=None
    )
    frame[TARGET] = grouped["사고건수"].shift(-1)
    frame[TARGET_YEAR] = frame["연도"] + 1

    numeric_features = [
        "연도",
        "사고건수",
        "사고건수_1년전",
        "사고건수_최근3년평균",
        "사고건수_전년증감",
        "사망자수",
        "부상자수",
        "운전자안전벨트착용률",
        "음주운전율",
        "자동차등록대수",
        "자동차등록대수_전년증감률",
        nightlife_column,
        *noise_columns,
    ]
    categorical_features = ["자치구"]
    return frame, numeric_features, categorical_features


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str], scale: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "district",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(
    numeric_features: list[str], categorical_features: list[str]
) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_features, categorical_features, True)),
                ("model", Ridge(alpha=100.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_features, categorical_features, False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_features, categorical_features, False)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        max_features=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def walk_forward_evaluate(
    labeled: pd.DataFrame,
    features: list[str],
    models: dict[str, Pipeline],
    test_target_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []

    for target_year in test_target_years:
        train = labeled[labeled[TARGET_YEAR] < target_year]
        test = labeled[labeled[TARGET_YEAR] == target_year]
        if train.empty or test.empty:
            continue

        fold_predictions: dict[str, np.ndarray] = {
            "persistence": test["사고건수"].to_numpy(dtype=float)
        }
        for name, model in models.items():
            model.fit(train[features], train[TARGET])
            fold_predictions[name] = np.clip(model.predict(test[features]), 0, None)

        for name, predicted in fold_predictions.items():
            result = test[["자치구", "연도", TARGET_YEAR, TARGET]].copy()
            result["모델"] = name
            result["예측_사고건수"] = predicted
            result["오차"] = result["예측_사고건수"] - result[TARGET]
            predictions.append(result)
            fold_metrics.append(
                {
                    "모델": name,
                    "평가연도": target_year,
                    "학습행수": len(train),
                    "평가행수": len(test),
                    **metrics(test[TARGET], predicted),
                }
            )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_metric_frame = pd.DataFrame(fold_metrics)
    return prediction_frame, fold_metric_frame


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in predictions.groupby("모델"):
        rows.append(
            {
                "모델": name,
                "평가행수": len(group),
                **metrics(group[TARGET], group["예측_사고건수"].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def save_report(
    summary: pd.DataFrame,
    selected_model_name: str,
    learned_model_name: str,
    importance: pd.DataFrame,
    train_rows: int,
) -> None:
    lines = [
        "# 다음 연도 자치구별 교통사고 건수 예측",
        "",
        "## 설계",
        "",
        "- 예측 단위: 서울 25개 자치구",
        "- 목표값: 같은 자치구의 다음 연도 교통사고 건수",
        "- 검증: 미래 정보가 학습에 들어가지 않도록 2021~2025년 순차 검증",
        f"- 최종 학습 행 수: {train_rows}",
        f"- 운영 선택 모델: `{selected_model_name}` (순차 검증 MAE 기준)",
        f"- 학습형 도전 모델: `{learned_model_name}` (학습형 모델 중 MAE 1위)",
        "- 2026년 값은 2025년까지의 자료로 만든 예측이며 실제값이 아님",
        "",
        "## 모델 비교",
        "",
        markdown_table(summary, decimals=3),
        "",
        "## 학습형 도전 모델 변수 중요도",
        "",
        markdown_table(importance.head(12), decimals=4),
        "",
        "## 해석 시 주의사항",
        "",
        "- 사고건수·부상자수 등 과거 사고 이력이 강한 예측 신호이지만 원인을 의미하지는 않습니다.",
        "- 행정민원은 2021년에 집계 체계가 바뀌어 제외했습니다.",
        "- 구급·구조는 2022년 이후만 있고 안전센터는 최신 연도가 없어 최종 피처에서 제외했습니다.",
        "- CCTV 파일은 현재 시점 시설 목록으로 보이며 과거 폐쇄 시설의 생존 편향 가능성이 있어 제외했습니다.",
    ]
    (OUTPUT_DIR / "MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, numeric_features, categorical_features = build_dataset()
    features = numeric_features + categorical_features
    labeled = frame[frame[TARGET].notna()].copy()

    models = candidate_models(numeric_features, categorical_features)
    test_target_years = [2021, 2022, 2023, 2024, 2025]
    predictions, fold_metrics = walk_forward_evaluate(
        labeled, features, models, test_target_years
    )
    summary = pooled_metrics(predictions)

    selected_model_name = str(summary.iloc[0]["모델"])
    learned_summary = summary[summary["모델"] != "persistence"]
    best_learned_model_name = str(learned_summary.iloc[0]["모델"])
    final_model = models[best_learned_model_name]

    # Importance is measured on the latest genuinely held-out year (2025).
    importance_train = labeled[labeled[TARGET_YEAR] < 2025]
    importance_test = labeled[labeled[TARGET_YEAR] == 2025]
    importance_model = candidate_models(numeric_features, categorical_features)[best_learned_model_name]
    importance_model.fit(importance_train[features], importance_train[TARGET])
    importance_result = permutation_importance(
        importance_model,
        importance_test[features],
        importance_test[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "변수": features,
            "MAE_증가량_평균": importance_result.importances_mean,
            "MAE_증가량_표준편차": importance_result.importances_std,
        }
    ).sort_values("MAE_증가량_평균", ascending=False)

    final_model.fit(labeled[features], labeled[TARGET])
    joblib.dump(final_model, OUTPUT_DIR / "learned_challenger_model.joblib")

    prediction_2026_input = frame[frame["연도"] == 2025].copy()
    if selected_model_name == "persistence":
        predicted_2026 = prediction_2026_input["사고건수"].to_numpy(dtype=float)
    else:
        predicted_2026 = np.clip(final_model.predict(prediction_2026_input[features]), 0, None)
    selected_cv = predictions[predictions["모델"] == selected_model_name]
    absolute_errors = selected_cv["오차"].abs().to_numpy()
    q80, q95 = np.quantile(absolute_errors, [0.80, 0.95])
    prediction_2026 = prediction_2026_input[["자치구", "연도", "사고건수"]].copy()
    prediction_2026 = prediction_2026.rename(
        columns={"연도": "입력기준연도", "사고건수": "2025_사고건수"}
    )
    prediction_2026["예측대상연도"] = 2026
    prediction_2026["예측_사고건수"] = np.rint(predicted_2026).astype(int)
    prediction_2026["80%_경험적하한"] = np.maximum(
        0, np.rint(predicted_2026 - q80)
    ).astype(int)
    prediction_2026["80%_경험적상한"] = np.rint(predicted_2026 + q80).astype(int)
    prediction_2026["95%_경험적하한"] = np.maximum(
        0, np.rint(predicted_2026 - q95)
    ).astype(int)
    prediction_2026["95%_경험적상한"] = np.rint(predicted_2026 + q95).astype(int)
    prediction_2026 = prediction_2026.sort_values("예측_사고건수", ascending=False)

    frame.to_csv(OUTPUT_DIR / "modeling_dataset.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        OUTPUT_DIR / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / "metrics_by_year.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    prediction_2026.to_csv(
        OUTPUT_DIR / "predictions_2026.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "target": TARGET,
        "target_definition": "district accident count in feature year + 1",
        "selected_operational_model": selected_model_name,
        "selected_model_rule": "prediction equals current-year accident count" if selected_model_name == "persistence" else "learned estimator",
        "best_learned_challenger": best_learned_model_name,
        "features": features,
        "training_rows": len(labeled),
        "training_feature_year_min": int(labeled["연도"].min()),
        "training_feature_year_max": int(labeled["연도"].max()),
        "validation_target_years": test_target_years,
        "prediction_interval_method": "absolute walk-forward residual quantiles",
        "absolute_error_q80": float(q80),
        "absolute_error_q95": float(q95),
        "random_state": RANDOM_STATE,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_report(
        summary,
        selected_model_name,
        best_learned_model_name,
        importance,
        len(labeled),
    )

    print(summary.to_string(index=False))
    print(f"\nSelected operational model: {selected_model_name}")
    print(f"Best learned challenger: {best_learned_model_name}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
