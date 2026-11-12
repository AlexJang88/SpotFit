"""자치구 패널로 유아 인프라의 연관성·시간 방향성과 다음 해 인구증가율을 분석한다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/training"))
from train_child_population_growth import CORE, load_population  # noqa: E402
from analyze_housing_school_causality import (  # noqa: E402
    births_by_district,
    district_age_mobility_panel,
    school_pressure_panel,
)


FEATURES = [
    "infant_population",
    "prior_infant_growth_pct",
    "births_per_1000",
    "net_migration_0_4_per_1000",
    "daycare_capacity_per_1000",
    "daycare_occupancy_rate",
    "public_daycare_capacity_share",
    "kindergarten_count_per_1000",
    "kindergarten_class_size",
    "apartment_households_per_1000",
]


def district_population() -> pd.DataFrame:
    population, _ = load_population(CORE)
    panel = population.groupby(["gu", "year"], as_index=False).agg(
        infant_population=("children_infant", "sum"),
        total_population=("total_population", "sum"),
    )
    panel = panel.sort_values(["gu", "year"])
    panel["infant_growth_pct"] = panel.groupby("gu")["infant_population"].pct_change() * 100
    panel["prior_infant_growth_pct"] = panel["infant_growth_pct"]
    panel["target_next_growth_pct"] = panel.groupby("gu")["infant_growth_pct"].shift(-1)
    return panel


def daycare_panel() -> pd.DataFrame:
    candidates = list((ROOT / "data/raw").glob("*/*/*2021_2025.csv"))
    path = next(path for path in candidates if "보육시설" in path.name)
    raw = pd.read_csv(path, dtype={"admdong_cd": str})
    raw = raw.rename(columns={"stats_yr": "year", "sigungu_nm": "gu"})
    raw["year"] = pd.to_numeric(raw["year"])
    raw["to_popl_cnt"] = pd.to_numeric(raw["to_popl_cnt"], errors="coerce")
    raw["current_popl_cnt"] = pd.to_numeric(raw["current_popl_cnt"], errors="coerce")
    district = raw[raw["gu"].ne("계")].copy()
    total = district[district["chldcare_type_nm"].eq("계")][
        ["gu", "year", "to_popl_cnt", "current_popl_cnt"]
    ].rename(columns={"to_popl_cnt": "daycare_capacity", "current_popl_cnt": "daycare_enrollment"})
    public = district[district["chldcare_type_nm"].eq("국공립")][
        ["gu", "year", "to_popl_cnt"]
    ].rename(columns={"to_popl_cnt": "public_daycare_capacity"})
    result = total.merge(public, on=["gu", "year"], how="left", validate="one_to_one")
    result["daycare_occupancy_rate"] = result["daycare_enrollment"] / result["daycare_capacity"]
    result["public_daycare_capacity_share"] = result["public_daycare_capacity"] / result["daycare_capacity"]
    return result


def kindergarten_panel() -> pd.DataFrame:
    school, _ = school_pressure_panel()
    return school[[
        "gu", "year", "school_count_infant", "school_students_infant",
        "school_classes_infant", "school_class_size_infant",
    ]].rename(columns={
        "school_count_infant": "kindergarten_count",
        "school_students_infant": "kindergarten_students",
        "school_classes_infant": "kindergarten_classes",
        "school_class_size_infant": "kindergarten_class_size",
    })


def housing_panel(population: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "data/interim/child_housing_observed/observed_apartment_panel_2021_2025.csv"
    housing = pd.read_csv(path, dtype={"dong_code": str})
    mapping = load_population(CORE)[0][["dong_code", "gu"]].drop_duplicates()
    mapping["district_code"] = mapping["dong_code"].str[:5]
    mapping = mapping[["district_code", "gu"]].drop_duplicates()
    if mapping.duplicated("district_code").any():
        raise ValueError("자치구 코드가 둘 이상의 자치구명에 대응합니다.")
    housing["district_code"] = housing["dong_code"].str[:5]
    housing = housing.merge(mapping, on="district_code", how="left", validate="many_to_one")
    if housing["gu"].isna().any():
        raise ValueError(f"주택 패널 자치구 미매핑 {housing['gu'].isna().sum()}행")
    result = housing.groupby(["gu", "year"], as_index=False).agg(
        apartment_households=("observed_apartment_households_by_area", "sum"),
    )
    return result


def build_panel() -> pd.DataFrame:
    panel = district_population()
    births, _ = births_by_district()
    mobility, _ = district_age_mobility_panel()
    mobility = mobility[mobility["age_band"].eq("0-4세")][
        ["gu", "year", "inflow", "outflow", "net_migration"]
    ]
    for frame in (births, mobility, daycare_panel(), kindergarten_panel(), housing_panel(panel)):
        panel = panel.merge(frame, on=["gu", "year"], how="left", validate="one_to_one")

    denominator = panel["infant_population"].replace(0, np.nan)
    panel["births_per_1000"] = panel["births"] / denominator * 1000
    panel["net_migration_0_4_per_1000"] = panel["net_migration"] / denominator * 1000
    panel["daycare_capacity_per_1000"] = panel["daycare_capacity"] / denominator * 1000
    panel["kindergarten_count_per_1000"] = panel["kindergarten_count"] / denominator * 1000
    panel["apartment_households_per_1000"] = panel["apartment_households"] / denominator * 1000
    panel = panel.sort_values(["gu", "year"])
    for feature in ["daycare_capacity_per_1000", "daycare_occupancy_rate",
                    "public_daycare_capacity_share", "kindergarten_count_per_1000"]:
        panel[f"delta_{feature}"] = panel.groupby("gu")[feature].diff()
        panel[f"next_delta_{feature}"] = panel.groupby("gu")[f"delta_{feature}"].shift(-1)
    panel["previous_growth_pct"] = panel.groupby("gu")["infant_growth_pct"].shift(0)
    return panel


def association_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = ["infant_growth_pct", "target_next_growth_pct", "net_migration_0_4_per_1000"]
    for feature in FEATURES[2:]:
        for outcome in outcomes:
            if feature == outcome:
                continue
            pair = panel[[feature, outcome]].dropna()
            rho, p = stats.spearmanr(pair[feature], pair[outcome]) if len(pair) >= 20 else (np.nan, np.nan)
            rows.append({"feature": feature, "outcome": outcome, "n": len(pair),
                         "spearman": rho, "p_nominal": p})
    result = pd.DataFrame(rows)
    result["abs_spearman"] = result["spearman"].abs()
    return result.sort_values(["outcome", "abs_spearman"], ascending=[True, False])


def two_way_demean(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame[column] - frame.groupby("gu")[column].transform("mean")
            - frame.groupby("year")[column].transform("mean") + frame[column].mean())


def fixed_effect_slope(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict[str, float]:
    columns = list(dict.fromkeys(["gu", "year", x, y, *controls]))
    work = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    regressors = [x, *[column for column in controls if column not in {x, y}]]
    matrix = np.column_stack([two_way_demean(work, column) for column in regressors])
    yd = two_way_demean(work, y).to_numpy()
    if len(work) < 20 or np.linalg.matrix_rank(matrix) < len(regressors):
        return {"n": len(work), "beta": np.nan, "p_value": np.nan, "r2_within": np.nan}
    coefficients = np.linalg.lstsq(matrix, yd, rcond=None)[0]
    residual = yd - matrix @ coefficients
    # HC1 이분산 강건 공분산. 표본이 작으므로 결과는 탐색적으로만 해석한다.
    bread = np.linalg.pinv(matrix.T @ matrix)
    meat = matrix.T @ (matrix * residual[:, None] ** 2)
    covariance = bread @ meat @ bread * len(work) / max(len(work) - len(regressors), 1)
    se = float(np.sqrt(max(covariance[0, 0], 0)))
    beta = float(coefficients[0])
    p = float(2 * stats.t.sf(abs(beta / se), df=max(len(work) - len(regressors), 1))) if se > 0 else np.nan
    tss = float(np.dot(yd, yd))
    r2 = 1 - float(np.dot(residual, residual)) / tss if tss > 0 else np.nan
    return {"n": len(work), "controls": ",".join(controls), "beta": beta,
            "robust_se": se, "p_value": p, "r2_within": r2}


def bh_adjust(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def directionality_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common_controls = ["infant_growth_pct", "births_per_1000",
                       "net_migration_0_4_per_1000", "apartment_households_per_1000"]
    for feature in ["daycare_capacity_per_1000", "daycare_occupancy_rate",
                    "public_daycare_capacity_share", "kindergarten_count_per_1000"]:
        forward = fixed_effect_slope(panel, f"delta_{feature}", "target_next_growth_pct", common_controls)
        reverse = fixed_effect_slope(
            panel, "infant_growth_pct", f"next_delta_{feature}",
            ["births_per_1000", "net_migration_0_4_per_1000", "apartment_households_per_1000"],
        )
        rows.append({"infrastructure": feature, "direction": "infra_change_t_to_population_growth_t+1", **forward})
        rows.append({"infrastructure": feature, "direction": "population_growth_t_to_infra_change_t+1", **reverse})
    result = pd.DataFrame(rows)
    result["bh_q_value"] = bh_adjust(result["p_value"])
    return result


def models() -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("model", DummyRegressor(strategy="median"))]),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                                                     max_features=.8, random_state=42, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5,
                                                                 max_features=.8, random_state=42, n_jobs=-1))]),
    }


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_pct_point": float(mean_absolute_error(actual, predicted)),
        "rmse_pct_point": float(mean_squared_error(actual, predicted) ** .5),
        "r2": float(r2_score(actual, predicted)),
        "direction_accuracy": float((np.sign(actual.to_numpy()) == np.sign(predicted)).mean()),
    }


def train_models(panel: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = panel.dropna(subset=["target_next_growth_pct"]).copy()
    train, test = work[work["year"].le(2023)], work[work["year"].eq(2024)]
    rows, fitted = [], {}
    district_groups = train["gu"]
    splitter = GroupKFold(n_splits=5)
    for name, model in models().items():
        fold_metrics = []
        for tr, va in splitter.split(train, groups=district_groups):
            model.fit(train.iloc[tr][FEATURES], train.iloc[tr]["target_next_growth_pct"])
            fold_metrics.append(metrics(train.iloc[va]["target_next_growth_pct"], model.predict(train.iloc[va][FEATURES])))
        model.fit(train[FEATURES], train["target_next_growth_pct"])
        holdout_pred = model.predict(test[FEATURES])
        holdout = metrics(test["target_next_growth_pct"], holdout_pred)
        rows.append({
            "model": name,
            "train_rows": len(train),
            "holdout_rows": len(test),
            "district_cv_mae": np.mean([x["mae_pct_point"] for x in fold_metrics]),
            "district_cv_rmse": np.mean([x["rmse_pct_point"] for x in fold_metrics]),
            "district_cv_r2": np.mean([x["r2"] for x in fold_metrics]),
            **{f"holdout_{key}": value for key, value in holdout.items()},
        })
        fitted[name] = (model, holdout_pred)
    result = pd.DataFrame(rows)
    non_dummy = result[result["model"].ne("dummy_median")]
    winner = str(non_dummy.sort_values(["district_cv_rmse", "holdout_rmse_pct_point"]).iloc[0]["model"])
    model, prediction = fitted[winner]
    joblib.dump({"model": model, "features": FEATURES, "target": "target_next_growth_pct",
                 "training_years": [2021, 2022, 2023]}, out / "infant_population_growth_model.joblib")
    pred = test[["gu", "year", "target_next_growth_pct"]].copy()
    pred["prediction_growth_pct"] = prediction
    pred["error_pct_point"] = pred["prediction_growth_pct"] - pred["target_next_growth_pct"]
    baseline = result[result["model"].eq("dummy_median")].iloc[0]
    selected = result[result["model"].eq(winner)].iloc[0]
    decision = {
        "selected_model": winner,
        "accuracy_definition": "R2 and percentage-point error; classification accuracy is not applicable",
        "holdout_year": "features at 2024 predict 2024->2025 growth",
        "holdout_r2": selected["holdout_r2"],
        "holdout_mae_pct_point": selected["holdout_mae_pct_point"],
        "holdout_rmse_pct_point": selected["holdout_rmse_pct_point"],
        "holdout_direction_accuracy": selected["holdout_direction_accuracy"],
        "baseline_holdout_rmse_pct_point": baseline["holdout_rmse_pct_point"],
        "rmse_improvement_vs_baseline": 1 - selected["holdout_rmse_pct_point"] / baseline["holdout_rmse_pct_point"],
        "production_ready": bool(selected["holdout_r2"] > 0 and selected["holdout_rmse_pct_point"] < baseline["holdout_rmse_pct_point"]),
    }
    return result, pred, decision


def markdown_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=np.number):
        printable[column] = printable[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    printable = printable.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in printable.to_numpy().tolist()]
    return "\n".join([header, divider, *body])


def write_report(out: Path, panel: pd.DataFrame, associations: pd.DataFrame,
                 directions: pd.DataFrame, scores: pd.DataFrame, decision: dict) -> None:
    top = associations[associations["outcome"].eq("target_next_growth_pct")].head(8)
    significant = directions[directions["bh_q_value"].lt(.05)]
    lines = [
        "# 유아 인프라 연관·방향성·예측 분석", "",
        f"- 생성시각: {datetime.now().astimezone().isoformat()}",
        f"- 패널: {panel['gu'].nunique()}개 자치구, {int(panel.year.min())}~{int(panel.year.max())}년, {len(panel)}행",
        "- 예측 목표: t년 자료로 t→t+1 유아(0~5세) 인구증가율 예측", "",
        "## 예측 성능", "", markdown_table(scores), "",
        f"선정 모델은 `{decision['selected_model']}`이다. 2024→2025 시간 외 검증에서 "
        f"MAE {decision['holdout_mae_pct_point']:.3f}%p, RMSE {decision['holdout_rmse_pct_point']:.3f}%p, "
        f"R² {decision['holdout_r2']:.3f}, 증감 방향 정확도 {decision['holdout_direction_accuracy']:.1%}다. "
        f"단순 중앙값 기준 대비 RMSE 개선률은 "
        f"{decision['rmse_improvement_vs_baseline']:.1%}다.", "",
        "## 다음 해 증가율과의 단변량 연관", "", markdown_table(top), "",
        "## 시간 방향성 고정효과 결과", "", markdown_table(directions), "",
        f"BH q<0.05 방향성 결과는 {len(significant)}건이다. 짧은 5개년, 정책의 비무작위 배치 때문에 "
        "유의한 결과도 인과효과의 확정 증거가 아니다.", "",
        "## 해석 제한", "",
        "- 25개 자치구와 4개 연간 전환만 있으므로 표본이 작다.",
        "- 어린이집 자료는 자치구 연말 관측이며 행정동별 과거 정원·현원은 아니다.",
        "- 예측 R²는 분류 정확도가 아니며, 음수이면 단순 평균 예측보다 못하다는 뜻이다.",
        "- 상관, 시간적 선행, 예측력은 각각 인과관계와 동일하지 않다.",
        "- 정책 개원일을 이용한 사건연구나 외생적 배정 기준이 있어야 인과 식별이 강해진다.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/infant_infrastructure_analysis")
    args = parser.parse_args()
    out = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    panel = build_panel()
    associations = association_table(panel)
    directions = directionality_table(panel)
    scores, predictions, decision = train_models(panel, out)
    panel.to_csv(out / "district_year_panel.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(out / "association_results.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(out / "directionality_fixed_effects.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(out / "model_scores.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "holdout_2024_to_2025_predictions.csv", index=False, encoding="utf-8-sig")
    (out / "model_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, panel, associations, directions, scores, decision)
    print(out)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
