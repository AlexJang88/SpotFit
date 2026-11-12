"""연령별 다음 해 자치구 순전입률을 분석하고 예측한다."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_analyze/child_population"))
from analyze_housing_school_causality import (  # noqa: E402
    births_by_district,
    district_age_mobility_panel,
    school_pressure_panel,
)
from analyze_infant_infrastructure import daycare_panel  # noqa: E402


AGE_GROUPS = {
    "infant": ("0-4세", 0, 4, "미취학·영유아(0~4세)"),
    "elementary": ("5-9세", 5, 9, "초등 진입기(5~9세)"),
    "middle": ("10-14세", 10, 14, "중등 진입기(10~14세)"),
    "high": ("15-19세", 15, 19, "고등·후기 학령기(15~19세)"),
}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def population_panel() -> pd.DataFrame:
    core = ROOT / "data/raw/아동/아동 데이터/핵심"
    frames = []
    for path in sorted(core.glob("*연령별인구현황19.csv")):
        match = re.search(r"(20\d{2})", path.name)
        if not match:
            continue
        year = int(match.group(1))
        raw = pd.read_csv(path, encoding="cp949", dtype=str)
        extracted = raw["행정구역"].str.extract(r"^(.*?)\s*\((\d{10})\)\s*$")
        raw["region_name"] = extracted[0].str.strip()
        raw["dong_code"] = extracted[1]
        raw = raw[raw["dong_code"].str.startswith("11", na=False)
                  & ~raw["dong_code"].str.endswith("00000", na=False)].copy()
        raw["gu"] = raw["region_name"].str.split().str[-2]
        raw["year"] = year
        raw["total_population"] = numeric(raw[f"{year}년_계_총인구수"])
        keep = ["gu", "year", "total_population"]
        for group, (_, start, end, _) in AGE_GROUPS.items():
            columns = [f"{year}년_계_{age}세" for age in range(start, end + 1)]
            raw[f"population_{group}"] = raw[columns].apply(numeric).sum(axis=1, min_count=len(columns))
            keep.append(f"population_{group}")
        frames.append(raw[keep].groupby(["gu", "year"], as_index=False).sum())
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["gu", "year"]).any():
        raise ValueError("인구 패널에 자치구·연도 중복이 있습니다.")
    return panel


def housing_panel(population: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "data/interim/child_housing_observed/observed_apartment_panel_2021_2025.csv"
    housing = pd.read_csv(path, dtype={"dong_code": str})
    mapping = population[["gu"]].drop_duplicates().copy()
    # 주민등록 원본에서 자치구 코드 매핑을 다시 만든다.
    core_path = sorted((ROOT / "data/raw/아동/아동 데이터/핵심").glob("*연령별인구현황19.csv"))[0]
    raw = pd.read_csv(core_path, encoding="cp949", dtype=str, usecols=["행정구역"])
    extracted = raw["행정구역"].str.extract(r"^(.*?)\s*\((\d{10})\)\s*$")
    names = extracted[0].str.strip().str.split()
    code_map = pd.DataFrame({"district_code": extracted[1].str[:5], "gu": names.str[-2]}).dropna().drop_duplicates()
    code_map = code_map[code_map["gu"].isin(mapping["gu"])]
    housing["district_code"] = housing["dong_code"].str[:5]
    housing = housing.merge(code_map, on="district_code", how="left", validate="many_to_one")
    housing["price_weight"] = numeric(housing["observed_apartment_households_by_price"])
    housing["weighted_price"] = numeric(housing["아파트_평균_시가"]) * housing["price_weight"]
    result = housing.groupby(["gu", "year"], as_index=False).agg(
        apartment_households=("observed_apartment_households_by_area", "sum"),
        weighted_price=("weighted_price", "sum"), price_weight=("price_weight", "sum"),
    )
    result["apartment_mean_price"] = result["weighted_price"].div(result["price_weight"].replace(0, np.nan))
    return result.drop(columns=["weighted_price", "price_weight"])


def build_panel() -> pd.DataFrame:
    population = population_panel()
    mobility, _ = district_age_mobility_panel()
    age_map = {value[0]: key for key, value in AGE_GROUPS.items()}
    mobility["age_group"] = mobility["age_band"].map(age_map)
    mobility["population_column"] = mobility["age_group"].map(lambda group: f"population_{group}")
    population_long = population.melt(
        id_vars=["gu", "year", "total_population"],
        value_vars=[f"population_{group}" for group in AGE_GROUPS],
        var_name="population_column", value_name="age_population",
    )
    mobility = mobility.merge(
        population_long, on=["gu", "year", "population_column"], how="left", validate="one_to_one"
    )
    mobility["net_migration_per_1000"] = mobility["net_migration"].div(
        mobility["age_population"].replace(0, np.nan)
    ) * 1000
    panel = mobility.copy()
    panel = panel.merge(housing_panel(population), on=["gu", "year"], how="left", validate="many_to_one")
    panel["apartment_households_per_1000"] = panel["apartment_households"].div(
        panel["total_population"].replace(0, np.nan)
    ) * 1000

    school, _ = school_pressure_panel()
    panel = panel.merge(school, on=["gu", "year"], how="left", validate="many_to_one")
    for group in AGE_GROUPS:
        panel[f"school_count_{group}_per_1000"] = panel[f"school_count_{group}"].div(
            panel["age_population"].replace(0, np.nan)
        ) * 1000
        panel[f"school_students_{group}_per_1000"] = panel[f"school_students_{group}"].div(
            panel["age_population"].replace(0, np.nan)
        ) * 1000

    births, _ = births_by_district()
    panel = panel.merge(births, on=["gu", "year"], how="left", validate="many_to_one")
    panel["births_per_1000"] = panel["births"].div(panel["age_population"].replace(0, np.nan)) * 1000
    daycare = daycare_panel()
    panel = panel.merge(daycare, on=["gu", "year"], how="left", validate="many_to_one")
    panel["daycare_capacity_per_1000"] = panel["daycare_capacity"].div(
        panel["age_population"].replace(0, np.nan)
    ) * 1000

    secondary_path = ROOT / "data/interim/secondary_education/secondary_education_panel_2021_2025.csv"
    secondary = pd.read_csv(secondary_path)
    secondary_keep = [
        "gu", "year", "four_year_college_progression_rate",
        "special_autonomous_high_school_count",
        "special_autonomous_supply_per_100_middle_graduates",
    ]
    panel = panel.merge(secondary[secondary_keep], on=["gu", "year"], how="left", validate="many_to_one")
    panel = panel.sort_values(["age_group", "gu", "year"])
    panel["target_next_net_migration_per_1000"] = panel.groupby(
        ["age_group", "gu"]
    )["net_migration_per_1000"].shift(-1)
    panel["target_year"] = panel["year"] + 1
    return panel


def features_for(group: str) -> list[str]:
    common = [
        "net_migration_per_1000", "age_population", "total_population",
        "apartment_households_per_1000", "apartment_mean_price",
    ]
    specific = {
        "infant": ["births_per_1000", "daycare_capacity_per_1000", "daycare_occupancy_rate",
                   "public_daycare_capacity_share", "school_count_infant_per_1000",
                   "school_class_size_infant"],
        "elementary": ["school_count_elementary_per_1000", "school_students_elementary_per_1000",
                       "school_class_size_elementary", "school_count_infant_per_1000"],
        "middle": ["school_count_middle_per_1000", "school_students_middle_per_1000",
                   "school_class_size_middle", "special_autonomous_high_school_count",
                   "special_autonomous_supply_per_100_middle_graduates"],
        "high": ["school_count_high_per_1000", "school_students_high_per_1000",
                 "school_class_size_high", "four_year_college_progression_rate"],
    }
    return [*common, *specific[group]]


def candidates() -> dict[str, Pipeline]:
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
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                             ("model", HistGradientBoostingRegressor(
                                                 max_iter=200, max_leaf_nodes=10, min_samples_leaf=10,
                                                 l2_regularization=2.0, random_state=42))]),
    }


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_per_1000": float(mean_absolute_error(actual, predicted)),
        "rmse_per_1000": float(mean_squared_error(actual, predicted) ** .5),
        "r2": float(r2_score(actual, predicted)),
        "direction_accuracy": float((np.sign(actual.to_numpy()) == np.sign(predicted)).mean()),
    }


def md_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=np.number):
        printable[column] = printable[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    printable = printable.fillna("").astype(str)
    return "\n".join([
        "| " + " | ".join(printable.columns) + " |",
        "| " + " | ".join(["---"] * len(printable.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in printable.to_numpy()],
    ])


def main() -> None:
    out = ROOT / "artifacts/child_net_migration" / datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = out / "models"
    models_dir.mkdir(parents=True)
    panel = build_panel()
    panel.to_csv(out / "district_age_migration_panel.csv", index=False, encoding="utf-8-sig")
    score_rows, association_rows, predictions, forecasts, ablation_rows, selections = [], [], [], [], [], {}

    for group, (age_band, _, _, label) in AGE_GROUPS.items():
        features = features_for(group)
        work = panel[panel["age_group"].eq(group)].replace([np.inf, -np.inf], np.nan).copy()
        supervised = work.dropna(subset=["target_next_net_migration_per_1000"])
        train, holdout = supervised[supervised.year.le(2023)], supervised[supervised.year.eq(2024)]
        for feature in features:
            pair = train[[feature, "target_next_net_migration_per_1000"]].dropna()
            rho, p = stats.spearmanr(pair[feature], pair["target_next_net_migration_per_1000"])
            association_rows.append({"age_group": group, "age_band": age_band, "feature": feature,
                                     "n": len(pair), "spearman": rho, "p_nominal": p})

        fitted, fold_rmse = {}, {}
        for name, model in candidates().items():
            fold_scores = []
            for validation_year in sorted(train.year.unique()):
                fold_train = train[train.year.ne(validation_year)]
                fold_valid = train[train.year.eq(validation_year)]
                model.fit(fold_train[features], fold_train["target_next_net_migration_per_1000"])
                fold_scores.append(metrics(
                    fold_valid["target_next_net_migration_per_1000"], model.predict(fold_valid[features])
                ))
            model.fit(train[features], train["target_next_net_migration_per_1000"])
            pred = model.predict(holdout[features])
            holdout_score = metrics(holdout["target_next_net_migration_per_1000"], pred)
            cv_rmse = float(np.mean([score["rmse_per_1000"] for score in fold_scores]))
            fold_rmse[name] = [score["rmse_per_1000"] for score in fold_scores]
            score_rows.append({"age_group": group, "model": name, "feature_count": len(features),
                               "train_rows": len(train), "holdout_rows": len(holdout),
                               "cv_rmse_per_1000": cv_rmse,
                               "cv_rmse_std": float(np.std(fold_rmse[name], ddof=1)), **holdout_score})
            fitted[name] = model

        group_scores = pd.DataFrame([row for row in score_rows if row["age_group"] == group])
        winner = str(group_scores.sort_values(["cv_rmse_per_1000", "cv_rmse_std"]).iloc[0].model)
        baseline_cv = float(group_scores[group_scores.model.eq("dummy_median")].iloc[0].cv_rmse_per_1000)
        winner_row = group_scores[group_scores.model.eq(winner)].iloc[0]
        improvement = 1 - float(winner_row.cv_rmse_per_1000) / baseline_cv
        wins = int(np.sum(np.array(fold_rmse[winner]) < np.array(fold_rmse["dummy_median"])))
        selections[group] = {"label": label, "age_band": age_band, "target": "next_year_net_migration_per_1000",
                             "model": winner, "features": features, "cv_improvement_vs_dummy": improvement,
                             "fold_wins": wins,
                             "decision": "exploratory_candidate" if improvement >= .05 and wins == 3 else "selection_deferred"}
        winner_model = fitted[winner]
        holdout_pred = winner_model.predict(holdout[features])
        prediction = holdout[["gu", "year", "target_year", "age_band", "target_next_net_migration_per_1000"]].copy()
        prediction["model"] = winner
        prediction["prediction_next_net_migration_per_1000"] = holdout_pred
        predictions.append(prediction)

        without_prior = [feature for feature in features if feature != "net_migration_per_1000"]
        ablation_model = candidates()[winner]
        ablation_model.fit(train[without_prior], train["target_next_net_migration_per_1000"])
        ablation_score = metrics(
            holdout["target_next_net_migration_per_1000"], ablation_model.predict(holdout[without_prior])
        )
        ablation_rows.append({"age_group": group, "model": winner,
                              "variant": "without_prior_net_migration", **ablation_score})

        # 검증이 끝난 동일 알고리즘을 2021~2024 전환 전체로 재학습해 2025→2026을 예측한다.
        final_model = candidates()[winner]
        final_model.fit(supervised[features], supervised["target_next_net_migration_per_1000"])
        joblib.dump({"model": final_model, "features": features, "target": "target_next_net_migration_per_1000",
                     "age_band": age_band}, models_dir / f"net_migration_{group}.joblib")
        latest = work[work.year.eq(2025)].copy()
        latest_forecast = latest[["gu", "age_band"]].copy()
        latest_forecast["feature_year"] = 2025
        latest_forecast["forecast_year"] = 2026
        latest_forecast["model"] = winner
        latest_forecast["forecast_net_migration_per_1000"] = final_model.predict(latest[features])
        forecasts.append(latest_forecast)

    scores = pd.DataFrame(score_rows)
    associations = pd.DataFrame(association_rows)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    forecasts_frame = pd.concat(forecasts, ignore_index=True)
    ablation = pd.DataFrame(ablation_rows)
    scores.to_csv(out / "model_scores.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(out / "associations.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(out / "holdout_2024_to_2025_predictions.csv", index=False, encoding="utf-8-sig")
    forecasts_frame.to_csv(out / "forecast_2026.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(out / "ablation_without_prior_migration.csv", index=False, encoding="utf-8-sig")
    (out / "selected_models.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

    winner_rows = scores[scores.apply(lambda row: row.model == selections[row.age_group]["model"], axis=1)][
        ["age_group", "model", "cv_rmse_per_1000", "mae_per_1000", "rmse_per_1000", "r2", "direction_accuracy"]
    ]
    sections = []
    for group, (_, _, _, label) in AGE_GROUPS.items():
        selected = selections[group]
        related = associations[associations.age_group.eq(group)].sort_values(
            "spearman", key=lambda series: series.abs(), ascending=False
        )
        ablation_group = ablation[ablation.age_group.eq(group)]
        sections.append(f"""## {label}

- Target: 다음 해 `{selected['age_band']}` 순전입률(해당 연령 인구 1,000명당)
- 선택 모델: `{selected['model']}`
- 판정: `{selected['decision']}`
- 사용 feature: `{', '.join(selected['features'])}`

### 연관관계

{md_table(related)}

### 검증 성능

{md_table(winner_rows[winner_rows.age_group.eq(group)])}

### 직전 순전입률 제거 민감도

{md_table(ablation_group)}
""")
    report = f"""# 연령별 순전입률 연관분석 및 다음 해 예측

- 생성시각: {datetime.now().astimezone().isoformat()}
- 분석 단위: 서울 25개 자치구 × 연도
- 기간: 2021~2025년
- 학습·교차검증: 2021~2023년 feature → 각각 다음 해 순전입률
- 최종 검증: 2024년 feature → 2025년 순전입률
- 미래 예측: 검증 후 2021~2024 전환으로 재학습, 2025년 feature → 2026년 순전입률

## Target 정의

`순전입률 = (연령별 총전입 - 총전출) / 해당 연령 주민등록인구 × 1,000`

양수는 순유입, 음수는 순유출이다. 인구증가율과 달리 연령 코호트가 다음 구간으로 넘어가는 효과를 정답에서 분리해 실제 지역 간 이동에 더 가까운 값을 측정한다.

## 검증 원칙

- 자치구 순전입률을 행정동에 복제하지 않았다.
- feature는 연령대별 분석 가설에 따라 사전 지정했고 최종 검증 정답으로 선택하지 않았다.
- Dummy median, Ridge, Random Forest, Extra Trees, HistGradientBoosting을 동일한 시간 폴드로 비교했다.
- CV RMSE가 가장 작은 알고리즘을 선택한 후 2024→2025를 한 번 평가했다.

{''.join(sections)}
## 해석 제한

- 독립적인 연간 전환은 네 번뿐이며 개발기간은 세 번이므로 결과 변동성이 크다.
- 15~19세에는 고등학생뿐 아니라 대학 진학·취업·입대 연령이 포함된다.
- 학교 소재 자치구와 실제 통학권·학군은 일치하지 않을 수 있다.
- 연관성과 예측력은 인프라의 인과효과가 아니다.
- 높은 예측력에는 직전 연도 순전입률의 지속성이 포함된다. 제거 민감도 결과를 함께 봐야 인프라만의 설명력을 과대평가하지 않는다.
- 2026 예측에는 아직 실제 정답이 없으며 계획·운영에 사용하기 전 추가 연도로 검증해야 한다.
- 현재 생존시설만으로 역산한 학원 수는 미래정보·생존편향 때문에 주 모델에서 제외했다.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(winner_rows.to_string(index=False))
    print(f"결과: {out.resolve()}")


if __name__ == "__main__":
    main()
