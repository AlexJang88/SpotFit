"""행정동별 보육·교육·주택 인프라로 다음 해 연령별 순전입률을 분석·예측한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training.train_child_population_growth import CORE, GROUPS, load_population, merge_features  # noqa: E402

MIGRATION = ROOT / "data/processed/child_population/admin_dong_age_migration/admin_dong_age_migration_annual_2021_2025.csv"

SCHOOL_LEVEL = {"infant": "유치원", "elementary": "초등학교", "middle": "중학교", "high": "고등학교"}


def group_features(group: str, columns: set[str]) -> tuple[list[str], list[str]]:
    """공통 혼란요인과 연령대에 맞는 인프라 후보를 분리한다."""
    common_candidates = [
        "current_net_migration_per_1000",
        "current_age_population",
        "total_population",
        "observed_apartment_households_by_area",
        "observed_apartment_households_by_price",
        "apartment_households_per_1000",
        "아파트_단지_수",
        "아파트_평균_면적",
        "아파트_평균_시가",
        "총_상주인구_수",
        "총_가구_수",
        "아파트_가구_수",
        "총_유동인구_수",
        "총_직장_인구_수",
        "교육_지출_총금액",
        "지하철_역_수",
        "버스_정거장_수",
    ]
    level = SCHOOL_LEVEL[group]
    infrastructure = [
        f"school_count_{level}",
        f"student_count_{level}",
        f"teacher_count_{level}",
        f"class_count_{level}",
        f"students_per_class_{level}",
    ]
    if group == "infant":
        infrastructure += [
            "daycare_count_reconstructed", "daycare_opened_observed_event", "daycare_closed_observed_event",
            "safety_zone_kindergarten", "safety_zone_daycare", "children_park_count_reconstructed",
        ]
    elif group == "elementary":
        infrastructure += ["safety_zone_elementary", "academy_survivor_count"]
    elif group == "middle":
        infrastructure += [
            "special_autonomous_high_school_count", "special_autonomous_supply_per_100_middle_graduates",
            "academy_survivor_count", "exam_academy_survivor_count", "language_academy_survivor_count",
        ]
    else:
        infrastructure += [
            "four_year_college_progression_rate", "academy_survivor_count",
            "exam_academy_survivor_count", "language_academy_survivor_count",
        ]
    return ([x for x in common_candidates if x in columns], [x for x in infrastructure if x in columns])


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    population, population_audit = load_population(CORE)
    features, feature_audit = merge_features(population)
    migration = pd.read_csv(MIGRATION, dtype={"dong_code": str})
    # 집계 산출물의 preschool(0~5세)을 기존 학습 파이프라인의 infant 키와 통일한다.
    migration["age_group"] = migration["age_group"].replace({"preschool": "infant"})
    migration = migration.rename(columns={
        "age_population_year_end": "current_age_population",
        "net_migration_per_1000_year_end_population": "current_net_migration_per_1000",
    })
    keep = [
        "year", "dong_code", "age_group", "age_band", "inflow", "outflow", "net_migration",
        "current_age_population", "current_net_migration_per_1000",
    ]
    panel = migration[keep].merge(features, on=["year", "dong_code"], how="left", validate="many_to_one")
    panel["apartment_households_per_1000"] = panel["observed_apartment_households_by_area"].div(
        panel["total_population"].replace(0, np.nan)
    ) * 1000
    panel = panel.sort_values(["age_group", "dong_code", "year"])
    next_rate = panel.groupby(["age_group", "dong_code"])["current_net_migration_per_1000"].shift(-1)
    next_year = panel.groupby(["age_group", "dong_code"])["year"].shift(-1)
    panel["target_next_net_migration_per_1000"] = next_rate.where(next_year.eq(panel["year"] + 1))
    panel["target_year"] = panel["year"] + 1
    audit = pd.concat([population_audit, feature_audit], ignore_index=True, sort=False)
    return panel, audit


def models() -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))]),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=8, max_features=.8, random_state=42, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=8, max_features=.8, random_state=42, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=12, min_samples_leaf=20, l2_regularization=2.0, random_state=42))]),
    }


def scores(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_per_1000": float(mean_absolute_error(actual, predicted)),
        "rmse_per_1000": float(mean_squared_error(actual, predicted) ** .5),
        "r2": float(r2_score(actual, predicted)),
        "direction_accuracy": float((np.sign(actual.to_numpy()) == np.sign(predicted)).mean()),
    }


def residualized_association(data: pd.DataFrame, feature: str, target: str, controls: list[str]) -> dict[str, float]:
    use = data[["dong_code", "year", feature, target, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 30 or use[feature].nunique() < 2:
        return {"n": len(use), "adjusted_beta_std": np.nan, "adjusted_spearman": np.nan,
                "p_nominal": np.nan, "twfe_beta_std": np.nan, "twfe_spearman": np.nan,
                "twfe_p_nominal": np.nan}
    x_controls = np.column_stack([np.ones(len(use)), *[stats.zscore(use[c].to_numpy()) for c in controls]])
    x = stats.zscore(use[feature].to_numpy())
    y = stats.zscore(use[target].to_numpy())
    x_resid = x - x_controls @ np.linalg.lstsq(x_controls, x, rcond=None)[0]
    y_resid = y - x_controls @ np.linalg.lstsq(x_controls, y, rcond=None)[0]
    beta = float(np.dot(x_resid, y_resid) / np.dot(x_resid, x_resid)) if np.dot(x_resid, x_resid) else np.nan
    rho, p = stats.spearmanr(x_resid, y_resid)

    # 행정동의 시간 불변 특성과 서울 전체 연도 충격을 제거한 within 변환.
    # 불균형 패널에도 수렴하도록 행정동·연도 평균 제거를 반복한다.
    within_columns = [feature, target, *controls]
    within = use[within_columns].astype(float).copy()
    for _ in range(12):
        within -= within.groupby(use["dong_code"]).transform("mean")
        within -= within.groupby(use["year"]).transform("mean")
    wx = within[feature].to_numpy()
    wy = within[target].to_numpy()
    wc = np.column_stack([np.ones(len(within)), *[within[c].to_numpy() for c in controls]])
    wx_resid = wx - wc @ np.linalg.lstsq(wc, wx, rcond=None)[0]
    wy_resid = wy - wc @ np.linalg.lstsq(wc, wy, rcond=None)[0]
    wx_sd, wy_sd = wx_resid.std(), wy_resid.std()
    if wx_sd == 0 or wy_sd == 0:
        twfe_beta, twfe_rho, twfe_p = np.nan, np.nan, np.nan
    else:
        wx_std, wy_std = wx_resid / wx_sd, wy_resid / wy_sd
        twfe_beta = float(np.dot(wx_std, wy_std) / np.dot(wx_std, wx_std))
        twfe_rho, twfe_p = stats.spearmanr(wx_resid, wy_resid)
    return {"n": len(use), "adjusted_beta_std": beta, "adjusted_spearman": float(rho),
            "p_nominal": float(p), "twfe_beta_std": twfe_beta,
            "twfe_spearman": float(twfe_rho), "twfe_p_nominal": float(twfe_p)}


def bh_adjust(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1)
    result.loc[valid.index] = adjusted
    return result


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
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env.dev", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/admin_dong_child_net_migration")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="admin-dong-child-net-migration")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output / stamp
    model_dir = output / "models"
    model_dir.mkdir(parents=True)
    panel, audit = build_panel()
    panel.to_csv(output / "training_panel.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "feature_source_audit.csv", index=False, encoding="utf-8-sig")

    associations, comparisons, holdout_predictions, forecasts, selections = [], [], [], [], {}
    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'artifacts/mlflow_admin_dong_child_net_migration.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"admin_dong_migration_{stamp}") as parent:
        parent_id = parent.info.run_id
        mlflow.log_params({
            "target": "next_year_net_migration_per_1000_year_end_age_population",
            "development_feature_years": "2021-2023", "holdout_feature_year": 2024,
            "forecast_feature_year": 2025, "spatial_unit": "Seoul admin dong",
        })
        mlflow.set_tags({"validation": "leave-one-year-out then 2024-to-2025 holdout", "causal_claim": "none"})

        for group, (_, _, label) in GROUPS.items():
            target = "target_next_net_migration_per_1000"
            work = panel[panel["age_group"].eq(group)].replace([np.inf, -np.inf], np.nan).copy()
            supervised = work.dropna(subset=[target]).query("current_age_population >= 30").copy()
            train = supervised[supervised["year"].le(2023)].copy()
            holdout = supervised[supervised["year"].eq(2024)].copy()
            common, infra = group_features(group, set(panel.columns))
            usable_infra = [f for f in infra if train[f].notna().mean() >= .6 and train[f].nunique(dropna=True) > 1]
            housing_candidates = [x for x in [
                "observed_apartment_households_by_area", "observed_apartment_households_by_price",
                "apartment_households_per_1000", "아파트_단지_수", "아파트_평균_면적", "아파트_평균_시가",
            ] if x in train and train[x].notna().mean() >= .6 and train[x].nunique(dropna=True) > 1]
            for feature in [*usable_infra, *housing_candidates]:
                raw = train[[feature, target]].dropna()
                rho, p = stats.spearmanr(raw[feature], raw[target]) if len(raw) >= 30 else (np.nan, np.nan)
                controls = [x for x in [
                    "current_net_migration_per_1000", "current_age_population", "total_population",
                    "apartment_households_per_1000",
                ] if x in train and x != feature]
                adjusted = residualized_association(train, feature, target, controls)
                associations.append({"age_group": group, "age_label": label, "feature": feature,
                                     "category": "infrastructure" if feature in usable_infra else "housing",
                                     "raw_spearman": rho, "raw_p_nominal": p, **adjusted})
            ranked = pd.DataFrame([x for x in associations if x["age_group"] == group and x["category"] == "infrastructure"]).sort_values(
                "adjusted_spearman", key=lambda s: s.abs(), ascending=False
            )
            selected_infra = ranked["feature"].head(8).tolist()
            base_features = [x for x in common if train[x].notna().mean() >= .6 and train[x].nunique(dropna=True) > 1]
            features = list(dict.fromkeys([*base_features, *selected_infra]))
            if not features or holdout.empty:
                raise ValueError(f"{group}: 학습 feature 또는 홀드아웃 자료가 없습니다.")

            fold_years = sorted(train["year"].unique())
            fitted = {}
            baseline_rmse = None
            for name, estimator in models().items():
                fold_metrics = []
                for validation_year in fold_years:
                    fit = train[train["year"].ne(validation_year)]
                    valid = train[train["year"].eq(validation_year)]
                    model = clone(estimator).fit(fit[features], fit[target])
                    fold_metrics.append(scores(valid[target], model.predict(valid[features])))
                rmses = np.array([x["rmse_per_1000"] for x in fold_metrics])
                if name == "dummy_median":
                    baseline_rmse = rmses.copy()
                final = clone(estimator).fit(train[features], train[target])
                test_score = scores(holdout[target], final.predict(holdout[features]))
                row = {"age_group": group, "model": name, "feature_count": len(features),
                       "cv_rmse_mean": float(rmses.mean()), "cv_rmse_std": float(rmses.std(ddof=1)),
                       "relative_improvement_vs_dummy": float(1 - rmses.mean() / baseline_rmse.mean()),
                       "fold_wins_vs_dummy": int((rmses < baseline_rmse).sum()),
                       **{f"holdout_{k}": v for k, v in test_score.items()}}
                comparisons.append(row)
                fitted[name] = final
                with mlflow.start_run(run_name=f"{group}_{name}", nested=True):
                    mlflow.log_params({"age_group": group, "model": name, "features": json.dumps(features, ensure_ascii=False)})
                    mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float)) and k not in {"feature_count"}})

            table = pd.DataFrame([x for x in comparisons if x["age_group"] == group])
            winner = str(table.sort_values(["cv_rmse_mean", "cv_rmse_std"]).iloc[0]["model"])
            winner_row = table[table["model"].eq(winner)].iloc[0]
            decision = "exploratory_candidate" if winner_row["relative_improvement_vs_dummy"] >= .05 and winner_row["fold_wins_vs_dummy"] == len(fold_years) else "selection_deferred"
            selections[group] = {"age_label": label, "target": target, "model": winner, "features": features,
                                 "decision": decision, "train_rows": len(train), "holdout_rows": len(holdout)}
            winner_model = fitted[winner]
            joblib.dump({"model": winner_model, "features": features, "target": target, "age_group": group}, model_dir / f"admin_dong_net_migration_{group}.joblib")

            pred = holdout[["dong_code", "gu", "dong", "year", "target_year", target]].copy()
            pred["age_group"] = group
            pred["prediction_next_net_migration_per_1000"] = winner_model.predict(holdout[features])
            holdout_predictions.append(pred)

            all_fit = supervised[supervised["year"].le(2024)]
            forecast_model = clone(models()[winner]).fit(all_fit[features], all_fit[target])
            latest = work[work["year"].eq(2025) & work["current_age_population"].ge(30)].dropna(subset=features, how="all")
            future = latest[["dong_code", "gu", "dong", "year", "age_band"]].copy()
            future["forecast_year"] = 2026
            future["age_group"] = group
            future["forecast_net_migration_per_1000"] = forecast_model.predict(latest[features])
            forecasts.append(future)

        association_df = pd.DataFrame(associations)
        association_df["pooled_bh_q"] = bh_adjust(association_df["p_nominal"])
        association_df["twfe_bh_q"] = bh_adjust(association_df["twfe_p_nominal"])
        comparison_df = pd.DataFrame(comparisons).sort_values(["age_group", "cv_rmse_mean"])
        prediction_df = pd.concat(holdout_predictions, ignore_index=True)
        forecast_df = pd.concat(forecasts, ignore_index=True)
        association_df.to_csv(output / "infrastructure_associations.csv", index=False, encoding="utf-8-sig")
        comparison_df.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
        prediction_df.to_csv(output / "holdout_predictions_2025.csv", index=False, encoding="utf-8-sig")
        forecast_df.to_csv(output / "forecast_2026.csv", index=False, encoding="utf-8-sig")
        (output / "selected_models.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

        winners = comparison_df[comparison_df.apply(lambda row: row["model"] == selections[row["age_group"]]["model"], axis=1)]
        top_associations = association_df.sort_values("twfe_spearman", key=lambda s: s.abs(), ascending=False).groupby(["age_group", "category"]).head(5)
        feature_sections = []
        for group, selected in selections.items():
            feature_sections.append(f"""### {selected['age_label']}

- Target: 다음 해 해당 연령대 순전입률(연말 해당 연령 인구 1,000명당)
- 선택 모델: `{selected['model']}`
- 판정: `{selected['decision']}`
- Feature: {', '.join(f'`{x}`' for x in selected['features'])}
""")
        report = f"""# 행정동별 보육·교육·주택 인프라와 연령별 순전입률

- 생성 시각: {datetime.now().astimezone().isoformat()}
- 공간 단위: 서울 행정동
- 연령대: 0~5세, 6~11세, 12~14세, 15~17세
- Target: `t+1년 순전입자 / t+1년 연말 해당 연령 인구 × 1,000`
- 개발기간: 2021~2023년 feature → 2022~2024년 target
- 최종 검증: 2024년 feature → 2025년 target
- 미래 예측: 2025년 feature → 2026년 순전입률

## 분석 설계

인프라가 결과보다 먼저 관측되도록 1년 시차를 적용했다. 연관분석은 직전 순전입률, 해당 연령 인구,
전체 인구, 아파트 가구 비율을 통제한다. `adjusted_*`는 지역 간 수준 차이도 포함한 통제 결과이고,
`twfe_*`는 행정동 고정효과와 연도 고정효과를 추가해 같은 동 내부의 시간 변화를 사용한 결과다.
BH q값은 전체 인프라 검정의 다중비교를 보정한다. 그래도 무작위 실험이나 자연실험이 아니므로
인과효과의 확정 증거가 아니다.

## 사용 Feature와 Target

{''.join(feature_sections)}

## 통제 후 인프라 연관 상위 결과

{md_table(top_associations)}

`twfe_bh_q`는 다중비교를 보정하지만 공간 자기상관과 행정동별 군집 오차까지 해결하지 않는다.
따라서 q<0.05만으로 인과관계를 선언하지 않는다.

## 모델 성능

{md_table(winners)}

- `MAE/RMSE` 단위는 해당 연령 인구 1,000명당 순전입자 수다.
- `R²`가 0 이하면 홀드아웃 평균 예측보다 못한 것이다.
- `direction_accuracy`는 순유입/순유출 방향을 맞힌 비율이며 회귀모델의 일반적인 정확도와 다르다.
- `selection_deferred` 모델은 운영·의사결정에 사용하지 않는다.

## 인과 해석 제한

- 주택공급, 보육·교육시설은 수요가 예상되는 지역에 먼저 배치될 수 있어 역인과가 남는다.
- 행정동 경계와 실제 통학권·생활권이 일치하지 않는다.
- 독립적인 연간 전환은 네 번뿐이며, 시설 과거자료 일부는 현재 생존시설과 사건일로 복원한 값이다.
- 주택의 영향을 통제한 뒤 교육·보육 feature가 남더라도 ‘주택은 영향이 없다’는 증명이 되지 않는다.
- 강한 인과 주장은 입주·개원·학교 신설 전후의 사건연구와 유사 대조지역 설계가 추가로 필요하다.
"""
        (output / "report.md").write_text(report, encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="evidence")
        for path in model_dir.iterdir():
            mlflow.log_artifact(str(path), artifact_path="models")

    print(winners.to_string(index=False))
    print(f"MLflow parent run: {parent_id}")
    print(f"결과: {output.resolve()}")


if __name__ == "__main__":
    main()
