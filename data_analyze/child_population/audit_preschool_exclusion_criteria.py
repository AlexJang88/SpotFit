"""미취학 만족도 모델을 교육환경 예측점수에서 제외할 객관적 근거를 감사한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from statsmodels.formula.api import mixedlm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training.train_child_population_growth import CORE, load_population  # noqa: E402
from ml.training.train_child_satisfaction_scores import (  # noqa: E402
    build_district_features, candidate_estimators, candidate_features, read_survey,
)

SURVEY_DIR = ROOT / "data/raw/아동/만족도 조사"
GEOJSON = ROOT / "frontend/public/seoul-dong.geojson"
PRACTICAL_MAE_IMPROVEMENT = .05
PRACTICAL_SATISFACTION_DIFFERENCE = .20
PRACTICAL_DISTRICT_SD = PRACTICAL_SATISFACTION_DIFFERENCE / 2
RANDOM_SEED = 42


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def code_to_gu() -> dict[str, str]:
    population, _ = load_population(CORE)
    mapping = population.assign(
        gu_code=population["dong_code"].str[:5].astype(int).astype(str).str[-3:].str.zfill(3)
    )[["gu_code", "gu"]].drop_duplicates()
    return mapping.set_index("gu_code")["gu"].to_dict()


def choose_geo_column(frame: pd.DataFrame) -> str | None:
    preferred = [c for c in frame if any(key in c.replace(" ", "") for key in
                                        ["구코드", "행정구역1", "지역소분류"])]
    for column in preferred:
        values = frame[column].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
        if values.nunique() == 25 and numeric(values).max() >= 100:
            return column
    return None


def choose_weight_column(frame: pd.DataFrame) -> str:
    standardized = [c for c in frame if "표준화" in c and "가중치" in c]
    if standardized:
        return standardized[0]
    originals = [c for c in frame if "가중치" in c]
    if not originals:
        raise ValueError("설문 가중치 컬럼이 없습니다.")
    return originals[0]


def satisfaction_columns(frame: pd.DataFrame) -> list[str]:
    def normalized(value: str) -> str:
        return value.replace(" ", "").replace("_", "").replace("-", "")
    return [c for c in frame if "보육시설" in normalized(c) and "만족도" in normalized(c)]


def load_household_satisfaction() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = code_to_gu()
    rows, audit = [], []
    for path in sorted(SURVEY_DIR.rglob("*.csv")):
        if not path.name[:4].isdigit():
            continue
        year = int(path.name[:4])
        if year in {2010, 2011, 2015}:
            audit.append({"year": year, "status": "excluded", "reason":
                          "unverified ordinal district code" if year < 2012 else "no childcare satisfaction item"})
            continue
        frame = read_survey(path)
        geo_column, sat_columns = choose_geo_column(frame), satisfaction_columns(frame)
        if geo_column is None or not sat_columns:
            audit.append({"year": year, "status": "excluded", "reason": "district or satisfaction item unavailable"})
            continue
        weights = numeric(frame[choose_weight_column(frame)])
        satisfaction_items = frame[sat_columns].apply(numeric)
        satisfaction = satisfaction_items.where(
            satisfaction_items.ge(1) & satisfaction_items.le(5)
        ).mean(axis=1)
        gu_code = frame[geo_column].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        selected = pd.DataFrame({"year": year, "gu_code": gu_code, "gu": gu_code.map(mapping),
                                 "satisfaction": satisfaction, "weight": weights}).dropna()
        selected = selected[selected["weight"].gt(0) & selected["satisfaction"].between(1, 5)]
        rows.append(selected)
        audit.append({"year": year, "status": "included", "reason": "verified 3-digit district code",
                      "households": len(selected), "satisfaction_columns": len(sat_columns)})
    result = pd.concat(rows, ignore_index=True)
    return result, pd.DataFrame(audit)


def weighted_group_summary(households: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, gu), part in households.groupby(["year", "gu"]):
        values, weights = part["satisfaction"].to_numpy(), part["weight"].to_numpy()
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean) ** 2, weights=weights)
        effective_n = weights.sum() ** 2 / np.square(weights).sum()
        rows.append({"year": year, "gu": gu, "target": mean, "survey_n": len(part),
                     "effective_n": effective_n, "within_variance": variance,
                     "measurement_se": np.sqrt(variance / effective_n)})
    return pd.DataFrame(rows)


def bootstrap_district_sd(panel: pd.DataFrame, n_boot: int = 3000) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    residual = panel["target"] - panel.groupby("year")["target"].transform("mean")
    work = panel.assign(residual=residual)
    district_effect = work.groupby("gu")["residual"].mean()
    observed = district_effect.std(ddof=1)
    estimates = []
    for _ in range(n_boot):
        simulated = work["target"].to_numpy() + rng.normal(0, work["measurement_se"].to_numpy())
        sim = work.assign(simulated=simulated)
        sim_residual = sim["simulated"] - sim.groupby("year")["simulated"].transform("mean")
        estimates.append(sim.assign(residual=sim_residual).groupby("gu")["residual"].mean().std(ddof=1))
    low, high = np.quantile(estimates, [.05, .95])
    return float(observed), float(low), float(high)


def district_polygons() -> dict[str, object]:
    data = json.loads(GEOJSON.read_text(encoding="utf-8"))
    grouped: dict[str, list] = {}
    for feature in data["features"]:
        grouped.setdefault(feature["properties"]["sggnm"], []).append(shape(feature["geometry"]))
    return {gu: unary_union(polygons) for gu, polygons in grouped.items()}


def adjacency_matrix(names: list[str]) -> np.ndarray:
    polygons = district_polygons()
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            if polygons[left].boundary.distance(polygons[right].boundary) < 1e-7:
                matrix[i, j] = matrix[j, i] = 1
    return matrix


def moran_test(values: np.ndarray, weights: np.ndarray, permutations: int = 999) -> tuple[float, float]:
    centered = values - values.mean()
    denominator, weight_sum = np.square(centered).sum(), weights.sum()
    if denominator == 0 or weight_sum == 0:
        return np.nan, np.nan
    observed = len(values) / weight_sum * np.sum(weights * np.outer(centered, centered)) / denominator
    rng = np.random.default_rng(RANDOM_SEED)
    permuted = []
    for _ in range(permutations):
        shuffled = rng.permutation(centered)
        permuted.append(len(values) / weight_sum * np.sum(weights * np.outer(shuffled, shuffled)) / denominator)
    p = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1)
    return float(observed), float(p)


def repeated_model_comparison(training: pd.DataFrame, features: list[str], repeats: int = 50) -> pd.DataFrame:
    rows = []
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=5, random_state=RANDOM_SEED)
    for split, (fit_index, valid_index) in enumerate(splitter.split(training, groups=training["gu"]), start=1):
        fit, valid = training.iloc[fit_index], training.iloc[valid_index]
        for name, model in candidate_estimators().items():
            fitted = clone(model).fit(fit[features], fit["target"])
            mae = np.abs(valid["target"].to_numpy() - fitted.predict(valid[features])).mean()
            rows.append({"split": split, "model": name, "mae": mae})
    result = pd.DataFrame(rows)
    baseline = result[result["model"].eq("dummy_mean")].set_index("split")["mae"]
    summaries = []
    for model, part in result.groupby("model"):
        values = part.set_index("split")["mae"]
        improvement = (baseline - values) / baseline
        summaries.append({"model": model, "repeats": len(values), "mae_mean": values.mean(),
                          "median_improvement_vs_dummy": improvement.median(),
                          "improvement_ci_low": improvement.quantile(.025),
                          "improvement_ci_high": improvement.quantile(.975),
                          "wins_vs_dummy": int((values < baseline).sum())})
    return pd.DataFrame(summaries).sort_values("mae_mean")


def ablation_test(training: pd.DataFrame, full: list[str], childcare: list[str], repeats: int = 100) -> pd.DataFrame:
    without = [x for x in full if x not in childcare]
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=5, random_state=RANDOM_SEED)
    rows = []
    ridge = candidate_estimators()["ridge_alpha_1"]
    for split, (fit_index, valid_index) in enumerate(splitter.split(training, groups=training["gu"]), start=1):
        fit, valid = training.iloc[fit_index], training.iloc[valid_index]
        scores = {}
        for variant, columns in (("with_childcare", full), ("without_childcare", without)):
            model = clone(ridge).fit(fit[columns], fit["target"])
            scores[variant] = np.abs(valid["target"].to_numpy() - model.predict(valid[columns])).mean()
        rows.append({"split": split, **scores,
                     "relative_improvement_from_childcare":
                         (scores["without_childcare"] - scores["with_childcare"]) / scores["without_childcare"]})
    frame = pd.DataFrame(rows)
    return pd.DataFrame([{
        "repeats": repeats, "mae_with_childcare": frame["with_childcare"].mean(),
        "mae_without_childcare": frame["without_childcare"].mean(),
        "median_relative_improvement": frame["relative_improvement_from_childcare"].median(),
        "improvement_ci_low": frame["relative_improvement_from_childcare"].quantile(.025),
        "improvement_ci_high": frame["relative_improvement_from_childcare"].quantile(.975),
        "fold_wins_with_childcare": int((frame["with_childcare"] < frame["without_childcare"]).sum()),
    }])


def md_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=np.number):
        printable[column] = printable[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    printable = printable.fillna("").astype(str)
    return "\n".join(["| " + " | ".join(printable.columns) + " |",
                      "| " + " | ".join(["---"] * len(printable.columns)) + " |",
                      *["| " + " | ".join(row) + " |" for row in printable.to_numpy()]])


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env.dev", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preschool_exclusion_audit")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output / stamp
    output.mkdir(parents=True)

    households, source_audit = load_household_satisfaction()
    survey_panel = weighted_group_summary(households)
    survey_panel["year_centered"] = survey_panel["target"] - survey_panel.groupby("year")["target"].transform("mean")

    # 1~2: 지역 분산과 측정오차 비교
    district_sd, sd_ci_low, sd_ci_high = bootstrap_district_sd(survey_panel)
    annual_between_sd = survey_panel.groupby("year")["target"].std()
    median_measurement_se = survey_panel["measurement_se"].median()
    signal_to_noise = annual_between_sd.median() / median_measurement_se

    # 3: 연도 고정효과 + 자치구 랜덤절편 계층모형
    mixed_model = mixedlm("target ~ C(year)", survey_panel, groups=survey_panel["gu"])
    try:
        mixed = mixed_model.fit(reml=True, method="lbfgs")
    except np.linalg.LinAlgError:
        # 랜덤효과 분산이 0 경계에 가까우면 Hessian이 특이해질 수 있다.
        mixed = mixed_model.fit(reml=True, method="powell", maxiter=2000)
    random_variance = float(mixed.cov_re.iloc[0, 0])
    residual_variance = float(mixed.scale)
    icc = random_variance / (random_variance + residual_variance)

    # 4: 연도별 공간 자기상관
    moran_rows = []
    for year, part in survey_panel.groupby("year"):
        part = part.sort_values("gu")
        weights = adjacency_matrix(part["gu"].tolist())
        moran_i, p = moran_test(part["year_centered"].to_numpy(), weights)
        moran_rows.append({"year": year, "moran_i": moran_i, "permutation_p": p})
    moran = pd.DataFrame(moran_rows)

    # 5~6: 2021~2023 예측 반복검증 및 보육 feature 제거
    district_features, feature_audit = build_district_features()
    model_panel = survey_panel.merge(district_features, on=["year", "gu"], how="inner", validate="one_to_one")
    development = model_panel[model_panel["year"].between(2021, 2023)].replace([np.inf, -np.inf], np.nan)
    candidates = [x for x in candidate_features("preschool") if x in development
                  and development[x].notna().mean() >= .7 and development[x].nunique(dropna=True) > 1]
    model_comparison = repeated_model_comparison(development, candidates)
    childcare = ["daycare_per_1000_preschool", "kindergarten_per_1000_preschool",
                 "kindergarten_students_per_class", "children_parks_per_1000_preschool"]
    ablation = ablation_test(development, candidates, childcare)

    # 7: 실무적 동등성. 지역효과 SD의 90% 상한이 0.10점보다 작아야 통과.
    equivalence_pass = sd_ci_high < PRACTICAL_DISTRICT_SD
    best_non_dummy = model_comparison[model_comparison["model"].ne("dummy_mean")].sort_values("mae_mean").iloc[0]
    prediction_pass = not (best_non_dummy["improvement_ci_low"] > 0 and
                           best_non_dummy["median_improvement_vs_dummy"] >= PRACTICAL_MAE_IMPROVEMENT)
    ablation_row = ablation.iloc[0]
    ablation_pass = not (ablation_row["improvement_ci_low"] > 0 and
                         ablation_row["median_relative_improvement"] >= .01)
    spatial_pass = int((moran["permutation_p"] < .05).sum()) == 0
    criteria = pd.DataFrame([
        {"criterion": "1_between_district_variance", "status": "pass" if district_sd < PRACTICAL_DISTRICT_SD else "fail",
         "evidence": f"district SD={district_sd:.4f}; threshold={PRACTICAL_DISTRICT_SD:.2f}"},
        {"criterion": "2_measurement_error_vs_signal", "status": "pass" if signal_to_noise < 2 else "fail",
         "evidence": f"median between-SD / median survey-SE={signal_to_noise:.3f}"},
        {"criterion": "3_hierarchical_district_effect", "status": "pass" if icc < .05 else "fail",
         "evidence": f"random variance={random_variance:.6f}; ICC={icc:.4f}; threshold=.05"},
        {"criterion": "4_spatial_autocorrelation", "status": "pass" if spatial_pass else "fail",
         "evidence": f"significant years={int((moran['permutation_p'] < .05).sum())}/{len(moran)}"},
        {"criterion": "5_repeated_cv_vs_mean", "status": "pass" if prediction_pass else "fail",
         "evidence": f"best={best_non_dummy['model']}; median improvement={best_non_dummy['median_improvement_vs_dummy']:.3%}; 95% CI [{best_non_dummy['improvement_ci_low']:.3%}, {best_non_dummy['improvement_ci_high']:.3%}]"},
        {"criterion": "6_childcare_feature_ablation", "status": "pass" if ablation_pass else "fail",
         "evidence": f"median improvement={ablation_row['median_relative_improvement']:.3%}; 95% CI [{ablation_row['improvement_ci_low']:.3%}, {ablation_row['improvement_ci_high']:.3%}]"},
        {"criterion": "7_practical_equivalence", "status": "pass" if equivalence_pass else "fail",
         "evidence": f"district SD 90% CI [{sd_ci_low:.4f}, {sd_ci_high:.4f}]; upper must be <{PRACTICAL_DISTRICT_SD:.2f}"},
        {"criterion": "8_admin_dong_or_facility_replication", "status": "unavailable",
         "evidence": "survey microdata contains district code only; no admin-dong/facility identifier"},
    ])
    passed = int(criteria["status"].eq("pass").sum())
    failed = int(criteria["status"].eq("fail").sum())
    decision = "exclude_from_predictive_score" if passed >= 5 and failed <= 2 else "evidence_insufficient_for_exclusion"

    households.to_csv(output / "household_childcare_satisfaction_2012_2024.csv", index=False, encoding="utf-8-sig")
    survey_panel.to_csv(output / "district_year_satisfaction_panel.csv", index=False, encoding="utf-8-sig")
    source_audit.to_csv(output / "survey_source_audit.csv", index=False, encoding="utf-8-sig")
    moran.to_csv(output / "spatial_moran_by_year.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(output / "repeated_cv_model_comparison.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output / "childcare_feature_ablation.csv", index=False, encoding="utf-8-sig")
    criteria.to_csv(output / "exclusion_criteria_decision.csv", index=False, encoding="utf-8-sig")
    metadata = {"decision": decision, "passed": passed, "failed": failed,
                "district_sd": district_sd, "district_sd_ci90": [sd_ci_low, sd_ci_high],
                "median_measurement_se": median_measurement_se, "signal_to_noise_ratio": signal_to_noise,
                "random_intercept_variance": random_variance, "residual_variance": residual_variance, "icc": icc,
                "practical_satisfaction_difference": PRACTICAL_SATISFACTION_DIFFERENCE,
                "practical_district_sd": PRACTICAL_DISTRICT_SD}
    (output / "decision.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 미취학 만족도 모델 제외 기준 감사

- 생성 시각: {datetime.now().astimezone().isoformat()}
- 지역 차이 자료: 2012~2014, 2016~2024년 서울 25개 자치구 보육시설 만족도
- 예측 자료: 2021~2023년 자치구×연도 75건
- 사전 실무 기준: 지역 간 의미 있는 만족도 차이 0.20점, 지역효과 SD 기준 0.10점, 예측 MAE 개선 기준 5%
- 최종 판정: **`{decision}`** ({passed}개 통과, {failed}개 실패, 1개 자료 없음)

## 기준별 판정

{md_table(criteria)}

## 1~3. 지역 분산·측정오차·계층모형

- 연도효과 제거 후 자치구 효과 SD: {district_sd:.4f}
- 측정오차를 반영한 SD 90% 구간: [{sd_ci_low:.4f}, {sd_ci_high:.4f}]
- 설문 측정오차 중앙값: {median_measurement_se:.4f}
- 연도별 지역 SD / 측정오차 비율: {signal_to_noise:.3f}
- 자치구 랜덤절편 분산: {random_variance:.6f}
- 잔차 분산: {residual_variance:.6f}
- ICC: {icc:.4f}

## 4. 공간 자기상관

{md_table(moran)}

## 5. 반복 교차검증

{md_table(model_comparison)}

평균 기준모델 대비 개선률의 95% 구간과 5% 실무 기준을 함께 사용했다. 단순히 한 번의 CV에서 MAE가 조금 낮다는 이유로 예측 가능하다고 판정하지 않았다.

## 6. 보육 feature 제거 검정

{md_table(ablation)}

보육 feature는 어린이집 수, 유치원 수, 유치원 학급당 원아 수, 어린이공원 수다. 이들을 모두 제거한 모델과 같은 분할에서 비교했다.

## 7. 동등성 기준

`지역효과 SD의 90% 상한 < 0.10점`일 때만 지역 차이가 실무적으로 무시 가능하다고 사전 정의했다. 일반 유의성 검정의 p>0.05를 차이 없음의 증거로 사용하지 않았다.

## 8. 행정동·시설 단위 재현 여부

보유한 서울서베이 원자료에는 자치구 식별자만 있고 행정동 또는 이용시설 식별자가 없다. 따라서 현재 자료로 재현 검정을 할 수 없다. 이 조건은 통과가 아니라 `unavailable`로 남겼다.

## 해석 원칙

- `exclude_from_predictive_score`는 미취학 생활이 중요하지 않다는 뜻이 아니다.
- 현재 확보한 자치구 자료로 만든 **교육환경 만족도 예측점수에서 미취학 만족도 모델을 제외**해도 된다는 제품 범위 판정이다.
- 미취학 영역은 만족도 예측등급 대신 보육시설 공급·접근성 현황으로 별도 제공할 수 있다.
- 행정동·시설 식별 만족도 자료가 확보되면 8번을 다시 실행해야 한다.
"""
    (output / "report.md").write_text(report, encoding="utf-8")

    tracking_uri = args.tracking_uri or f"sqlite:///{(ROOT / 'artifacts/mlflow_preschool_exclusion.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("preschool-exclusion-audit")
    with mlflow.start_run(run_name=f"preschool_exclusion_{stamp}") as run:
        mlflow.log_params({"years": "2012-2014,2016-2024", "spatial_unit": "Seoul district",
                           "practical_difference_points": PRACTICAL_SATISFACTION_DIFFERENCE,
                           "practical_district_sd": PRACTICAL_DISTRICT_SD,
                           "practical_mae_improvement": PRACTICAL_MAE_IMPROVEMENT,
                           "decision": decision})
        mlflow.log_metrics({"criteria_passed": passed, "criteria_failed": failed, "district_effect_sd": district_sd,
                            "district_sd_ci90_upper": sd_ci_high, "measurement_se_median": median_measurement_se,
                            "signal_to_noise": signal_to_noise, "district_icc": icc,
                            "significant_moran_years": int((moran["permutation_p"] < .05).sum())})
        mlflow.set_tags({"causal_claim": "none", "purpose": "objective exclusion decision"})
        mlflow.log_artifacts(str(output), artifact_path="evidence")
        print(json.dumps({"output": str(output), "run_id": run.info.run_id, **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
