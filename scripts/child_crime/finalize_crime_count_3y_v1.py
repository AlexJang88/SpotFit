"""Finalize the selected crime-count candidate and produce review artifacts."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.training import compare_child_crime_borough_v2 as old
from ml.training import optimize_borough_crime_count_3y as exp


def count_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(actual_log)
    predicted = np.expm1(predicted_log)
    return {
        "log_RMSE": float(mean_squared_error(actual_log, predicted_log) ** .5),
        "crime_RMSE": float(mean_squared_error(actual, predicted) ** .5),
        "crime_MAE": float(mean_absolute_error(actual, predicted)),
        "crime_MAPE_pct": float(np.mean(np.abs(predicted - actual) / actual) * 100),
        "crime_R2": float(r2_score(actual, predicted)),
        "Spearman": float(spearmanr(actual, predicted).statistic),
    }


def previous_component_benchmark() -> dict[str, object]:
    panel = pd.read_csv(old.OUT / "panel.csv")
    groups = json.loads((old.OUT / "feature_groups.json").read_text(encoding="utf-8"))
    columns = list(dict.fromkeys(old.CORE + [column for values in groups.values() for column in values]))
    gus = sorted(panel.gu.unique())
    pairs = old.make_pairs(panel, 3)
    row = pd.read_csv(old.OUT / "3y" / "best_by_method_cv.csv").query("method == 'components'").iloc[0]
    features = ast.literal_eval(row.features)
    indices = [columns.index(column) for column in features]

    actual_cv: list[float] = []
    predicted_cv: list[float] = []
    folds = []
    holdout_metrics = None
    for target_year in (*exp.CV_YEARS, exp.HOLDOUT_YEAR):
        test = pairs[pairs.target_year == target_year]
        train = pairs[pairs.target_year <= target_year - exp.HORIZON]
        model = old.fit(old.context(train, test, columns, gus), indices, "components", float(row.alpha))
        _, detail = old.predict(model, test, 3)
        actual = test.log_crime_future.to_numpy()
        predicted = np.log1p(detail["crime"])
        metrics = count_metrics(actual, predicted)
        if target_year in exp.CV_YEARS:
            actual_cv.extend(actual)
            predicted_cv.extend(predicted)
            folds.append({"target_year": target_year, **metrics})
        else:
            holdout_metrics = metrics
    return {
        "description": "previous Y component model's crime-count submodel",
        "groups": ast.literal_eval(row.groups),
        "features": features,
        "alpha": float(row.alpha),
        "cv": count_metrics(np.asarray(actual_cv), np.asarray(predicted_cv)),
        "folds": folds,
        "holdout": holdout_metrics,
    }


def main() -> int:
    report_path = exp.OUT / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    winner = report["winner_cv"]
    previous = previous_component_benchmark()
    report["previous_component_benchmark"] = previous
    report["comparison_note"] = (
        "The selected candidate beats the previous component submodel on 2022-2024 pooled CV, "
        "but the previous submodel is better on the untouched 2025 year. Keep both experimental."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    panel = pd.read_csv(exp.BASE / "panel.csv")
    groups = json.loads((exp.BASE / "feature_groups.json").read_text(encoding="utf-8"))
    panel, crime_pattern, type_counts = exp.add_crime_history(panel)
    groups = {"범죄유형·검거·추세": crime_pattern, **groups}
    pairs = exp.make_pairs(panel, type_counts)
    latest = panel[(panel.year == 2025) & panel.log_crime.notna()].copy()
    latest["target_year"] = 2028
    prediction, operational = exp.fit_predict(
        pairs, latest, winner["features"], winner["spec"], winner["mode"], type_counts
    )
    exp.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": operational,
        "features": winner["features"],
        "spec": winner["spec"],
        "mode": winner["mode"],
        "type_counts": type_counts,
        "training_target_year_max": int(pairs.target_year.max()),
        "forecast_horizon": 3,
        "metadata": report,
    }
    joblib.dump(artifact, exp.MODEL_OUT / "operational_crime_count_3y.joblib")
    future = latest[["gu", "year", "target_year", "crime"]].copy()
    future["predicted_crime"] = np.maximum(0, np.expm1(prediction))
    future.to_csv(exp.OUT / "forecast_2028.csv", index=False, encoding="utf-8-sig")

    best_method = pd.read_csv(exp.OUT / "best_by_method.csv")
    comparisons = pd.read_csv(exp.OUT / "all_model_comparisons.csv")
    history = pd.read_csv(exp.OUT / "improvement_history.csv")
    holdout = pd.read_csv(exp.OUT / "holdout_2025_predictions.csv")
    fold_frame = pd.DataFrame(winner["folds"])
    selected_features = pd.DataFrame(
        [{"group": group, "feature": feature} for group in winner["groups"] for feature in groups[group]]
        + [{"group": "공통", "feature": "log_crime"}]
    ).drop_duplicates("feature")
    summary = pd.DataFrame([
        {"model": "Persistence", "period": "CV 2022-2024", **report["baseline_cv"]},
        {"model": "기존 Y 분리모델의 범죄 하위모델", "period": "CV 2022-2024", **previous["cv"]},
        {"model": "신규 범죄유형별 예측·합산모델", "period": "CV 2022-2024", **winner},
        {"model": "Persistence", "period": "Holdout 2025", **report["baseline_holdout"]},
        {"model": "기존 Y 분리모델의 범죄 하위모델", "period": "Holdout 2025", **previous["holdout"]},
        {"model": "신규 범죄유형별 예측·합산모델", "period": "Holdout 2025", **report["winner_holdout"]},
    ])
    with pd.ExcelWriter(exp.OUT / "범죄건수_3년예측_모델비교.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="최종비교", index=False)
        fold_frame.to_excel(writer, sheet_name="신규모델_연도별CV", index=False)
        best_method.to_excel(writer, sheet_name="기법별최상위", index=False)
        history.to_excel(writer, sheet_name="성능개선이력", index=False)
        selected_features.to_excel(writer, sheet_name="선택Feature", index=False)
        holdout.to_excel(writer, sheet_name="2025실제예측", index=False)
        future.to_excel(writer, sheet_name="2028예측", index=False)
        comparisons.to_excel(writer, sheet_name="전체13032개설정", index=False)

    cv_old = previous["cv"]
    hold_old = previous["holdout"]
    text = f"""# 3년 뒤 5대 범죄건수 모델 최적화 결과

## 결론

- 선택 모델: 5대 범죄 유형별 `log(1+건수)`를 Ridge(alpha=10)로 각각 예측한 뒤 건수를 합산
- 선택 Feature 그룹: 유흥시설, 가구구조, 산업조직구조 + 현재 총 범죄건수
- 전체 비교: 10개 그룹의 1,024개 조합, 총 {report['search']['total_configurations']:,}개 설정
- 선택 구간: 2022~2024년, 최종 확인: 2025년 25개 자치구

## 성능

| 구간 | 모델 | log RMSE | 건수 RMSE | 건수 MAE | MAPE |
|---|---|---:|---:|---:|---:|
| CV | Persistence | {report['baseline_cv']['log_RMSE']:.4f} | {report['baseline_cv']['crime_RMSE']:.1f} | {report['baseline_cv']['crime_MAE']:.1f} | {report['baseline_cv']['crime_MAPE_pct']:.2f}% |
| CV | 기존 분리모델의 범죄 하위모델 | {cv_old['log_RMSE']:.4f} | {cv_old['crime_RMSE']:.1f} | {cv_old['crime_MAE']:.1f} | {cv_old['crime_MAPE_pct']:.2f}% |
| CV | 신규 유형별 합산모델 | {winner['log_RMSE']:.4f} | {winner['crime_RMSE']:.1f} | {winner['crime_MAE']:.1f} | {winner['crime_MAPE_pct']:.2f}% |
| 2025 | Persistence | {report['baseline_holdout']['log_RMSE']:.4f} | {report['baseline_holdout']['crime_RMSE']:.1f} | {report['baseline_holdout']['crime_MAE']:.1f} | {report['baseline_holdout']['crime_MAPE_pct']:.2f}% |
| 2025 | 기존 분리모델의 범죄 하위모델 | {hold_old['log_RMSE']:.4f} | {hold_old['crime_RMSE']:.1f} | {hold_old['crime_MAE']:.1f} | {hold_old['crime_MAPE_pct']:.2f}% |
| 2025 | 신규 유형별 합산모델 | {report['winner_holdout']['log_RMSE']:.4f} | {report['winner_holdout']['crime_RMSE']:.1f} | {report['winner_holdout']['crime_MAE']:.1f} | {report['winner_holdout']['crime_MAPE_pct']:.2f}% |

신규 모델은 CV에서 Persistence보다 {report['cv_improvement_pct']:.2f}%, 기존 범죄 하위모델보다 {(cv_old['log_RMSE']-winner['log_RMSE'])/cv_old['log_RMSE']*100:.2f}% 개선됐다. 2025년에는 Persistence보다 {report['holdout_improvement_pct']:.2f}% 개선됐지만 기존 범죄 하위모델보다 log RMSE가 {(report['winner_holdout']['log_RMSE']-hold_old['log_RMSE'])/hold_old['log_RMSE']*100:.2f}% 나빴다. 따라서 신규 모델은 통합 영향력 실험의 후보로 사용하되 기존 모델을 즉시 폐기하지 않는다.

## 누수 통제

예측 기준연도가 t이면 학습 정답은 `target_year <= t`인 자료만 사용했다. 2025년은 Feature·기법 선택이 끝난 뒤 한 번만 평가했다. 외부 Feature는 기존 패널과 동일하게 1년 지연값을 사용했다.
"""
    (exp.OUT / "experiment_report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"operational_model": str(exp.MODEL_OUT / 'operational_crime_count_3y.joblib'), "report": str(exp.OUT / 'experiment_report.md')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
