"""다음 연도 범죄안전 대체점수를 직접 예측하고 기존 표준모델과 비교한다."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "ml" / "training" / "train_crime_safety_1y.py"
spec = importlib.util.spec_from_file_location("component_model", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)

OUT = ROOT / "artifacts" / "crime_safety_forecast" / "direct_score_1y_2018_2023"


def make_pairs(data: pd.DataFrame) -> pd.DataFrame:
    pairs = base.make_pairs(data)
    future = data[["연도", "자치구", "범죄안전_대체점수"]].copy()
    future["기준연도"] = future["연도"] - 1
    future = future.drop(columns="연도").rename(columns={"범죄안전_대체점수": "target_score"})
    return pairs.merge(future, on=["기준연도", "자치구"], how="inner", validate="one_to_one")


def grade(frame: pd.DataFrame, score_column: str, prefix: str) -> None:
    frame[f"{prefix}_순위"] = frame[score_column].rank(ascending=False, method="first").astype(int)
    frame[f"{prefix}_등급"] = ((frame[f"{prefix}_순위"] - 1) // 5 + 1).astype(int)
    frame[f"{prefix}_등급문자"] = frame[f"{prefix}_등급"].map(dict(enumerate("ABCDE", 1)))


def metrics(frame: pd.DataFrame, prediction: str, prefix: str) -> dict[str, float]:
    actual, pred = frame["실제점수"], frame[prediction]
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)),
        "순위_Spearman": float(spearmanr(actual, pred).statistic),
        "등급_정확도": float((frame["실제_등급"] == frame[f"{prefix}_등급"]).mean()),
        "인접등급포함_정확도": float((frame["실제_등급"].sub(frame[f"{prefix}_등급"]).abs() <= 1).mean()),
        "A등급_적중률": float(((frame["실제_등급"] == 1) & (frame[f"{prefix}_등급"] == 1)).sum() / 5),
    }


def main() -> int:
    data = pd.read_csv(base.INPUT, encoding="utf-8-sig")
    pairs = make_pairs(data)
    train_validation = pairs[pairs["목표연도"] <= 2021]
    validation = pairs[pairs["목표연도"] == 2022]
    final_train = pairs[pairs["목표연도"] <= 2022]
    test = pairs[pairs["목표연도"] == 2023].copy()

    comparison = [{
        "모델": "Persistence",
        "검증_RMSE": mean_squared_error(validation["target_score"], validation["범죄안전_대체점수"]) ** 0.5,
    }]
    for name, estimator in base.candidates().items():
        model = base.pipeline(estimator)
        model.fit(train_validation[base.FEATURES], train_validation["target_score"])
        comparison.append({
            "모델": name,
            "검증_RMSE": mean_squared_error(validation["target_score"], model.predict(validation[base.FEATURES])) ** 0.5,
        })
    comparison = pd.DataFrame(comparison).sort_values("검증_RMSE")
    selected_name = comparison.iloc[0]["모델"]
    best_ml_name = comparison[comparison["모델"] != "Persistence"].iloc[0]["모델"]

    fitted = {}
    predictions = {}
    for name in [selected_name, best_ml_name]:
        if name == "Persistence":
            predictions[name] = test["범죄안전_대체점수"].to_numpy()
        else:
            model = base.pipeline(base.candidates()[name])
            model.fit(final_train[base.FEATURES], final_train["target_score"])
            fitted[name] = model
            predictions[name] = model.predict(test[base.FEATURES])

    result = test[["목표연도", "자치구", "target_score", "범죄안전_대체점수"]].rename(
        columns={"target_score": "실제점수", "범죄안전_대체점수": "전년도점수"}
    )
    result["검증선택모델_예측점수"] = predictions[selected_name]
    result["최우수ML_예측점수"] = predictions[best_ml_name]
    grade(result, "실제점수", "실제")
    grade(result, "전년도점수", "기준")
    grade(result, "검증선택모델_예측점수", "선택")
    grade(result, "최우수ML_예측점수", "최우수ML")

    report = {
        "검증선택모델": selected_name,
        "최우수머신러닝": best_ml_name,
        "전년도유지": metrics(result, "전년도점수", "기준"),
        "검증선택모델": {"모델명": selected_name, **metrics(result, "검증선택모델_예측점수", "선택")},
        "최우수머신러닝": {"모델명": best_ml_name, **metrics(result, "최우수ML_예측점수", "최우수ML")},
    }
    # 실제값이 없는 2024 운영 예측은 검증에서 선택된 모델로 생성한다.
    current = data[data["연도"] == 2023].copy().rename(columns={"연도": "기준연도"})
    previous = data[data["연도"] == 2022].set_index("자치구")
    for column in base.RATE_COLUMNS:
        current[f"{column}_전년차"] = current[column] - current["자치구"].map(previous[column])
    forecast = current[["자치구"]].copy()
    forecast.insert(0, "예측연도", 2024)
    if selected_name == "Persistence":
        forecast["예측점수"] = current["범죄안전_대체점수"].to_numpy()
    else:
        operational = base.pipeline(base.candidates()[selected_name])
        operational.fit(pairs[base.FEATURES], pairs["target_score"])
        fitted["operational"] = operational
        forecast["예측점수"] = operational.predict(current[base.FEATURES])
    grade(forecast, "예측점수", "예측")
    forecast = forecast.sort_values("예측_순위")

    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / "직접점수_모델비교_검증2022.csv", index=False, encoding="utf-8-sig")
    result.sort_values("선택_순위").to_csv(OUT / "2023_직접점수_예측대실제.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(OUT / "2024_직접점수_예측.csv", index=False, encoding="utf-8-sig")
    (OUT / "평가지표.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"selected": selected_name, "best_ml": best_ml_name, "models": fitted,
                 "features": base.FEATURES}, OUT / "direct_score_1y_models.joblib")
    print(comparison.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
