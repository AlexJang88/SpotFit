"""범죄안전 1년 예측모델의 상승·하락 및 상대등급 전환 탐지 성능 평가."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "ml" / "training" / "train_crime_safety_1y.py"
spec = importlib.util.spec_from_file_location("crime_component", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(base)

OUT = ROOT / "artifacts" / "crime_safety_forecast" / "grade_transition_rolling_2021_2023"
TEST_YEARS = (2021, 2022, 2023)


def add_grade(frame: pd.DataFrame, score: str, prefix: str) -> None:
    frame[f"{prefix}_순위"] = frame.groupby("목표연도")[score].rank(
        ascending=False, method="first"
    ).astype(int)
    frame[f"{prefix}_등급"] = ((frame[f"{prefix}_순위"] - 1) // 5 + 1).astype(int)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    actual_changed = frame["실제_등급"] != frame["전년_등급"]
    predicted_changed = frame[f"{prefix}_등급"] != frame["전년_등급"]
    tp = int((actual_changed & predicted_changed).sum())
    fp = int((~actual_changed & predicted_changed).sum())
    fn = int((actual_changed & ~predicted_changed).sum())
    tn = int((~actual_changed & ~predicted_changed).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    # 등급 숫자가 작아지면 개선, 커지면 악화다.
    actual_direction = np.sign(frame["전년_등급"] - frame["실제_등급"])
    predicted_direction = np.sign(frame["전년_등급"] - frame[f"{prefix}_등급"])
    changed_mask = actual_direction != 0
    score_actual_direction = np.sign(frame["실제점수"] - frame["전년점수"])
    score_pred_direction = np.sign(frame[f"{prefix}_예측점수"] - frame["전년점수"])

    return {
        "표본수": int(len(frame)),
        "실제_등급전환수": int(actual_changed.sum()),
        "예측_등급전환수": int(predicted_changed.sum()),
        "등급전환_TP": tp,
        "등급전환_FP": fp,
        "등급전환_FN": fn,
        "등급전환_TN": tn,
        "등급전환_정밀도": precision,
        "등급전환_탐지율_재현율": recall,
        "등급전환_F1": safe_div(2 * precision * recall, precision + recall),
        "등급전환_이진정확도": safe_div(tp + tn, len(frame)),
        "등급_정확도": float((frame["실제_등급"] == frame[f"{prefix}_등급"]).mean()),
        "인접등급포함_정확도": float(
            (frame["실제_등급"].sub(frame[f"{prefix}_등급"]).abs() <= 1).mean()
        ),
        "등급방향_3분류정확도": float((actual_direction == predicted_direction).mean()),
        "실제전환구_방향정확도": float(
            (actual_direction[changed_mask] == predicted_direction[changed_mask]).mean()
        ) if changed_mask.any() else 0.0,
        "점수상승하락_방향정확도": float((score_actual_direction == score_pred_direction).mean()),
        "MAE": float(mean_absolute_error(frame["실제점수"], frame[f"{prefix}_예측점수"])),
        "RMSE": float(mean_squared_error(frame["실제점수"], frame[f"{prefix}_예측점수"]) ** 0.5),
        "R2": float(r2_score(frame["실제점수"], frame[f"{prefix}_예측점수"])),
    }


def predict_components(
    train: pd.DataFrame, validation: pd.DataFrame, final_train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    result = test[["목표연도", "자치구"]].copy()
    selections: list[dict[str, object]] = []
    for target in base.RATE_COLUMNS:
        y = f"target_{target}"
        options = [("Persistence", mean_squared_error(validation[y], validation[target]) ** 0.5)]
        for name, estimator in base.candidates().items():
            model = base.pipeline(estimator)
            model.fit(train[base.FEATURES], train[y])
            options.append((name, mean_squared_error(validation[y], model.predict(validation[base.FEATURES])) ** 0.5))
        chosen, validation_rmse = min(options, key=lambda item: item[1])
        if chosen == "Persistence":
            prediction = test[target].to_numpy()
        else:
            model = base.pipeline(base.candidates()[chosen])
            model.fit(final_train[base.FEATURES], final_train[y])
            prediction = model.predict(test[base.FEATURES])
        result[f"component_{target}"] = prediction
        selections.append({"목표연도": int(test["목표연도"].iloc[0]), "예측지표": target,
                           "선택모델": chosen, "검증_RMSE": float(validation_rmse)})
    result = base.score_frame(result, "component_")
    return result, selections


def predict_direct(
    train: pd.DataFrame, validation: pd.DataFrame, final_train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, str, str]:
    options = [("Persistence", mean_squared_error(validation["target_score"], validation["범죄안전_대체점수"]) ** 0.5)]
    for name, estimator in base.candidates().items():
        model = base.pipeline(estimator)
        model.fit(train[base.FEATURES], train["target_score"])
        options.append((name, mean_squared_error(validation["target_score"], model.predict(validation[base.FEATURES])) ** 0.5))
    selected = min(options, key=lambda item: item[1])[0]
    best_ml = min((item for item in options if item[0] != "Persistence"), key=lambda item: item[1])[0]

    predictions: dict[str, np.ndarray] = {}
    for name in {selected, best_ml}:
        if name == "Persistence":
            predictions[name] = test["범죄안전_대체점수"].to_numpy()
        else:
            model = base.pipeline(base.candidates()[name])
            model.fit(final_train[base.FEATURES], final_train["target_score"])
            predictions[name] = model.predict(test[base.FEATURES])
    return predictions[selected], predictions[best_ml], selected, best_ml


def main() -> int:
    data = pd.read_csv(base.INPUT, encoding="utf-8-sig")
    pairs = base.make_pairs(data)
    future_score = data[["연도", "자치구", "범죄안전_대체점수"]].copy()
    future_score["기준연도"] = future_score["연도"] - 1
    future_score = future_score.drop(columns="연도").rename(columns={"범죄안전_대체점수": "target_score"})
    pairs = pairs.merge(future_score, on=["기준연도", "자치구"], validate="one_to_one")

    folds, selections, direct_selections = [], [], []
    for target_year in TEST_YEARS:
        validation_year = target_year - 1
        train = pairs[pairs["목표연도"] < validation_year]
        validation = pairs[pairs["목표연도"] == validation_year]
        final_train = pairs[pairs["목표연도"] <= validation_year]
        test = pairs[pairs["목표연도"] == target_year].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"{target_year} 롤링 백테스트 자료가 부족합니다.")

        component, fold_selections = predict_components(train, validation, final_train, test)
        selected_pred, best_ml_pred, selected_name, best_ml_name = predict_direct(
            train, validation, final_train, test
        )
        selections.extend(fold_selections)
        direct_selections.append({"목표연도": target_year, "검증선택모델": selected_name,
                                  "검증최우수ML": best_ml_name})

        fold = test[["목표연도", "자치구", "target_score", "범죄안전_대체점수"]].rename(
            columns={"target_score": "실제점수", "범죄안전_대체점수": "전년점수"}
        )
        fold["Persistence_예측점수"] = fold["전년점수"]
        fold["Component_예측점수"] = component["component_안전점수"].to_numpy()
        fold["DirectSelected_예측점수"] = selected_pred
        fold["DirectBestML_예측점수"] = best_ml_pred
        folds.append(fold)

    result = pd.concat(folds, ignore_index=True)
    add_grade(result, "실제점수", "실제")
    add_grade(result, "전년점수", "전년")
    for prefix in ("Persistence", "Component", "DirectSelected", "DirectBestML"):
        add_grade(result, f"{prefix}_예측점수", prefix)

    metrics = {prefix: evaluate(result, prefix) for prefix in (
        "Persistence", "Component", "DirectSelected", "DirectBestML"
    )}
    yearly = []
    for year, frame in result.groupby("목표연도"):
        for prefix in metrics:
            yearly.append({"목표연도": int(year), "모델": prefix, **evaluate(frame, prefix)})

    detail_columns = ["목표연도", "자치구", "실제점수", "전년점수", "실제_등급", "전년_등급"]
    for prefix in metrics:
        result[f"{prefix}_실제전환"] = result["실제_등급"] != result["전년_등급"]
        result[f"{prefix}_예측전환"] = result[f"{prefix}_등급"] != result["전년_등급"]
        detail_columns += [f"{prefix}_예측점수", f"{prefix}_등급", f"{prefix}_실제전환", f"{prefix}_예측전환"]

    OUT.mkdir(parents=True, exist_ok=True)
    result[detail_columns].to_csv(OUT / "등급전환_롤링백테스트_상세.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "연도별_등급전환_성능.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / "지표별_롤링선택모델.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(direct_selections).to_csv(OUT / "직접점수_롤링선택모델.csv", index=False, encoding="utf-8-sig")
    (OUT / "등급전환_평가지표.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
