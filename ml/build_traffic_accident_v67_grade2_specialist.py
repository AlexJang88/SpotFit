from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.build_traffic_accident_v61_v65_segment_analysis import fold_prediction
from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "교통사고"
V59 = MODEL_ROOT / "51차~60차" / "59차"
OUTPUT = MODEL_ROOT / "67차"
POOL_START, POOL_END = 4, 16


def scaled(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def model_defs():
    result = {}
    for c in (0.03, 0.1, 0.3, 1, 3):
        result[f"Logistic_C{c:g}"] = scaled(
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=42)
        )
    for leaf in (2, 5, 10):
        result[f"ExtraTrees_leaf{leaf}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=0.7, class_weight="balanced", random_state=42, n_jobs=1)),
        ])
        result[f"RandomForest_leaf{leaf}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=0.7, class_weight="balanced", random_state=42, n_jobs=1)),
        ])
    for depth in (1, 2):
        result[f"GradientBoosting_d{depth}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=depth, random_state=42)),
        ])
    return result


def component_importance_order(component):
    fitted = component["model"].named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    else:
        coefficients = np.abs(np.asarray(fitted.coef_, dtype=float))
        values = coefficients if coefficients.ndim == 1 else coefficients.mean(axis=0)
    return [
        feature for feature, _ in sorted(
            zip(component["features"], values), key=lambda item: item[1], reverse=True
        )
    ]


def feature_sets(bundle):
    scores = {}
    components = [bundle["component_a"], bundle["component_b"]]
    for component in components:
        for position, feature in enumerate(component_importance_order(component), 1):
            scores[feature] = scores.get(feature, 0.0) + 1 / position
    ranked = [feature for feature, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    union = list(dict.fromkeys(components[0]["features"] + components[1]["features"]))
    return {
        "top15": ranked[:15],
        "top20": ranked[:20],
        "top25": ranked[:25],
        "top30": ranked[:30],
        "all": union,
    }


def grade2_probability(model, frame, features):
    probabilities = model.predict_proba(frame[features])
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def correct_grade2(base_score, probability):
    """Reorder safety ranks 4..16 so five selected districts become safety grade 2."""
    # The base score is risk-oriented, so ascending risk is descending safety.
    order = np.argsort(np.asarray(base_score), kind="stable")
    pool = list(order[POOL_START - 1 : POOL_END])
    chosen = sorted(pool, key=lambda index: (-probability[index], list(order).index(index)))[:5]
    chosen_set = set(chosen)
    remaining = [index for index in pool if index not in chosen_set]
    chosen = sorted(chosen, key=lambda index: list(order).index(index))
    # Safety ranks 4-5 remain grade 1, ranks 6-10 are the selected grade 2,
    # and ranks 11-16 retain their relative 59차 order.
    new_pool = remaining[:2] + chosen + remaining[2:]
    corrected_order = list(order)
    corrected_order[POOL_START - 1 : POOL_END] = new_pool
    corrected = np.empty(len(order), dtype=int)
    corrected[np.asarray(corrected_order)] = np.arange(len(order)) // 5 + 1
    return corrected


def calculate_metrics(frame):
    grouped = frame.groupby(["예측간격", "검증목표연도"], as_index=False).agg(
        exact=("정확여부", "mean"),
        within1=("한등급이내", "mean"),
        error=("절대오차", "mean"),
    )
    summary = grouped.groupby("예측간격", as_index=False).agg(
        exact=("exact", "mean"), within1=("within1", "mean"),
        error=("error", "mean"), min_exact=("exact", "min")
    )
    recall = frame.groupby(["예측간격", "실제등급"], as_index=False)["정확여부"].mean().rename(columns={"정확여부": "등급재현율"})
    return grouped, summary, recall


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    bundles = {
        2: joblib.load(V59 / "교통사고_59차_2027년_2년후_모델.joblib"),
        3: joblib.load(V59 / "교통사고_59차_2028년_3년후_모델.joblib"),
    }
    definitions = model_defs()
    all_trials = []
    best_by_horizon = {}
    best_details = []

    for horizon, bundle in bundles.items():
        target_grade = f"{horizon}년후_위험등급"
        year_col = f"{horizon}년후_목표연도"
        labeled = data[data[target_grade].notna()].copy()
        sets = feature_sets(bundle)
        config_rows = {}

        for set_name, features in sets.items():
            for model_name, estimator in definitions.items():
                config = f"{model_name}@{set_name}"
                rows = []
                for year in CV_YEARS[horizon]:
                    train = labeled[labeled[year_col] < year]
                    valid = labeled[labeled[year_col] == year].copy()
                    # Internal risk grade 4 is external safety grade 2.
                    binary = (train[target_grade].astype(int) == 4).astype(int)
                    specialist = clone(estimator).fit(train[features], binary)
                    probability = grade2_probability(specialist, valid, features)
                    base_score = fold_prediction(bundle, train, valid, horizon)
                    base_grade = 6 - grades_from_risk(base_score)
                    corrected_grade = correct_grade2(base_score, probability)
                    actual = 6 - valid[target_grade].astype(int).to_numpy()
                    for district, actual_grade, before, after in zip(valid["자치구"], actual, base_grade, corrected_grade):
                        rows.append({
                            "예측간격": horizon, "검증목표연도": year, "자치구": district,
                            "실제등급": int(actual_grade), "59차예측등급": int(before), "67차예측등급": int(after),
                            "정확여부": int(actual_grade == after), "한등급이내": int(abs(actual_grade-after) <= 1),
                            "절대오차": int(abs(actual_grade-after)),
                        })
                detail = pd.DataFrame(rows)
                _, summary, recall = calculate_metrics(detail)
                metric = summary.iloc[0].to_dict()
                grade2 = float(recall.loc[recall["실제등급"].eq(2), "등급재현율"].iloc[0])
                grade3 = float(recall.loc[recall["실제등급"].eq(3), "등급재현율"].iloc[0])
                trial = {"예측간격": horizon, "모델": config, "변수수": len(features), **metric, "2등급재현율": grade2, "3등급재현율": grade3}
                all_trials.append(trial)
                config_rows[config] = (trial, detail, estimator, features)

        baseline = {2: {"exact": .592, "within1": .912, "error": .496, "grade2": .60, "grade3": .56}, 3: {"exact": .60, "within1": .90, "error": .50, "grade2": .50, "grade3": .55}}[horizon]
        eligible = [value for value in config_rows.values() if value[0]["exact"] + 1e-12 >= baseline["exact"]]
        pool = eligible if eligible else list(config_rows.values())
        best = max(pool, key=lambda value: (value[0]["exact"], value[0]["2등급재현율"], value[0]["3등급재현율"], value[0]["within1"], -value[0]["error"], -value[0]["변수수"]))
        trial, detail, estimator, features = best
        improved = trial["exact"] > baseline["exact"] + 1e-12 or (abs(trial["exact"]-baseline["exact"]) < 1e-12 and trial["2등급재현율"] > baseline["grade2"] + 1e-12)
        best_by_horizon[horizon] = {**trial, "채택": improved}
        best_details.append(detail.assign(전문모델=trial["모델"], 채택=improved))

        full_binary = (labeled[target_grade].astype(int) == 4).astype(int)
        fitted_specialist = clone(estimator).fit(labeled[features], full_binary)
        wrapper = {
            "version": 67, "base_version": 59, "horizon": horizon, "forecast_year": 2025+horizon,
            "kind": "safety_grade2_boundary_specialist", "base_model": bundle,
            "candidate_rank_start": POOL_START, "candidate_rank_end": POOL_END,
            "specialist_model": fitted_specialist, "specialist_features": features,
            "grade_definition": "안전 1등급=가장 안전, 안전 5등급=가장 위험",
            "selection_rule": "안전순위 4~16위 중 안전 2등급 확률 상위 5개를 안전 2등급으로 재배치",
            "adopted": improved, "metrics": trial,
        }
        joblib.dump(wrapper, OUTPUT / f"교통사고_67차_{2025+horizon}년_{horizon}년후_2등급보정모델.joblib")
        print(json.dumps({"horizon": horizon, "baseline": baseline, "best": best_by_horizon[horizon]}, ensure_ascii=False), flush=True)

    trials = pd.DataFrame(all_trials).sort_values(["예측간격", "exact", "2등급재현율", "3등급재현율"], ascending=[True, False, False, False])
    details = pd.concat(best_details, ignore_index=True)
    best_frame = pd.DataFrame([best_by_horizon[2], best_by_horizon[3]])
    with pd.ExcelWriter(OUTPUT / "67차_2등급전문보정_검증결과.xlsx", engine="openpyxl") as writer:
        best_frame.to_excel(writer, sheet_name="최고후보", index=False)
        trials.to_excel(writer, sheet_name="전체후보", index=False)
        details.to_excel(writer, sheet_name="연도자치구별상세", index=False)
        _, summary, recall = calculate_metrics(details)
        summary.to_excel(writer, sheet_name="채택후보성능", index=False)
        recall.to_excel(writer, sheet_name="등급별재현율", index=False)
    info = {
        "모델명": "교통사고 67차 안전 2등급 경계 전문 보정 모델", "기반": "59차",
        "등급정의": "안전 1등급=가장 안전, 안전 5등급=가장 위험",
        "방식": "59차 안전순위 산출 후 안전 4~16위만 재분석하여 안전 2등급 5개 재선정",
        "2년후": best_by_horizon[2], "3년후": best_by_horizon[3],
    }
    (OUTPUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
