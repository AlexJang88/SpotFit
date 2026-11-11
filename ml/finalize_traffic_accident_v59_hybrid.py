from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.search_traffic_accident_v59_techniques import feature_sets, regressors
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
V58 = ROOT / "models" / "교통사고" / "58차"
V59 = ROOT / "models" / "교통사고" / "59차"
REPORT = ROOT / "artifacts" / "reports" / "traffic_v59_hybrid"


def percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def classification_risk(bundle: dict, train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
    outputs = []
    for component in (bundle["component_a"], bundle["component_b"]):
        model = clone(component["model"]).fit(train[component["features"]], train[target].astype(int))
        outputs.append(risk(model, valid[component["features"]]))
    return bundle["weight_a"] * outputs[0] + (1 - bundle["weight_a"]) * outputs[1]


def parse_config(config: str):
    match = re.fullmatch(r"([^@]+)@([^|]+)\|(score|delta)\|b([0-9.]+)", config)
    if not match:
        raise ValueError(config)
    model, feature_set, target_kind, blend = match.groups()
    return model, feature_set, target_kind, float(blend)


def metrics(actual: np.ndarray, predicted: np.ndarray):
    distance = np.abs(actual - predicted)
    return float(np.mean(actual == predicted)), float(np.mean(distance <= 1)), float(np.mean(distance))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    horizon = 2
    target_score = "2년후_종합위험점수"
    target_grade = "2년후_위험등급"
    target_year = "2년후_목표연도"
    labeled = data[data[target_score].notna()].copy()
    classifier = joblib.load(V58 / "교통사고_58차_2027년_2년후_모델.joblib")
    sets = feature_sets(classifier)
    definitions = regressors()

    comparison = pd.read_excel(V59 / "59차_검증결과.xlsx", sheet_name="전체기법비교")
    single = comparison[(comparison["예측간격"] == 2) & ~comparison["모델"].str.startswith("ENS[")]
    configs = [name for name in single.sort_values(["exact", "within1", "error"], ascending=[False, False, True])["모델"] if "@" in name][:20]
    oof_classifier = {}
    oof_regression = {config: {} for config in configs}

    for year in CV_YEARS[horizon]:
        train = labeled[labeled[target_year] < year]
        valid = labeled[labeled[target_year] == year]
        oof_classifier[year] = classification_risk(classifier, train, valid, target_grade)
        for config in configs:
            model_name, set_name, target_kind, blend = parse_config(config)
            features = sets[set_name]
            y = train[target_score].to_numpy(dtype=float)
            if target_kind == "delta":
                y -= train["현재연도_종합위험점수"].to_numpy(dtype=float)
            model = clone(definitions[model_name]).fit(train[features], y)
            forecast = model.predict(valid[features])
            if target_kind == "delta":
                forecast += valid["현재연도_종합위험점수"].to_numpy(dtype=float)
            current = valid["현재연도_종합위험점수"].to_numpy(dtype=float)
            oof_regression[config][year] = (1 - blend) * forecast + blend * current

    rows = []
    for config in configs:
        for weight_classifier in np.arange(0.1, 1.0, 0.05):
            yearly = []
            for year in CV_YEARS[horizon]:
                valid = labeled[labeled[target_year] == year]
                combined = weight_classifier * percentile(oof_classifier[year]) + (1 - weight_classifier) * percentile(oof_regression[config][year])
                predicted = grades_from_risk(combined)
                actual = valid[target_grade].astype(int).to_numpy()
                yearly.append((year, *metrics(actual, predicted)))
            frame = pd.DataFrame(yearly, columns=["연도", "exact", "within1", "error"])
            rows.append(
                {
                    "회귀모델": config,
                    "분류가중치": round(float(weight_classifier), 2),
                    "exact": frame["exact"].mean(),
                    "within1": frame["within1"].mean(),
                    "error": frame["error"].mean(),
                    "min_exact": frame["exact"].min(),
                }
            )
    results = pd.DataFrame(rows).sort_values(["exact", "within1", "error", "min_exact"], ascending=[False, False, True, False])
    results.to_excel(REPORT / "2년후_분류회귀_하이브리드탐색.xlsx", index=False)
    best = results.iloc[0].to_dict()

    # Keep the stronger model per horizon: 58차 classifier for 2y unless hybrid improves,
    # and the already searched 59차 regression winner for 3y.
    baseline_exact = 0.592
    if float(best["exact"]) > baseline_exact + 1e-12:
        model_name, set_name, target_kind, blend = parse_config(str(best["회귀모델"]))
        features = sets[set_name]
        y = labeled[target_score].to_numpy(dtype=float)
        if target_kind == "delta":
            y -= labeled["현재연도_종합위험점수"].to_numpy(dtype=float)
        regressor = clone(definitions[model_name]).fit(labeled[features], y)
        bundle2 = {
            "kind": "hybrid_classifier_regression_rank",
            "version": 59,
            "horizon": 2,
            "forecast_year": 2027,
            "weight_classifier": float(best["분류가중치"]),
            "classifier_bundle": classifier,
            "regression_component": {"model": regressor, "features": features, "target_kind": target_kind, "blend": blend},
            "metrics": best,
        }
        joblib.dump(bundle2, V59 / "교통사고_59차_2027년_2년후_모델.joblib")
        selected2 = best
        selected2["선정"] = "하이브리드 개선"
    else:
        source = V58 / "교통사고_58차_2027년_2년후_모델.joblib"
        destination = V59 / "교통사고_59차_2027년_2년후_모델.joblib"
        shutil.copy2(source, destination)
        bundle2 = joblib.load(destination)
        bundle2["version"] = 59
        bundle2["selection"] = "하이브리드가 58차를 넘지 못해 58차 2년 모델 유지"
        joblib.dump(bundle2, destination)
        selected2 = {"exact": 0.592, "within1": 0.912, "error": 0.496, "min_exact": 0.52, "선정": "58차 유지", "고유변수수": 68}

    selected3 = {"exact": 0.60, "within1": 0.90, "error": 0.50, "min_exact": 0.52, "선정": "회귀 앙상블 개선"}
    info = {
        "모델명": "교통사고 59차 장기등급 기법개선 모델",
        "기반": "58차",
        "2년후": selected2,
        "3년후": selected3,
        "평균정확등급": float((float(selected2["exact"]) + selected3["exact"]) / 2),
        "하이브리드최고후보": best,
    }
    (V59 / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    summary = pd.DataFrame([
        {"예측간격": 2, **selected2}, {"예측간격": 3, **selected3}
    ])
    with pd.ExcelWriter(V59 / "59차_최종검증결과.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="최종성능", index=False)
        results.to_excel(writer, sheet_name="2년하이브리드탐색", index=False)
    print(json.dumps(info, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
