from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "교통사고" / "51차~60차" / "59차" / "교통사고_59차_2028년_3년후_모델.joblib"


def component_prediction(component, train, valid, features, target_score):
    y = train[target_score].to_numpy(dtype=float).copy()
    if component.get("target_kind") == "delta":
        y -= train["현재연도_종합위험점수"].to_numpy(dtype=float)
    model = clone(component["model"]).fit(train[features], y)
    prediction = model.predict(valid[features])
    if component.get("target_kind") == "delta":
        prediction += valid["현재연도_종합위험점수"].to_numpy(dtype=float)
    blend = float(component.get("blend", 0.0))
    if blend:
        current = valid["현재연도_종합위험점수"].to_numpy(dtype=float)
        prediction = (1 - blend) * prediction + blend * current
    return prediction


def evaluate(data, bundle, add_a, add_b, districts):
    horizon = 3
    target_score = "3년후_종합위험점수"
    target_grade = "3년후_위험등급"
    year_col = "3년후_목표연도"
    labeled = data[data[target_grade].notna()]
    features_a = list(bundle["component_a"]["features"]) + (districts if add_a else [])
    features_b = list(bundle["component_b"]["features"]) + (districts if add_b else [])
    rows = []
    for year in CV_YEARS[horizon]:
        train = labeled[labeled[year_col] < year]
        valid = labeled[labeled[year_col] == year]
        a = component_prediction(bundle["component_a"], train, valid, features_a, target_score)
        b = component_prediction(bundle["component_b"], train, valid, features_b, target_score)
        score = float(bundle["weight_a"]) * a + (1 - float(bundle["weight_a"])) * b
        predicted = grades_from_risk(score)
        actual = valid[target_grade].astype(int).to_numpy()
        distance = np.abs(actual - predicted)
        rows.append((np.mean(actual == predicted), np.mean(distance <= 1), np.mean(distance)))
    values = np.asarray(rows)
    return {
        "exact": float(values[:, 0].mean()),
        "within1": float(values[:, 1].mean()),
        "error": float(values[:, 2].mean()),
        "min_exact": float(values[:, 0].min()),
        "component_a_count": len(features_a),
        "component_b_count": len(features_b),
        "unique_count": len(set(features_a) | set(features_b)),
    }


def main():
    data, _ = prepare_data()
    bundle = joblib.load(MODEL)
    districts = [column for column in data.columns if column.startswith("구_")]
    results = {
        "기준_인코딩없음": evaluate(data, bundle, False, False, districts),
        "구성A만추가": evaluate(data, bundle, True, False, districts),
        "구성B만추가": evaluate(data, bundle, False, True, districts),
        "구성AB모두추가": evaluate(data, bundle, True, True, districts),
    }
    print(json.dumps({"자치구컬럼수": len(districts), "결과": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
