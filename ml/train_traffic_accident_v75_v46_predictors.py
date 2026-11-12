from __future__ import annotations

import json

import joblib
import pandas as pd

import ml.train_traffic_accident_v74_all_available as engine


ROOT = engine.ROOT
VERSION = 75
engine.VERSION = VERSION
engine.OUT = ROOT / "models" / "교통사고" / "75차.46차32개보조X"
engine.PROCESSED = ROOT / "data" / "processed" / "교통사고_75차_46차32개보조X_2027_2028.xlsx"


def v46_feature_frame(panel: pd.DataFrame, target_features: list[str]):
    info_path = ROOT / "models" / "교통사고" / "41차~49차" / "46차" / "model_info.json"
    predictors = json.loads(info_path.read_text(encoding="utf-8"))["최종변수"]
    missing = [column for column in predictors if column not in panel.columns]
    if missing:
        raise ValueError(f"46차 보조 X 누락: {missing}")

    base = panel[["연도", "자치구", *predictors]].copy().sort_values(["자치구", "연도"])
    out = base.copy()
    group = base.groupby("자치구", sort=False)
    for feature in target_features:
        out[f"전년__{feature}"] = group[feature].shift(1)
        out[f"증감__{feature}"] = group[feature].diff()
        out[f"3년평균__{feature}"] = group[feature].transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
    districts = pd.get_dummies(out["자치구"], prefix="자치구", dtype=float)
    out = pd.concat([out.drop(columns="자치구"), districts], axis=1)
    return out, predictors, list(districts.columns)


engine.raw_feature_frame = v46_feature_frame


def load_v74_maps():
    model_dir = ROOT / "models" / "교통사고" / "74차.보유전체데이터활용"
    result = {}
    for horizon in (2, 3):
        bundle = joblib.load(model_dir / f"교통사고_74차_{horizon}년후_13지표예측모델.joblib")
        result[horizon] = {feature: payload["spec"] for feature, payload in bundle.items()}
    return result


engine.COMPARISON_MAPS = load_v74_maps()


if __name__ == "__main__":
    engine.main()
