from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "58차"
OUT = ROOT / "artifacts" / "reports" / "traffic_v60_weight_calibration"


def percentile(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data()
    overall = {}
    for horizon in (2, 3):
        bundle = joblib.load(SOURCE / f"교통사고_58차_{2025+horizon}년_{horizon}년후_모델.joblib")
        target = f"{horizon}년후_위험등급"
        year_col = f"{horizon}년후_목표연도"
        labeled = data[data[target].notna()]
        fold = {}
        for year in CV_YEARS[horizon]:
            train = labeled[labeled[year_col] < year]
            valid = labeled[labeled[year_col] == year]
            outputs = []
            for component in (bundle["component_a"], bundle["component_b"]):
                model = clone(component["model"]).fit(train[component["features"]], train[target].astype(int))
                outputs.append(risk(model, valid[component["features"]]))
            fold[year] = (outputs[0], outputs[1], valid["현재연도_종합위험점수"].to_numpy(dtype=float), valid[target].astype(int).to_numpy())

        rows = []
        for mode in ("raw", "percentile"):
          for weight_a in np.arange(0, 1.001, 0.01):
            blend_values = (0.0,) if mode == "raw" else np.arange(0, 0.501, 0.05)
            for current_blend in blend_values:
                year_rows = []
                for year in CV_YEARS[horizon]:
                    a, b, current, actual = fold[year]
                    if mode == "raw":
                        combined = weight_a * a + (1 - weight_a) * b
                    else:
                        learned = weight_a * percentile(a) + (1 - weight_a) * percentile(b)
                        combined = (1-current_blend) * learned + current_blend * percentile(current)
                    predicted = grades_from_risk(combined)
                    distance = np.abs(actual-predicted)
                    year_rows.append((np.mean(actual==predicted),np.mean(distance<=1),np.mean(distance)))
                values=np.asarray(year_rows)
                rows.append({"예측간격":horizon,"혼합방식":mode,"구성A가중치":round(float(weight_a),2),"현재점수혼합":round(float(current_blend),2),"exact":float(values[:,0].mean()),"within1":float(values[:,1].mean()),"error":float(values[:,2].mean()),"min_exact":float(values[:,0].min())})
        result=pd.DataFrame(rows).sort_values(["exact","within1","error","min_exact"],ascending=[False,False,True,False])
        result.to_excel(OUT/f"{horizon}년후_가중치보정탐색.xlsx",index=False)
        overall[horizon]=result.iloc[0].to_dict()
        print(json.dumps({"horizon":horizon,"best":overall[horizon]},ensure_ascii=False),flush=True)
    (OUT/"최고결과.json").write_text(json.dumps(overall,ensure_ascii=False,indent=2,default=float),encoding="utf-8")


if __name__=="__main__": main()
