from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT=Path(__file__).resolve().parents[1]; START=ROOT/"models"/"교통사고"/"54차"; REPORT=ROOT/"artifacts"/"reports"/"traffic_v54_horizon_prune"; REPORT.mkdir(parents=True,exist_ok=True)


def comps(bundle): return [bundle["component"]] if bundle["kind"] in {"classifier_rank","classifier_direct"} else [bundle["component_a"],bundle["component_b"]]


def predict(bundle,tr,va,target,removed):
    values=[]
    for c in comps(bundle):
        features=[f for f in c["features"] if f not in removed]; model=clone(c["model"]).fit(tr[features],tr[target].astype(int)); values.append(risk(model,va[features]))
    return values[0] if len(values)==1 else bundle["weight_a"]*values[0]+(1-bundle["weight_a"])*values[1]


def evaluate(data,bundle,h,removed):
    target=f"{h}년후_위험등급"; year_col=f"{h}년후_목표연도"; labeled=data[data[target].notna()]; rows=[]
    for year in CV_YEARS[h]:
        tr=labeled[labeled[year_col]<year]; va=labeled[labeled[year_col]==year]; pred=grades_from_risk(predict(bundle,tr,va,target,removed)); actual=va[target].astype(int).to_numpy()
        rows.append({"예측간격":h,"검증목표연도":year,"정확등급정확도":accuracy_score(actual,pred),"±1등급정확도":np.mean(np.abs(actual-pred)<=1),"평균등급오차":np.mean(np.abs(actual-pred))})
    d=pd.DataFrame(rows); return {"exact":float(d.정확등급정확도.mean()),"within1":float(d["±1등급정확도"].mean()),"error":float(d.평균등급오차.mean()),"min_exact":float(d.정확등급정확도.min())},d


def key(m): return m["exact"],m["within1"],-m["error"],m["min_exact"]


def importance(bundle):
    score={}
    for c in comps(bundle):
        final=c["model"].named_steps["model"]
        if hasattr(final,"feature_importances_"): values=np.asarray(final.feature_importances_)
        elif hasattr(final,"coef_"): values=np.abs(np.asarray(final.coef_)).mean(axis=0)
        else: continue
        if len(values)!=len(c["features"]): continue
        for f,v in zip(c["features"],values/(values.sum() or 1)): score[f]=score.get(f,0)+float(v)
    return [f for f,_ in sorted(score.items(),key=lambda item:item[1]) if not f.startswith("구_")]


def refit(bundle,data,h,removed):
    target=f"{h}년후_위험등급"; labeled=data[data[target].notna()]
    def one(c):
        features=[f for f in c["features"] if f not in removed]; result={k:v for k,v in c.items() if k not in {"model","features"}}; result.update(features=features,model=clone(c["model"]).fit(labeled[features],labeled[target].astype(int))); return result
    result={k:v for k,v in bundle.items() if k not in {"component","component_a","component_b"}}
    if bundle["kind"] in {"classifier_rank","classifier_direct"}: result["component"]=one(bundle["component"])
    else: result.update(component_a=one(bundle["component_a"]),component_b=one(bundle["component_b"])); return result
    return result


def save(version,data,bundles,removed_by_h,metrics_by_h,detail_by_h,history):
    out=ROOT/"models"/"교통사고"/f"{version}차"; out.mkdir(parents=True,exist_ok=True)
    for h in (2,3):
        fitted=refit(bundles[h],data,h,removed_by_h[h]); fitted.update(version=version,horizon=h,forecast_year=2025+h,removed_features=sorted(removed_by_h[h])); joblib.dump(fitted,out/f"교통사고_{version}차_{2025+h}년_{h}년후_모델.joblib")
    overall={"2년후":metrics_by_h[2],"3년후":metrics_by_h[3],"평균정확등급":np.mean([metrics_by_h[2]["exact"],metrics_by_h[3]["exact"]]),"평균_1칸이내":np.mean([metrics_by_h[2]["within1"],metrics_by_h[3]["within1"]]),"평균등급오차":np.mean([metrics_by_h[2]["error"],metrics_by_h[3]["error"]])}
    info={"모델명":f"교통사고 {version}차 장기등급 모델","기반":"54차","방식":"예측기간별 독립 후진제거","2년후제거":sorted(removed_by_h[2]),"3년후제거":sorted(removed_by_h[3]),"성능":overall}
    (out/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    with pd.ExcelWriter(out/f"{version}차_검증결과.xlsx",engine="openpyxl") as w: pd.concat(detail_by_h.values(),ignore_index=True).to_excel(w,sheet_name="연도별검증",index=False); pd.DataFrame(history).to_excel(w,sheet_name="개선이력",index=False)


def main():
    data,_=prepare_data(); bundles={2:joblib.load(START/"교통사고_54차_2027년_2년후_모델.joblib"),3:joblib.load(START/"교통사고_54차_2028년_3년후_모델.joblib")}
    removed={2:set(),3:set()}; metrics={}; details={}; history=[]; version=54
    for h in (2,3): metrics[h],details[h]=evaluate(data,bundles[h],h,removed[h])
    history.append({"차수":54,"변경":"54차 기준","2년정확도":metrics[2]["exact"],"3년정확도":metrics[3]["exact"]})
    print("BASE54",metrics,flush=True)
    for h in (2,3):
        for round_no in range(1,9):
            candidates=[f for f in importance(bundles[h]) if f not in removed[h]][:35]; trials=[]
            for f in candidates:
                m,d=evaluate(data,bundles[h],h,removed[h]|{f}); trials.append({"제거후보":f,**m,"detail":d})
            trials.sort(key=lambda r:key(r),reverse=True); pd.DataFrame([{k:v for k,v in r.items() if k!="detail"} for r in trials]).to_excel(REPORT/f"h{h}_round{round_no:02d}.xlsx",index=False)
            if not trials or key(trials[0])<=key(metrics[h]): break
            best=trials[0]; f=best["제거후보"]; removed[h].add(f); metrics[h]={k:best[k] for k in metrics[h]}; details[h]=best["detail"]; version+=1
            history.append({"차수":version,"변경":f"{h}년 모델 제거: {f}","2년정확도":metrics[2]["exact"],"3년정확도":metrics[3]["exact"]}); save(version,data,bundles,removed,metrics,details,history)
            print("ACCEPT",version,h,f,metrics[h],flush=True)
    pd.DataFrame(history).to_excel(REPORT/"54차부터_최종차수_개선이력.xlsx",index=False)
    print("FINAL",json.dumps({"version":version,"removed":{str(k):sorted(v) for k,v in removed.items()},"metrics":metrics},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
