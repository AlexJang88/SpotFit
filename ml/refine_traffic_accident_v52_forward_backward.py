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


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"models"/"교통사고"/"52차"
OUT=ROOT/"artifacts"/"reports"/"traffic_v52_refine"; OUT.mkdir(parents=True,exist_ok=True)
MODELS={2:BASE/"교통사고_52차_2027년_2년후_모델.joblib",3:BASE/"교통사고_52차_2028년_3년후_모델.joblib"}


def components(bundle):
    return [bundle["component"]] if bundle["kind"] in {"classifier_rank","classifier_direct"} else [bundle["component_a"],bundle["component_b"]]


def component_risk(component,train,valid,target,added,removed):
    features=[f for f in component["features"] if f not in removed]
    features.extend(f for f in added if f not in features and f not in removed)
    model=clone(component["model"]).fit(train[features],train[target].astype(int))
    return risk(model,valid[features])


def predict(bundle,train,valid,target,added,removed):
    comps=components(bundle); values=[component_risk(c,train,valid,target,added,removed) for c in comps]
    if len(values)==1: return values[0]
    return bundle["weight_a"]*values[0]+(1-bundle["weight_a"])*values[1]


def evaluate(data,bundles,added,removed):
    rows=[]
    for h,bundle in bundles.items():
        target=f"{h}년후_위험등급"; year_col=f"{h}년후_목표연도"; labeled=data[data[target].notna()]
        for year in CV_YEARS[h]:
            tr=labeled[labeled[year_col]<year]; va=labeled[labeled[year_col]==year]
            pred=grades_from_risk(predict(bundle,tr,va,target,added,removed)); actual=va[target].astype(int).to_numpy()
            rows.append({"예측간격":h,"검증목표연도":year,"정확등급정확도":accuracy_score(actual,pred),"±1등급정확도":np.mean(np.abs(actual-pred)<=1),"평균등급오차":np.mean(np.abs(actual-pred))})
    d=pd.DataFrame(rows); s=d.groupby("예측간격",as_index=False).agg(exact=("정확등급정확도","mean"),within1=("±1등급정확도","mean"),error=("평균등급오차","mean"),min_exact=("정확등급정확도","min"))
    m={"exact_avg":float(s.exact.mean()),"within1_avg":float(s.within1.mean()),"error_avg":float(s.error.mean()),"min_exact":float(s.min_exact.min()),
       "h2_exact":float(s.loc[s.예측간격.eq(2),"exact"].iloc[0]),"h3_exact":float(s.loc[s.예측간격.eq(3),"exact"].iloc[0]),
       "h2_within1":float(s.loc[s.예측간격.eq(2),"within1"].iloc[0]),"h3_within1":float(s.loc[s.예측간격.eq(3),"within1"].iloc[0])}
    return m,d


def better(a,b):
    return (a["exact_avg"],a["within1_avg"],-a["error_avg"],a["min_exact"])>(b["exact_avg"],b["within1_avg"],-b["error_avg"],b["min_exact"])


def full_component(component,data,h,added,removed):
    target=f"{h}년후_위험등급"; labeled=data[data[target].notna()]; features=[f for f in component["features"] if f not in removed]
    features.extend(f for f in added if f not in features and f not in removed)
    result={k:v for k,v in component.items() if k not in {"model","features"}}; result.update(features=features,model=clone(component["model"]).fit(labeled[features],labeled[target].astype(int))); return result


def full_bundle(bundle,data,h,added,removed):
    result={k:v for k,v in bundle.items() if k not in {"component","component_a","component_b"}}
    if bundle["kind"] in {"classifier_rank","classifier_direct"}: result["component"]=full_component(bundle["component"],data,h,added,removed)
    else:
        result["component_a"]=full_component(bundle["component_a"],data,h,added,removed); result["component_b"]=full_component(bundle["component_b"],data,h,added,removed)
    return result


def save(version,phase,change,data,bundles,added,removed,metrics,detail,history):
    directory=ROOT/"models"/"교통사고"/f"{version}차"; directory.mkdir(parents=True,exist_ok=True)
    for h in (2,3):
        b=full_bundle(bundles[h],data,h,added,removed); b.update(version=version,horizon=h,forecast_year=2025+h,added_features=added,removed_features=sorted(removed))
        joblib.dump(b,directory/f"교통사고_{version}차_{2025+h}년_{h}년후_모델.joblib")
    info={"모델명":f"교통사고 {version}차 장기등급 모델","기반":"52차","단계":phase,"변경":change,"추가변수":added,"제거변수":sorted(removed),"성능":metrics}
    (directory/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    with pd.ExcelWriter(directory/f"{version}차_검증결과.xlsx",engine="openpyxl") as w: detail.to_excel(w,sheet_name="연도별검증",index=False); pd.DataFrame(history).to_excel(w,sheet_name="개선이력",index=False)


def importance_order(bundles):
    scores={}
    for bundle in bundles.values():
        for component in components(bundle):
            model=component["model"]; final=model.named_steps["model"]
            if hasattr(final,"feature_importances_"): values=np.asarray(final.feature_importances_)
            elif hasattr(final,"coef_"): values=np.abs(np.asarray(final.coef_)).mean(axis=0)
            else: continue
            if "impute" in model.named_steps and len(values)!=len(component["features"]): continue
            total=values.sum() or 1
            for f,v in zip(component["features"],values/total): scores[f]=scores.get(f,0)+float(v)
    return [f for f,_ in sorted(scores.items(),key=lambda item:item[1]) if not f.startswith("구_")]


def main():
    data,candidates=prepare_data(); bundles={h:joblib.load(p) for h,p in MODELS.items()}; added=[]; removed=set(); version=52; history=[]
    current,detail=evaluate(data,bundles,added,removed); history.append({"차수":52,"단계":"기준","변경":"52차 분류앙상블",**current}); print("BASE52",json.dumps(current,ensure_ascii=False),flush=True)
    used=set(f for b in bundles.values() for c in components(b) for f in c["features"])
    screen=pd.read_csv(ROOT/"artifacts"/"reports"/"traffic_v52_search"/"forward_01_screen.csv")
    ordered=[c for c in screen.후보변수 if c in candidates and c not in used]
    # Forward verify the 20 strongest unused candidates and their top pairs.
    trials=[]
    for candidate in ordered[:20]:
        m,d=evaluate(data,bundles,[candidate],removed); trials.append({"변경":candidate,"features":[candidate],**m,"detail":d})
    trials.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in {"detail","features"}} for r in trials]).to_excel(OUT/"forward_single.xlsx",index=False)
    if trials and better(trials[0],current):
        best=trials[0]; added=best["features"]; version+=1; current={k:best[k] for k in current}; detail=best["detail"]; change="추가: "+best["변경"]
        history.append({"차수":version,"단계":"전진추가","변경":change,**current}); save(version,"전진추가",change,data,bundles,added,removed,current,detail,history); print("ACCEPT",version,change,current,flush=True)
    else:
        pair=[]; top=ordered[:10]
        for i,a in enumerate(top):
            for b in top[i+1:]:
                m,d=evaluate(data,bundles,[a,b],removed); pair.append({"변경":f"{a} + {b}","features":[a,b],**m,"detail":d})
        pair.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
        pd.DataFrame([{k:v for k,v in r.items() if k not in {"detail","features"}} for r in pair]).to_excel(OUT/"forward_pairs.xlsx",index=False)
        if pair and better(pair[0],current):
            best=pair[0]; added=best["features"]; version+=1; current={k:best[k] for k in current}; detail=best["detail"]; change="추가: "+best["변경"]
            history.append({"차수":version,"단계":"전진추가","변경":change,**current}); save(version,"전진추가",change,data,bundles,added,removed,current,detail,history); print("ACCEPT",version,change,current,flush=True)
    # Backward refinement. Re-rank low-importance variables after every accepted deletion.
    for round_no in range(1,9):
        order=[f for f in importance_order(bundles) if f not in removed]
        removal=[]
        for feature in order[:24]:
            m,d=evaluate(data,bundles,added,removed|{feature}); removal.append({"변경":feature,**m,"detail":d})
        removal.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
        pd.DataFrame([{k:v for k,v in r.items() if k!="detail"} for r in removal]).to_excel(OUT/f"backward_{round_no:02d}.xlsx",index=False)
        if not removal or not better(removal[0],current): break
        best=removal[0]; feature=best["변경"]; removed.add(feature); version+=1; current={k:best[k] for k in current}; detail=best["detail"]; change="제거: "+feature
        history.append({"차수":version,"단계":"후진제거","변경":change,**current}); save(version,"후진제거",change,data,bundles,added,removed,current,detail,history); print("ACCEPT",version,change,current,flush=True)
    pd.DataFrame(history).to_excel(OUT/"52차부터_최종차수_개선이력.xlsx",index=False)
    print("FINAL",json.dumps({"version":version,"added":added,"removed":sorted(removed),"metrics":current},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
