from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"reports"/"traffic_v52_techniques"
MODEL_OUT=ROOT/"models"/"교통사고"/"52차"


def scaled(model): return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",model)])


def model_defs():
    result={}
    for c in (.01,.03,.1,.3,1,3,10,30):
        result[f"Logistic_C{c:g}"]=scaled(LogisticRegression(C=c,max_iter=5000,class_weight="balanced",random_state=42))
    for a in (.1,1,3,10,30,100,300,1000): result[f"RidgeClassifier_{a:g}"]=scaled(RidgeClassifier(alpha=a,class_weight="balanced"))
    for c in (.03,.1,.3,1,3,10):
        result[f"SVC_linear_{c:g}"]=scaled(SVC(C=c,kernel="linear",class_weight="balanced",probability=True,random_state=42))
        result[f"SVC_rbf_{c:g}"]=scaled(SVC(C=c,kernel="rbf",class_weight="balanced",probability=True,random_state=42))
    for leaf in (2,4,7,10):
        result[f"ExtraTrees_{leaf}"]=Pipeline([("impute",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=leaf,max_features=.7,class_weight="balanced",random_state=42,n_jobs=1))])
        result[f"RandomForest_{leaf}"]=Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=leaf,max_features=.7,class_weight="balanced",random_state=42,n_jobs=1))])
    for depth in (1,2,3): result[f"GBC_{depth}"]=Pipeline([("impute",SimpleImputer(strategy="median")),("model",GradientBoostingClassifier(n_estimators=200,learning_rate=.03,max_depth=depth,random_state=42))])
    return result


def risk(model, x):
    if hasattr(model,"predict_proba"):
        p=model.predict_proba(x); classes=model.classes_.astype(float); return -(p@classes)
    d=model.decision_function(x); classes=model.classes_.astype(float)
    if d.ndim==1: return d
    d=d-d.max(axis=1,keepdims=True); p=np.exp(d); p/=p.sum(axis=1,keepdims=True); return -(p@classes)


def feature_sets(data,candidates):
    base_history=["현재연도_종합위험점수","현재연도_위험등급","위험점수_lag1","위험등급_lag1","위험점수_lag2","위험등급_lag2","위험점수_lag3","위험등급_lag3","점수_증감1","점수_증감2","점수_평균2","점수_평균3","점수_변동3"]
    raw13=list(pd.read_excel(ROOT/"data"/"processed"/"교통사고_51차_최종모델데이터.xlsx",sheet_name="변수가중치")["변수"])
    scores=[f"점수_{f}" for f in raw13]
    districts=[c for c in data.columns if c.startswith("구_")]
    screen=pd.read_csv(ROOT/"artifacts"/"reports"/"traffic_v52_search"/"forward_01_screen.csv")
    top10=list(screen.head(10).후보변수); top25=list(screen.head(25).후보변수)
    return {
        "history":base_history,
        "raw_history":raw13+base_history,
        "scores_history":scores+base_history,
        "base_all":raw13+scores+base_history,
        "base_district":raw13+scores+base_history+districts,
        "base_top10":raw13+scores+base_history+top10,
        "base_top25":raw13+scores+base_history+top25,
        "master_history":candidates+base_history,
        "master_district":candidates+base_history+districts,
    }


def metrics(actual,pred): return accuracy_score(actual,pred),float(np.mean(np.abs(actual-pred)<=1)),float(np.mean(np.abs(actual-pred)))


def main():
    OUT.mkdir(parents=True,exist_ok=True); data,candidates=prepare_data(); sets=feature_sets(data,candidates); defs=model_defs()
    all_detail=[]; fitted_specs={}; oof={}
    for horizon in (2,3):
        grade_target=f"{horizon}년후_위험등급"; year_col=f"{horizon}년후_목표연도"; labeled=data[data[grade_target].notna()]
        for fs_name,features in sets.items():
            for model_name,estimator in defs.items():
                config=f"{model_name}@{fs_name}"; fold_risk={}
                for year in CV_YEARS[horizon]:
                    tr=labeled[labeled[year_col]<year]; va=labeled[labeled[year_col]==year]
                    fitted=clone(estimator).fit(tr[features],tr[grade_target].astype(int)); r=risk(fitted,va[features]); fold_risk[year]=r
                    actual=va[grade_target].astype(int).to_numpy(); direct=fitted.predict(va[features]).astype(int); ranked=grades_from_risk(r)
                    for method,pred in (("direct",direct),("rank",ranked)):
                        m=metrics(actual,pred); all_detail.append([horizon,config,method,year,*m])
                oof[(horizon,config)]=fold_risk; fitted_specs[(horizon,config)]=(estimator,features)
    detail=pd.DataFrame(all_detail,columns=["예측간격","모델","방식","검증목표연도","정확등급정확도","±1등급정확도","평균등급오차"])
    summary=detail.groupby(["예측간격","모델","방식"],as_index=False).agg(exact=("정확등급정확도","mean"),within1=("±1등급정확도","mean"),error=("평균등급오차","mean"),min_exact=("정확등급정확도","min"))
    # Rank ensembles among the 30 best models for each horizon.
    ensemble_detail=[]
    for horizon in (2,3):
        top=list(summary[(summary.예측간격==horizon)&(summary.방식=="rank")].sort_values(["exact","within1","error"],ascending=[False,False,True]).head(30).모델)
        grade_target=f"{horizon}년후_위험등급"; year_col=f"{horizon}년후_목표연도"; labeled=data[data[grade_target].notna()]
        for i,a in enumerate(top):
            for b in top[i+1:]:
                for wa in (.2,.35,.5,.65,.8):
                    name=f"ENS[{a}]+[{b}]|w{wa:g}"
                    for year in CV_YEARS[horizon]:
                        va=labeled[labeled[year_col]==year]; actual=va[grade_target].astype(int).to_numpy()
                        pred=grades_from_risk(wa*oof[(horizon,a)][year]+(1-wa)*oof[(horizon,b)][year]); m=metrics(actual,pred)
                        ensemble_detail.append([horizon,name,"rank",year,*m])
    detail=pd.concat([detail,pd.DataFrame(ensemble_detail,columns=detail.columns)],ignore_index=True)
    summary=detail.groupby(["예측간격","모델","방식"],as_index=False).agg(exact=("정확등급정확도","mean"),within1=("±1등급정확도","mean"),error=("평균등급오차","mean"),min_exact=("정확등급정확도","min"))
    summary=summary.sort_values(["예측간격","exact","within1","error"],ascending=[True,False,False,True]).reset_index(drop=True)
    winners=summary.groupby("예측간격",as_index=False).head(1).copy()
    with pd.ExcelWriter(OUT/"기법전체탐색.xlsx",engine="openpyxl") as writer:
        winners.to_excel(writer,sheet_name="최종후보",index=False); summary.to_excel(writer,sheet_name="전체요약",index=False); detail.to_excel(writer,sheet_name="연도별검증",index=False)
    avg_exact=float(winners.exact.mean())
    print(winners.to_string(index=False),flush=True)
    if avg_exact<=.555+1e-12:
        print(json.dumps({"accepted":False,"average_exact":avg_exact},ensure_ascii=False)); return
    MODEL_OUT.mkdir(parents=True,exist_ok=True)
    for _,winner in winners.iterrows():
        horizon=int(winner.예측간격); name=winner.모델; grade_target=f"{horizon}년후_위험등급"; labeled=data[data[grade_target].notna()]
        def fit_one(config):
            estimator,features=fitted_specs[(horizon,config)]; return {"config":config,"features":features,"model":clone(estimator).fit(labeled[features],labeled[grade_target].astype(int))}
        if name.startswith("ENS["):
            import re
            m=re.fullmatch(r"ENS\[(.+)\]\+\[(.+)\]\|w([0-9.]+)",name); a,b,w=m.groups(); bundle={"kind":"classifier_ensemble_rank","weight_a":float(w),"component_a":fit_one(a),"component_b":fit_one(b)}
        else: bundle={"kind":"classifier_rank" if winner.방식=="rank" else "classifier_direct","component":fit_one(name)}
        bundle.update({"version":52,"horizon":horizon,"forecast_year":2025+horizon,"metrics":winner.to_dict()})
        joblib.dump(bundle,MODEL_OUT/f"교통사고_52차_{2025+horizon}년_{horizon}년후_모델.joblib")
    info={"모델명":"교통사고 52차 장기등급 모델","기반":"51차","개선방식":"전체 변수군 + 분류기법 + 순위앙상블 탐색","성능":{"2년후":winners.iloc[0].to_dict(),"3년후":winners.iloc[1].to_dict(),"평균정확등급":avg_exact}}
    (MODEL_OUT/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"accepted":True,"version":52,"average_exact":avg_exact},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
