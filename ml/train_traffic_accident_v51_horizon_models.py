from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.improve_traffic_accident_v51_advanced import DATA, MODEL_DIR, add_features, grades_from_risk


REPORT = MODEL_DIR / "51차_2027_2028_직접예측모델평가.xlsx"
MODEL_2027 = MODEL_DIR / "교통사고_51차_2027년_2년후_직접예측모델.joblib"
MODEL_2028 = MODEL_DIR / "교통사고_51차_2028년_3년후_직접예측모델.joblib"


def scaled(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])


def models():
    result = {f"Ridge_{a:g}": scaled(Ridge(alpha=a)) for a in (.1, 1, 3, 10, 30, 100, 300, 1000)}
    for depth in (1, 2, 3):
        for leaf in (2, 4, 7):
            result[f"GBR_d{depth}_leaf{leaf}"] = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=.025, max_depth=depth,
                                                       min_samples_leaf=leaf, loss="huber", random_state=42)),
            ])
    for leaf in (3, 6, 10):
        result[f"ExtraTrees_leaf{leaf}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=leaf, max_features=.7,
                                            random_state=42, n_jobs=1)),
        ])
        result[f"RandomForest_leaf{leaf}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=leaf, max_features=.7,
                                              random_state=42, n_jobs=1)),
        ])
    return result


def prepare() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    raw = pd.read_excel(DATA, sheet_name="51차점수등급데이터")
    data, feature_sets = add_features(raw)
    data = data.sort_values(["자치구", "연도"]).copy()
    group = data.groupby("자치구", sort=False)
    for horizon in (2, 3):
        data[f"{horizon}년후_종합위험점수"] = group["현재연도_종합위험점수"].shift(-horizon)
        data[f"{horizon}년후_위험등급"] = group["현재연도_위험등급"].shift(-horizon)
        data[f"{horizon}년후_목표연도"] = data["연도"] + horizon
    return data.sort_values(["연도", "자치구"]).reset_index(drop=True), feature_sets


def evaluate_horizon(data: pd.DataFrame, feature_sets: dict[str, list[str]], horizon: int):
    score_target = f"{horizon}년후_종합위험점수"; grade_target = f"{horizon}년후_위험등급"; year_target = f"{horizon}년후_목표연도"
    labeled = data[data[grade_target].notna()].copy()
    cv_years = tuple(range(2021 if horizon == 2 else 2022, 2026))
    rows, risks = [], {}
    selected_sets = {k:v for k,v in feature_sets.items() if k in ("history","scores_history","raw_history","all","all_district")}
    for fs_name, features in selected_sets.items():
        for model_name, estimator in models().items():
            for target_kind in ("score", "delta"):
                for target_year in cv_years:
                    train = labeled[labeled[year_target] < target_year]
                    valid = labeled[labeled[year_target] == target_year]
                    if train.empty or valid.empty: continue
                    y = train[score_target].to_numpy().copy()
                    if target_kind == "delta": y -= train["현재연도_종합위험점수"].to_numpy()
                    fitted = deepcopy(estimator).fit(train[features], y)
                    pred = fitted.predict(valid[features])
                    if target_kind == "delta": pred += valid["현재연도_종합위험점수"].to_numpy()
                    current = valid["현재연도_종합위험점수"].to_numpy()
                    actual = valid[grade_target].astype(int).to_numpy()
                    for blend in (0, .2, .4, .6, .8, 1.0):
                        risk = (1-blend)*pred + blend*current
                        grade = grades_from_risk(risk)
                        key = f"{model_name}|{target_kind}|b{blend:g}"
                        rows.append([horizon,key,fs_name,target_year,accuracy_score(actual,grade),np.mean(np.abs(actual-grade)<=1),np.mean(np.abs(actual-grade))])
                        risks[(key,fs_name,target_year)] = risk
    detail = pd.DataFrame(rows, columns=["예측간격","모델","변수세트","검증목표연도","정확등급정확도","±1등급정확도","평균등급오차"])
    summary = detail.groupby(["예측간격","모델","변수세트"],as_index=False).agg(
        exact=("정확등급정확도","mean"), min_exact=("정확등급정확도","min"),
        within1=("±1등급정확도","mean"), error=("평균등급오차","mean")).rename(columns={
            "exact":"정확등급정확도","min_exact":"최저연도정확도","within1":"±1등급정확도","error":"평균등급오차"})
    summary = summary.sort_values(["정확등급정확도","±1등급정확도","평균등급오차"],ascending=[False,False,True]).reset_index(drop=True)
    # 상위 단일 모델을 결합해 장기예측 순위를 안정화한다.
    top = [(r["모델"],r["변수세트"]) for _,r in summary.head(25).iterrows()]
    ens=[]
    for i,a in enumerate(top):
        for b in top[i+1:]:
            for wa in (.25,.5,.75):
                name=f"ENS[{a[0]}@{a[1]}]+[{b[0]}@{b[1]}]|w{wa:g}"
                for year in cv_years:
                    ra=risks.get((a[0],a[1],year)); rb=risks.get((b[0],b[1],year))
                    if ra is None or rb is None: continue
                    valid=labeled[labeled[year_target]==year]; actual=valid[grade_target].astype(int).to_numpy()
                    grade=grades_from_risk(wa*ra+(1-wa)*rb)
                    ens.append([horizon,name,"ensemble",year,accuracy_score(actual,grade),np.mean(np.abs(actual-grade)<=1),np.mean(np.abs(actual-grade))])
    if ens: detail=pd.concat([detail,pd.DataFrame(ens,columns=detail.columns)],ignore_index=True)
    summary = detail.groupby(["예측간격","모델","변수세트"],as_index=False).agg(
        exact=("정확등급정확도","mean"), min_exact=("정확등급정확도","min"),
        within1=("±1등급정확도","mean"), error=("평균등급오차","mean")).rename(columns={
            "exact":"정확등급정확도","min_exact":"최저연도정확도","within1":"±1등급정확도","error":"평균등급오차"})
    summary["선정점수"] = summary["정확등급정확도"] + .08*summary["최저연도정확도"] + .03*summary["±1등급정확도"] - .01*summary["평균등급오차"]
    summary=summary.sort_values(["정확등급정확도","±1등급정확도","평균등급오차"],ascending=[False,False,True]).reset_index(drop=True)
    return labeled, detail, summary


def fit_spec(spec: str, fs: str, labeled: pd.DataFrame, feature_sets: dict[str,list[str]], horizon: int):
    base,target_kind,blend_text=spec.split("|"); features=feature_sets[fs]
    y=labeled[f"{horizon}년후_종합위험점수"].to_numpy().copy()
    if target_kind=="delta": y-=labeled["현재연도_종합위험점수"].to_numpy()
    fitted=models()[base].fit(labeled[features],y)
    return {"model_name":base,"target_kind":target_kind,"blend":float(blend_text[1:]),"feature_set":fs,"features":features,"model":fitted}


def fit_winner(best: pd.Series, labeled: pd.DataFrame, feature_sets: dict[str,list[str]], horizon: int):
    import re
    name,fs=best["모델"],best["변수세트"]
    if fs!="ensemble": return {"kind":"single", "component":fit_spec(name,fs,labeled,feature_sets,horizon)}
    m=re.fullmatch(r"ENS\[(.+)@([^@]+)\]\+\[(.+)@([^@]+)\]\|w([0-9.]+)",name)
    sa,fa,sb,fb,w=m.groups()
    return {"kind":"ensemble","weight_a":float(w),"component_a":fit_spec(sa,fa,labeled,feature_sets,horizon),"component_b":fit_spec(sb,fb,labeled,feature_sets,horizon)}


def predict_component(component: dict, frame: pd.DataFrame):
    pred=component["model"].predict(frame[component["features"]]); current=frame["현재연도_종합위험점수"].to_numpy()
    if component["target_kind"]=="delta": pred+=current
    return (1-component["blend"])*pred+component["blend"]*current


def predict(bundle: dict, frame: pd.DataFrame):
    if bundle["kind"]=="single": return predict_component(bundle["component"],frame)
    return bundle["weight_a"]*predict_component(bundle["component_a"],frame)+(1-bundle["weight_a"])*predict_component(bundle["component_b"],frame)


def main():
    data,feature_sets=prepare(); all_summary=[]; all_detail=[]; forecasts=[]
    for horizon,year,path in ((2,2027,MODEL_2027),(3,2028,MODEL_2028)):
        labeled,detail,summary=evaluate_horizon(data,feature_sets,horizon); best=summary.iloc[0]
        bundle=fit_winner(best,labeled,feature_sets,horizon); bundle.update({"version":51,"horizon":horizon,"forecast_year":year,"best_model":best.to_dict()})
        joblib.dump(bundle,path)
        future=data[data["연도"]==2025].copy(); risk=predict(bundle,future); grade=grades_from_risk(risk)
        forecast=future[["자치구","현재연도_종합위험점수","현재연도_위험등급"]].copy()
        forecast["예측연도"]=year; forecast["예측위험점수"]=risk; forecast["예측위험등급"]=grade
        forecast["예측위험순위"]=pd.Series(risk).rank(ascending=False,method="first").astype(int).to_numpy(); forecasts.append(forecast)
        all_summary.append(summary); all_detail.append(detail)
    summary=pd.concat(all_summary,ignore_index=True); detail=pd.concat(all_detail,ignore_index=True); forecast=pd.concat(forecasts,ignore_index=True)
    with pd.ExcelWriter(REPORT,engine="openpyxl") as writer:
        summary.to_excel(writer,sheet_name="전체후보비교",index=False); detail.to_excel(writer,sheet_name="연도별검증",index=False)
        summary.groupby("예측간격",as_index=False).head(1).to_excel(writer,sheet_name="최종모델성능",index=False)
        forecast.to_excel(writer,sheet_name="2027_2028예측",index=False)
    final=summary.groupby("예측간격",as_index=False).head(1)
    with pd.ExcelWriter(DATA,engine="openpyxl",mode="a",if_sheet_exists="replace") as writer:
        final.to_excel(writer,sheet_name="51차_장기예측성능",index=False); forecast.to_excel(writer,sheet_name="51차_2027_2028예측",index=False)
        data.to_excel(writer,sheet_name="51차_2년3년목표데이터",index=False)
    print(final.to_string(index=False)); print(forecast.sort_values(["예측연도","예측위험순위"]).head(10).to_string(index=False))


if __name__=="__main__": main()
