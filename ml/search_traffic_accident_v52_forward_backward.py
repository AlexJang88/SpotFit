from __future__ import annotations

"""Build model versions after v51 with forward addition and backward pruning.

Every raw candidate is screened with leakage-safe rolling validation.  The best
candidates are then rechecked with the exact v51 two-component ensembles.  Each
accepted improvement is persisted as the next model version.
"""

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
V51 = ROOT / "models" / "교통사고" / "51차"
PROCESSED = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
REPORT_ROOT = ROOT / "artifacts" / "reports" / "traffic_v52_search"
CHECKPOINT = REPORT_ROOT / "search_checkpoint.json"
BASE_MODELS = {
    2: V51 / "교통사고_51차_2027년_2년후_직접예측모델.joblib",
    3: V51 / "교통사고_51차_2028년_3년후_직접예측모델.joblib",
}
CV_YEARS = {2: (2021, 2022, 2023, 2024, 2025), 3: (2022, 2023, 2024, 2025)}
EXCLUDE_PARTS = ("다음연도", "2년후", "3년후", "예측연도", "목표연도")


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_excel(PROCESSED, sheet_name="51차_2년3년목표데이터")
    master = prepare_panel()
    existing = set(base.columns)
    candidates = []
    for column in master.columns:
        if column in {"연도", "자치구"} or column in existing or any(part in column for part in EXCLUDE_PARTS):
            continue
        numeric = pd.to_numeric(master[column], errors="coerce")
        if numeric.notna().mean() < .75 or numeric.nunique(dropna=True) <= 1:
            continue
        master[column] = numeric
        candidates.append(column)
    merged = base.merge(master[["연도", "자치구", *candidates]], on=["연도", "자치구"], how="left", validate="one_to_one")
    return merged.sort_values(["연도", "자치구"]).reset_index(drop=True), candidates


def component_prediction(component: dict, train: pd.DataFrame, valid: pd.DataFrame, target: str, extra: list[str], removed: set[str]):
    features = [f for f in component["features"] if f not in removed]
    features.extend(f for f in extra if f not in features and f not in removed)
    y = train[target].to_numpy().copy()
    if component["target_kind"] == "delta":
        y -= train["현재연도_종합위험점수"].to_numpy()
    estimator = clone(component["model"])
    estimator.fit(train[features], y)
    prediction = estimator.predict(valid[features])
    current = valid["현재연도_종합위험점수"].to_numpy()
    if component["target_kind"] == "delta":
        prediction += current
    prediction = (1-component["blend"])*prediction + component["blend"]*current
    return prediction


def fold_prediction(bundle: dict, train: pd.DataFrame, valid: pd.DataFrame, target: str, extra: list[str], removed: set[str]):
    if bundle["kind"] == "single":
        return component_prediction(bundle["component"], train, valid, target, extra, removed)
    a = component_prediction(bundle["component_a"], train, valid, target, extra, removed)
    b = component_prediction(bundle["component_b"], train, valid, target, extra, removed)
    return bundle["weight_a"]*a + (1-bundle["weight_a"])*b


def evaluate_exact(data: pd.DataFrame, bundles: dict[int, dict], extra: list[str], removed: set[str] | None = None):
    removed = removed or set(); rows=[]
    for horizon,bundle in bundles.items():
        target=f"{horizon}년후_종합위험점수"; grade_target=f"{horizon}년후_위험등급"; year_col=f"{horizon}년후_목표연도"
        labeled=data[data[grade_target].notna()].copy()
        for year in CV_YEARS[horizon]:
            train=labeled[labeled[year_col]<year]; valid=labeled[labeled[year_col]==year]
            risk=fold_prediction(bundle,train,valid,target,extra,removed)
            pred=grades_from_risk(risk); actual=valid[grade_target].astype(int).to_numpy()
            rows.append({"예측간격":horizon,"검증목표연도":year,"정확등급정확도":accuracy_score(actual,pred),
                         "±1등급정확도":float(np.mean(np.abs(actual-pred)<=1)),"평균등급오차":float(np.mean(np.abs(actual-pred)))})
    detail=pd.DataFrame(rows)
    summary=detail.groupby("예측간격",as_index=False).agg(exact=("정확등급정확도","mean"),within1=("±1등급정확도","mean"),error=("평균등급오차","mean"),min_exact=("정확등급정확도","min"))
    metrics={
        "exact_avg":float(summary.exact.mean()), "within1_avg":float(summary.within1.mean()),
        "error_avg":float(summary.error.mean()), "min_exact":float(summary.min_exact.min()),
        "h2_exact":float(summary.loc[summary.예측간격.eq(2),"exact"].iloc[0]),
        "h3_exact":float(summary.loc[summary.예측간격.eq(3),"exact"].iloc[0]),
        "h2_within1":float(summary.loc[summary.예측간격.eq(2),"within1"].iloc[0]),
        "h3_within1":float(summary.loc[summary.예측간격.eq(3),"within1"].iloc[0]),
    }
    return metrics,detail


def better(candidate: dict, current: dict) -> bool:
    if candidate["exact_avg"] > current["exact_avg"] + 1e-12: return True
    if abs(candidate["exact_avg"]-current["exact_avg"]) < 1e-12 and candidate["within1_avg"] > current["within1_avg"] + .004: return True
    if abs(candidate["exact_avg"]-current["exact_avg"]) < 1e-12 and abs(candidate["within1_avg"]-current["within1_avg"]) < 1e-12 and candidate["error_avg"] < current["error_avg"]-1e-12: return True
    return False


def screen(data: pd.DataFrame, candidates: list[str], selected: list[str]):
    base_features=["현재연도_종합위험점수","현재연도_위험등급","위험점수_lag1","위험점수_lag2","점수_증감1","점수_평균3"]
    rows=[]
    for candidate in candidates:
        features=[*base_features,*selected,candidate]; horizon_scores=[]; within=[]; errors=[]
        for horizon in (2,3):
            target=f"{horizon}년후_종합위험점수"; grade_target=f"{horizon}년후_위험등급"; year_col=f"{horizon}년후_목표연도"
            labeled=data[data[grade_target].notna()]
            best_h=None
            for alpha in (1,10,100,300):
                fold=[]
                for year in CV_YEARS[horizon]:
                    tr=labeled[labeled[year_col]<year]; va=labeled[labeled[year_col]==year]
                    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
                    model.fit(tr[features],tr[target]); risk=.6*model.predict(va[features])+.4*va["현재연도_종합위험점수"].to_numpy()
                    pred=grades_from_risk(risk); actual=va[grade_target].astype(int).to_numpy()
                    fold.append((accuracy_score(actual,pred),np.mean(np.abs(actual-pred)<=1),np.mean(np.abs(actual-pred))))
                aggregate=np.mean(fold,axis=0)
                if best_h is None or tuple([aggregate[0],aggregate[1],-aggregate[2]])>tuple([best_h[0],best_h[1],-best_h[2]]): best_h=aggregate
            horizon_scores.append(best_h[0]); within.append(best_h[1]); errors.append(best_h[2])
        rows.append({"후보변수":candidate,"screen_exact":float(np.mean(horizon_scores)),"screen_within1":float(np.mean(within)),"screen_error":float(np.mean(errors))})
    return pd.DataFrame(rows).sort_values(["screen_exact","screen_within1","screen_error"],ascending=[False,False,True]).reset_index(drop=True)


def train_full_component(component: dict, data: pd.DataFrame, horizon: int, extra: list[str], removed: set[str]):
    target=f"{horizon}년후_종합위험점수"; labeled=data[data[target].notna()]
    features=[f for f in component["features"] if f not in removed]
    features.extend(f for f in extra if f not in features and f not in removed)
    y=labeled[target].to_numpy().copy()
    if component["target_kind"]=="delta": y-=labeled["현재연도_종합위험점수"].to_numpy()
    fitted=clone(component["model"]).fit(labeled[features],y)
    result={k:v for k,v in component.items() if k not in {"model","features"}}
    result.update({"features":features,"model":fitted})
    return result


def train_full_bundle(bundle: dict, data: pd.DataFrame, horizon: int, extra: list[str], removed: set[str]):
    if bundle["kind"]=="single": return {**bundle,"component":train_full_component(bundle["component"],data,horizon,extra,removed)}
    return {**bundle,"component_a":train_full_component(bundle["component_a"],data,horizon,extra,removed),"component_b":train_full_component(bundle["component_b"],data,horizon,extra,removed)}


def save_version(version: int, phase: str, change: str, data: pd.DataFrame, base_bundles: dict[int,dict], selected: list[str], removed: set[str], metrics: dict, detail: pd.DataFrame, history: list[dict]):
    out=ROOT/"models"/"교통사고"/f"{version}차"; out.mkdir(parents=True,exist_ok=True)
    for horizon,year in ((2,2027),(3,2028)):
        fitted=train_full_bundle(base_bundles[horizon],data,horizon,selected,removed)
        fitted.update({"version":version,"horizon":horizon,"forecast_year":year,"selected_extra_features":selected,"removed_features":sorted(removed)})
        joblib.dump(fitted,out/f"교통사고_{version}차_{year}년_{horizon}년후_모델.joblib")
    info={"모델명":f"교통사고 {version}차 장기등급 모델","기반":"51차","단계":phase,"변경":change,
          "추가변수":selected,"제거변수":sorted(removed),"성능":metrics,"검증":"목표연도 기준 순차검증"}
    (out/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    with pd.ExcelWriter(out/f"{version}차_검증결과.xlsx",engine="openpyxl") as writer:
        detail.to_excel(writer,sheet_name="연도별검증",index=False); pd.DataFrame(history).to_excel(writer,sheet_name="개선이력",index=False)


def main():
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    data,candidates=prepare_data(); base_bundles={h:joblib.load(p) for h,p in BASE_MODELS.items()}
    selected=[]; removed=set(); version=51; history=[]
    current,current_detail=evaluate_exact(data,base_bundles,selected,removed)
    history.append({"차수":51,"단계":"기준","변경":"51차 기준","추가변수":"","제거변수":"",**current})
    print("BASE",json.dumps(current,ensure_ascii=False),flush=True)

    # Forward: screen every remaining variable, then fully verify the top 18.
    for round_no in range(1,11):
        remaining=[c for c in candidates if c not in selected and c not in removed]
        screening=screen(data,remaining,selected); screening.to_csv(REPORT_ROOT/f"forward_{round_no:02d}_screen.csv",index=False,encoding="utf-8-sig")
        full=[]
        for candidate in screening.head(18).후보변수:
            metrics,detail=evaluate_exact(data,base_bundles,[*selected,candidate],removed)
            full.append({"후보변수":candidate,**metrics,"detail":detail})
        full.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
        best=full[0]
        pd.DataFrame([{k:v for k,v in row.items() if k!="detail"} for row in full]).to_excel(REPORT_ROOT/f"forward_{round_no:02d}_full.xlsx",index=False)
        if not better(best,current):
            # Individual additions stalled: evaluate pairs among the best screen candidates.
            pair_rows=[]; top=list(screening.head(10).후보변수)
            for i,a in enumerate(top):
                for b in top[i+1:]:
                    metrics,detail=evaluate_exact(data,base_bundles,[*selected,a,b],removed)
                    pair_rows.append({"후보변수":f"{a} + {b}","pair":[a,b],**metrics,"detail":detail})
            pair_rows.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
            pd.DataFrame([{k:v for k,v in row.items() if k not in {"detail","pair"}} for row in pair_rows]).to_excel(REPORT_ROOT/f"forward_{round_no:02d}_pairs.xlsx",index=False)
            if not pair_rows or not better(pair_rows[0],current): break
            chosen=pair_rows[0]["pair"]; best=pair_rows[0]
        else: chosen=[best["후보변수"]]
        selected.extend(chosen); version+=1; current={k:best[k] for k in current}; current_detail=best["detail"]
        change="추가: "+", ".join(chosen); history.append({"차수":version,"단계":"전진추가","변경":change,"추가변수":" | ".join(selected),"제거변수":"",**current})
        save_version(version,"전진추가",change,data,base_bundles,selected,removed,current,current_detail,history)
        CHECKPOINT.write_text(json.dumps({"version":version,"selected":selected,"removed":sorted(removed),"metrics":current,"history":history},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"ACCEPT V{version}",change,json.dumps(current,ensure_ascii=False),flush=True)

    # Backward: remove one feature group at a time, keeping only improvements.
    removable=set(selected)
    for bundle in base_bundles.values():
        components=[bundle["component"]] if bundle["kind"]=="single" else [bundle["component_a"],bundle["component_b"]]
        for component in components: removable.update(f for f in component["features"] if not f.startswith("구_"))
    for round_no in range(1,16):
        trials=[]
        for feature in sorted(removable-removed):
            metrics,detail=evaluate_exact(data,base_bundles,selected,removed|{feature})
            trials.append({"제거후보":feature,**metrics,"detail":detail})
        trials.sort(key=lambda r:(r["exact_avg"],r["within1_avg"],-r["error_avg"],r["min_exact"]),reverse=True)
        pd.DataFrame([{k:v for k,v in row.items() if k!="detail"} for row in trials]).to_excel(REPORT_ROOT/f"backward_{round_no:02d}.xlsx",index=False)
        if not trials or not better(trials[0],current): break
        best=trials[0]; feature=best["제거후보"]; removed.add(feature); version+=1; current={k:best[k] for k in current}; current_detail=best["detail"]
        change="제거: "+feature; history.append({"차수":version,"단계":"후진제거","변경":change,"추가변수":" | ".join(selected),"제거변수":" | ".join(sorted(removed)),**current})
        save_version(version,"후진제거",change,data,base_bundles,selected,removed,current,current_detail,history)
        CHECKPOINT.write_text(json.dumps({"version":version,"selected":selected,"removed":sorted(removed),"metrics":current,"history":history},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"ACCEPT V{version}",change,json.dumps(current,ensure_ascii=False),flush=True)

    pd.DataFrame(history).to_excel(REPORT_ROOT/"51차부터_최종차수_개선이력.xlsx",index=False)
    print("FINAL",json.dumps({"version":version,"selected":selected,"removed":sorted(removed),"metrics":current},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
