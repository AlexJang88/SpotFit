"""Optimize 3-year child crime exposure specifically for borough ranking.

The crime and child-population components are trained separately. Candidate
selection uses only target years 2022-2024; 2025 remains a locked holdout.
"""
from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training import compare_child_crime_borough_v2 as old
from ml.training import optimize_borough_crime_count_3y as crime_new

OUT = ROOT / "artifacts" / "child_crime_exposure_rank_3y_v1"
MODEL_OUT = ROOT / "models" / "experiments" / "child_crime_exposure_rank_3y_v1"
CV_YEARS = (2022, 2023, 2024)
HOLDOUT = 2025
HORIZON = 3


def exposure(log_crime: np.ndarray, log_child: np.ndarray) -> np.ndarray:
    crime = np.maximum(0, np.expm1(log_crime))
    child = np.maximum(1, np.expm1(log_child))
    return np.log1p(crime / child * 10000)


def rank_metrics(actual: np.ndarray, predicted: np.ndarray, years: np.ndarray) -> dict[str, float]:
    correlations, errors, grades, top5, top10 = [], [], [], [], []
    for year in np.unique(years):
        mask = years == year
        a, p = actual[mask], predicted[mask]
        # Rank 1 is highest exposure/risk.
        ar, pr = rankdata(-a, method="ordinal"), rankdata(-p, method="ordinal")
        correlations.append(spearmanr(a, p).statistic)
        errors.extend(np.abs(ar - pr))
        grades.extend(np.floor((ar - 1) / 5) == np.floor((pr - 1) / 5))
        top5.append(len(set(np.argsort(-a)[:5]) & set(np.argsort(-p)[:5])) / 5)
        top10.append(len(set(np.argsort(-a)[:10]) & set(np.argsort(-p)[:10])) / 10)
    spearman = float(np.mean(correlations))
    rank_mae = float(np.mean(errors))
    exact = float(np.mean(grades) * 100)
    top5_pct = float(np.mean(top5) * 100)
    top10_pct = float(np.mean(top10) * 100)
    ranking_score = 100 * (.40 * ((spearman + 1) / 2) + .25 * (1 - rank_mae / 24) + .20 * (top5_pct / 100) + .15 * (exact / 100))
    return {
        "ranking_score": ranking_score, "Spearman": spearman, "rank_MAE": rank_mae,
        "exact_quintile_pct": exact, "top20_overlap_pct": top5_pct, "top40_overlap_pct": top10_pct,
        "y_RMSE": float(mean_squared_error(actual, predicted) ** .5),
        "y_MAE": float(mean_absolute_error(actual, predicted)),
    }


def prep(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)


def population_estimator(spec: str):
    name, value = spec.split(":", 1)
    values = [float(v) for v in value.split(",")]
    if name == "ridge": return Ridge(alpha=values[0])
    if name == "elastic": return ElasticNet(alpha=values[0], l1_ratio=values[1], max_iter=30000, random_state=42)
    if name == "extra": return ExtraTreesRegressor(n_estimators=120, max_depth=int(values[0]), min_samples_leaf=int(values[1]), max_features=values[2], n_jobs=1, random_state=42)
    if name == "forest": return RandomForestRegressor(n_estimators=140, max_depth=int(values[0]), min_samples_leaf=int(values[1]), max_features=values[2], n_jobs=1, random_state=42)
    if name == "hist": return HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=int(values[0]), min_samples_leaf=int(values[1]), l2_regularization=values[2], learning_rate=.04, early_stopping=False, random_state=42)
    if name == "gbr": return GradientBoostingRegressor(n_estimators=150, max_depth=int(values[0]), min_samples_leaf=int(values[1]), learning_rate=.03, loss="huber", random_state=42)
    raise ValueError(spec)


def fit_population(train: pd.DataFrame, test: pd.DataFrame, features: list[str], spec: str, mode: str):
    columns = ["gu", *features]
    model = Pipeline([("prep", prep(features)), ("model", population_estimator(spec))])
    if mode == "direct": target = train.target_log_child
    elif mode == "delta": target = train.target_log_child - train.log_child
    else: raise ValueError(mode)
    model.fit(train[columns], target)
    prediction = model.predict(test[columns])
    if mode == "delta": prediction += test.log_child.to_numpy()
    return prediction, model


def fitted_population_predict(model, frame: pd.DataFrame, features: list[str], mode: str) -> np.ndarray:
    result = model.predict(frame[["gu", *features]])
    return result + frame.log_child.to_numpy() if mode == "delta" else result


def crime_setup():
    panel = pd.read_csv(old.OUT / "panel.csv")
    groups = json.loads((old.OUT / "feature_groups.json").read_text(encoding="utf-8"))
    columns = list(dict.fromkeys(old.CORE + [c for values in groups.values() for c in values]))
    pairs = old.make_pairs(panel, 3)
    row = pd.read_csv(old.OUT / "3y" / "best_by_method_cv.csv").query("method == 'components'").iloc[0]
    old_features = ast.literal_eval(row.features)
    new_result = json.loads((crime_new.OUT / "result.json").read_text(encoding="utf-8"))
    new_winner = new_result["winner_cv"]
    new_panel = pd.read_csv(crime_new.BASE / "panel.csv")
    new_panel, _, type_counts = crime_new.add_crime_history(new_panel)
    new_pairs = crime_new.make_pairs(new_panel, type_counts)
    return panel, pairs, columns, old_features, float(row.alpha), new_panel, new_pairs, type_counts, new_winner, groups


def crime_predictions(years, pairs, columns, old_features, old_alpha, new_pairs, type_counts, new_winner):
    rows = []
    for target_year in years:
        origin = target_year - 3
        test = pairs[pairs.target_year == target_year].sort_values("gu")
        train = pairs[pairs.target_year <= origin]
        ctx = old.context(train, test, columns, sorted(train.gu.unique()))
        old_model = old.fit(ctx, [columns.index(c) for c in old_features], "components", old_alpha)
        _, old_detail = old.predict(old_model, test, 3)
        nt = new_pairs[new_pairs.target_year == target_year].sort_values("gu")
        ntrain = new_pairs[new_pairs.target_year <= origin]
        new_pred, _ = crime_new.fit_predict(ntrain, nt, new_winner["features"], new_winner["spec"], new_winner["mode"], type_counts)
        assert test.gu.tolist() == nt.gu.tolist()
        for gu, current, actual, op, npred in zip(test.gu, test.log_crime, test.log_crime_future, np.log1p(old_detail["crime"]), new_pred):
            rows.append({"gu": gu, "target_year": target_year, "current": current, "actual": actual, "old": op, "new": npred})
    result = pd.DataFrame(rows)
    result["ensemble84_log"] = .16 * result.old + .84 * result.new
    result["average_log"] = .5 * result.old + .5 * result.new
    return result


def population_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    future = panel[["gu", "year", "log_child", "y"]].copy()
    future.year -= 3
    future = future.rename(columns={"log_child": "target_log_child", "y": "target_y"})
    result = panel.merge(future, on=["gu", "year"], validate="one_to_one")
    result["target_year"] = result.year + 3
    return result[result.year >= 2015].sort_values(["year", "gu"]).reset_index(drop=True)


def evaluate_population(pairs, crime_cv, features, spec, mode):
    predicted, actual_child, actual_y, years, gus = [], [], [], [], []
    for target_year in CV_YEARS:
        origin = target_year - 3
        train = pairs[pairs.target_year <= origin]
        test = pairs[pairs.target_year == target_year].sort_values("gu")
        p, _ = fit_population(train, test, features, spec, mode)
        predicted.extend(p); actual_child.extend(test.target_log_child); actual_y.extend(test.target_y)
        years.extend(test.target_year); gus.extend(test.gu)
    pred = np.asarray(predicted); actual_child = np.asarray(actual_child); actual_y = np.asarray(actual_y); years = np.asarray(years)
    aligned = crime_cv.set_index(["target_year", "gu"]).loc[list(zip(years, gus))].reset_index()
    combinations = []
    for crime_name in ("current", "old", "new", "ensemble84_log", "average_log"):
        py = exposure(aligned[crime_name].to_numpy(), pred)
        combinations.append({"crime_candidate": crime_name, **rank_metrics(actual_y, py, years)})
    child_rmse = float(mean_squared_error(actual_child, pred) ** .5)
    child_mape = float(np.mean(np.abs(np.expm1(pred)-np.expm1(actual_child))/np.expm1(actual_child))*100)
    return combinations, pred, {"child_log_RMSE": child_rmse, "child_MAPE_pct": child_mape}


def fitted_crime_models(train_old, test_old, columns, old_features, old_alpha, train_new, test_new, type_counts, winner):
    ctx = old.context(train_old, test_old, columns, sorted(train_old.gu.unique()))
    om = old.fit(ctx, [columns.index(c) for c in old_features], "components", old_alpha)
    _, od = old.predict(om, test_old, 3)
    npred, nm = crime_new.fit_predict(train_new, test_new, winner["features"], winner["spec"], winner["mode"], type_counts)
    return {"old_model": om, "new_model": nm, "old_log": np.log1p(od["crime"]), "new_log": npred}


def predict_selected_crime(name, models, old_frame, new_frame, winner, type_counts):
    _, od = old.predict(models["old_model"], old_frame, 3)
    op = np.log1p(od["crime"])
    nm = models["new_model"]
    raw = nm.predict(new_frame[["gu", *winner["features"]]])
    if winner["mode"] == "type_sum": npred = np.log1p(np.maximum(0, np.expm1(raw)).sum(axis=1))
    elif winner["mode"] == "delta": npred = new_frame.log_crime.to_numpy() + raw
    else: npred = raw
    if name == "current": return old_frame.log_crime.to_numpy()
    if name == "old": return op
    if name == "new": return npred
    if name == "ensemble84_log": return .16 * op + .84 * npred
    if name == "average_log": return .5 * op + .5 * npred
    raise ValueError(name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); MODEL_OUT.mkdir(parents=True, exist_ok=True)
    panel, crime_pairs, crime_columns, old_features, old_alpha, new_panel, new_pairs, type_counts, new_winner, groups = crime_setup()
    pop_pairs = population_pairs(panel)
    crime_cv = crime_predictions(CV_YEARS, crime_pairs, crime_columns, old_features, old_alpha, new_pairs, type_counts, new_winner)
    crime_holdout = crime_predictions((HOLDOUT,), crime_pairs, crime_columns, old_features, old_alpha, new_pairs, type_counts, new_winner)
    names = list(groups)
    ridge_specs = [f"ridge:{a}" for a in (.01, .1, 1, 10, 100, 1000)]
    results, prediction_cache = [], {}
    print(f"population stage1 subsets={2**len(names)} configs={2**len(names)*len(ridge_specs)*2}", flush=True)
    for mask in range(1 << len(names)):
        active = [name for i, name in enumerate(names) if mask & (1 << i)]
        features = list(dict.fromkeys(["log_child", *[c for name in active for c in groups[name]]]))
        for spec in ridge_specs:
            for mode in ("direct", "delta"):
                combos, pred, child = evaluate_population(pop_pairs, crime_cv, features, spec, mode)
                key = (mask, spec, mode); prediction_cache[key] = pred
                for combo in combos:
                    results.append({"stage": "all_subsets_linear", "mask": mask, "groups": active, "features": features, "spec": spec, "mode": mode, **child, **combo})
        if (mask+1) % 64 == 0:
            best=max(results,key=lambda r:(r["ranking_score"],r["Spearman"],-r["rank_MAE"]));print(mask+1,best["ranking_score"],best["groups"],best["crime_candidate"],best["spec"],best["mode"],flush=True)

    # Nonlinear methods only on population subsets appearing in the top rank candidates or child-RMSE candidates.
    top_masks=[]
    for row in sorted(results,key=lambda r:(-r["ranking_score"],-r["Spearman"],r["rank_MAE"],r["child_log_RMSE"])) + sorted(results,key=lambda r:r["child_log_RMSE"]):
        if row["mask"] not in top_masks: top_masks.append(row["mask"])
        if len(top_masks)==24: break
    nonlinear=["elastic:.001,.1","elastic:.001,.5","elastic:.01,.1","elastic:.01,.5","extra:5,3,.7","extra:9,2,1.0","forest:6,3,.8","hist:7,10,5","gbr:2,5"]
    print("population stage2",len(top_masks),flush=True)
    for mask in top_masks:
        active=[name for i,name in enumerate(names) if mask&(1<<i)]
        features=list(dict.fromkeys(["log_child",*[c for name in active for c in groups[name]]]))
        for spec in nonlinear:
            for mode in ("direct","delta"):
                combos,pred,child=evaluate_population(pop_pairs,crime_cv,features,spec,mode)
                prediction_cache[(mask,spec,mode)]=pred
                for combo in combos: results.append({"stage":"shortlist_nonlinear","mask":mask,"groups":active,"features":features,"spec":spec,"mode":mode,**child,**combo})

    results.sort(key=lambda r:(-r["ranking_score"],-r["Spearman"],r["rank_MAE"],-r["top20_overlap_pct"],-r["exact_quintile_pct"],r["y_RMSE"]))
    winner=results[0]
    # Locked holdout evaluation.
    hold_test=pop_pairs[pop_pairs.target_year==HOLDOUT].sort_values("gu");hold_train=pop_pairs[pop_pairs.target_year<=HOLDOUT-3]
    child_hold,child_model=fit_population(hold_train,hold_test,winner["features"],winner["spec"],winner["mode"])
    ch=crime_holdout.sort_values("gu")
    assert ch.gu.tolist()==hold_test.gu.tolist()
    hold_pred_y=exposure(ch[winner["crime_candidate"]].to_numpy(),child_hold)
    hold_metrics=rank_metrics(hold_test.target_y.to_numpy(),hold_pred_y,hold_test.target_year.to_numpy())
    baseline_cv=[];baseline_hold=[]
    for target_year in CV_YEARS:
        t=pop_pairs[pop_pairs.target_year==target_year]
        baseline_cv.append(t)
    bc=pd.concat(baseline_cv);base_cv=rank_metrics(bc.target_y.to_numpy(),bc.y.to_numpy(),bc.target_year.to_numpy())
    base_hold=rank_metrics(hold_test.target_y.to_numpy(),hold_test.y.to_numpy(),hold_test.target_year.to_numpy())

    compact=pd.DataFrame([{k:v for k,v in row.items() if k not in ("features",)} for row in results])
    compact.to_csv(OUT/"all_rank_comparisons.csv",index=False,encoding="utf-8-sig")
    compact.drop_duplicates(["crime_candidate","spec","mode"]).to_csv(OUT/"best_by_method.csv",index=False,encoding="utf-8-sig")
    # Refit final operational components using every completed label through 2025.
    latest_old=panel[panel.year==2025].sort_values("gu").copy();latest_old["target_year"]=2028
    latest_new=new_panel[new_panel.year==2025].sort_values("gu").copy();latest_new["target_year"]=2028
    crime_models=fitted_crime_models(crime_pairs,latest_old,crime_columns,old_features,old_alpha,new_pairs,latest_new,type_counts,new_winner)
    final_crime=predict_selected_crime(winner["crime_candidate"],crime_models,latest_old,latest_new,new_winner,type_counts)
    final_child,pop_operational=fit_population(pop_pairs,latest_old,winner["features"],winner["spec"],winner["mode"])
    final_y=exposure(final_crime,final_child)

    # Model-agnostic integrated sensitivity: +1 historical SD, measured on CV folds.
    member_features=set(winner["features"])
    if winner["crime_candidate"] in ("old","ensemble84_log","average_log"): member_features.update(old_features)
    if winner["crime_candidate"] in ("new","ensemble84_log","average_log"): member_features.update(new_winner["features"])
    impacts={feature:[] for feature in member_features if feature in panel.columns or feature in new_panel.columns}
    for target_year in CV_YEARS:
        origin=target_year-3
        ot=crime_pairs[crime_pairs.target_year==target_year].sort_values("gu");otr=crime_pairs[crime_pairs.target_year<=origin]
        nt=new_pairs[new_pairs.target_year==target_year].sort_values("gu");ntr=new_pairs[new_pairs.target_year<=origin]
        pt=pop_pairs[pop_pairs.target_year==target_year].sort_values("gu");ptr=pop_pairs[pop_pairs.target_year<=origin]
        cm=fitted_crime_models(otr,ot,crime_columns,old_features,old_alpha,ntr,nt,type_counts,new_winner)
        cp=predict_selected_crime(winner["crime_candidate"],cm,ot,nt,new_winner,type_counts)
        pp,pm=fit_population(ptr,pt,winner["features"],winner["spec"],winner["mode"])
        base=exposure(cp,pp)
        for feature in impacts:
            o2=ot.copy();n2=nt.copy();p2=pt.copy()
            source=ptr[feature] if feature in ptr else (otr[feature] if feature in otr else ntr[feature])
            step=float(pd.to_numeric(source,errors="coerce").std())
            if not np.isfinite(step) or step==0: continue
            for frame in (o2,n2,p2):
                if feature in frame: frame[feature]=pd.to_numeric(frame[feature],errors="coerce")+step
            changed_crime=predict_selected_crime(winner["crime_candidate"],cm,o2,n2,new_winner,type_counts)
            changed_child=fitted_population_predict(pm,p2,winner["features"],winner["mode"])
            delta=exposure(changed_crime,changed_child)-base
            impacts[feature].extend(delta.tolist())
    impact_rows=[]
    derived_prefix=("y","log_crime","log_child")
    for feature,values in impacts.items():
        if not values: continue
        values=np.asarray(values); magnitude=float(np.mean(np.abs(values)));signed=float(np.mean(values))
        eligible=not feature.startswith(derived_prefix)
        impact_rows.append({"feature":feature,"mean_abs_y_change":magnitude,"mean_signed_y_change":signed,"direction":"higher_risk" if signed>=0 else "lower_risk","score_eligible":eligible})
    impact=pd.DataFrame(impact_rows).sort_values("mean_abs_y_change",ascending=False)
    impact["model_weight_all_pct"]=impact.mean_abs_y_change/impact.mean_abs_y_change.sum()*100
    eligible_total=impact.loc[impact.score_eligible,"mean_abs_y_change"].sum()
    impact["score_weight_pct"]=np.where(impact.score_eligible,impact.mean_abs_y_change/eligible_total*100,0)
    impact.to_csv(OUT/"integrated_feature_weights.csv",index=False,encoding="utf-8-sig")

    # Latest borough score: weighted quintile points for score-eligible inputs.
    score=pd.DataFrame({"gu":latest_old.gu,"predicted_y_2028":final_y,"predicted_crime_2028":np.expm1(final_crime),"predicted_child_2028":np.expm1(final_child)})
    score["predicted_y_risk_rank"]=rankdata(-score.predicted_y_2028,method="ordinal")
    contributions=[]
    for row in impact[impact.score_eligible].itertuples():
        source=latest_old if row.feature in latest_old else latest_new
        values=pd.to_numeric(source[row.feature],errors="coerce").to_numpy()
        order=-values if row.direction=="higher_risk" else values
        ranks=rankdata(order,method="ordinal")
        points=np.array([100,80,60,40,20])[(np.minimum(ranks-1,24)//5).astype(int)]
        contribution=points*row.score_weight_pct/100
        score[f"points__{row.feature}"]=points;contributions.append(contribution)
    score["integrated_risk_score"]=np.sum(contributions,axis=0) if contributions else np.nan
    score["integrated_risk_rank"]=rankdata(-score.integrated_risk_score,method="ordinal")
    score.to_csv(OUT/"borough_integrated_scores_2025_inputs.csv",index=False,encoding="utf-8-sig")
    predictions=hold_test[["gu","year","target_year","y","target_y"]].copy();predictions["predicted_y"]=hold_pred_y
    predictions["actual_risk_rank"]=rankdata(-predictions.target_y,method="ordinal");predictions["predicted_risk_rank"]=rankdata(-predictions.predicted_y,method="ordinal")
    predictions.to_csv(OUT/"holdout_2025_predictions.csv",index=False,encoding="utf-8-sig")
    metadata={"objective":"rank final Y=log1p(crime/child*10000)","ranking_score_formula":"40% Spearman rescaled + 25% normalized rank MAE + 20% top20 overlap + 15% exact quintile","cv_years":list(CV_YEARS),"holdout_year":HOLDOUT,"configuration_count":len(results),"baseline_cv":base_cv,"winner_cv":winner,"baseline_holdout":base_hold,"winner_holdout":hold_metrics,"influence_method":"mean absolute final-Y change from +1 training SD perturbation across locked CV folds; directions are signed average changes, not causal effects","score_rule":"score-eligible external features only; annual borough quintiles 100/80/60/40/20; weights normalized to 100"}
    (OUT/"result.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    joblib.dump({"crime_candidate":winner["crime_candidate"],"crime_models":crime_models,"population_model":pop_operational,"population_features":winner["features"],"population_spec":winner["spec"],"population_mode":winner["mode"],"new_crime_winner":new_winner,"type_counts":type_counts,"integrated_weights":impact,"metadata":metadata},MODEL_OUT/"operational_exposure_rank_3y.joblib")
    with pd.ExcelWriter(OUT/"exposure_rank_model_and_weights.xlsx",engine="openpyxl") as writer:
        pd.DataFrame([{"model":"Persistence",**base_cv},{"model":"Winner",**{k:v for k,v in winner.items() if k not in ("features","groups")}}]).to_excel(writer,sheet_name="CV_summary",index=False)
        pd.DataFrame([{"model":"Persistence",**base_hold},{"model":"Winner",**hold_metrics}]).to_excel(writer,sheet_name="holdout_summary",index=False)
        impact.to_excel(writer,sheet_name="integrated_weights",index=False);score.to_excel(writer,sheet_name="borough_scores",index=False);predictions.to_excel(writer,sheet_name="holdout_predictions",index=False)
        compact.to_excel(writer,sheet_name="all_comparisons",index=False)
    print(json.dumps({"baseline_cv":base_cv,"winner":{k:v for k,v in winner.items() if k not in ("features",)},"baseline_holdout":base_hold,"winner_holdout":hold_metrics},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
