from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import objective, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import add_persistence_cache, parse_spec
from ml.train_traffic_accident_v83_poisson_exposure_grade import POP, TARGET_COUNT, TARGET_POP, VICTIMS, population_forecast
from ml.train_traffic_accident_v84_poisson_forward_raw import poisson_weights, raw_candidates


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "교통사고_85차_86차_기간별최적화.xlsx"
CONFIG = {2: (85, "2년최적화"), 3: (86, "3년최적화")}
MAX_ADDITIONS = 14


def evaluate_one(panel, features, coef, weights, horizon, specs, cache, years):
    return evaluate_specs(panel, features, coef, weights, horizon, years, specs, cache)


def search_horizon(panel, core, candidates, alpha, horizon, core_specs, cache):
    _, core_coef, core_weights = poisson_weights(panel, core, alpha)
    selected, specs = list(core), dict(core_specs)
    current, _, _ = evaluate_one(panel, selected, core_coef, core_weights, horizon, specs, cache, DEV_YEARS)
    accepted = [{"단계": 0, "작업": "83차핵심시작", "변수": "-", "지표수": len(selected), **current}]
    trials = []
    for step in range(1, MAX_ADDITIONS+1):
        best = None
        for added in candidates:
            if added in selected:
                continue
            trial_features = [*selected, added]
            _, coef, weights = poisson_weights(panel, trial_features, alpha)
            trial_specs = {**specs, added: "Persistence"}
            metrics, _, _ = evaluate_one(panel, trial_features, coef, weights, horizon, trial_specs, cache, DEV_YEARS)
            trials.append({"예측기간": f"{horizon}년 후", "단계": step, "작업": "추가후보",
                           "변수": added, "지표수": len(trial_features), **metrics})
            record = (objective(metrics), added, metrics, trial_features, coef, weights, trial_specs)
            if best is None or record[0] > best[0]:
                best = record
        if best is None or best[2]["정확등급정확도"] <= current["정확등급정확도"] + 1e-9:
            accepted.append({"단계": step, "작업": "추가중단", "변수": best[1] if best else "-",
                             "지표수": len(selected), **(best[2] if best else current)})
            break
        _, added, current, selected, _, _, specs = best
        accepted.append({"단계": step, "작업": "추가채택", "변수": added,
                         "지표수": len(selected), **current})

    # 추가된 변수만 후진 제거한다. 83차 핵심 13개는 근간으로 고정한다.
    changed = True; remove_step = 0
    while changed:
        changed = False; best = None
        for removed in [f for f in selected if f not in core]:
            trial_features = [f for f in selected if f != removed]
            _, coef, weights = poisson_weights(panel, trial_features, alpha)
            trial_specs = {f: s for f, s in specs.items() if f != removed}
            metrics, _, _ = evaluate_one(panel, trial_features, coef, weights, horizon, trial_specs, cache, DEV_YEARS)
            trials.append({"예측기간": f"{horizon}년 후", "단계": remove_step, "작업": "제거후보",
                           "변수": removed, "지표수": len(trial_features), **metrics})
            record = (objective(metrics), removed, metrics, trial_features, coef, weights, trial_specs)
            if best is None or record[0] > best[0]:
                best = record
        if best is not None and best[0] > objective(current):
            _, removed, current, selected, _, _, specs = best
            remove_step += 1; changed = True
            accepted.append({"단계": remove_step, "작업": "제거채택", "변수": removed,
                             "지표수": len(selected), **current})
    model, coef, weights = poisson_weights(panel, selected, alpha)
    return selected, specs, model, coef, weights, pd.DataFrame(accepted), pd.DataFrame(trials)


def main():
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["자치구", "연도"]).reset_index(drop=True)
    group = panel.groupby("자치구", sort=False)
    panel[TARGET_COUNT] = group[VICTIMS].shift(-1); panel[TARGET_POP] = group[POP].shift(-1)
    v83_dir = ROOT / "models" / "교통사고" / "83차.피해자수Poisson인구노출등급"
    v83_info = json.loads((v83_dir / "model_info.json").read_text(encoding="utf-8"))
    v83 = joblib.load(v83_dir / "교통사고_83차_피해자수Poisson등급모델.joblib")
    core = list(v83["features"]); alpha = float(v83_info["alpha"])
    candidates = raw_candidates(panel, core)
    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    base_specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in CONFIG}

    caches, expanded, parts = {}, {}, {}
    for horizon in CONFIG:
        _, methods, cache, _, _ = build_prediction_cache(panel, core, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, core, horizon, cache, methods)
        add_persistence_cache(panel, candidates, horizon, cache)
        caches[horizon], expanded[horizon], parts[horizon] = cache, data, (predictor_cols, pool)

    summaries, weights_all, accepted_all, trials_all, metrics_all, details_all, forecasts_all = [], [], [], [], [], [], []
    for horizon, (version, suffix) in CONFIG.items():
        selected, specs, fitted, coef, weights, accepted, trials = search_horizon(
            panel, core, candidates, alpha, horizon, base_specs[horizon], caches[horizon]
        )
        accepted.insert(0, "차수", version); accepted.insert(1, "예측기간", f"{horizon}년 후")
        trials.insert(0, "차수", version); accepted_all.append(accepted); trials_all.append(trials)
        dev, dev_rows, dev_detail = evaluate_one(panel, selected, coef, weights, horizon, specs, caches[horizon], DEV_YEARS)
        hold, hold_rows, hold_detail = evaluate_one(panel, selected, coef, weights, horizon, specs, caches[horizon], [HOLDOUT_YEAR])
        dev_rows["차수"] = version; dev_rows["구분"] = "개발"
        hold_rows["차수"] = version; hold_rows["구분"] = "2025독립"
        metrics_all.extend([dev_rows, hold_rows]); dev_detail["차수"] = version; hold_detail["차수"] = version
        details_all.extend([dev_detail, hold_detail])
        promoted = [f for f in selected if f not in core]
        weights_all.append(pd.DataFrame({"차수": version, "예측기간": f"{horizon}년 후", "지표": selected,
                                         "구분": ["83차핵심" if f in core else "Raw추가" for f in selected],
                                         "Poisson표준화계수": coef,
                                         "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                         "가중치_pct": weights*100}))
        predictor_cols, pool = parts[horizon]
        core_future, bundle = fit_forecast(panel, expanded[horizon], predictor_cols, pool, core, horizon, base_specs[horizon])
        latest = panel[panel["연도"] == 2025].set_index("자치구")
        future = core_future.copy()
        for added in promoted:
            future[added] = future["자치구"].map(latest[added]); bundle[added] = {"spec": "Persistence", "components": {}}
        scored = score_frame(future, selected, coef, weights, "예측_")
        output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        pop_future = population_forecast(panel, 2025+horizon); output = output.merge(pop_future, on="자치구", validate="one_to_one")
        output["예측_어린이피해자율_1만명당"] = fitted.predict(output[selected])*10000
        output["예측_어린이피해자수"] = output["예측_어린이피해자율_1만명당"]/10000*output["예측_어린이인구수"]
        output.insert(0, "차수", version); output.insert(1, "예측기간", f"{horizon}년 후"); forecasts_all.append(output)

        out = ROOT / "models" / "교통사고" / f"{version}차.피해자수Poisson{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        info = {"모델명": f"교통사고 {version}차 피해자수 Poisson {suffix}", "예측기간": f"{horizon}년 후",
                "Y": "다음 연도 어린이 교통사고 사망자 수+부상자 수",
                "불변근간": "83차 핵심13개 고정→기간별 Raw 전진선택·추가변수 후진제거→5분위→Poisson 가중치→안전등급",
                "최종지표수": len(selected), "추가지표": promoted,
                "개발성능": dev, "2025독립성능": hold,
                "비교기준": {"74차": .56, "83차": .60 if horizon == 2 else .44, "84차": .68 if horizon == 2 else .44}}
        joblib.dump({"version": version, "horizon": horizon, "features": selected, "poisson_model": fitted,
                     "coefficients": dict(zip(selected, coef)), "weights": dict(zip(selected, weights)),
                     "forecast_model": bundle}, out / f"교통사고_{version}차_{suffix}.joblib")
        (out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({"차수": version, "예측기간": f"{horizon}년 후", "지표수": len(selected),
                          "추가지표수": len(promoted), **{f"개발_{k}": v for k, v in dev.items()},
                          **{f"2025_{k}": v for k, v in hold.items()}})

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="85_86요약", index=False)
        pd.concat(weights_all, ignore_index=True).to_excel(writer, sheet_name="기간별지표가중치", index=False)
        pd.concat(accepted_all, ignore_index=True).to_excel(writer, sheet_name="최고성능갱신", index=False)
        pd.concat(trials_all, ignore_index=True).to_excel(writer, sheet_name="전체후보평가", index=False)
        pd.concat(metrics_all, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(details_all, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(forecasts_all, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
