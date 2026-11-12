from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import prediction_frame
from ml.train_traffic_accident_v74_all_available import add_expanded_cache
from ml.train_traffic_accident_v76_promoted_indicators import parse_spec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "교통사고" / "전체-어린이교통안전등급_앙상블1차"
PROCESSED = ROOT / "data" / "processed" / "전체-어린이교통안전등급_앙상블1차.xlsx"
CHILD_XLSX = ROOT / "data" / "processed" / "교통사고_76차_보조지표메인승격_2027_2028.xlsx"
TOTAL_XLSX = ROOT / "data" / "processed" / "전체교통안전등급_1차_2027_2028.xlsx"
HORIZONS = (2, 3)
WEIGHTS = np.arange(0, 1.0001, .05)  # 전체 모델 위험순위의 비중


def load_bundle(path):
    data = joblib.load(path)
    features = list(data["features"])
    coef = np.array([data["coefficients"][f] for f in features], float)
    weights = np.array([data["weights"][f] for f in features], float)
    return features, coef, weights


def load_specs(path, method_col):
    table = pd.read_excel(path, sheet_name="지표별미래예측법")
    result = {}
    for horizon in HORIZONS:
        rows = table[table["예측기간"] == f"{horizon}년 후"]
        result[horizon] = dict(zip(rows["지표"], rows[method_col].map(parse_spec)))
    return result


def risk_rank(score):
    # 기존 risk_grade와 동일하게 동점은 원래 자치구 정렬 순서로 확정한다.
    return score.rank(method="first", pct=True).astype(float)


def safety_grade(risk_score):
    risk_grade = np.ceil(risk_score.rank(method="first", ascending=False, pct=True) * 5).clip(1, 5).astype(int)
    return 6 - risk_grade


def metric(actual, pred):
    a, p = np.asarray(actual, int), np.asarray(pred, int)
    return {"정확등급정확도": float(np.mean(a == p)),
            "1등급이내정확도": float(np.mean(np.abs(a-p) <= 1)),
            "평균등급오차": float(np.mean(np.abs(a-p)))}


def objective(row):
    return (row["정확등급정확도"], row["1등급이내정확도"], -row["평균등급오차"])


def build_scores(panel, features, coef, weights, specs, horizon):
    _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
    _, _, _, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
    rows = []
    for year in (*DEV_YEARS, HOLDOUT_YEAR):
        actual = panel[panel["연도"] == year][["연도", "자치구", *features]].sort_values("자치구")
        predicted = prediction_frame(panel, features, year, specs[horizon], cache).sort_values("자치구")
        a = score_frame(actual, features, coef, weights, "실제_")
        p = score_frame(predicted, features, coef, weights, "예측_")
        joined = a.merge(p, on=["연도", "자치구"], validate="one_to_one")
        rows.append(joined[["연도", "자치구", "실제_종합위험점수", "실제_안전등급",
                            "예측_종합위험점수", "예측_안전등급"]])
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    child_model = ROOT / "models" / "교통사고" / "76차.보조지표메인승격" / "교통사고_76차_보조지표메인승격모델.joblib"
    total_model = ROOT / "models" / "교통사고" / "전체교통안전등급_1차" / "전체교통안전등급_1차.joblib"
    cf, cc, cw = load_bundle(child_model); tf, tc, tw = load_bundle(total_model)
    child_specs = load_specs(CHILD_XLSX, "예측법")
    total_specs = load_specs(TOTAL_XLSX, "선정예측법")

    results, search_rows, details = {}, [], []
    for horizon in HORIZONS:
        child = build_scores(panel, cf, cc, cw, child_specs, horizon)
        total = build_scores(panel, tf, tc, tw, total_specs, horizon)
        merged = child.merge(total, on=["연도", "자치구"], suffixes=("_어린이", "_전체"), validate="one_to_one")
        merged["예측위험순위_어린이"] = merged.groupby("연도")["예측_종합위험점수_어린이"].transform(risk_rank)
        merged["예측위험순위_전체"] = merged.groupby("연도")["예측_종합위험점수_전체"].transform(risk_rank)
        horizon_result = {}
        for target_name, actual_col in (("전체교통안전", "실제_안전등급_전체"),
                                        ("어린이교통안전", "실제_안전등급_어린이")):
            trials = []
            dev = merged[merged["연도"].isin(DEV_YEARS)].copy()
            for w in WEIGHTS:
                dev["혼합위험"] = w*dev["예측위험순위_전체"] + (1-w)*dev["예측위험순위_어린이"]
                dev["혼합안전등급"] = dev.groupby("연도")["혼합위험"].transform(safety_grade)
                m = metric(dev[actual_col], dev["혼합안전등급"])
                trials.append({"목표": target_name, "예측기간": f"{horizon}년 후",
                               "전체모델비중": float(w), "어린이76차비중": float(1-w), **m})
            best = max(trials, key=objective); search_rows.extend(trials)
            hold = merged[merged["연도"] == HOLDOUT_YEAR].copy()
            w = best["전체모델비중"]
            hold["혼합위험"] = w*hold["예측위험순위_전체"] + (1-w)*hold["예측위험순위_어린이"]
            hold["앙상블_안전등급"] = safety_grade(hold["혼합위험"])
            hold_metrics = metric(hold[actual_col], hold["앙상블_안전등급"])
            base_col = "예측_안전등급_전체" if target_name == "전체교통안전" else "예측_안전등급_어린이"
            base_metrics = metric(hold[actual_col], hold[base_col])
            adopted = objective(hold_metrics) > objective(base_metrics)
            horizon_result[target_name] = {"개발선정비중_전체": float(w),
                                           "개발선정비중_어린이76차": float(1-w),
                                           "2025앙상블": hold_metrics, "2025단독기준": base_metrics,
                                           "개선여부": adopted}
            detail = hold[["연도", "자치구", actual_col, base_col, "앙상블_안전등급",
                           "예측위험순위_전체", "예측위험순위_어린이", "혼합위험"]].copy()
            detail.insert(0, "목표", target_name); detail.insert(1, "예측기간", f"{horizon}년 후")
            details.append(detail)
        results[str(horizon)] = horizon_result

    # 이미 생성된 2027·2028 위험점수를 같은 방식으로 혼합한다.
    child_future = pd.read_excel(CHILD_XLSX, sheet_name="2027_2028예측")
    total_future = pd.read_excel(TOTAL_XLSX, sheet_name="2027_2028예측")
    future = child_future[["예측기간", "연도", "자치구", "예측_종합위험점수"]].merge(
        total_future[["예측기간", "연도", "자치구", "예측_종합위험점수"]],
        on=["예측기간", "연도", "자치구"], suffixes=("_어린이", "_전체"), validate="one_to_one")
    future["예측위험순위_어린이"] = future.groupby("연도")["예측_종합위험점수_어린이"].transform(risk_rank)
    future["예측위험순위_전체"] = future.groupby("연도")["예측_종합위험점수_전체"].transform(risk_rank)
    future_outputs = []
    for target_name in ("전체교통안전", "어린이교통안전"):
        part = future.copy(); part.insert(0, "목표", target_name)
        part["전체모델비중"] = part["예측기간"].map(
            {f"{h}년 후": results[str(h)][target_name]["개발선정비중_전체"] for h in HORIZONS})
        part["혼합위험"] = part["전체모델비중"]*part["예측위험순위_전체"] + (1-part["전체모델비중"])*part["예측위험순위_어린이"]
        part["앙상블_안전등급"] = part.groupby("연도")["혼합위험"].transform(safety_grade)
        future_outputs.append(part)

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        pd.DataFrame(search_rows).to_excel(writer, sheet_name="개발혼합비탐색", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025독립검증상세", index=False)
        pd.concat(future_outputs, ignore_index=True).to_excel(writer, sheet_name="2027_2028앙상블", index=False)
    info = {"모델명": "전체-어린이 교통안전등급 양방향 앙상블 1차",
            "결합방식": "각 모델 종합위험점수를 연도별 자치구 위험순위 백분율로 변환 후 가중합",
            "비중선정": "2021~2024 개발구간", "최종검증": "2025 독립검증",
            "결과": results, "processed": str(PROCESSED.relative_to(ROOT))}
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"results": results, "child_model": str(child_model.relative_to(ROOT)),
                 "total_model": str(total_model.relative_to(ROOT))}, OUT / "전체-어린이교통안전등급_앙상블1차.joblib")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
