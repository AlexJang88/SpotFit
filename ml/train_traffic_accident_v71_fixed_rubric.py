from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.train_traffic_accident_v44 import prepare_panel


ROOT = Path(__file__).resolve().parents[1]
VERSION = 71
OUT = ROOT / "models" / "교통사고" / f"{VERSION}차.49차고정등급표"
PROCESSED = ROOT / "data" / "processed" / f"교통사고_{VERSION}차_49차고정등급표_2027_2028.xlsx"
HORIZONS = (2, 3)
STRATEGIES = ("persistence", "mean2", "mean3", "trend2", "trend3_damped", "trend4_damped")


def load_v49():
    base = ROOT / "models" / "교통사고" / "41차~49차" / "49차"
    info = json.loads((base / "model_info.json").read_text(encoding="utf-8"))
    model_file = next(base.glob("*.joblib"))
    bundle = joblib.load(model_file)
    features = info["최종변수"]
    coef = np.asarray(bundle["model"].named_steps["model"].coef_, float)
    weights = np.abs(coef) / np.abs(coef).sum()
    return features, coef, weights


def risk_quintile(values: pd.Series, positive: bool) -> pd.Series:
    pct = values.rank(method="average", ascending=not positive, pct=True)
    return (120 - np.ceil(pct * 5).clip(1, 5) * 20).astype(int)


def risk_grade(score: pd.Series) -> pd.Series:
    pct = score.rank(method="first", ascending=False, pct=True)
    return np.ceil(pct * 5).clip(1, 5).astype(int)


def score_frame(frame: pd.DataFrame, features, coef, weights, prefix="") -> pd.DataFrame:
    out = frame[["연도", "자치구"]].copy()
    contributions = []
    for feature, c, w in zip(features, coef, weights):
        q = risk_quintile(frame[feature], c >= 0)
        out[f"{prefix}5분위점수_{feature}"] = q
        col = f"{prefix}가중점수_{feature}"
        out[col] = q * w
        contributions.append(col)
    out[f"{prefix}종합위험점수"] = out[contributions].sum(axis=1)
    out[f"{prefix}위험등급"] = risk_grade(out[f"{prefix}종합위험점수"])
    out[f"{prefix}안전등급"] = 6 - out[f"{prefix}위험등급"]
    return out


def predict_series(history: pd.Series, strategy: str, global_low: float, global_high: float) -> float:
    s = history.dropna().astype(float)
    if s.empty:
        return float((global_low + global_high) / 2)
    if strategy == "persistence":
        value = s.iloc[-1]
    elif strategy == "mean2":
        value = s.tail(2).mean()
    elif strategy == "mean3":
        value = s.tail(3).mean()
    else:
        n = 2 if strategy == "trend2" else (3 if strategy == "trend3_damped" else 4)
        tail = s.tail(n)
        if len(tail) < 2:
            value = tail.iloc[-1]
        else:
            slope = np.polyfit(np.arange(len(tail)), tail.to_numpy(), 1)[0]
            damping = 1.0 if strategy == "trend2" else .5
            value = tail.iloc[-1] + damping * slope
    return float(np.clip(value, global_low, global_high))


def make_predictions(panel, features, origin_year, target_year, strategy_by_feature):
    rows = []
    train = panel[panel["연도"] <= origin_year]
    for district in sorted(panel["자치구"].unique()):
        row = {"연도": target_year, "기준연도": origin_year, "자치구": district}
        district_history = train[train["자치구"] == district].sort_values("연도")
        for feature in features:
            observed = train[feature].dropna().astype(float)
            low, high = observed.quantile(.01), observed.quantile(.99)
            row[feature] = predict_series(
                district_history[feature], strategy_by_feature[feature], float(low), float(high)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(actual, predicted, features, coef, weights):
    actual_score = score_frame(actual, features, coef, weights, "실제_")
    pred_score = score_frame(predicted, features, coef, weights, "예측_")
    joined = actual_score.merge(pred_score, on=["연도", "자치구"], validate="one_to_one")
    a = joined["실제_안전등급"].to_numpy()
    p = joined["예측_안전등급"].to_numpy()
    metrics = {
        "정확등급정확도": float(np.mean(a == p)),
        "1등급이내정확도": float(np.mean(np.abs(a - p) <= 1)),
        "평균등급오차": float(np.mean(np.abs(a - p))),
        "종합점수_MAE": float(mean_absolute_error(joined["실제_종합위험점수"], joined["예측_종합위험점수"])),
    }
    return metrics, joined


def objective(metrics):
    return (metrics["정확등급정확도"], metrics["1등급이내정확도"], -metrics["평균등급오차"], -metrics["종합점수_MAE"])


def backtest(panel, features, coef, weights, horizon, strategy_map, target_years):
    rows, details = [], []
    for target_year in target_years:
        origin = target_year - horizon
        pred = make_predictions(panel, features, origin, target_year, strategy_map)
        actual = panel[panel["연도"] == target_year][["연도", "자치구", *features]].copy()
        metrics, joined = evaluate(actual, pred, features, coef, weights)
        rows.append({"예측기간": f"{horizon}년 후", "검증연도": target_year, **metrics})
        joined.insert(0, "예측기간", f"{horizon}년 후")
        joined.insert(1, "기준연도", origin)
        details.append(joined)
    summary = {
        key: float(np.mean([r[key] for r in rows]))
        for key in ["정확등급정확도", "1등급이내정확도", "평균등급오차", "종합점수_MAE"]
    }
    return summary, pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def select_strategies(panel, features, coef, weights, horizon, dev_years):
    candidates = []
    for strategy in STRATEGIES:
        mapping = {f: strategy for f in features}
        metrics, _, _ = backtest(panel, features, coef, weights, horizon, mapping, dev_years)
        candidates.append({"단계": "공통전략", "변수": "전체", "전략": strategy, **metrics})
    best = max(candidates, key=lambda r: objective(r))
    selected = {f: best["전략"] for f in features}
    current = best
    # 가중치가 큰 지표부터 한 번씩 검토한다. 동일 자료에서 무제한 반복해
    # 등급 정확도를 맞추면 과적합되므로, 개선이 확인되는 변경만 1회 허용한다.
    for feature in [features[i] for i in np.argsort(-np.abs(weights))]:
        best_trial = current
        best_strategy = selected[feature]
        for strategy in STRATEGIES:
            if strategy == selected[feature]:
                continue
            trial_map = dict(selected); trial_map[feature] = strategy
            metrics, _, _ = backtest(panel, features, coef, weights, horizon, trial_map, dev_years)
            trial = {"단계": "변수별개선", "변수": feature, "전략": strategy, **metrics}
            if objective(trial) > objective(best_trial):
                best_trial, best_strategy = trial, strategy
        if best_strategy != selected[feature]:
            selected[feature] = best_strategy
            current = best_trial
            candidates.append(best_trial)
    return selected, pd.DataFrame(candidates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features, coef, weights = load_v49()
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    missing = [f for f in features if f not in panel]
    if missing:
        raise ValueError(f"49차 지표 누락: {missing}")

    weight_table = pd.DataFrame({
        "지표": features, "49차표준화계수": coef,
        "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
        "고정가중치_pct": weights * 100,
    }).sort_values("고정가중치_pct", ascending=False)

    all_selection, all_metrics, all_details, all_forecasts = [], [], [], []
    model_payload = {}
    for horizon in HORIZONS:
        # 2025년은 완전히 독립적인 최종 검증으로 남긴다.
        dev_years = list(range(max(2020, 2017 + horizon), 2025))
        selected, selection_log = select_strategies(panel, features, coef, weights, horizon, dev_years)
        selection_log.insert(0, "예측기간", f"{horizon}년 후")
        all_selection.append(selection_log)

        dev_metrics, dev_rows, dev_detail = backtest(panel, features, coef, weights, horizon, selected, dev_years)
        hold_metrics, hold_rows, hold_detail = backtest(panel, features, coef, weights, horizon, selected, [2025])
        persist = {f: "persistence" for f in features}
        base_metrics, base_rows, _ = backtest(panel, features, coef, weights, horizon, persist, [2025])
        dev_rows["구분"] = "개발 롤링검증"
        hold_rows["구분"] = "2025 독립검증"
        base_rows["구분"] = "2025 Persistence"
        all_metrics.extend([dev_rows, hold_rows, base_rows])
        dev_detail["구분"] = "개발 롤링검증"; hold_detail["구분"] = "2025 독립검증"
        all_details.extend([dev_detail, hold_detail])

        # 개발구간 최적화 조합이 독립검증에서 Persistence를 이기지 못하면
        # 단순하고 재현 가능한 Persistence를 최종 채택한다.
        final_selected = selected if objective(hold_metrics) > objective(base_metrics) else persist
        final_reason = (
            "개발 최적화 조합이 2025 독립검증에서 Persistence보다 우수"
            if final_selected is selected else
            "개발 최적화 조합이 2025 독립검증에서 Persistence보다 낮아 과적합으로 판단"
        )
        target_year = 2025 + horizon
        forecast_values = make_predictions(panel, features, 2025, target_year, final_selected)
        forecast_scores = score_frame(forecast_values, features, coef, weights, "예측_")
        forecast = forecast_values.merge(forecast_scores, on=["연도", "자치구"], validate="one_to_one")
        forecast.insert(0, "예측기간", f"{horizon}년 후")
        all_forecasts.append(forecast)
        model_payload[str(horizon)] = {
            "final_strategies": final_selected, "selection_reason": final_reason,
            "experimental_strategies": selected, "development_metrics": dev_metrics,
            "holdout_2025_metrics": hold_metrics, "persistence_2025_metrics": base_metrics,
        }

    joblib.dump({
        "version": VERSION, "architecture": "13개 지표 개별예측→5분위→49차 고정가중치→종합점수→안전등급",
        "features": features, "coefficients": dict(zip(features, coef)),
        "weights": dict(zip(features, weights)), "horizons": model_payload,
        "safety_grade_orientation": "1=가장 안전, 5=가장 위험",
    }, OUT / "교통사고_71차_49차고정등급표모델.joblib")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    selection_df = pd.concat(all_selection, ignore_index=True)
    details_df = pd.concat(all_details, ignore_index=True)
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="49차고정가중치", index=False)
        pd.DataFrame([{"예측기간": f"{h}년 후", "지표": f, "선정예측법": s}
                      for h, payload in model_payload.items() for f, s in payload["final_strategies"].items()]).to_excel(
            writer, sheet_name="지표별선정모델", index=False)
        metrics_df.to_excel(writer, sheet_name="성능평가", index=False)
        selection_df.to_excel(writer, sheet_name="모델선정과정", index=False)
        details_df.to_excel(writer, sheet_name="검증상세", index=False)
        forecasts_df.to_excel(writer, sheet_name="2027_2028예측", index=False)

    metadata = {
        "모델명": "교통사고 71차 49차 고정등급표 미래지표 예측모델",
        "구조": "13개 지표 각각 예측 → 연도별 5분위 → 49차 고정 가중치 → 종합위험점수 → 안전등급",
        "안전등급방향": "1등급=가장 안전, 5등급=가장 위험",
        "모델선정원칙": "2020~2024 롤링검증 정확등급 정확도 최우선, ±1등급 정확도·등급오차·점수MAE 순",
        "독립검증": "2025년은 전략 선택에서 제외",
        "가중치변경": False, "지표변경": False,
        "결과": model_payload,
        "예측파일": str(PROCESSED.relative_to(ROOT)),
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
