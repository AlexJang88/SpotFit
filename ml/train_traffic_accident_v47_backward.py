from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel
from ml.train_traffic_accident_v42 import CV_YEARS, metric_values, pipeline
from ml.train_traffic_accident_v44 import SCHOOL_FEATURE, prepare_panel


VERSION = 47
START_VERSION = 46
OUT = ROOT / "models" / "교통사고" / "41차~49차" / f"{VERSION}차"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0)
CORR_LIMIT = 0.90
# 46차 CV MAE 대비 최대 1% 이내의 손실만 허용하고 더 단순한 모델을 선택한다.
MAX_CV_MAE_INCREASE = 0.01
PRUNE_WITHIN_TOLERANCE = False
PRIORITIZE_CORRELATED = True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    base_info = json.loads((ROOT / "models" / "교통사고" / "41차~49차" / f"{START_VERSION}차" / "model_info.json").read_text(encoding="utf-8"))
    selected = list(base_info["최종변수"])
    all_features = selected.copy()
    feature_index = {feature: index for index, feature in enumerate(all_features)}

    fold_data = []
    for year in CV_YEARS:
        train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < year)]
        valid = panel[panel["예측연도"] == year]
        xtr = train[all_features].to_numpy(float)
        xva = valid[all_features].to_numpy(float)
        median = np.nanmedian(xtr, axis=0)
        median = np.where(np.isnan(median), 0, median)
        xtr = np.where(np.isnan(xtr), median, xtr)
        xva = np.where(np.isnan(xva), median, xva)
        mean = xtr.mean(axis=0)
        scale = xtr.std(axis=0)
        scale[scale == 0] = 1
        xtr = (xtr - mean) / scale
        xva = (xva - mean) / scale
        ytr = train[TARGET].to_numpy(float)
        fold_data.append({
            "gram": xtr.T @ xtr,
            "cross": xtr.T @ (ytr - ytr.mean()),
            "xva": xva,
            "yva": valid[TARGET].to_numpy(float),
            "ymean": ytr.mean(),
        })

    cache: dict[tuple[str, ...], dict] = {}

    def evaluate(features: list[str]) -> dict:
        key = tuple(features)
        if key in cache:
            return cache[key]
        indices = np.array([feature_index[feature] for feature in features])
        best = None
        for alpha in ALPHAS:
            yearly = []
            for fold in fold_data:
                gram = fold["gram"][np.ix_(indices, indices)]
                beta = np.linalg.solve(gram + alpha * np.eye(len(indices)), fold["cross"][indices])
                prediction = np.maximum(fold["ymean"] + fold["xva"][:, indices] @ beta, 0)
                yearly.append(metric_values(fold["yva"], prediction))
            result = {
                "alpha": alpha,
                "평균_MAE": float(np.mean([m["MAE"] for m in yearly])),
                "평균_RMSE": float(np.mean([m["RMSE"] for m in yearly])),
                "평균_R2": float(np.mean([m["R2"] for m in yearly])),
                "평균_WMAPE_pct": float(np.mean([m["WMAPE_pct"] for m in yearly])),
            }
            if best is None or (result["평균_MAE"], result["평균_RMSE"]) < (best["평균_MAE"], best["평균_RMSE"]):
                best = result
        cache[key] = best
        return best

    def high_corr_pairs(features: list[str]) -> list[tuple[float, str, str]]:
        corr = panel[features].corr().abs()
        pairs = []
        for i, left in enumerate(features):
            for right in features[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and value >= CORR_LIMIT:
                    pairs.append((float(value), left, right))
        return sorted(pairs, reverse=True)

    baseline = evaluate(selected)
    allowed_mae = baseline["평균_MAE"] * (1 + MAX_CV_MAE_INCREASE)
    history = [{"단계": 0, "제거변수": f"{START_VERSION}차 기준", "제거사유": "시작", "변수수": len(selected),
                "고상관쌍수": len(high_corr_pairs(selected)), **baseline}]

    step = 0
    while len(selected) > 2:
        pairs = high_corr_pairs(selected)
        if pairs and PRIORITIZE_CORRELATED:
            priority = sorted({feature for _, left, right in pairs for feature in (left, right)})
            reason = f"상관계수 {CORR_LIMIT:.2f} 이상 중복"
        else:
            priority = selected.copy()
            reason = "일반 후진제거"

        trials = []
        for removed in priority:
            features = [feature for feature in selected if feature != removed]
            result = evaluate(features)
            trials.append((result["평균_MAE"], result["평균_RMSE"], removed, features, result))
        best_mae, _, removed, features, result = min(trials)

        # 중복이 남아 있으면 46차 대비 1% 이내에서 반드시 정리한다.
        # 중복 정리 후에는 MAE가 현재 단계보다 실제 개선되는 경우만 추가 제거한다.
        accept = best_mae <= allowed_mae if ((pairs and PRIORITIZE_CORRELATED) or PRUNE_WITHIN_TOLERANCE) else best_mae < history[-1]["평균_MAE"] - 1e-12
        if not accept:
            print(f"제거 종료: 다음 후보={removed}, CV MAE={best_mae:.6f}", flush=True)
            break

        selected = features
        step += 1
        remaining_pairs = high_corr_pairs(selected)
        history.append({"단계": step, "제거변수": removed, "제거사유": reason, "변수수": len(selected),
                        "고상관쌍수": len(remaining_pairs), **result})
        print(f"{step}단계 제거: {removed}, 변수={len(selected)}, 고상관쌍={len(remaining_pairs)}, CV MAE={best_mae:.6f}", flush=True)

    final_cv = evaluate(selected)
    alpha = float(final_cv["alpha"])
    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2024)]
    holdout = panel[panel["예측연도"] == 2025]
    holdout_model = pipeline(Ridge(alpha=alpha), True).fit(train[selected], train[TARGET])
    prediction = np.maximum(holdout_model.predict(holdout[selected]), 0)
    holdout_metrics = metric_values(holdout[TARGET], prediction)

    labeled = panel[panel[TARGET].notna()]
    future = panel[panel[TARGET].isna()]
    final_model = pipeline(Ridge(alpha=alpha), True).fit(labeled[selected], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = np.maximum(final_model.predict(future[selected]), 0)
    joblib.dump({"version": VERSION, "model_name": f"Ridge_{alpha:g}", "target": TARGET,
                 "features": selected, "model": final_model}, OUT / f"교통사고_{VERSION}차_예측모델.joblib")

    comparison = pd.DataFrame([
        {"구분": f"{VERSION}차", **holdout_metrics},
        {"구분": f"{START_VERSION}차", **base_info["2025_독립검증"]},
    ])
    with pd.ExcelWriter(OUT / f"{VERSION}차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="2025성능비교", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="단계별변수제거", index=False)
        pd.DataFrame({"최종변수": selected}).to_excel(writer, sheet_name="최종변수", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)

    metadata = {
        "모델명": f"교통사고 {VERSION}차 경량화 모델",
        "설계": (
            f"{START_VERSION}차에서 고상관 중복변수를 우선 제거한 뒤 순차검증 기반 후진제거"
            if PRIORITIZE_CORRELATED
            else f"{START_VERSION}차의 전체 변수를 대상으로 순차검증 기반 전수 후진제거"
        ),
        "평가조합수": len(cache),
        "제거단계수": step,
        "제거변수": [row["제거변수"] for row in history[1:]],
        "alpha": alpha,
        "최종변수수": len(selected),
        "최종변수": selected,
        "최종_고상관쌍수": len(high_corr_pairs(selected)),
        "2021_2024_순차검증": final_cv,
        "2025_독립검증": holdout_metrics,
        f"2025_{START_VERSION}차": base_info["2025_독립검증"],
        f"{START_VERSION}차대비_MAE개선율_pct": (base_info["2025_독립검증"]["MAE"] - holdout_metrics["MAE"]) / base_info["2025_독립검증"]["MAE"] * 100,
    }
    (OUT / "model_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
