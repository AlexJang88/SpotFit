from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel


OUT = ROOT / "models" / "교통사고" / "41차" / "전체124개_부분집합탐색"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RANDOM_SEED = 41
RANDOM_ITERATIONS = 60_000
MAX_FEATURES = 35


def metric_values(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - prediction)
    return {
        "MAE": float(error.mean()),
        "RMSE": float(np.sqrt(np.mean((actual - prediction) ** 2))),
        "R2": float(1 - np.sum((actual - prediction) ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "WMAPE_pct": float(error.sum() / actual.sum() * 100),
    }


def main() -> None:
    started = time.time()
    rng = random.Random(RANDOM_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    panel, features, audit = build_panel()
    if len(features) != 124:
        raise ValueError(f"예상한 124개 feature가 아닙니다: {len(features)}")

    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] <= 2023)].copy()
    valid = panel[panel["예측연도"] == 2024].copy()
    if len(train) != 150 or len(valid) != 25:
        raise ValueError(f"분할 행 수 오류: 학습={len(train)}, 검증={len(valid)}")

    x_train_raw = train[features].to_numpy(float)
    x_valid_raw = valid[features].to_numpy(float)
    median = np.nanmedian(x_train_raw, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    x_train_raw = np.where(np.isnan(x_train_raw), median, x_train_raw)
    x_valid_raw = np.where(np.isnan(x_valid_raw), median, x_valid_raw)
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train_raw - mean) / scale
    x_valid = (x_valid_raw - mean) / scale
    y_train = train[TARGET].to_numpy(float)
    y_valid = valid[TARGET].to_numpy(float)
    y_mean = y_train.mean()
    gram = x_train.T @ x_train
    cross = x_train.T @ (y_train - y_mean)

    rate_2023 = panel[panel["연도"] == 2023].set_index("자치구")[CURRENT_RATE]
    baseline_prediction = valid["자치구"].map(rate_2023).to_numpy(float)
    baseline = metric_values(y_valid, baseline_prediction)

    cache: dict[tuple[int, ...], tuple[float, float, dict[str, float]]] = {}
    records: list[dict] = []
    global_best: tuple[int, ...] = ()
    global_best_mae = baseline["MAE"]

    def evaluate(indices) -> tuple[float, float, dict[str, float]]:
        nonlocal global_best, global_best_mae
        key = tuple(sorted(set(indices)))
        if not key or len(key) > MAX_FEATURES:
            return math.inf, math.nan, {}
        if key in cache:
            return cache[key]
        ix = np.fromiter(key, dtype=int)
        g = gram[np.ix_(ix, ix)]
        c = cross[ix]
        identity = np.eye(len(key))
        best = (math.inf, math.nan, {})
        for alpha in ALPHAS:
            beta = np.linalg.solve(g + alpha * identity, c)
            prediction = np.maximum(y_mean + x_valid[:, ix] @ beta, 0)
            result = metric_values(y_valid, prediction)
            if result["MAE"] < best[0]:
                best = (result["MAE"], alpha, result)
        cache[key] = best
        if best[0] < global_best_mae - 1e-12:
            global_best_mae = best[0]
            global_best = key
            records.append({
                "평가조합수": len(cache),
                "탐색단계": current_stage[0],
                "변수수": len(key),
                "alpha": best[1],
                **best[2],
                "초기모델대비_MAE개선율_pct": (baseline["MAE"] - best[0]) / baseline["MAE"] * 100,
                "변수": " | ".join(features[i] for i in key),
            })
        return best

    current_stage = ["단일변수"]
    # 단일 변수 전체를 평가한 뒤 가장 좋은 변수에서 출발한다.
    singles = [(evaluate((i,))[0], (i,)) for i in range(len(features))]
    current = min(singles)[1]

    # 전진 선택: 각 단계에서 아직 없는 123개 이하의 변수를 모두 시험한다.
    current_stage[0] = "전진선택"
    no_improvement = 0
    for size in range(2, MAX_FEATURES + 1):
        candidates = []
        selected = set(current)
        for candidate in range(len(features)):
            if candidate not in selected:
                key = tuple(sorted((*current, candidate)))
                candidates.append((evaluate(key)[0], key))
        candidate_mae, candidate_key = min(candidates)
        current_mae = evaluate(current)[0]
        current = candidate_key
        if candidate_mae < current_mae - 1e-12:
            no_improvement = 0
        else:
            no_improvement += 1
        print(f"전진선택 {size}개 완료: MAE={candidate_mae:.6f}, 전체최고={global_best_mae:.6f}", flush=True)
        if no_improvement >= 5:
            break

    # 최고 조합에서 제거와 추가를 반복하는 floating local search.
    current_stage[0] = "부동선택"
    current = global_best
    changed = True
    while changed:
        changed = False
        current_mae = evaluate(current)[0]
        neighbors = []
        if len(current) > 1:
            for item in current:
                key = tuple(i for i in current if i != item)
                neighbors.append((evaluate(key)[0], key))
        selected = set(current)
        if len(current) < MAX_FEATURES:
            for item in range(len(features)):
                if item not in selected:
                    key = tuple(sorted((*current, item)))
                    neighbors.append((evaluate(key)[0], key))
        best_neighbor_mae, best_neighbor = min(neighbors)
        if best_neighbor_mae < current_mae - 1e-12:
            current = best_neighbor
            changed = True

    # 최고점 주변에서 1~3개 변수를 교체하는 확률 탐색.
    current_stage[0] = "확률교체탐색"
    current = global_best
    current_mae = global_best_mae
    for iteration in range(1, RANDOM_ITERATIONS + 1):
        proposal = set(current)
        move = rng.random()
        if move < 0.30 and len(proposal) < MAX_FEATURES:
            proposal.add(rng.randrange(len(features)))
        elif move < 0.55 and len(proposal) > 1:
            proposal.remove(rng.choice(tuple(proposal)))
        else:
            if proposal:
                proposal.remove(rng.choice(tuple(proposal)))
            proposal.add(rng.randrange(len(features)))
        proposal_key = tuple(sorted(proposal))
        proposal_mae = evaluate(proposal_key)[0]
        temperature = max(0.002, 0.08 * (1 - iteration / RANDOM_ITERATIONS))
        if proposal_mae < current_mae or rng.random() < math.exp((current_mae - proposal_mae) / temperature):
            current, current_mae = proposal_key, proposal_mae
        if iteration % 10_000 == 0:
            current = global_best
            current_mae = global_best_mae
            print(
                f"확률탐색 {iteration:,}/{RANDOM_ITERATIONS:,}: "
                f"고유조합={len(cache):,}, 최고 MAE={global_best_mae:.6f}",
                flush=True,
            )

    top = sorted(
        (
            {
                "변수수": len(key),
                "alpha": value[1],
                **value[2],
                "초기모델대비_MAE개선율_pct": (baseline["MAE"] - value[0]) / baseline["MAE"] * 100,
                "변수": " | ".join(features[i] for i in key),
            }
            for key, value in cache.items()
        ),
        key=lambda row: (row["MAE"], row["변수수"]),
    )[:1000]
    for rank, row in enumerate(top, 1):
        row["순위"] = rank
    top_frame = pd.DataFrame(top)
    top_frame = top_frame[["순위", *[c for c in top_frame.columns if c != "순위"]]]
    history = pd.DataFrame(records)
    feature_table = audit.copy()
    feature_table.insert(0, "번호", range(1, len(feature_table) + 1))
    baseline_table = pd.DataFrame([{"모델": "2023년 값 고정", **baseline}])

    top_frame.to_csv(OUT / "상위_1000개_부분집합.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUT / "최고성능_갱신이력.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "124개_feature_부분집합탐색.xlsx", engine="openpyxl") as writer:
        baseline_table.to_excel(writer, sheet_name="초기모델", index=False)
        top_frame.to_excel(writer, sheet_name="상위1000조합", index=False)
        history.to_excel(writer, sheet_name="최고성능갱신", index=False)
        feature_table.to_excel(writer, sheet_name="전체124개feature", index=False)

    summary = {
        "method": "전진선택 + 부동선택 + 확률적 추가·제거·교체 탐색",
        "exhaustive": False,
        "candidate_feature_count": len(features),
        "evaluated_unique_subsets": len(cache),
        "ridge_alphas": ALPHAS,
        "training_target_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "selection_validation_year": 2024,
        "untouched_final_holdout_year": 2025,
        "baseline": baseline,
        "best": top[0],
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
