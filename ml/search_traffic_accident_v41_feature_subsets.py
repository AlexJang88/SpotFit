from __future__ import annotations

import heapq
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.train_best_child_traffic_accident_rate import CURRENT_RATE, ROOT, TARGET, build_panel


OUT = ROOT / "models" / "교통사고" / "41차" / "변수부분집합탐색"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
TOP_K = 1000

# 현재 확보한 데이터에서 확정한 19개 입력 변수다.
FEATURES = [
    CURRENT_RATE,
    "어린이피해자율_최근3년평균",
    "어린이피해자율_최근3년추세",
    "어린이피해자율_최근3년변동성",
    "보행어린이교통사고율_어린이1만명당",
    "보호구역보행어린이교통사고율_어린이1만명당",
    "보행어린이다발_지점수",
    "보호구역어린이다발_지점수",
    "어린이1만명당_자동차등록대수",
    "승용차_낮_평균속도(km/h)",
    "자동차1만대당_전체교통사고건수",
    "도로km당_전체교통사고건수",
    "도로율(%)",
    "어린이인구비율_pct",
    "도로면적당_전체CCTV",
    "자동차음주운전경험률(%)",
    "운전자안전벨트착용률(%)",
    "유치원수",
    "초등학교수",
]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - predicted)
    return {
        "MAE": float(error.mean()),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "R2": float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "WMAPE_pct": float(error.sum() / actual.sum() * 100),
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _ = build_panel()
    missing = [feature for feature in FEATURES if feature not in panel.columns]
    if missing:
        raise ValueError(f"41차 부분집합 탐색 변수가 없습니다: {missing}")

    # Y가 2023년까지인 자료만 학습하고 X(2023) -> Y(2024)를 변수 선택에 사용한다.
    train = panel[panel["예측연도"] <= 2023].copy()
    valid = panel[panel["예측연도"] == 2024].copy()
    if len(train) != 150 or len(valid) != 25:
        raise ValueError(f"예상하지 않은 분할 크기: 학습={len(train)}, 2024검증={len(valid)}")

    x_train_raw = train[FEATURES].to_numpy(dtype=float)
    x_valid_raw = valid[FEATURES].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_valid = valid[TARGET].to_numpy(dtype=float)

    median = np.nanmedian(x_train_raw, axis=0)
    x_train_raw = np.where(np.isnan(x_train_raw), median, x_train_raw)
    x_valid_raw = np.where(np.isnan(x_valid_raw), median, x_valid_raw)
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train_raw - mean) / scale
    x_valid = (x_valid_raw - mean) / scale
    y_mean = y_train.mean()
    y_centered = y_train - y_mean
    gram = x_train.T @ x_train
    cross = x_train.T @ y_centered

    # 사용자가 지정한 2023년 고정 기준과 일반 직전연도 유지 기준을 함께 남긴다.
    rate_2023 = panel[panel["연도"] == 2023].set_index("자치구")[CURRENT_RATE]
    fixed_2023_prediction = valid["자치구"].map(rate_2023).to_numpy(dtype=float)
    persistence_prediction = valid[CURRENT_RATE].to_numpy(dtype=float)
    baselines = pd.DataFrame([
        {"모델": "2023년 값 고정", **metrics(y_valid, fixed_2023_prediction)},
        {"모델": "직전연도 값 유지", **metrics(y_valid, persistence_prediction)},
    ])
    baseline_mae = float(baselines.loc[baselines["모델"] == "2023년 값 고정", "MAE"].iloc[0])

    top: list[tuple[float, int, float, tuple[int, ...], dict[str, float]]] = []
    improvements: list[dict] = []
    best_mae = baseline_mae
    evaluated = 0

    for size in range(1, len(FEATURES) + 1):
        for indices in itertools.combinations(range(len(FEATURES)), size):
            index = np.fromiter(indices, dtype=int)
            subset_gram = gram[np.ix_(index, index)]
            subset_cross = cross[index]
            identity = np.eye(size)
            for alpha in RIDGE_ALPHAS:
                beta = np.linalg.solve(subset_gram + alpha * identity, subset_cross)
                prediction = np.maximum(y_mean + x_valid[:, index] @ beta, 0)
                result = metrics(y_valid, prediction)
                mae = result["MAE"]
                evaluated += 1
                item = (-mae, size, alpha, indices, result)
                if len(top) < TOP_K:
                    heapq.heappush(top, item)
                elif mae < -top[0][0]:
                    heapq.heapreplace(top, item)
                if mae < best_mae - 1e-12:
                    best_mae = mae
                    improvements.append({
                        "평가순번": evaluated,
                        "변수수": size,
                        "alpha": alpha,
                        "MAE": mae,
                        "기준대비_MAE개선율_pct": (baseline_mae - mae) / baseline_mae * 100,
                        "변수": " | ".join(FEATURES[i] for i in indices),
                    })
        print(f"{size}개 변수 조합 완료: 누적 {evaluated:,}회, 현재 최고 MAE={best_mae:.6f}", flush=True)

    top_rows = []
    for negative_mae, size, alpha, indices, result in sorted(top, key=lambda item: -item[0]):
        top_rows.append({
            "순위": len(top_rows) + 1,
            "변수수": size,
            "alpha": alpha,
            **result,
            "기준대비_MAE개선율_pct": (baseline_mae - result["MAE"]) / baseline_mae * 100,
            "변수": " | ".join(FEATURES[i] for i in indices),
        })
    top_frame = pd.DataFrame(top_rows)
    improvement_frame = pd.DataFrame(improvements)
    feature_frame = pd.DataFrame({"번호": range(1, 20), "변수": FEATURES})

    top_frame.to_csv(OUT / "상위_1000개_변수조합.csv", index=False, encoding="utf-8-sig")
    improvement_frame.to_csv(OUT / "최고성능_갱신이력.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "41차_전체부분집합_탐색결과.xlsx", engine="openpyxl") as writer:
        baselines.to_excel(writer, sheet_name="초기기준모델", index=False)
        top_frame.to_excel(writer, sheet_name="상위1000조합", index=False)
        improvement_frame.to_excel(writer, sheet_name="최고성능갱신", index=False)
        feature_frame.to_excel(writer, sheet_name="19개변수", index=False)

    best = top_frame.iloc[0].to_dict()
    summary = {
        "selection_validation_year": 2024,
        "untouched_final_holdout_year": 2025,
        "training_target_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "feature_subset_count": 2 ** len(FEATURES) - 1,
        "ridge_alphas": RIDGE_ALPHAS,
        "total_model_fits": evaluated,
        "baseline": baselines.to_dict(orient="records"),
        "best": best,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
