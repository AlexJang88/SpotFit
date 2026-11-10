from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from ml.train_traffic_accident_1y_best import DATA_PATH, INDICATORS, TARGET
from ml.train_traffic_accident_incremental_models import (
    CURRENT,
    MODEL_SPECS,
    TREND_FEATURES,
    TYPE_FEATURES,
    add_external_features,
    candidates,
    metrics,
)
from ml.train_traffic_accident_combination_models import ADVANCED_SPECS, COMBINATION_SPECS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "models" / "교통사고" / "22차~40차"
BASE_FEATURES = list(INDICATORS)
CV_TARGET_YEARS = [2021, 2022, 2023, 2024]


SOURCE_SPECS = {**MODEL_SPECS, **COMBINATION_SPECS, **ADVANCED_SPECS}
VERSION_SPECS = {
    old_version + 20: {
        "source_version": old_version,
        "name": spec["name"],
        "features": list(spec["features"]),
    }
    for old_version, spec in sorted(SOURCE_SPECS.items())
}


def train_one(version: int, frame: pd.DataFrame) -> dict:
    spec = VERSION_SPECS[version]
    features = list(dict.fromkeys([*BASE_FEATURES, *spec["features"]]))
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"{version}차 필수 변수가 없습니다: {missing}")

    labeled = frame[frame[TARGET].notna()].copy()
    future = frame[frame[TARGET].isna()].copy()
    # 모든 차수는 명시된 입력 변수 전체에서 관계를 학습하는 ML 모델만 비교한다.
    # 현재 사고 건수만 그대로 반환하는 persistence 기준모형은 비교 대상에서 제외한다.
    pool = candidates(include_persistence=False)

    fold_rows: list[dict] = []
    cv_predictions: list[pd.DataFrame] = []
    for candidate in pool:
        for target_year in CV_TARGET_YEARS:
            train = labeled[labeled["예측연도"] < target_year]
            valid = labeled[labeled["예측연도"] == target_year]
            fitted = clone(candidate.estimator).fit(train[features], train[TARGET])
            prediction = np.maximum(fitted.predict(valid[features]), 0)
            fold_rows.append({
                "후보명": candidate.name,
                "알고리즘": candidate.family,
                "검증연도": target_year,
                "학습행수": len(train),
                **metrics(valid[TARGET], prediction),
            })
            output = valid[["기준연도", "예측연도", "자치구", TARGET]].copy()
            output["후보명"] = candidate.name
            output["예측값"] = prediction
            cv_predictions.append(output)

    folds = pd.DataFrame(fold_rows)
    comparison = folds.groupby(["후보명", "알고리즘"], as_index=False).agg(
        평균_MAE=("MAE", "mean"),
        평균_RMSE=("RMSE", "mean"),
        평균_R2=("R2", "mean"),
        평균_WMAPE_pct=("WMAPE_pct", "mean"),
        평균_오차기반정확도_pct=("오차기반정확도_pct", "mean"),
    ).sort_values(["평균_MAE", "평균_RMSE"]).reset_index(drop=True)
    selected_name = comparison.iloc[0]["후보명"]
    selected = next(candidate for candidate in pool if candidate.name == selected_name)

    # 2025년은 모델 선택에 사용하지 않는 최종 홀드아웃이다.
    train_to_2024 = labeled[labeled["예측연도"] <= 2024]
    holdout = labeled[labeled["예측연도"] == 2025]
    holdout_model = clone(selected.estimator).fit(train_to_2024[features], train_to_2024[TARGET])
    holdout_prediction = np.maximum(holdout_model.predict(holdout[features]), 0)
    holdout_metrics = metrics(holdout[TARGET], holdout_prediction)
    holdout_output = holdout[["기준연도", "예측연도", "서울시", "자치구", TARGET, *features]].copy()
    holdout_output["예측값"] = np.round(holdout_prediction, 1)
    holdout_output["절대오차"] = np.abs(holdout_output[TARGET] - holdout_output["예측값"])

    final_model = clone(selected.estimator).fit(labeled[features], labeled[TARGET])
    future_prediction = np.maximum(final_model.predict(future[features]), 0)
    forecast = future[["기준연도", "예측연도", "서울시", "자치구", *features]].copy()
    forecast["예측_다음연도_교통사고건수"] = np.round(future_prediction, 1)

    importance = permutation_importance(
        holdout_model,
        holdout[features],
        holdout[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=50,
        random_state=42,
        n_jobs=1,
    )
    importance_output = pd.DataFrame({
        "특성": features,
        "순열중요도_MAE증가": importance.importances_mean,
        "표준편차": importance.importances_std,
    }).sort_values("순열중요도_MAE증가", ascending=False)

    output_dir = OUTPUT_ROOT / f"{version}차"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "UnweightedRawIndicatorsPlusAddedFeatures",
        "version": version,
        "source_version": spec["source_version"],
        "horizon_years": 1,
        "target": TARGET,
        "model": final_model,
        "features": features,
        "selected_candidate": selected_name,
        "manual_weights": None,
        "selection_rule": "2021~2024 시계열 확장 교차검증 평균 MAE 최소",
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
    joblib.dump(payload, output_dir / f"교통사고_{version}차_예측모델.joblib")

    metadata = {
        "차수": version,
        "대응_기존차수": spec["source_version"],
        "설계": f"안전 원천지표 6개(무가중) + {spec['name']}",
        "가중치적용": "없음; 모델이 데이터에서 관계를 직접 학습",
        "예측대상": TARGET,
        "입력특성": features,
        "공통원천특성": BASE_FEATURES,
        "추가특성": spec["features"],
        "선정모델": selected_name,
        "교차검증최고": comparison.iloc[0].to_dict(),
        "2025_홀드아웃성능": holdout_metrics,
        "학습행수": len(labeled),
        "2026_예측행수": len(forecast),
    }
    (output_dir / "model_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    variable_audit = pd.DataFrame([
        *({"구분": "공통 안전 원천지표(무가중)", "특성": feature} for feature in BASE_FEATURES),
        *({"구분": "차수별 추가 특성", "특성": feature} for feature in spec["features"]),
    ])
    with pd.ExcelWriter(output_dir / "모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        variable_audit.to_excel(writer, sheet_name="사용변수", index=False)
        comparison.to_excel(writer, sheet_name="후보모델비교", index=False)
        folds.to_excel(writer, sheet_name="연도별교차검증", index=False)
        pd.concat(cv_predictions, ignore_index=True).to_excel(writer, sheet_name="교차검증예측", index=False)
        holdout_output.to_excel(writer, sheet_name="2025홀드아웃검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        importance_output.to_excel(writer, sheet_name="특성중요도", index=False)

    (output_dir / "README.md").write_text(
        f"# 교통사고 {version}차 모델\n\n"
        f"- 기존 대응 모델: {spec['source_version']}차\n"
        f"- 구성: 안전 원천지표 6개(공식 가중치 미적용) + {spec['name']}\n"
        f"- 선정 모델: {selected_name}\n"
        f"- 2025 홀드아웃 MAE: {holdout_metrics['MAE']:.2f}건\n"
        f"- WMAPE: {holdout_metrics['WMAPE_pct']:.2f}%\n"
        f"- 오차 기반 참고 정확도: {holdout_metrics['오차기반정확도_pct']:.2f}%\n"
        f"- R²: {holdout_metrics['R2']:.4f}\n",
        encoding="utf-8",
    )
    print(
        f"{version}차 완료: {selected_name}, "
        f"MAE={holdout_metrics['MAE']:.2f}, 정확도={holdout_metrics['오차기반정확도_pct']:.2f}%",
        flush=True,
    )
    return {
        "차수": f"{version}차",
        "대응_기존차수": f"{spec['source_version']}차",
        "추가변수군": spec["name"],
        "선정모델": selected_name,
        **{f"홀드아웃_{key}": value for key, value in holdout_metrics.items()},
        "입력특성수": len(features),
    }


def main() -> None:
    raw = pd.read_excel(DATA_PATH, sheet_name="모델데이터")
    required = ["기준연도", "예측연도", "서울시", "자치구", TARGET, *BASE_FEATURES]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    if raw.duplicated(["기준연도", "자치구"]).any():
        raise ValueError("기준연도·자치구 중복이 있습니다.")

    # 가중합 점수나 점수기여 컬럼을 만들지 않고 원천지표를 그대로 사용한다.
    frame = add_external_features(raw)
    summaries = [train_one(version, frame) for version in sorted(VERSION_SPECS)]
    summary = pd.DataFrame(summaries)
    design = pd.DataFrame([
        {
            "차수": f"{version}차",
            "대응_기존차수": f"{spec['source_version']}차",
            "공통변수": ", ".join(BASE_FEATURES),
            "추가변수": ", ".join(spec["features"]),
            "수동가중치": "없음",
        }
        for version, spec in sorted(VERSION_SPECS.items())
    ])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_ROOT / "모델_22차_40차_성능비교.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능비교", index=False)
        design.to_excel(writer, sheet_name="차수별설계", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
