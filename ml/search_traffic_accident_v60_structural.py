from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor

from ml.lighten_traffic_accident_v57 import importance_order
from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "58차"
OUTPUT = ROOT / "models" / "교통사고" / "60차"
REPORT = ROOT / "artifacts" / "reports" / "traffic_v60_structural"


def add_temporal_features(data: pd.DataFrame, sources: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = data.sort_values(["자치구", "연도"]).copy()
    group = frame.groupby("자치구", sort=False)
    added = []
    for source in sources:
        if source not in frame or not pd.api.types.is_numeric_dtype(frame[source]):
            continue
        lag1 = f"시계열_{source}_lag1"
        lag2 = f"시계열_{source}_lag2"
        diff = f"시계열_{source}_증감1"
        mean3 = f"시계열_{source}_평균3"
        trend = f"시계열_{source}_추세3"
        frame[lag1] = group[source].shift(1)
        frame[lag2] = group[source].shift(2)
        frame[diff] = frame[source] - frame[lag1]
        frame[mean3] = pd.concat([frame[source], frame[lag1], frame[lag2]], axis=1).mean(axis=1)
        frame[trend] = (frame[source] - frame[lag2]) / 2
        added.extend([lag1, lag2, diff, mean3, trend])
    frame["자치구_과거위험점수평균"] = group["현재연도_종합위험점수"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    frame["현재연도_서울내위험순위pct"] = frame.groupby("연도")["현재연도_종합위험점수"].rank(
        ascending=False, method="average", pct=True
    )
    added.extend(["자치구_과거위험점수평균", "현재연도_서울내위험순위pct"])
    return frame.sort_values(["연도", "자치구"]).reset_index(drop=True), added


def base_feature_order(bundle: dict) -> list[str]:
    scores = {}
    for component in (bundle["component_a"], bundle["component_b"]):
        order = importance_order(component)
        for index, feature in enumerate(order):
            scores[feature] = scores.get(feature, 0.0) + 1 / (index + 1)
    return [feature for feature, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def ordinal_score(train_x, train_y, valid_x):
    expected = np.ones(len(valid_x), dtype=float)
    for threshold in (1, 2, 3, 4):
        binary = (train_y > threshold).astype(int)
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=42)),
        ])
        model.fit(train_x, binary)
        expected += model.predict_proba(valid_x)[:, 1]
    return -expected


def evaluate_metrics(actual, score):
    predicted = grades_from_risk(score)
    distance = np.abs(actual - predicted)
    return float(np.mean(actual == predicted)), float(np.mean(distance <= 1)), float(np.mean(distance))


def ranker_defs():
    result = {}
    for depth in (1, 2, 3):
        for leaf in (5, 15):
            name = f"XGBRank_d{depth}_leaf{leaf}"
            result[name] = dict(
                objective="rank:pairwise", n_estimators=180, learning_rate=0.035,
                max_depth=depth, min_child_weight=leaf, subsample=0.85,
                colsample_bytree=0.8, reg_lambda=10, reg_alpha=0.2,
                random_state=42, n_jobs=1,
            )
    return result


def regressor_defs():
    result = {}
    for depth in (1, 2, 3):
        for leaf in (5, 15):
            name = f"XGBReg_d{depth}_leaf{leaf}"
            result[name] = dict(
                objective="reg:squarederror", n_estimators=220, learning_rate=0.03,
                max_depth=depth, min_child_weight=leaf, subsample=0.85,
                colsample_bytree=0.8, reg_lambda=10, reg_alpha=0.2,
                random_state=42, n_jobs=1,
            )
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    original, _ = prepare_data()
    bundles = {
        horizon: joblib.load(SOURCE / f"교통사고_58차_{2025+horizon}년_{horizon}년후_모델.joblib")
        for horizon in (2, 3)
    }
    combined_order = []
    for horizon in (2, 3):
        combined_order.extend(base_feature_order(bundles[horizon]))
    combined_order = list(dict.fromkeys(combined_order))
    raw_sources = []
    for feature in combined_order:
        candidate = feature.removeprefix("점수_")
        if candidate in original and not candidate.startswith(("구_", "위험", "현재연도_")):
            raw_sources.append(candidate)
    raw_sources = list(dict.fromkeys(raw_sources))[:18]
    data, temporal = add_temporal_features(original, raw_sources)

    all_detail = []
    all_summary = []
    winners = {}
    oof_store = {}
    spec_store = {}

    for horizon in (2, 3):
        order = base_feature_order(bundles[horizon])
        sets = {
            "base20_temporal10": list(dict.fromkeys(order[:20] + temporal[:52])),
            "base30_temporal15": list(dict.fromkeys(order[:30] + temporal[:77])),
            "base40_temporal18": list(dict.fromkeys(order[:40] + temporal)),
            "compact25": order[:25],
            "compact35": order[:35],
        }
        target_score = f"{horizon}년후_종합위험점수"
        target_grade = f"{horizon}년후_위험등급"
        year_col = f"{horizon}년후_목표연도"
        labeled = data[data[target_grade].notna()].copy()
        rows = []

        for set_name, features in sets.items():
            for model_name, params in ranker_defs().items():
                config = f"{model_name}@{set_name}"
                for year in CV_YEARS[horizon]:
                    train = labeled[labeled[year_col] < year].sort_values([year_col, "자치구"])
                    valid = labeled[labeled[year_col] == year].sort_values("자치구")
                    groups = train.groupby(year_col, sort=True).size().to_numpy()
                    model = XGBRanker(**params)
                    model.fit(train[features], train[target_score], group=groups, verbose=False)
                    score = model.predict(valid[features])
                    actual = valid[target_grade].astype(int).to_numpy()
                    exact, within1, error = evaluate_metrics(actual, score)
                    rows.append([horizon, config, year, exact, within1, error, len(features)])
                    oof_store[(horizon, config, year)] = score
                spec_store[(horizon, config)] = ("ranker", params, features)

            for model_name, params in regressor_defs().items():
                for target_kind in ("score", "delta"):
                    config = f"{model_name}@{set_name}|{target_kind}"
                    for year in CV_YEARS[horizon]:
                        train = labeled[labeled[year_col] < year]
                        valid = labeled[labeled[year_col] == year]
                        y = train[target_score].to_numpy(dtype=float).copy()
                        if target_kind == "delta":
                            y -= train["현재연도_종합위험점수"].to_numpy(dtype=float)
                        model = XGBRegressor(**params).fit(train[features], y, verbose=False)
                        score = model.predict(valid[features])
                        if target_kind == "delta":
                            score += valid["현재연도_종합위험점수"].to_numpy(dtype=float)
                        actual = valid[target_grade].astype(int).to_numpy()
                        exact, within1, error = evaluate_metrics(actual, score)
                        rows.append([horizon, config, year, exact, within1, error, len(features)])
                        oof_store[(horizon, config, year)] = score
                    spec_store[(horizon, config)] = ("regressor", params, features, target_kind)

            config = f"OrdinalLogistic@{set_name}"
            for year in CV_YEARS[horizon]:
                train = labeled[labeled[year_col] < year]
                valid = labeled[labeled[year_col] == year]
                score = ordinal_score(train[features], train[target_grade].astype(int).to_numpy(), valid[features])
                actual = valid[target_grade].astype(int).to_numpy()
                exact, within1, error = evaluate_metrics(actual, score)
                rows.append([horizon, config, year, exact, within1, error, len(features)])
                oof_store[(horizon, config, year)] = score
            spec_store[(horizon, config)] = ("ordinal", None, features)

        detail = pd.DataFrame(rows, columns=["예측간격", "모델", "검증목표연도", "exact", "within1", "error", "변수수"])
        summary = detail.groupby(["예측간격", "모델", "변수수"], as_index=False).agg(
            exact=("exact", "mean"), within1=("within1", "mean"), error=("error", "mean"), min_exact=("exact", "min")
        ).sort_values(["exact", "within1", "error", "min_exact"], ascending=[False, False, True, False])

        # OOF stacking: blend rank-percentiles of the strongest structurally different candidates.
        top = list(summary.head(20)["모델"])
        stack_rows = []
        for i, left in enumerate(top):
            for right in top[i + 1:]:
                for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
                    name = f"STACK[{left}]+[{right}]|w{weight:g}"
                    for year in CV_YEARS[horizon]:
                        valid = labeled[labeled[year_col] == year].sort_values("자치구")
                        a = pd.Series(oof_store[(horizon, left, year)]).rank(pct=True).to_numpy()
                        b = pd.Series(oof_store[(horizon, right, year)]).rank(pct=True).to_numpy()
                        score = weight * a + (1-weight) * b
                        actual = valid[target_grade].astype(int).to_numpy()
                        exact, within1, error = evaluate_metrics(actual, score)
                        stack_rows.append([horizon, name, year, exact, within1, error, -1])
                    spec_store[(horizon, name)] = ("stack", left, right, weight)
        detail = pd.concat([detail, pd.DataFrame(stack_rows, columns=detail.columns)], ignore_index=True)
        summary = detail.groupby(["예측간격", "모델", "변수수"], as_index=False).agg(
            exact=("exact", "mean"), within1=("within1", "mean"), error=("error", "mean"), min_exact=("exact", "min")
        ).sort_values(["exact", "within1", "error", "min_exact"], ascending=[False, False, True, False])
        winners[horizon] = summary.iloc[0].to_dict()
        all_detail.append(detail)
        all_summary.append(summary)
        print(json.dumps({"horizon": horizon, "winner": winners[horizon]}, ensure_ascii=False), flush=True)

    # A shared configuration is also reported as the multi-horizon common-structure result.
    shared = pd.concat(all_summary)
    shared["구조"] = shared["모델"].str.replace(r"@.*", "", regex=True)
    shared_summary = shared.groupby("구조", as_index=False).agg(평균정확도=("exact", "mean"), 최저정확도=("min_exact", "min"))
    shared_summary = shared_summary.sort_values(["평균정확도", "최저정확도"], ascending=[False, False])

    with pd.ExcelWriter(OUTPUT / "60차_구조개선탐색결과.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([winners[2], winners[3]]).to_excel(writer, sheet_name="최고후보", index=False)
        pd.concat(all_summary, ignore_index=True).to_excel(writer, sheet_name="전체요약", index=False)
        pd.concat(all_detail, ignore_index=True).to_excel(writer, sheet_name="연도별검증", index=False)
        shared_summary.to_excel(writer, sheet_name="2년3년공통구조", index=False)
        pd.DataFrame({"시계열원천변수": raw_sources}).to_excel(writer, sheet_name="시계열파생원천", index=False)
    info = {
        "탐색명": "교통사고 60차 구조개선 탐색",
        "방법": ["원천변수 lag·증감·평균·추세", "자치구 과거추세", "XGB pairwise ranking", "순서형 분류", "XGB 미래점수·증감 회귀", "OOF 순위 스태킹", "2년·3년 공통구조 비교"],
        "최고후보": {"2년후": winners[2], "3년후": winners[3]},
        "59차기준": {"2년후": 0.592, "3년후": 0.600, "평균": 0.596},
    }
    (OUTPUT / "search_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
