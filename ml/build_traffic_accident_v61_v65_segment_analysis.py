from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.search_traffic_accident_v52_forward_backward import CV_YEARS, prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "교통사고"
V59 = MODEL_ROOT / "51차~60차" / "59차"
VERSION_CUTOFFS = {61: 1, 62: 2, 63: 3, 64: 4, 65: 5}


def regression_prediction(component, train, valid, target_score):
    features = component["features"]
    y = train[target_score].to_numpy(dtype=float).copy()
    if component.get("target_kind") == "delta":
        y -= train["현재연도_종합위험점수"].to_numpy(dtype=float)
    model = clone(component["model"]).fit(train[features], y)
    prediction = model.predict(valid[features])
    if component.get("target_kind") == "delta":
        prediction += valid["현재연도_종합위험점수"].to_numpy(dtype=float)
    blend = float(component.get("blend", 0.0))
    if blend:
        current = valid["현재연도_종합위험점수"].to_numpy(dtype=float)
        prediction = (1 - blend) * prediction + blend * current
    return np.asarray(prediction, dtype=float)


def fold_prediction(bundle, train, valid, horizon):
    target_score = f"{horizon}년후_종합위험점수"
    target_grade = f"{horizon}년후_위험등급"
    if bundle["kind"] == "classifier_ensemble_rank":
        values = []
        for component in (bundle["component_a"], bundle["component_b"]):
            model = clone(component["model"]).fit(
                train[component["features"]], train[target_grade].astype(int)
            )
            values.append(risk(model, valid[component["features"]]))
    elif bundle["kind"] == "ensemble_rank":
        values = [
            regression_prediction(bundle["component_a"], train, valid, target_score),
            regression_prediction(bundle["component_b"], train, valid, target_score),
        ]
    else:
        raise ValueError(f"지원하지 않는 59차 모델 형식: {bundle['kind']}")
    return float(bundle["weight_a"]) * values[0] + (1 - float(bundle["weight_a"])) * values[1]


def make_oof(data, bundles):
    rows = []
    for horizon, bundle in bundles.items():
        target_grade = f"{horizon}년후_위험등급"
        year_col = f"{horizon}년후_목표연도"
        labeled = data[data[target_grade].notna()]
        for target_year in CV_YEARS[horizon]:
            train = labeled[labeled[year_col] < target_year]
            valid = labeled[labeled[year_col] == target_year].copy()
            score = fold_prediction(bundle, train, valid, horizon)
            predicted = 6 - grades_from_risk(score)
            actual = 6 - valid[target_grade].astype(int).to_numpy()
            for district, actual_grade, predicted_grade, predicted_score in zip(
                valid["자치구"], actual, predicted, score
            ):
                rows.append(
                    {
                        "예측간격": horizon,
                        "검증목표연도": target_year,
                        "자치구": district,
                        "실제등급": int(actual_grade),
                        "예측등급": int(predicted_grade),
                        "내부예측위험점수": float(predicted_score),
                        "정확여부": int(actual_grade == predicted_grade),
                        "±1등급이내": int(abs(actual_grade - predicted_grade) <= 1),
                        "절대등급오차": int(abs(actual_grade - predicted_grade)),
                    }
                )
    return pd.DataFrame(rows)


def segment_tables(oof, cutoff):
    selected = oof[oof["실제등급"] <= cutoff].copy()
    yearly = selected.groupby(["예측간격", "검증목표연도"], as_index=False).agg(
        표본수=("자치구", "size"),
        정확등급정확도=("정확여부", "mean"),
        한등급이내정확도=("±1등급이내", "mean"),
        평균등급오차=("절대등급오차", "mean"),
    )
    summary = yearly.groupby("예측간격", as_index=False).agg(
        대상등급수=("표본수", "sum"),
        연평균_정확등급정확도=("정확등급정확도", "mean"),
        연평균_한등급이내정확도=("한등급이내정확도", "mean"),
        연평균_평균등급오차=("평균등급오차", "mean"),
        최저연도_정확도=("정확등급정확도", "min"),
    )
    by_grade = selected.groupby(["예측간격", "실제등급"], as_index=False).agg(
        표본수=("자치구", "size"),
        정확등급재현율=("정확여부", "mean"),
        한등급이내정확도=("±1등급이내", "mean"),
        평균등급오차=("절대등급오차", "mean"),
    )
    confusion_rows = []
    for horizon in (2, 3):
        part = selected[selected["예측간격"] == horizon]
        table = pd.crosstab(part["실제등급"], part["예측등급"], dropna=False).reindex(
            index=range(1, cutoff + 1), columns=range(1, 6), fill_value=0
        )
        for actual_grade, row in table.iterrows():
            record = {"예측간격": horizon, "실제등급": actual_grade}
            record.update({f"예측{grade}등급": int(row[grade]) for grade in range(1, 6)})
            confusion_rows.append(record)
    return selected, yearly, summary, by_grade, pd.DataFrame(confusion_rows)


def main():
    data, _ = prepare_data()
    bundles = {
        2: joblib.load(V59 / "교통사고_59차_2027년_2년후_모델.joblib"),
        3: joblib.load(V59 / "교통사고_59차_2028년_3년후_모델.joblib"),
    }
    oof = make_oof(data, bundles)
    comparison = []

    for version, cutoff in VERSION_CUTOFFS.items():
        top_percent = cutoff * 20
        output = MODEL_ROOT / f"{version}차"
        output.mkdir(parents=True, exist_ok=True)
        selected, yearly, summary, by_grade, confusion = segment_tables(oof, cutoff)
        summary.insert(0, "차수", version)
        summary.insert(2, "분석구간", f"실제 안전 상위 {top_percent}% (안전 1~{cutoff}등급)")
        comparison.append(summary.copy())

        with pd.ExcelWriter(output / f"{version}차_상위{top_percent}퍼센트_등급분석.xlsx", engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="성능요약", index=False)
            yearly.to_excel(writer, sheet_name="연도별성능", index=False)
            by_grade.to_excel(writer, sheet_name="실제등급별성능", index=False)
            confusion.to_excel(writer, sheet_name="혼동행렬", index=False)
            selected.to_excel(writer, sheet_name="검증상세", index=False)

        wrapper = {
            "version": version,
            "base_version": 59,
            "analysis_type": "5등급 유지·실제 안전구간 누적 성능분석",
            "grade_definition": "안전 1등급=가장 안전, 안전 5등급=가장 위험",
            "top_percent": top_percent,
            "included_actual_grades": list(range(1, cutoff + 1)),
            "horizon_models": bundles,
            "summary": summary.to_dict(orient="records"),
        }
        joblib.dump(wrapper, output / f"교통사고_{version}차_상위{top_percent}퍼센트_분석모델.joblib")
        info = {
            "모델명": f"교통사고 {version}차 안전 상위 {top_percent}% 구간분석",
            "기반모델": "59차",
            "5등급제유지": True,
            "분석대상실제등급": list(range(1, cutoff + 1)),
            "등급정의": "안전 1등급=가장 안전, 안전 5등급=가장 위험",
            "주의": "새로 재학습한 모델이 아니라 59차 OOF 예측을 실제 안전구간별로 진단한 분석 모델",
            "성능": summary.to_dict(orient="records"),
        }
        (output / "model_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )

    comparison_frame = pd.concat(comparison, ignore_index=True)
    combined = MODEL_ROOT / "61차~65차_구간별성능비교.xlsx"
    with pd.ExcelWriter(combined, engine="openpyxl") as writer:
        comparison_frame.to_excel(writer, sheet_name="누적구간비교", index=False)
        oof.to_excel(writer, sheet_name="59차_OOF전체예측", index=False)
    print(json.dumps({"comparison": str(combined), "summary": comparison_frame.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
