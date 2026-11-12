from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "교통사고" / "87차.피해자수Poisson기간별최종"
PROCESSED = ROOT / "data" / "processed" / "교통사고_87차_피해자수Poisson기간별최종_2027_2028.xlsx"


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    v84_dir = ROOT / "models" / "교통사고" / "84차.피해자수PoissonRaw전진선택"
    v86_dir = ROOT / "models" / "교통사고" / "86차.피해자수Poisson3년최적화"
    v84_info = json.loads((v84_dir / "model_info.json").read_text(encoding="utf-8"))
    v86_info = json.loads((v86_dir / "model_info.json").read_text(encoding="utf-8"))
    v84_bundle = joblib.load(v84_dir / "교통사고_84차_피해자수PoissonRaw전진선택.joblib")
    v86_bundle = joblib.load(v86_dir / "교통사고_86차_3년최적화.joblib")

    v84_xlsx = ROOT / v84_info["processed"]
    old_compare = ROOT / "data" / "processed" / "교통사고_85차_86차_기간별최적화.xlsx"
    w2 = pd.read_excel(v84_xlsx, sheet_name="최종지표와가중치")
    w2.insert(0, "적용모델", "87차_2년(84차검증조합)")
    weights_old = pd.read_excel(old_compare, sheet_name="기간별지표가중치")
    w3 = weights_old[weights_old["차수"] == 86].copy()
    w3.insert(0, "적용모델", "87차_3년(86차검증조합)")

    perf2 = pd.read_excel(v84_xlsx, sheet_name="성능평가")
    perf2 = perf2[perf2["예측기간"] == "2년 후"].copy(); perf2.insert(0, "적용모델", "87차_2년")
    perf_old = pd.read_excel(old_compare, sheet_name="성능평가")
    perf3 = perf_old[perf_old["차수"] == 86].copy(); perf3.insert(0, "적용모델", "87차_3년")

    detail2 = pd.read_excel(v84_xlsx, sheet_name="검증상세")
    # 84차 상세 시트는 2년 블록 뒤에 3년 블록을 같은 행 수로 저장했다.
    detail2 = detail2.iloc[:len(detail2)//2].copy(); detail2.insert(0, "적용모델", "87차_2년")
    detail_old = pd.read_excel(old_compare, sheet_name="검증상세")
    detail3 = detail_old[detail_old["차수"] == 86].copy(); detail3.insert(0, "적용모델", "87차_3년")

    future2 = pd.read_excel(v84_xlsx, sheet_name="2027_2028피해자수와등급")
    future2 = future2[future2["예측기간"] == "2년 후"].copy(); future2.insert(0, "적용모델", "87차_2년")
    future_old = pd.read_excel(old_compare, sheet_name="2027_2028예측")
    future3 = future_old[future_old["차수"] == 86].copy(); future3.insert(0, "적용모델", "87차_3년")

    summary = pd.DataFrame([
        {"적용모델": "87차_2년", "예측연도": 2027, "원검증조합": "84차", "지표수": v84_info["최종지표수"],
         "정확등급정확도": v84_info["기간별2025성능"]["2"]["정확등급정확도"],
         "1등급이내정확도": v84_info["기간별2025성능"]["2"]["1등급이내정확도"],
         "평균등급오차": v84_info["기간별2025성능"]["2"]["평균등급오차"]},
        {"적용모델": "87차_3년", "예측연도": 2028, "원검증조합": "86차", "지표수": v86_info["최종지표수"],
         "정확등급정확도": v86_info["2025독립성능"]["정확등급정확도"],
         "1등급이내정확도": v86_info["2025독립성능"]["1등급이내정확도"],
         "평균등급오차": v86_info["2025독립성능"]["평균등급오차"]},
    ])
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="87차요약", index=False)
        pd.concat([w2, w3], ignore_index=True).to_excel(writer, sheet_name="기간별지표가중치", index=False)
        pd.concat([perf2, perf3], ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat([detail2, detail3], ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat([future2, future3], ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    info = {
        "모델명": "교통사고 87차 피해자수 Poisson 기간별 최종모델",
        "Y": "다음 연도 어린이 교통사고 사망자 수+부상자 수",
        "인구처리": "어린이 인구 Poisson exposure",
        "불변근간": "미래지표→5분위→Poisson 계수가중치→종합점수→안전 1~5등급",
        "2년모델": {"근거조합": "84차", "지표수": v84_info["최종지표수"], "추가지표": v84_info["추가지표"],
                     "2025독립성능": v84_info["기간별2025성능"]["2"]},
        "3년모델": {"근거조합": "86차", "지표수": v86_info["최종지표수"], "추가지표": v86_info["추가지표"],
                     "2025독립성능": v86_info["2025독립성능"]},
        "기간별통합정확등급": .68, "기간별1등급이내정확도": 1.0,
        "안전등급방향": "1=가장 안전, 5=가장 위험",
        "processed": str(PROCESSED.relative_to(ROOT)),
    }
    joblib.dump({"version": 87, "two_year_model": v84_bundle, "three_year_model": v86_bundle,
                 "routing": {2: "two_year_model", 3: "three_year_model"}},
                OUT / "교통사고_87차_피해자수Poisson기간별최종.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
