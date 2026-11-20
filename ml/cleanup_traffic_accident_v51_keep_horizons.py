from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "교통사고" / "51차"
DATA = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
KEEP_SHEETS = {"변수가중치", "51차_장기예측성능", "51차_2027_2028예측", "51차_2년3년목표데이터"}


def main() -> None:
    workbook = load_workbook(DATA)
    removed = []
    for name in list(workbook.sheetnames):
        if name not in KEEP_SHEETS:
            workbook.remove(workbook[name]); removed.append(name)
    workbook.save(DATA)

    performance = pd.read_excel(DATA, sheet_name="51차_장기예측성능")
    rows = {}
    for _, row in performance.iterrows():
        horizon = int(row["예측간격"])
        rows[str(horizon)] = {
            "정확등급정확도_pct": round(float(row["정확등급정확도"] * 100), 1),
            "1칸이내정확도_pct": round(float(row["±1등급정확도"] * 100), 1),
            "평균등급오차": round(float(row["평균등급오차"]), 3),
            "최종모델": row["모델"],
        }
    info = {
        "모델명": "교통사고 51차 2027·2028 직접예측모델",
        "기준입력연도": 2025,
        "등급체계": "서울 25개 자치구 상대순위 5분위, 1등급 고위험·5등급 저위험",
        "2027년_2년후모델": rows["2"],
        "2028년_3년후모델": rows["3"],
        "상태": "51차 최종 장기 직접예측 모델 2종",
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REMOVED_SHEETS=" + ",".join(removed))
    print("KEPT_SHEETS=" + ",".join(workbook.sheetnames))


if __name__ == "__main__":
    main()
