from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "교통사고"
PERSISTENCE_BASELINE = {
    "정의": "현재 안전등급이 예측시점에도 그대로 유지된다고 예측",
    "2년후": {"정확등급정확도": 0.488, "한등급이내정확도": 0.848, "평균등급오차": 0.672},
    "3년후": {"정확등급정확도": 0.340, "한등급이내정확도": 0.850, "평균등급오차": 0.820},
}


def normalize_v51_processed() -> str:
    path = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
    workbook = load_workbook(path)
    sheet = workbook["51차_2027_2028예측"]
    headers = {cell.value: cell.column for cell in sheet[1]}
    new_headers = ["예측안전점수", "예측안전등급", "예측안전순위"]
    for header in new_headers:
        if header not in headers:
            column = sheet.max_column + 1
            sheet.cell(1, column, header)
            headers[header] = column
    for row in range(2, sheet.max_row + 1):
        risk_score = float(sheet.cell(row, headers["예측위험점수"]).value)
        risk_grade = int(sheet.cell(row, headers["예측위험등급"]).value)
        risk_rank = int(sheet.cell(row, headers["예측위험순위"]).value)
        sheet.cell(row, headers["예측안전점수"], 120.0 - risk_score)
        sheet.cell(row, headers["예측안전등급"], 6 - risk_grade)
        sheet.cell(row, headers["예측안전순위"], 26 - risk_rank)
    workbook.save(path)
    return str(path)


def main() -> None:
    paths = sorted((MODEL_ROOT / "51차~60차").glob("*/model_info.json"))
    updated = []
    for path in paths:
        info = json.loads(path.read_text(encoding="utf-8"))
        info["내부모델등급방향"] = "위험등급: 1=가장 위험, 5=가장 안전"
        info["최종표시등급"] = "안전등급: 1=가장 안전, 5=가장 위험"
        info["안전등급변환"] = "안전등급 = 6 - 내부 위험등급"
        info["안전순위변환"] = "안전순위 = 26 - 내부 위험순위"
        info["공식기준모델"] = "Persistence"
        info["Persistence기준성능"] = PERSISTENCE_BASELINE
        if path.parent.name == "51차":
            info["Persistence대비정확도개선"] = {"2년후_pp": 7.2, "3년후_pp": 21.0}
        elif path.parent.name == "59차":
            info["Persistence대비정확도개선"] = {"2년후_pp": 10.4, "3년후_pp": 26.0}
        if "등급체계" in info:
            info["등급체계"] = "서울 25개 자치구 상대순위 5분위, 안전 1등급=가장 안전·안전 5등급=가장 위험"
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        updated.append(str(path))
    print(json.dumps({"updated": updated, "processed": normalize_v51_processed()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
