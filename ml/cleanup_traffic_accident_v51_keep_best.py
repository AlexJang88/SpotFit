from pathlib import Path

from openpyxl import load_workbook
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx"
KEEP = {"51차점수등급데이터", "변수가중치", "51차고도화성능", "고도화2026등급예측"}
REPORT = ROOT / "models" / "교통사고" / "51차" / "51차_고도화등급예측평가.xlsx"
INFO = ROOT / "models" / "교통사고" / "51차" / "model_info.json"


def main() -> None:
    workbook = load_workbook(BOOK)
    removed = []
    for name in list(workbook.sheetnames):
        if name not in KEEP:
            workbook.remove(workbook[name])
            removed.append(name)
    workbook.save(BOOK)
    performance = pd.read_excel(BOOK, sheet_name="51차고도화성능")
    performance = performance[performance["모델"] == "고도화 51차"]
    with pd.ExcelWriter(BOOK, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        performance.to_excel(writer, sheet_name="51차고도화성능", index=False)

    sheets = pd.read_excel(REPORT, sheet_name=None)
    comparison = sheets["모델비교"]
    best_name = comparison.iloc[0]["모델"]
    best_summary = comparison.iloc[[0]]
    best_years = sheets["연도별검증"][sheets["연도별검증"]["모델"] == best_name]
    forecast = sheets["2026고도화예측"]
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        best_summary.to_excel(writer, sheet_name="최종모델성능", index=False)
        best_years.to_excel(writer, sheet_name="연도별검증", index=False)
        forecast.to_excel(writer, sheet_name="2026고도화예측", index=False)

    info = json.loads(INFO.read_text(encoding="utf-8"))
    for key in ["점수예측모델", "개선등급예측모델", "2021_2025_평균_정확등급정확도_pct",
                "2021_2025_평균_1칸이내정확도_pct", "평균등급오차"]:
        info.pop(key, None)
    info["현재최종모델"] = info.pop("고도화모델")
    info["현재최종변수세트"] = info.pop("고도화변수세트")
    info["현재_정확등급정확도_pct"] = info.pop("고도화_2021_2025_평균_정확등급정확도_pct")
    info["현재_1칸이내정확도_pct"] = info.pop("고도화_2021_2025_평균_1칸이내정확도_pct")
    info["현재_평균등급오차"] = info.pop("고도화_평균등급오차")
    info["상태"] = "51차 최종 고도화 모델"
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REMOVED_SHEETS=" + ",".join(removed))
    print("KEPT_SHEETS=" + ",".join(workbook.sheetnames))


if __name__ == "__main__":
    main()
