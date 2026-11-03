"""Build a leakage-safe 2017-2025 traffic-safety-criteria workbook."""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / "data" / "interim" / "안전" / "교통사고"
OUTPUT = SAFE / "교통사고_안전기준표_2017_2025.xlsx"
KEYS = ["연도", "자치구"]


def style(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col in ws.columns:
        values = [str(c.value or "") for c in list(col)[:300]]
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(
            max(max(map(len, values), default=8) + 2, 10), 45
        )


def main():
    accident = pd.read_excel(SAFE / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    driver = pd.read_csv(SAFE / "자치구별_운전자안전_2차.csv")
    population = pd.read_csv(SAFE / "자치구별_주민등록인구_2017_2025.csv")
    workplaces = pd.read_csv(SAFE / "자치구별_의료보장사업장수_2024.csv")

    for frame in (accident, driver, population, workplaces):
        frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)

    accident_cols = KEYS + [
        "사고건수", "사망자수", "부상자수", "사상자수", "치사율(%)",
        "어린이사고_사고건수", "어린이사고_사망자수", "어린이사고_부상자수",
    ]
    driver_cols = KEYS + [
        "운전자안전벨트착용률", "음주운전율", "자동차등록대수",
        "교통단속CCTV누적", "과속신호카메라누적", "음주단속적발건수",
    ]
    panel = accident.loc[accident["연도"].between(2017, 2025), accident_cols].merge(
        driver.loc[driver["연도"].between(2017, 2025), driver_cols],
        on=KEYS, how="left", validate="one_to_one",
    ).merge(
        population[["연도", "자치구", "주민등록인구_계"]],
        on=KEYS, how="left", validate="one_to_one",
    ).merge(
        workplaces[["연도", "자치구", "의료보장사업장수"]],
        on=KEYS, how="left", validate="one_to_one",
    )

    pop = panel["주민등록인구_계"].replace(0, np.nan)
    cars = panel["자동차등록대수"].replace(0, np.nan)
    panel["기준1_인구1만명당_교통사고사망자수"] = panel["사망자수"] / pop * 10_000
    panel["기준2_도로면적당_교통단속CCTV대수"] = np.nan
    panel["기준2_대리지표_인구1만명당_교통단속CCTV누적"] = panel["교통단속CCTV누적"] / pop * 10_000
    panel["기준3_운전자안전벨트착용률"] = panel["운전자안전벨트착용률"]
    panel["기준4_음주운전경험률"] = panel["음주운전율"]
    panel["기준5_자동차등록대수당_교통법규위반건수"] = np.nan
    panel["기준5_참고_자동차1만대당_음주단속적발건수"] = panel["음주단속적발건수"] / cars * 10_000
    panel["기준6_인구1만명당_재난약자수"] = np.nan
    panel["기준7_인구1만명당_의료보장사업장수"] = panel["의료보장사업장수"] / pop * 10_000
    panel["기준8_지역교통안전환경개선사업지출액"] = np.nan
    panel = panel.sort_values(KEYS).reset_index(drop=True)

    target = panel[["연도", "자치구", "사고건수"]].copy()
    target["연도"] -= 1
    target = target.rename(columns={"사고건수": "y_다음연도_교통사고건수"})
    model = panel.merge(target, on=KEYS, how="left", validate="one_to_one")
    model = model[model["연도"].between(2017, 2024)].copy()

    criteria = pd.DataFrame([
        [1, "위험", "인구1만명당 교통사고 사망자수", 50.00, "확보", "2017~2025", "공식 주민등록인구 사용"],
        [2, "경감", "도로면적당 교통단속 CCTV대수", 17.24, "미완성", "-", "과거 도로면적 없음; 대리지표만 수록"],
        [3, "의식", "운전자 안전벨트 착용률", 8.99, "확보", "2017~2025", "원자료 값"],
        [4, "의식", "음주운전 경험률", -8.70, "확보", "2017~2025", "음주운전율로 대응"],
        [5, "의식", "자동차등록대수당 교통법규위반건수", -2.31, "미완성", "-", "전체 위반건수 없음; 음주단속 참고값만 수록"],
        [6, "취약", "인구1만명당 재난약자수", 7.91, "미확보", "-", "65세 이상+유치원·초등학생 결합 필요"],
        [7, "취약", "인구1만명당 의료보장사업장수", 2.09, "부분", "2024", "건강보험통계연보"],
        [8, "경감", "지역교통 안전환경개선사업 지출액", 2.76, "미확보", "-", "행정안전부 내부자료"],
    ], columns=["번호", "유형", "안전기준표 지표", "가중치", "확보상태", "확보기간", "설명"])

    quality = pd.DataFrame([
        {
            "컬럼명": col,
            "전체행수": len(panel),
            "비결측행수": int(panel[col].notna().sum()),
            "결측행수": int(panel[col].isna().sum()),
            "커버리지_pct": round(panel[col].notna().mean() * 100, 1),
        }
        for col in panel.columns
    ])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="기준패널_2017_2025", index=False)
        model.to_excel(writer, sheet_name="다음연도예측_2017_2024", index=False)
        criteria.to_excel(writer, sheet_name="기준표_확보현황", index=False)
        quality.to_excel(writer, sheet_name="데이터품질", index=False)
        for ws in writer.book.worksheets:
            style(ws)

    print(f"saved={OUTPUT}")
    print(f"panel_rows={len(panel)}, model_rows={len(model)}, duplicate_keys={panel.duplicated(KEYS).sum()}")


if __name__ == "__main__":
    main()
