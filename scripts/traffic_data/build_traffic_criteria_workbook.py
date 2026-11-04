"""Create one Excel workbook for all circled traffic-safety criteria data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "data" / "interim" / "안전" / "교통사고"
OUTPUT = SAFE / "교통사고_안전기준표_전체통합.xlsx"
KEYS = ["연도", "자치구"]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SAFE / name, encoding="utf-8-sig")


def format_sheet(ws) -> None:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for column in ws.columns:
        values = [str(cell.value) if cell.value is not None else "" for cell in list(column)[:200]]
        width = min(max(max(map(len, values), default=8) + 2, 10), 38)
        ws.column_dimensions[get_column_letter(column[0].column)].width = width


def main() -> None:
    accident = pd.read_excel(SAFE / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    driver = read_csv("자치구별_운전자안전_2차.csv")
    population = read_csv("자치구별_인구활동량_1차.csv")
    weighted = read_csv("자치구별_교통사고안전_가중점수_2017_2025.csv")
    road = read_csv("자치구별_도로교통시설_2026_1차.csv")
    registered = read_csv("자치구별_주민등록인구_2017_2025.csv")
    workplaces = read_csv("자치구별_의료보장사업장수_2024.csv")
    cctv_latest = read_csv("자치구별_교통단속CCTV_최신.csv")
    cctv_detail = read_csv("서울_교통단속CCTV_최신_상세.csv")

    for frame in (accident, driver, population, weighted, road, registered, workplaces):
        frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)

    accident_cols = KEYS + [
        "사고건수", "사망자수", "부상자수", "사상자수", "치사율(%)",
        "어린이사고_사고건수", "어린이사고_사망자수", "어린이사고_부상자수",
    ]
    driver_cols = KEYS + [
        "운전자안전벨트착용률", "음주운전율", "자동차등록대수",
        "CCTV전체누적", "과속신호카메라누적", "교통단속CCTV누적", "음주단속적발건수",
    ]
    population_cols = KEYS + [
        "생활인구_연평균", "생활인구_연중시간최대", "생활인구_월변동성",
        "상주인구_연평균", "유동인구_연평균",
    ]

    integrated = accident[accident_cols].merge(
        driver[driver_cols], on=KEYS, how="left", validate="one_to_one"
    ).merge(population[population_cols], on=KEYS, how="left", validate="one_to_one")
    integrated = integrated[integrated["연도"].between(2017, 2025)].copy()
    pop = integrated["생활인구_연평균"].replace(0, np.nan)
    integrated["인구1만명당_교통사고사망자수_생활인구대리"] = integrated["사망자수"] / pop * 10_000
    integrated["인구1만명당_자동차등록대수_대리"] = integrated["자동차등록대수"] / pop * 10_000
    integrated["인구1만명당_교통단속CCTV누적_대리"] = integrated["교통단속CCTV누적"] / pop * 10_000

    score_cols = KEYS + [
        "공통4지표_원가중치점수_70점", "공통4지표_환산안전점수_100점", "공통4지표_연도내순위",
        "CCTV포함5지표_원가중치점수_87.24점", "CCTV포함5지표_환산안전점수_100점",
        "CCTV포함5지표_연도내순위_14개구내",
    ]
    integrated = integrated.merge(weighted[score_cols], on=KEYS, how="left", validate="one_to_one")
    integrated = integrated.merge(
        registered[["연도", "자치구", "주민등록인구_계"]],
        on=KEYS, how="left", validate="one_to_one"
    ).merge(
        workplaces[["연도", "자치구", "의료보장사업장수"]],
        on=KEYS, how="left", validate="one_to_one"
    )
    official_pop = integrated["주민등록인구_계"].replace(0, np.nan)
    integrated["인구1만명당_교통사고사망자수"] = integrated["사망자수"] / official_pop * 10_000
    integrated["인구1만명당_의료보장사업장수"] = integrated["의료보장사업장수"] / official_pop * 10_000
    integrated = integrated.sort_values(KEYS).reset_index(drop=True)

    definitions = pd.DataFrame([
        ["위험", "인구만명당 교통사고 사망자수", 50.00, "완전", "사망자수÷공식 주민등록인구×10,000", "2017~2025 산출"],
        ["경감", "도로면적당 교통단속 CCTV대수", 17.24, "부분", "과속·신호위반 CCTV 최신 집계와 상세 별도 수록", "도로면적 미확보; CCTV는 2026 최신 스냅샷"],
        ["의식", "운전자 안전벨트 착용률", 8.99, "완전", "운전자안전벨트착용률", "기준표와 직접 일치"],
        ["의식", "음주운전 경험률", -8.70, "완전", "음주운전율", "기준표와 직접 대응"],
        ["의식", "자동차등록대수당 교통법규위반건수", -2.31, "부분", "자동차등록대수와 차량밀도 대리지표 수록", "전체 교통법규위반건수 없음"],
        ["취약", "인구만명당 재난약자수", 7.91, "미확보", "-", "관련 원자료 없음"],
        ["취약", "인구만명당 의료보장 사업장수", 2.09, "완전(2024)", "의료보장사업장수÷공식 주민등록인구×10,000", "2024 건강보험통계연보"],
        ["경감", "지역교통 안전환경개선사업 지출액", 2.76, "미확보", "-", "관련 원자료 없음"],
    ], columns=["지표유형", "안전기준표 항목", "가중치", "확보상태", "통합파일 적용", "주의사항"])

    quality_rows = []
    key_count = len(integrated)
    for column in integrated.columns:
        if column in KEYS:
            continue
        non_null = int(integrated[column].notna().sum())
        quality_rows.append({
            "컬럼명": column,
            "비결측행수": non_null,
            "전체행수": key_count,
            "커버리지_pct": round(non_null / key_count * 100, 1),
            "최초연도": int(integrated.loc[integrated[column].notna(), "연도"].min()) if non_null else None,
            "최종연도": int(integrated.loc[integrated[column].notna(), "연도"].max()) if non_null else None,
        })
    quality = pd.DataFrame(quality_rows)

    sources = pd.DataFrame([
        ["교통사고", "자치구별_교통사고_2차.xlsx", "2010~2025", "사고·사망·부상 및 어린이 사고"],
        ["운전자·차량·단속", "자치구별_운전자안전_2차.csv", "2006~2025", "안전벨트·음주운전·자동차·CCTV"],
        ["인구 대리자료", "자치구별_인구활동량_1차.csv", "2017~2025", "생활·상주·유동인구"],
        ["가중점수", "자치구별_교통사고안전_가중점수_2017_2025.csv", "2017~2025", "안전기준표 가중평가 결과"],
        ["도로시설", "자치구별_도로교통시설_2026_1차.csv", "2026", "교차로·횡단보도 현황; 연도 불일치로 통합데이터에는 미병합"],
        ["공식 인구", "자치구별_주민등록인구_2017_2025.csv", "2017~2025", "인구 1만명당 지표의 공식 분모"],
        ["의료보장 사업장", "자치구별_의료보장사업장수_2024.csv", "2024", "건강보험통계연보 제Ⅰ-5표"],
        ["교통단속 CCTV", "자치구별_교통단속CCTV_최신.csv", "2026 최신", "경찰 관리 과속·신호위반 CCTV; 과거 모델 입력 금지"],
    ], columns=["분류", "원본파일", "기간", "내용"])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        integrated.to_excel(writer, sheet_name="통합데이터_2017_2025", index=False)
        definitions.to_excel(writer, sheet_name="기준표_지표정의", index=False)
        quality.to_excel(writer, sheet_name="데이터품질", index=False)
        accident[accident_cols].to_excel(writer, sheet_name="교통사고_장기", index=False)
        driver[driver_cols].to_excel(writer, sheet_name="운전자_CCTV_장기", index=False)
        population[population_cols].to_excel(writer, sheet_name="인구활동량", index=False)
        road.to_excel(writer, sheet_name="도로시설_2026", index=False)
        registered.to_excel(writer, sheet_name="공식인구_2017_2025", index=False)
        workplaces.to_excel(writer, sheet_name="의료보장사업장_2024", index=False)
        cctv_latest.to_excel(writer, sheet_name="교통단속CCTV_최신집계", index=False)
        cctv_detail.to_excel(writer, sheet_name="교통단속CCTV_최신상세", index=False)
        sources.to_excel(writer, sheet_name="출처_구성", index=False)
        for ws in writer.book.worksheets:
            format_sheet(ws)

    print(f"saved={OUTPUT}")
    print(f"integrated_rows={len(integrated)} columns={len(integrated.columns)}")
    print(f"sheets={12}")


if __name__ == "__main__":
    main()
