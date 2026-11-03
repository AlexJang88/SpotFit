"""Build the traffic-accident basic safety score requested by the user.

The official road-area denominator and traffic-law-violation count are not
available. The workbook therefore keeps the strict official fields empty and
publishes a clearly named operational proxy score alongside them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "interim" / "안전" / "교통사고"
OUT_DIR = ROOT / "data" / "processed" / "안전" / "교통사고"
OUT_XLSX = OUT_DIR / "교통사고_기본모델_2024.xlsx"
OUT_CSV = OUT_DIR / "교통사고_기본모델_2024.csv"


def safety_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Cross-sectional rank score from 0 (worst) to 100 (best)."""
    value = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=value.index, dtype=float)
    valid = value.notna()
    count = int(valid.sum())
    if count == 0:
        return result
    if count == 1 or value[valid].nunique() == 1:
        result.loc[valid] = 50.0
        return result
    rank = value[valid].rank(method="average", ascending=True)
    ascending_score = (rank - 1) / (count - 1) * 100
    result.loc[valid] = ascending_score if higher_is_better else 100 - ascending_score
    return result


def style(ws) -> None:
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
            max(max(map(len, values), default=8) + 2, 11), 48
        )


def main() -> None:
    accident = pd.read_excel(SOURCE / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    driver = pd.read_csv(SOURCE / "자치구별_운전자안전_2차.csv", encoding="utf-8-sig")
    population = pd.read_csv(SOURCE / "자치구별_주민등록인구_2017_2025.csv", encoding="utf-8-sig")
    workplace = pd.read_csv(SOURCE / "자치구별_의료보장사업장수_2024.csv", encoding="utf-8-sig")
    cctv = pd.read_csv(SOURCE / "자치구별_교통단속CCTV_최신.csv", encoding="utf-8-sig")

    for frame in (accident, driver, population, workplace):
        frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)

    keys = ["연도", "자치구"]
    model = accident.loc[accident["연도"].eq(2024), keys + ["사망자수"]].merge(
        population.loc[population["연도"].eq(2024), keys + ["주민등록인구_계"]],
        on=keys, how="left", validate="one_to_one",
    ).merge(
        workplace.loc[workplace["연도"].eq(2024), keys + ["의료보장사업장수"]],
        on=keys, how="left", validate="one_to_one",
    ).merge(
        driver.loc[driver["연도"].eq(2024), keys + ["운전자안전벨트착용률", "음주운전율"]],
        on=keys, how="left", validate="one_to_one",
    ).merge(cctv[["자치구", "교통단속CCTV대수_최신", "데이터기준일_최신"]],
            on="자치구", how="left", validate="one_to_one")

    pop = model["주민등록인구_계"].replace(0, np.nan)
    model["위해_인구만명당_교통사고사망자수"] = model["사망자수"] / pop * 10_000
    model["취약_인구만명당_의료보장사업장수"] = model["의료보장사업장수"] / pop * 10_000
    model["경감_도로면적당_교통단속CCTV대수_공식"] = np.nan
    model["경감_대체_인구만명당_최신교통단속CCTV대수"] = model["교통단속CCTV대수_최신"] / pop * 10_000
    model["의식_자동차등록대수당_교통법규위반건수"] = np.nan

    model["안전점수_위해"] = safety_rank(model["위해_인구만명당_교통사고사망자수"], False)
    model["안전점수_취약"] = safety_rank(model["취약_인구만명당_의료보장사업장수"], False)
    model["안전점수_경감_대체"] = safety_rank(model["경감_대체_인구만명당_최신교통단속CCTV대수"], True)
    model["안전점수_의식_안전벨트"] = safety_rank(model["운전자안전벨트착용률"], True)
    model["안전점수_의식_음주운전"] = safety_rank(model["음주운전율"], False)

    # The unavailable -2.31 item is excluded. The remaining 8.99 and 8.70
    # weights are proportionally rescaled so that the consciousness group is 20.
    consciousness_total = 8.99 + 8.70
    seatbelt_weight = 20 * 8.99 / consciousness_total
    drunk_weight = 20 * 8.70 / consciousness_total
    model["적용가중치_위해"] = 50.0
    model["적용가중치_취약"] = 10.0
    model["적용가중치_경감_대체"] = 20.0
    model["적용가중치_의식_안전벨트"] = seatbelt_weight
    model["적용가중치_의식_음주운전"] = drunk_weight

    model["가중기여_위해"] = model["안전점수_위해"] * 0.50
    model["가중기여_취약"] = model["안전점수_취약"] * 0.10
    model["가중기여_경감_대체"] = model["안전점수_경감_대체"] * 0.20
    model["가중기여_의식_안전벨트"] = model["안전점수_의식_안전벨트"] * seatbelt_weight / 100
    model["가중기여_의식_음주운전"] = model["안전점수_의식_음주운전"] * drunk_weight / 100
    contribution_cols = [
        "가중기여_위해", "가중기여_취약", "가중기여_경감_대체",
        "가중기여_의식_안전벨트", "가중기여_의식_음주운전",
    ]
    model["교통사고_기본모델점수_대체산식"] = model[contribution_cols].sum(axis=1, min_count=len(contribution_cols))
    model["교통사고_기본모델순위"] = model["교통사고_기본모델점수_대체산식"].rank(
        method="min", ascending=False
    ).astype("Int64")
    model["공식산식_완전충족"] = False
    model["모델주의"] = (
        "도로면적 미확보로 인구만명당 최신 CCTV를 경감 대체지표로 사용; "
        "교통법규위반 지표 미확보로 안전벨트·음주운전 가중치를 의식 20점에 비례 재배분"
    )
    model = model.sort_values("교통사고_기본모델순위").reset_index(drop=True)

    weights = pd.DataFrame([
        ["위해", "인구만명당 교통사고 사망자수", 50.00, 50.00, "확보", "낮을수록 안전"],
        ["취약", "인구만명당 재난약자수", 7.91, 0.00, "미확보", "같은 지표군 내 의료보장 사업장수에 재배분"],
        ["취약", "인구만명당 의료보장 사업장수", 2.09, 10.00, "2024 확보", "낮을수록 안전"],
        ["경감", "도로면적당 교통단속 CCTV대수", 17.24, 20.00, "분자만 확보", "도로면적이 없어 인구 기준 대체값 사용"],
        ["경감", "지역교통 안전환경개선사업 지출액", 2.76, 0.00, "미확보", "같은 지표군 내 CCTV에 재배분"],
        ["의식", "운전자 안전벨트 착용률", 8.99, seatbelt_weight, "확보", "높을수록 안전"],
        ["의식", "음주운전 경험률", -8.70, drunk_weight, "확보", "낮을수록 안전"],
        ["의식", "자동차등록대수당 교통법규위반건수", -2.31, 0.00, "미확보", "남은 두 항목에 의식 20점을 비례 재배분"],
    ], columns=["지표군", "지표", "기준표_원가중치", "기본모델_적용가중치", "확보상태", "처리"])

    method = pd.DataFrame([
        ["대상", "2024년 서울 25개 자치구"],
        ["표준화", "자치구 횡단면 순위점수: 최저 안전 0, 최고 안전 100, 동률 평균순위"],
        ["총점", "위해 50 + 취약 10 + 경감 20 + 의식 20 = 100"],
        ["가중치 재배분", "같은 지표군 안에서만 비례 재배분; 다른 지표군으로 이동하지 않음"],
        ["경감 대체", "공식 도로면적당 CCTV 대신 인구만명당 최신 CCTV 사용"],
        ["해석", "점수가 높을수록 상대적으로 안전"],
    ], columns=["항목", "내용"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        model.to_excel(writer, sheet_name="기본모델_결과", index=False)
        weights.to_excel(writer, sheet_name="가중치_재배분", index=False)
        method.to_excel(writer, sheet_name="계산방법_주의", index=False)
        for ws in writer.book.worksheets:
            style(ws)

    print(f"saved={OUT_XLSX}")
    print(f"saved={OUT_CSV}")
    print(f"rows={len(model)}, score_nulls={model['교통사고_기본모델점수_대체산식'].isna().sum()}")
    print(model[["교통사고_기본모델순위", "자치구", "교통사고_기본모델점수_대체산식"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
