from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "AI수집" / "안전" / "교통사고"
RAW_TRAFFIC = ROOT / "data" / "raw" / "안전" / "교통사고"
INTERIM = ROOT / "data" / "interim" / "안전" / "교통사고"


def kosis_file(pattern: str) -> Path:
    # 사용자가 raw/안전/교통사고에 확정 배치한 파일을 우선 사용하고,
    # 없을 때만 AI수집 원본을 찾는다.
    matches = list(RAW_TRAFFIC.rglob(pattern))
    if not matches:
        matches = list(RAW.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {matches}")
    return matches[0]


def to_long(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    year_cols = [c for c in df.columns if str(c).strip().endswith("년")]
    out = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="연도", value_name="값")
    out["연도"] = out["연도"].str.extract(r"(\d{4})").astype(int)
    out["값"] = pd.to_numeric(out["값"], errors="coerce")
    return out.dropna(subset=["값"])


def build_workplaces() -> pd.DataFrame:
    src = kosis_file("KOSIS_시군구별_적용인구현황_2017_2024*.csv")
    df = pd.read_csv(src, encoding="utf-8")
    # KOSIS 표는 시도와 그 하위 시군구가 한 열에 계층 순서로 놓인다.
    # 이름이 같은 중구·서구 등이 전국에 반복되므로 서울~부산 사이 구간만 취한다.
    locations = df["시군구별"]
    seoul_start = locations[locations.eq("서울특별시")].index.min()
    busan_start = locations[locations.eq("부산광역시")].index.min()
    df = df.loc[(df.index > seoul_start) & (df.index < busan_start)]
    df = df[(df["적용인구별"] == "사업장수") & df["시군구별"].str.endswith("구", na=False)]
    long = to_long(df, ["시군구별", "적용인구별"])
    out = long.rename(columns={"시군구별": "자치구", "값": "의료보장사업장수"})[
        ["연도", "자치구", "의료보장사업장수"]
    ]
    out["의료보장사업장수"] = out["의료보장사업장수"].astype(int)

    pop = pd.read_csv(INTERIM / "자치구별_주민등록인구_2017_2025.csv")
    out = out.merge(pop[["연도", "자치구", "주민등록인구_계"]], on=["연도", "자치구"], validate="one_to_one")
    out["인구만명당_의료보장사업장수"] = out["의료보장사업장수"] / out["주민등록인구_계"] * 10_000
    out["출처_사업장"] = "KOSIS 국민건강보험공단 시군구별 적용인구 현황(TX_35001_A008)"
    out["출처_인구"] = "서울 열린데이터광장 주민등록인구(연도별)"
    out.to_csv(INTERIM / "자치구별_인구만명당_의료보장사업장수_2017_2024.csv", index=False, encoding="utf-8-sig")
    return out


def build_roads() -> pd.DataFrame:
    src = kosis_file("KOSIS_서울시_도로현황_도로율_2017_2025*.csv")
    df = pd.read_csv(src, encoding="utf-8")
    df = df[(df["항목"] == "도로면적[㎢]") & df["자치구별"].str.endswith("구", na=False)]
    long = to_long(df, ["자치구별", "항목"])
    out = long.rename(columns={"자치구별": "자치구", "값": "도로면적_km2"})[["연도", "자치구", "도로면적_km2"]]
    out["도로면적_m2"] = out["도로면적_km2"] * 1_000_000
    out["출처"] = "KOSIS 서울시 도로현황(도로율)(DT_201004_K050002)"
    out.to_csv(INTERIM / "자치구별_도로면적_2017_2021.csv", index=False, encoding="utf-8-sig")
    return out


def build_cctv_backcast() -> pd.DataFrame:
    detail = pd.read_csv(INTERIM / "서울_교통단속CCTV_최신_상세.csv")
    eligible = detail[detail["지역안전지수_포함"].astype(str).str.lower().eq("true")].copy()
    eligible["설치연도"] = pd.to_numeric(eligible["설치연도"], errors="coerce")
    districts = sorted(detail["자치구"].dropna().unique())
    rows = []
    for year in range(2017, 2026):
        counts = eligible[eligible["설치연도"].le(year)].groupby("자치구").size()
        rows.extend((year, gu, int(counts.get(gu, 0))) for gu in districts)
    out = pd.DataFrame(rows, columns=["연도", "자치구", "교통단속CCTV대수_현존목록역산"])
    out["산출주의"] = "2026 최신 목록 중 설치연도≤해당연도인 장비 수; 철거·교체된 과거 장비는 포함되지 않아 당시 실제 보유대수와 다를 수 있음"
    out["출처"] = "공공데이터포털 전국무인교통단속카메라표준데이터(기준일 2026-10-20)"
    out.to_csv(INTERIM / "자치구별_교통단속CCTV_현존목록역산_2017_2025.csv", index=False, encoding="utf-8-sig")
    return out


def build_road_cctv(roads: pd.DataFrame, cctv: pd.DataFrame) -> pd.DataFrame:
    out = roads.merge(cctv, on=["연도", "자치구"], how="left", validate="one_to_one", suffixes=("_도로", "_CCTV"))
    out["도로면적당_교통단속CCTV대수_대_km2"] = out["교통단속CCTV대수_현존목록역산"] / out["도로면적_km2"]
    out["완성도"] = "부분: 도로면적은 해당연도 공식값, CCTV는 최신 현존목록 설치연도 역산"
    out.to_csv(INTERIM / "자치구별_도로면적당_교통단속CCTV_2017_2021_부분.csv", index=False, encoding="utf-8-sig")
    return out


def build_status(workplaces: pd.DataFrame, roads: pd.DataFrame, cctv: pd.DataFrame) -> pd.DataFrame:
    status = pd.DataFrame(
        [
            ["인구만명당 의료보장 사업장수", "완료", "2017-2024", len(workplaces), "공식 사업장수+공식 주민등록인구"],
            ["도로면적", "완료(공표 연도)", "2017-2021", len(roads), "2022-2025 자치구 값은 내려받은 공식 표에 미수록"],
            ["교통단속 CCTV대수", "부분", "2017-2025", len(cctv), "2026 현존목록의 설치연도 역산; 당시 실제 스냅샷 아님"],
            ["도로면적당 교통단속 CCTV대수", "부분", "2017-2021", len(roads), "공식 도로면적+CCTV 현존목록 역산"],
            ["자동차등록대수", "기확보", "2017-2025", 225, "기존 공식 자료"],
            ["교통법규위반건수", "미확보", "-", 0, "공개자료는 전국 합계이며 자치구별 단속건수는 확인되지 않음"],
            ["자동차등록대수당 교통법규위반건수", "미완성", "-", 0, "분자인 자치구별 교통법규위반건수 필요"],
        ],
        columns=["지표_원문", "상태", "연도", "행수", "설명"],
    )
    status.to_csv(INTERIM / "교통사고_세모지표_수집현황.csv", index=False, encoding="utf-8-sig")
    return status


def build_combined(workplaces: pd.DataFrame, roads: pd.DataFrame, cctv: pd.DataFrame, status: pd.DataFrame) -> None:
    pop = pd.read_csv(INTERIM / "자치구별_주민등록인구_2017_2025.csv")
    panel = pop[["연도", "자치구", "주민등록인구_계"]].copy()
    panel = panel.merge(
        workplaces[["연도", "자치구", "의료보장사업장수", "인구만명당_의료보장사업장수"]],
        on=["연도", "자치구"], how="left", validate="one_to_one",
    )
    panel = panel.merge(roads[["연도", "자치구", "도로면적_km2", "도로면적_m2"]],
                        on=["연도", "자치구"], how="left", validate="one_to_one")
    panel = panel.merge(cctv[["연도", "자치구", "교통단속CCTV대수_현존목록역산"]],
                        on=["연도", "자치구"], how="left", validate="one_to_one")
    cars = pd.read_csv(ROOT / "data" / "raw" / "안전" / "교통사고" / "자치구별_자동차등록대수_2006_2025.csv")
    panel = panel.merge(cars[["연도", "자치구", "자동차등록대수"]],
                        on=["연도", "자치구"], how="left", validate="one_to_one")
    panel["도로면적당_교통단속CCTV대수_대_km2"] = (
        panel["교통단속CCTV대수_현존목록역산"] / panel["도로면적_km2"]
    )
    panel["교통법규위반건수"] = pd.NA
    panel["자동차등록대수당_교통법규위반건수"] = pd.NA
    panel["CCTV자료등급"] = "세모: 2026 현존목록 설치연도 역산"
    panel["교통법규위반자료상태"] = "미확보: 자치구별 공개 원자료 확인 불가"
    panel.to_csv(INTERIM / "교통사고_세모지표_결합_2017_2025.csv", index=False, encoding="utf-8-sig")

    workbook = INTERIM / "교통사고_세모지표_원자료결합_2017_2025.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="전체결합", index=False)
        workplaces.to_excel(writer, sheet_name="의료보장_인구", index=False)
        roads.to_excel(writer, sheet_name="도로면적", index=False)
        cctv.to_excel(writer, sheet_name="CCTV_현존목록역산", index=False)
        status.to_excel(writer, sheet_name="수집현황_주의", index=False)


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    workplaces = build_workplaces()
    roads = build_roads()
    cctv = build_cctv_backcast()
    road_cctv = build_road_cctv(roads, cctv)
    status = build_status(workplaces, roads, cctv)
    build_combined(workplaces, roads, cctv, status)
    print(f"workplaces={len(workplaces)}, roads={len(roads)}, cctv={len(cctv)}, road_cctv={len(road_cctv)}")


if __name__ == "__main__":
    main()
