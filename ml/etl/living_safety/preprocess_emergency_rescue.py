"""구급·구조 3개 원본을 확인하고 서울 자치구·연도 단위로 정리한다."""

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/안전/생활안전")
OUTPUT = Path("data/interim/구급_구조_작업.csv")
MONTHS = [f"{month}월" for month in range(1, 13)]
SEOUL_GU = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False), errors="coerce"
    )


def inspect_original_emergency(path: Path) -> set[int]:
    """중복 구급 원본도 누락 없이 읽히는지 확인하되 합산에는 사용하지 않는다."""
    years: set[int] = set()
    workbook = pd.ExcelFile(path)
    for sheet in workbook.sheet_names:
        digits = "".join(character for character in sheet if character.isdigit())
        if len(digits) != 4:
            raise ValueError(f"연도를 확인할 수 없는 시트입니다: {sheet}")
        data = pd.read_excel(path, sheet_name=sheet)
        if data.empty or not set(MONTHS).issubset(data.columns):
            raise ValueError(f"구급 원본 시트 형식 오류: {sheet}")
        years.add(int(digits))
    return years


def load_district_totals(path: Path, data_type: str) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=0)
    required = {"연도", "자치구", "행유형", *MONTHS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 누락: {sorted(missing)}")

    data = data[data["자치구"].isin(SEOUL_GU)].copy()
    data["연도"] = numeric(data["연도"]).astype("Int64")
    for column in MONTHS:
        data[column] = numeric(data[column])

    # 자치구 합계가 있는 연도는 그 행을 사용하고, 없는 연도만 동 상세를 합산한다.
    selected_parts = []
    for _, year_frame in data.groupby("연도"):
        district_rows = year_frame[year_frame["행유형"].eq("자치구합계")]
        selected_parts.append(district_rows if not district_rows.empty else year_frame)
    data = pd.concat(selected_parts, ignore_index=True)

    keys = ["연도", "자치구"]
    result = data.groupby(keys, as_index=False)[MONTHS].sum(min_count=1)
    result[f"{data_type}건수"] = result[MONTHS].sum(axis=1, min_count=1).astype("Int64")
    result[f"{data_type}_관측개월수"] = result[MONTHS].notna().sum(axis=1).astype("int64")
    result[f"{data_type}_연간완전관측여부"] = result[f"{data_type}_관측개월수"].eq(12)
    return result[[
        *keys,
        f"{data_type}건수",
        f"{data_type}_관측개월수",
        f"{data_type}_연간완전관측여부",
    ]]


def preprocess(raw_dir: Path = RAW_DIR, output: Path = OUTPUT) -> pd.DataFrame:
    original_years = inspect_original_emergency(raw_dir / "생활안전_구급자료.xlsx")
    emergency = load_district_totals(raw_dir / "안전한동네_구급.xlsx", "구급")
    rescue = load_district_totals(raw_dir / "안전한동네_구조.xlsx", "구조")

    normalized_years = set(emergency["연도"].dropna().astype(int))
    if not original_years.issubset(normalized_years):
        raise ValueError("구급 원본 연도가 정리된 구급 데이터에 모두 포함되지 않습니다")

    keys = ["연도", "자치구"]
    result = emergency.merge(rescue, on=keys, how="outer")
    for data_type in ("구급", "구조"):
        result[f"{data_type}건수"] = result[f"{data_type}건수"].astype("Int64")
        result[f"{data_type}_관측개월수"] = (
            result[f"{data_type}_관측개월수"].fillna(0).astype("int64")
        )
        result[f"{data_type}_연간완전관측여부"] = (
            result[f"{data_type}_연간완전관측여부"].fillna(False).astype("bool")
        )

    result = result.sort_values(keys).reset_index(drop=True)
    if result.duplicated(keys).any():
        raise ValueError("연도·자치구 중복 키가 있습니다")
    if not set(result["자치구"]).issubset(SEOUL_GU):
        raise ValueError("서울 외 자치구가 포함되어 있습니다")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    frame = preprocess()
    print(f"저장 완료: {OUTPUT} | {len(frame):,}행")
