"""행정민원 다중 헤더를 연도·자치구 패널로 정제한다."""

from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/안전/생활안전/생활안전_행정민원자료.xlsx")
OUTPUT = Path("data/interim/행정민원_작업.csv")

COMMON_MAP = {
    "인가/허가": "인가허가_민원건수",
    "특허/면허": "특허면허_민원건수",
    "승인.지정": "승인지정_민원건수",
    "신고.등록": "신고등록_민원건수",
    "시험.검사": "시험검사_민원건수",
    "확인증명/교부": "확인증명교부_민원건수",
    "고충민원": "고충민원_건수",
    "기타": "기타민원_건수",
}
OUTPUT_COLUMNS = [
    "연도", "자치구", "집계기준", "행정민원_총건수",
    "인가허가_민원건수", "특허면허_민원건수", "승인지정_민원건수",
    "신고등록_민원건수", "시험검사_민원건수", "확인증명교부_민원건수",
    "고충민원_건수", "기타민원_건수", "일반민원_건수", "법정민원_건수",
    "질의건의기타민원_건수", "현장민원_건수",
]


def number(value: object) -> int | pd.NA:
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan"}:
        return pd.NA
    parsed = pd.to_numeric(text, errors="coerce")
    return pd.NA if pd.isna(parsed) else int(parsed)


def old_name(path: tuple[str, ...]) -> str | None:
    name = path[-1]
    if name == "계":
        return "행정민원_총건수"
    return COMMON_MAP.get(name)


def new_name(path: tuple[str, ...]) -> str | None:
    if path == ("총계", "소계"):
        return "행정민원_총건수"
    if path == ("일반민원", "소계"):
        return "일반민원_건수"
    if path == ("일반민원", "법정민원", "소계"):
        return "법정민원_건수"
    if path == ("일반민원", "질의 건의 기타 민원", "소계"):
        return "질의건의기타민원_건수"
    if path == ("현장민원", "소계"):
        return "현장민원_건수"
    if path == ("고충민원", "소계"):
        return "고충민원_건수"
    if len(path) == 3 and path[:2] == ("일반민원", "법정민원"):
        return COMMON_MAP.get(path[-1])
    return None


def parse_sheet(
    path: Path, sheet: str, header_rows: int, standard: str
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    records: dict[tuple[int, str], dict[str, object]] = {}
    mapper = old_name if standard.startswith("구형") else new_name

    for column in range(2, raw.shape[1]):
        year = int(raw.iloc[0, column])
        hierarchy: list[str] = []
        for row in range(1, header_rows):
            value = str(raw.iloc[row, column]).strip()
            if value not in {"", "nan"} and (not hierarchy or hierarchy[-1] != value):
                hierarchy.append(value)
        output_name = mapper(tuple(hierarchy))
        if output_name is None:
            continue
        for row in range(header_rows, raw.shape[0]):
            gu = str(raw.iloc[row, 1]).strip()
            if not gu.endswith("구"):
                continue
            key = (year, gu)
            record = records.setdefault(
                key, {"연도": year, "자치구": gu, "집계기준": standard}
            )
            if output_name in record:
                raise ValueError(f"중복 지표: {sheet}, {year}, {gu}, {output_name}")
            record[output_name] = number(raw.iloc[row, column])
    return pd.DataFrame(records.values())


def preprocess(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    old = parse_sheet(input_path, "행정민원_2006_2020", 2, "구형_2006_2020")
    new = parse_sheet(input_path, "행정민원_2021_2025", 4, "신형_2021_2025")
    result = pd.concat([old, new], ignore_index=True).reindex(columns=OUTPUT_COLUMNS)
    result = result.sort_values(["연도", "자치구"]).reset_index(drop=True)

    if result.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 키가 있습니다")
    if result[["연도", "자치구", "집계기준", "행정민원_총건수"]].isna().any().any():
        raise ValueError("필수 키 또는 총건수에 결측이 있습니다")
    numeric_columns = [c for c in OUTPUT_COLUMNS if c.endswith("건수")]
    result[numeric_columns] = result[numeric_columns].astype("Int64")
    if (result[numeric_columns] < 0).any().any():
        raise ValueError("민원 건수에 음수가 있습니다")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    frame = preprocess()
    print(
        f"저장 완료: {OUTPUT} | {len(frame):,}행 | "
        f"{frame['연도'].min()}~{frame['연도'].max()}년 | "
        f"자치구 {frame['자치구'].nunique()}개"
    )
