"""소음·진동 민원 자료를 연도·자치구 패널로 정제한다."""

from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/안전/생활안전/생활안전_소음진동자료.xlsx")
OUTPUT = Path("data/interim/소음진동_작업.csv")


def preprocess(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    raw = pd.read_excel(input_path, sheet_name=0, header=None)
    records: list[dict[str, object]] = []

    for row in range(3, len(raw)):
        gu = str(raw.iloc[row, 1]).strip()
        if not gu.endswith("구"):
            continue
        for column in range(2, raw.shape[1]):
            year = int(raw.iloc[0, column])
            upper = str(raw.iloc[1, column]).strip()
            lower = str(raw.iloc[2, column]).strip()
            metric = upper if lower == "소계" else lower
            value = pd.to_numeric(
                str(raw.iloc[row, column]).strip().replace(",", ""), errors="coerce"
            )
            records.append({"연도": year, "자치구": gu, "지표": metric, "값": value})

    long = pd.DataFrame(records)
    if long.duplicated(["연도", "자치구", "지표"]).any():
        raise ValueError("연도·자치구·지표 중복 키가 있습니다")

    result = long.pivot(index=["연도", "자치구"], columns="지표", values="값").reset_index()
    result.columns.name = None
    result = result.rename(columns={
        "환경관련 전체민원": "환경관련_전체민원건수",
        "소음진동 민원": "소음진동_전체민원건수",
        "공장": "공장소음진동_민원건수",
        "교통": "교통소음진동_민원건수",
        "생활": "생활소음진동_민원건수",
    })
    result = result[[
        "연도", "자치구", "환경관련_전체민원건수",
        "소음진동_전체민원건수", "공장소음진동_민원건수",
        "교통소음진동_민원건수", "생활소음진동_민원건수",
    ]].sort_values(["연도", "자치구"]).reset_index(drop=True)

    required_values = [
        "환경관련_전체민원건수", "소음진동_전체민원건수", "생활소음진동_민원건수"
    ]
    if result[required_values].isna().any().any():
        raise ValueError("유지 대상 민원 건수에 결측이 있습니다")
    result[required_values] = result[required_values].astype("int64")
    optional_values = ["공장소음진동_민원건수", "교통소음진동_민원건수"]
    result[optional_values] = result[optional_values].astype("Int64")

    if result.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 키가 있습니다")
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
