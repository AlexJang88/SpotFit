"""유흥·단란주점 연도별 자치구 자료를 정제한다."""

from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/안전/생활안전/유흥_단란주점_자치구_연도별_2010_2026.csv")
OUTPUT = Path("data/interim/유흥_단란주점_작업.csv")

COLUMN_NAMES = {
    "단란주점_연말_영업업소수": "단란주점_연말기준_영업중업소수",
    "유흥주점_연말_영업업소수": "유흥주점_연말기준_영업중업소수",
    "유흥_단란주점_합계": "유흥주점및단란주점_연말기준_영업중업소수",
}
COUNT_COLUMNS = list(COLUMN_NAMES.values())


def preprocess(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"연도", "자치구", "복원방식", *COLUMN_NAMES}
    if missing := required - set(data.columns):
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    result = data.drop(columns="복원방식").rename(columns=COLUMN_NAMES)
    result["연도"] = pd.to_numeric(result["연도"], errors="raise").astype("int64")
    result["자치구"] = result["자치구"].astype("string").str.strip()
    for column in COUNT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")

    if result.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 키가 있습니다")
    if result[["연도", "자치구", *COUNT_COLUMNS]].isna().any().any():
        raise ValueError("필수 값에 결측이 있습니다")
    if (result[COUNT_COLUMNS] < 0).any().any():
        raise ValueError("업소 수에 음수가 있습니다")
    if not (
        result["단란주점_연말기준_영업중업소수"]
        + result["유흥주점_연말기준_영업중업소수"]
        == result["유흥주점및단란주점_연말기준_영업중업소수"]
    ).all():
        raise ValueError("단란주점과 유흥주점의 합계가 원본 합계와 다릅니다")

    result = result.sort_values(["연도", "자치구"]).reset_index(drop=True)
    total = "유흥주점및단란주점_연말기준_영업중업소수"
    previous = result.groupby("자치구")[total].shift(1)
    result["유흥주점및단란주점_전년대비_증감수"] = (result[total] - previous).astype("Int64")
    result["유흥주점및단란주점_전년대비_증감률_pct"] = (
        (result[total] - previous).div(previous).mul(100).round(2)
    )
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
