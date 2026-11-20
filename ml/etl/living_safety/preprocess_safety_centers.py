"""안전센터 시설 목록을 연도별 최신 기준일의 서울 자치구별 센터 수로 정리한다."""

from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/안전/생활안전/안전한동네_안전센터.xlsx")
OUTPUT = Path("data/interim/안전센터_작업.csv")
SEOUL_GU = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
}


def preprocess(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    data = pd.read_excel(input_path, sheet_name="안전센터_통합")
    required = {"기준일", "소방서", "안전센터명", "위치"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    # 엑셀 일련번호를 실제 날짜로 변환한다.
    data["기준일"] = pd.to_datetime(
        pd.to_numeric(data["기준일"], errors="coerce"),
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    )

    # 센터명이나 위치가 없는 빈 시설 행은 실제 센터로 볼 수 없어 제외한다.
    data = data.dropna(subset=["기준일", "안전센터명", "위치"]).copy()
    for column in ["소방서", "안전센터명", "위치"]:
        data[column] = data[column].astype("string").str.strip()
    data = data[
        data["안전센터명"].ne("") & data["위치"].ne("")
    ].copy()

    # 위치 문자열에서 서울 자치구명을 추출한다.
    gu_pattern = "(" + "|".join(sorted(SEOUL_GU, key=len, reverse=True)) + ")"
    data["자치구"] = data["위치"].str.extract(gu_pattern, expand=False)
    data = data[data["자치구"].isin(SEOUL_GU)].copy()

    # 같은 기준일에 동일 센터가 반복된 경우 한 시설로 센다.
    data = data.drop_duplicates(
        subset=["기준일", "소방서", "안전센터명", "위치"]
    )
    data["연도"] = data["기준일"].dt.year.astype("int64")

    # 같은 연도에 조사일이 여러 번이면 가장 최근 시설 목록만 사용한다.
    latest_dates = data.groupby("연도")["기준일"].transform("max")
    latest = data[data["기준일"].eq(latest_dates)].copy()

    result = (
        latest.groupby(["연도", "기준일", "자치구"], as_index=False)
        .size()
        .rename(columns={"size": "안전센터수"})
        .sort_values(["연도", "자치구"])
        .reset_index(drop=True)
    )
    result["기준일"] = result["기준일"].dt.strftime("%Y-%m-%d")
    result["안전센터수"] = result["안전센터수"].astype("int64")

    keys = ["연도", "자치구"]
    if result.duplicated(keys).any():
        raise ValueError("연도·자치구 중복 키가 있습니다")
    if not set(result["자치구"]).issubset(SEOUL_GU):
        raise ValueError("서울 외 지역이 포함되어 있습니다")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    frame = preprocess()
    print(f"저장 완료: {OUTPUT} | {len(frame):,}행")
