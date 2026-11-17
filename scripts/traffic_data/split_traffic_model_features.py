"""Split the all-in-one traffic feature table into related interim datasets."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "안전" / "교통사고" / "교통사고_모델보강_전체통합.csv"
OUTPUT = ROOT / "data" / "interim" / "교통사고_모델보강"
KEYS = ["year", "district"]

GROUPS = {
    "01_인구_활동량": [
        "생활인구_연평균", "생활인구_연중시간최대", "생활인구_월변동성",
        "상주인구_연평균", "유동인구_연평균",
    ],
    "02_기상": [
        "강수량합계", "강수일수", "폭우일수", "적설량합계", "적설일수",
        "폭염일수", "결빙일수", "최고기온평균", "최저기온평균",
    ],
    "03_대중교통": [
        "대중교통_승객수_Q4합계", "버스_승객수_Q4합계", "지하철_승객수_Q4합계",
        "버스정류장수_Q4평균", "지하철역수_Q4평균",
        "버스노선수_Q4평균", "지하철노선수_Q4평균",
    ],
    "04_단속_CCTV": [
        "CCTV전체누적", "과속신호카메라누적", "교통단속CCTV누적", "음주단속적발건수",
    ],
    "05_어린이교통사고": [
        "어린이사고_사고건수", "어린이사고_사망자수", "어린이사고_부상자수",
    ],
    "06_도로_교통시설": [
        "교차로수", "횡단보도수", "신호횡단보도수", "시설현황_기준연도",
    ],
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT, encoding="utf-8-sig")
    source["year"] = pd.to_numeric(source["year"], errors="raise").astype(int)

    expected = set(KEYS)
    for columns in GROUPS.values():
        expected.update(columns)
    missing = expected.difference(source.columns)
    unclassified = set(source.columns).difference(expected)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if unclassified:
        raise ValueError(f"Unclassified columns: {sorted(unclassified)}")

    dictionary_rows = []
    summary_rows = []
    for group, columns in GROUPS.items():
        subset = source[KEYS + columns].copy()
        subset = subset.dropna(subset=columns, how="all")
        subset = subset.sort_values(KEYS).reset_index(drop=True)
        if subset.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in {group}")

        output_path = OUTPUT / f"{group}.csv"
        subset.to_csv(output_path, index=False, encoding="utf-8-sig")
        summary_rows.append({
            "파일": output_path.name,
            "행수": len(subset),
            "특성수": len(columns),
            "시작연도": int(subset.year.min()),
            "종료연도": int(subset.year.max()),
            "자치구수": int(subset.district.nunique()),
        })
        for column in columns:
            dictionary_rows.append({"분류": group, "컬럼명": column})

    pd.DataFrame(dictionary_rows).to_csv(
        OUTPUT / "00_컬럼_분류표.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "00_파일_요약.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
