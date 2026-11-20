"""Build the long-panel features for the child regional-crime exposure model.

Only sources with at least ten years of Seoul borough-level observations are
included in the core panel.  The output deliberately keeps the untransformed
measure and the log1p target so that model training and service display do not
confuse the two scales.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CHILD_CRIME = RAW / "아동범죄"
AGE_DIR = CHILD_CRIME / "연령별인구현황 2010~2025"
CRIME_PATH = CHILD_CRIME / "서울시_5대범죄_2014_2025_원자료.csv"
REGISTERED_PATH = RAW / "안전" / "범죄" / "서울시_자치구_연도별_등록인구.csv"
BAR_PATH = RAW / "안전" / "범죄" / "파생" / "유흥_단란주점_자치구_연도별_2010_2026.csv"
SOURCE_DIR = CHILD_CRIME / "feature_sources"
OUTPUT_DIR = CHILD_CRIME / "processed"


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unsupported encoding: {path}")


def district_name(value: object) -> str | None:
    match = re.search(r"서울특별시\s+(.+?구)\s*\(", str(value))
    return match.group(1).strip() if match else None


def load_child_population() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(AGE_DIR.glob("20*.csv")):
        year = int(path.name[:4])
        frame = read_csv(path)
        frame["자치구"] = frame["행정구역"].map(district_name)
        frame = frame[frame["자치구"].notna()].copy()

        for age in range(20):
            col = f"{year}년_계_{age}세"
            frame[f"age_{age}"] = numeric(frame[col])

        frame["아동인구_0_17세"] = frame[[f"age_{age}" for age in range(18)]].sum(axis=1)
        frame["영유아인구_0_5세"] = frame[[f"age_{age}" for age in range(6)]].sum(axis=1)
        frame["초등연령인구_6_12세"] = frame[[f"age_{age}" for age in range(6, 13)]].sum(axis=1)
        frame["청소년인구_13_17세"] = frame[[f"age_{age}" for age in range(13, 18)]].sum(axis=1)
        frame["연도"] = year
        frames.append(
            frame[
                [
                    "연도",
                    "자치구",
                    "아동인구_0_17세",
                    "영유아인구_0_5세",
                    "초등연령인구_6_12세",
                    "청소년인구_13_17세",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def load_registered_population() -> pd.DataFrame:
    frame = read_csv(REGISTERED_PATH)
    frame = frame.rename(columns={"통계연도": "연도", "구명": "자치구"})
    frame = frame[frame["자치구"].astype(str).str.endswith("구")].copy()
    keep = ["연도", "자치구", "세대수", "인구수", "남자인구수", "여자인구수"]
    frame = frame[keep].copy()
    for col in keep[2:]:
        frame[col] = numeric(frame[col])
    return frame


def load_bars() -> pd.DataFrame:
    frame = read_csv(BAR_PATH)
    return frame.drop(columns=["복원방식"], errors="ignore")


def load_crime() -> pd.DataFrame:
    frame = read_csv(CRIME_PATH)
    frame["발생건수"] = numeric(frame["발생건수"])
    return (
        frame.groupby(["년도", "시군구"], as_index=False)["발생건수"]
        .sum()
        .rename(columns={"년도": "연도", "시군구": "자치구", "발생건수": "5대범죄_발생건수"})
    )


def validate_complete(frame: pd.DataFrame, label: str, start: int, end: int) -> None:
    scoped = frame[frame["연도"].between(start, end)]
    counts = scoped.groupby("연도")["자치구"].nunique()
    bad = counts[counts != 25]
    duplicates = scoped.duplicated(["연도", "자치구"]).sum()
    if len(counts) != end - start + 1 or not bad.empty or duplicates:
        raise ValueError(
            f"{label} is not a complete {start}-{end} x 25 panel: "
            f"year_count={len(counts)}, bad={bad.to_dict()}, duplicates={duplicates}"
        )


def copy_sources() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (REGISTERED_PATH, BAR_PATH):
        target = SOURCE_DIR / path.name
        if not target.exists() or target.read_bytes() != path.read_bytes():
            shutil.copy2(path, target)


def main() -> None:
    child = load_child_population()
    registered = load_registered_population()
    bars = load_bars()
    crime = load_crime()

    validate_complete(child, "child population", 2010, 2025)
    validate_complete(registered, "registered population", 2010, 2025)
    validate_complete(bars, "bars", 2010, 2025)
    validate_complete(crime, "crime", 2014, 2025)

    panel = child.merge(registered, on=["연도", "자치구"], how="inner", validate="one_to_one")
    panel = panel.merge(bars, on=["연도", "자치구"], how="inner", validate="one_to_one")
    panel = panel[panel["연도"].between(2010, 2025)].copy()
    panel["아동인구비율"] = panel["아동인구_0_17세"] / panel["인구수"]
    panel["세대당인구"] = panel["인구수"] / panel["세대수"]
    panel["남성비율"] = panel["남자인구수"] / panel["인구수"]
    panel["유흥_단란주점_인구1만명당"] = panel["유흥_단란주점_합계"] / panel["인구수"] * 10_000
    panel["유흥_단란주점_아동1만명당"] = panel["유흥_단란주점_합계"] / panel["아동인구_0_17세"] * 10_000

    panel = panel.sort_values(["자치구", "연도"]).reset_index(drop=True)
    for col in ("아동인구_0_17세", "인구수", "세대수", "유흥_단란주점_합계"):
        panel[f"{col}_전년변화율"] = panel.groupby("자치구")[col].pct_change(fill_method=None)
        panel[f"{col}_3년변화율"] = panel.groupby("자치구")[col].pct_change(3, fill_method=None)

    model = panel.merge(crime, on=["연도", "자치구"], how="left", validate="one_to_one")
    model["아동지역범죄노출률_1만명당"] = (
        model["5대범죄_발생건수"] / model["아동인구_0_17세"] * 10_000
    )
    model["Y_log1p_아동지역범죄노출률"] = np.log1p(model["아동지역범죄노출률_1만명당"])
    for lag in (1, 2, 3):
        model[f"아동지역범죄노출률_lag{lag}"] = model.groupby("자치구")[
            "아동지역범죄노출률_1만명당"
        ].shift(lag)
    model = model[model["연도"].between(2014, 2025)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "아동지역범죄노출_장기후보피처_2010_2025.csv", index=False, encoding="utf-8-sig")
    model.to_csv(OUTPUT_DIR / "아동지역범죄노출_모델패널_2014_2025.csv", index=False, encoding="utf-8-sig")
    copy_sources()

    print(f"feature_panel_rows={len(panel)} years={panel['연도'].min()}-{panel['연도'].max()}")
    print(f"model_panel_rows={len(model)} years={model['연도'].min()}-{model['연도'].max()}")
    print(f"districts={model['자치구'].nunique()} target_missing={model['Y_log1p_아동지역범죄노출률'].isna().sum()}")


if __name__ == "__main__":
    main()
