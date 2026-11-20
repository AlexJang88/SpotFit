"""중·고등 학군 성과와 학원 접근성의 자치구·연도 패널을 만든다."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCHOOL_DIR = ROOT / "data/raw/school"
ACADEMY_PATH = ROOT / "data/raw/아동/전연령_교육인프라_공식통계/서울시_25개자치구_학원교습소_20261026.json"
OUTPUT = ROOT / "data/interim/secondary_education/secondary_education_panel_2021_2025.csv"


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def read_school_file(path: Path) -> pd.DataFrame:
    preview = pd.read_excel(path, sheet_name="학교별 주요통계", header=None, nrows=25)
    header_rows = preview.index[preview.iloc[:, 0].astype(str).str.strip().eq("조사기준일")]
    if len(header_rows) != 1:
        raise ValueError(f"헤더 행을 찾을 수 없습니다: {path}")
    frame = pd.read_excel(path, sheet_name="학교별 주요통계", header=int(header_rows[0]))
    frame.columns = [str(column).replace("\n", "").strip() for column in frame.columns]
    return frame


def school_outcomes() -> pd.DataFrame:
    candidates: dict[int, Path] = {}
    for path in SCHOOL_DIR.glob("*.xlsx"):
        match = re.search(r"(20\d{2})", path.name)
        if not match or ("상반기" not in path.name and "졸업후상황추가" not in path.name):
            continue
        year = int(match.group(1))
        if year not in candidates or "졸업후상황추가" in path.name:
            candidates[year] = path

    rows = []
    for year, path in sorted(candidates.items()):
        if not 2021 <= year <= 2025:
            continue
        frame = read_school_file(path)
        for column in ["졸업자_계", "국내_대학_계", "국외_대학_계", "입학자_계"]:
            frame[column] = numeric(frame[column])

        high = frame[frame["학교급"].eq("고등학교")].copy()
        high["four_year_entrants"] = high["국내_대학_계"].fillna(0) + high["국외_대학_계"].fillna(0)
        high_summary = high.groupby("행정구", as_index=False).agg(
            high_school_graduates=("졸업자_계", "sum"),
            four_year_entrants=("four_year_entrants", "sum"),
        )
        high_summary["four_year_college_progression_rate"] = (
            high_summary["four_year_entrants"]
            .div(high_summary["high_school_graduates"].replace(0, np.nan)) * 100
        )

        middle = frame[frame["학교급"].eq("중학교")].groupby("행정구", as_index=False).agg(
            middle_school_graduates=("졸업자_계", "sum")
        )
        special = high[high["고등학교유형"].isin(["특수목적고등학교", "자율고등학교"])].copy()
        special_summary = special.groupby("행정구", as_index=False).agg(
            special_autonomous_high_school_count=("학교명", "nunique"),
            special_autonomous_high_entrants=("입학자_계", "sum"),
        )
        summary = high_summary.merge(middle, on="행정구", how="outer").merge(
            special_summary, on="행정구", how="left"
        )
        fill = ["special_autonomous_high_school_count", "special_autonomous_high_entrants"]
        summary[fill] = summary[fill].fillna(0)
        summary["special_autonomous_supply_per_100_middle_graduates"] = (
            summary["special_autonomous_high_entrants"]
            .div(summary["middle_school_graduates"].replace(0, np.nan)) * 100
        )
        summary = summary.rename(columns={"행정구": "gu"})
        summary["year"] = year
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def surviving_academies() -> pd.DataFrame:
    payload = json.loads(ACADEMY_PATH.read_text(encoding="utf-8"))
    raw = pd.DataFrame(payload["rows"])
    raw["opening_date"] = pd.to_datetime(raw["ESTBL_YMD"], format="%Y%m%d", errors="coerce")
    raw["gu"] = raw["DISTRICT"].fillna(raw["ADMDST_NM"])
    raw["is_exam_academy"] = raw["FLD_NM"].astype(str).str.contains("입시|검정|보습", regex=True)
    raw["is_language_academy"] = raw["FLD_NM"].astype(str).str.contains("국제화", regex=True)
    rows = []
    for year in range(2021, 2026):
        observed = raw[raw["opening_date"].le(pd.Timestamp(year, 12, 31))]
        summary = observed.groupby("gu", as_index=False).agg(
            academy_survivor_count=("PEI_DSGN_NO", "nunique"),
            exam_academy_survivor_count=("is_exam_academy", "sum"),
            language_academy_survivor_count=("is_language_academy", "sum"),
        )
        summary["year"] = year
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    panel = school_outcomes().merge(surviving_academies(), on=["gu", "year"], how="left")
    panel["seoul_national_university_progression_rate"] = np.nan
    panel["special_autonomous_progression_rate"] = np.nan
    panel["academy_history_quality"] = "current_survivors_reconstructed_from_opening_date"
    panel["special_autonomous_metric_quality"] = "local_supply_proxy_not_origin_school_progression_rate"
    panel["snu_metric_quality"] = "unavailable_in_official_public_school_outcome_data"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT} ({len(panel):,}행, {panel['gu'].nunique()}개 자치구)")


if __name__ == "__main__":
    main()
