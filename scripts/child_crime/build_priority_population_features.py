"""Build release-safe borough features for 3-year child-population forecasts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw"
BASE = ROOT / "artifacts" / "child_crime_borough_v2"
OUT = ROOT / "artifacts" / "child_population_priority_v1"


def number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def raw_dir() -> Path:
    matches = [p for p in RAW_ROOT.iterdir() if p.is_dir() and any(x.name.startswith("2008-2025_") for x in p.glob("*.xlsx"))]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely identify raw directory: {matches}")
    return matches[0]


def exact_book(folder: Path, marker: str) -> Path:
    matches = [p for p in folder.glob("*.xlsx") if marker in p.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one workbook containing {marker!r}: {matches}")
    return matches[0]


def load_single_age(folder: Path, gus: list[str]) -> pd.DataFrame:
    age_dirs = [p for p in folder.iterdir() if p.is_dir() and any(x.name.startswith("201012_") for x in p.glob("*.csv"))]
    if len(age_dirs) != 1:
        raise RuntimeError(f"Cannot uniquely identify single-age directory: {age_dirs}")
    records = []
    for path in sorted(age_dirs[0].glob("*.csv")):
        year = int(path.name[:4])
        frame = pd.read_csv(path, encoding="cp949", dtype=str)
        admin = frame.columns[0]
        for gu in gus:
            # Published borough aggregate row, never the sum of aggregate + dong rows.
            hit = frame[frame[admin].str.contains(fr"서울특별시\s+{re.escape(gu)}\s*\(", regex=True, na=False)]
            if len(hit) != 1:
                raise ValueError(f"{year} {gu}: expected one borough row, got {len(hit)}")
            row = hit.iloc[0]
            ages = {}
            for age in range(101):
                suffix = f"_{age}세" if age < 100 else "_100세 이상"
                cols = [c for c in frame.columns if c.endswith(suffix) and "_계_" in c]
                if len(cols) != 1:
                    raise ValueError(f"{year}: cannot identify age column {suffix}")
                ages[age] = number(row[cols[0]])
            records.append({"year": year, "gu": gu, **{f"age_{a}": v for a, v in ages.items()}})
    detail = pd.DataFrame(records).sort_values(["gu", "year"])
    for lo, hi, name in [(0, 2, "age0_2"), (0, 14, "age0_14"), (15, 17, "age15_17")]:
        detail[name] = detail[[f"age_{a}" for a in range(lo, hi + 1)]].sum(axis=1)
    detail["cohort_retention_1y"] = detail[[f"age_{a}" for a in range(1, 18)]].sum(axis=1) / detail.groupby("gu")[[f"age_{a}" for a in range(17)]].shift(1).sum(axis=1)
    detail["cohort_retention_3y"] = detail[[f"age_{a}" for a in range(3, 18)]].sum(axis=1) / detail.groupby("gu")[[f"age_{a}" for a in range(15)]].shift(3).sum(axis=1)
    detail["aging_out_3y_share"] = detail.age15_17 / detail[[f"age_{a}" for a in range(18)]].sum(axis=1)
    detail["young_entry_share"] = detail.age0_2 / detail[[f"age_{a}" for a in range(18)]].sum(axis=1)
    for lo in (0, 3, 6, 9, 12):
        hi = lo + 2
        current = detail[[f"age_{a}" for a in range(lo + 3, hi + 4)]].sum(axis=1)
        past = detail.groupby("gu")[[f"age_{a}" for a in range(lo, hi + 1)]].shift(3).sum(axis=1)
        detail[f"cohort_{lo}_{hi}_retention_3y"] = current / past
    keep = ["year", "gu", "age0_2", "age0_14", "age15_17", "cohort_retention_1y", "cohort_retention_3y", "aging_out_3y_share", "young_entry_share"]
    keep += [f"cohort_{lo}_{lo+2}_retention_3y" for lo in (0, 3, 6, 9, 12)]
    return detail[keep]


def parse_student_counts(path: Path, gus: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    rows = []
    for col in range(2, raw.shape[1]):
        year = pd.to_numeric(raw.iat[0, col], errors="coerce")
        if pd.isna(year):
            continue
        level = str(raw.iat[1, col])
        major, minor = str(raw.iat[2, col]), str(raw.iat[3, col])
        if minor == "소계" and ("학생수" in major or "원아수" in major):
            field = {"유치원": "kindergarten_students", "초등학교": "elementary_students", "중학교": "middle_students", "고등학교": "high_students"}.get(level)
        elif "학급수" in minor:
            field = {"유치원": "kindergarten_classes", "초등학교": "elementary_classes", "중학교": "middle_classes", "고등학교": "high_classes"}.get(level)
        else:
            field = None
        if not field:
            continue
        for row in range(5, raw.shape[0]):
            gu = str(raw.iat[row, 1]).strip()
            if gu in gus:
                rows.append((int(year), gu, field, number(raw.iat[row, col])))
    long = pd.DataFrame(rows, columns=["year", "gu", "field", "value"])
    return long.pivot(index=["year", "gu"], columns="field", values="value").reset_index()


def parse_student_moves(path: Path, gus: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    rows = []
    seen = {}
    for col in range(2, raw.shape[1]):
        year = pd.to_numeric(raw.iat[0, col], errors="coerce")
        if pd.isna(year):
            continue
        level = {"초등학교": "elem", "중학교": "middle", "고등학교": "high"}.get(str(raw.iat[1, col]))
        if not level:
            continue
        key = (int(year), level)
        pos = seen.get(key, 0)
        # KOSIS column order: outbound-within, outbound-outside,
        # inbound-within, inbound-outside. Header labels are duplicated.
        suffix = ("out_within", "out_outside", "in_within", "in_outside")[pos % 4]
        seen[key] = pos + 1
        field = f"student_{level}_{suffix}"
        for row in range(5, raw.shape[0]):
            gu = str(raw.iat[row, 1]).strip()
            if gu in gus:
                rows.append((int(year), gu, field, number(raw.iat[row, col])))
    long = pd.DataFrame(rows, columns=["year", "gu", "field", "value"])
    wide = long.pivot(index=["year", "gu"], columns="field", values="value").reset_index()
    for level in ("elem", "middle", "high"):
        wide[f"student_{level}_net"] = wide[f"student_{level}_in_within"] + wide[f"student_{level}_in_outside"] - wide[f"student_{level}_out_within"] - wide[f"student_{level}_out_outside"]
    return wide


def parse_stress(path: Path, gus: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    province = raw.iloc[:, 2].ffill()
    years = pd.to_numeric(raw.iloc[11, 4:], errors="coerce").ffill()
    rows = []
    for col in range(4, raw.shape[1]):
        year = years.iat[col - 4]
        if pd.isna(year) or str(raw.iat[12, col]).strip() != "표준화율":
            continue
        for row in range(13, raw.shape[0]):
            gu = str(raw.iat[row, 3]).strip()
            if str(province.iat[row]).strip() == "서울" and gu in gus:
                rows.append((int(year), gu, number(raw.iat[row, col])))
    result = pd.DataFrame(rows, columns=["year", "gu", "stress_standardized_rate"])
    if result.duplicated(["year", "gu"]).any():
        raise ValueError("Duplicate stress observations")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / "panel.csv")
    gus = sorted(panel.gu.unique().tolist())
    folder = raw_dir()
    cohort = load_single_age(folder, gus)
    students = parse_student_counts(exact_book(folder, "학급당"), gus)
    moves = parse_student_moves(exact_book(folder, "학생변동상황"), gus)
    stress = parse_stress(exact_book(folder, "스트레스"), gus)
    result = panel[["year", "gu"]].merge(cohort, on=["year", "gu"], how="left", validate="one_to_one")
    for source in (students, moves, stress):
        result = result.merge(source, on=["year", "gu"], how="left", validate="one_to_one")
    groups = {
        "cohort": [c for c in cohort.columns if c not in ("year", "gu")],
        "student_count": [c for c in students.columns if c not in ("year", "gu")],
        "student_movement": [c for c in moves.columns if c not in ("year", "gu")],
        "stress": ["stress_standardized_rate"],
    }
    # Administrative survey/school statistics are only available after the
    # reference year. A one-year availability lag prevents release leakage.
    for group in ("student_count", "student_movement", "stress"):
        lagged = []
        for col in groups[group]:
            new = f"{col}_available_lag1"
            result[new] = result.groupby("gu")[col].shift(1)
            lagged.append(new)
        groups[group] = lagged
    result.to_csv(OUT / "priority_features.csv", index=False, encoding="utf-8-sig")
    (OUT / "feature_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage = result.groupby("year")[[c for values in groups.values() for c in values]].count()
    coverage.to_csv(OUT / "feature_coverage.csv", encoding="utf-8-sig")
    audit = {
        "cohort": {"years": [int(cohort.year.min()), int(cohort.year.max())], "rows": len(cohort), "availability": "same-year resident-register population"},
        "student_count": {"years": [int(students.year.min()), int(students.year.max())], "rows": len(students), "applied_lag": 1},
        "student_movement": {"years": [int(moves.year.min()), int(moves.year.max())], "rows": len(moves), "applied_lag": 1},
        "stress": {"years": [int(stress.year.min()), int(stress.year.max())], "rows": len(stress), "measure": "age-standardized rate", "applied_lag": 1},
    }
    (OUT / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "shape": result.shape, "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
