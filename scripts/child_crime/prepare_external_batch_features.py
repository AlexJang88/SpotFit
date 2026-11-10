"""Validate and prepare the externally acquired child-exposure feature batch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "raw" / "아동범죄" / "external_features"
OUT = BASE / "prepared"
OUT.mkdir(parents=True, exist_ok=True)


def read_raw(name: str) -> pd.DataFrame:
    path = BASE / name
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def district_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame["구명"].astype(str).str.endswith("구")].copy()
    result["통계연도"] = numeric(result["통계연도"]).astype("Int64")
    return result[result["통계연도"].notna()].copy()


def validate_panel(frame: pd.DataFrame, label: str, minimum_years: int = 10) -> None:
    counts = frame.groupby("연도")["자치구"].nunique()
    if frame.duplicated(["연도", "자치구"]).any():
        raise ValueError(f"{label}: duplicate district-year keys")
    if len(counts) < minimum_years:
        raise ValueError(f"{label}: only {len(counts)} distinct years")
    incomplete = counts[counts.ne(25)]
    if not incomplete.empty:
        raise ValueError(f"{label}: incomplete years {incomplete.to_dict()}")


def prepare_household_size() -> pd.DataFrame:
    frame = district_rows(read_raw("서울시_가구원수별가구_구별_원본.csv"))
    frame["가구수"] = numeric(frame["가구수"])
    grouped = (
        frame.groupby(["통계연도", "구명", "가구구성원규모"], as_index=False)["가구수"]
        .sum(min_count=1)
    )
    pivot = grouped.pivot(index=["통계연도", "구명"], columns="가구구성원규모", values="가구수")
    pivot.columns = [f"가구_{column}" for column in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"통계연도": "연도", "구명": "자치구"})
    # From 2024 the source no longer publishes the 6-person and 7+-person
    # categories. Drop schema-inconsistent columns instead of treating them as zero.
    incomplete_columns = [column for column in pivot.columns if pivot[column].isna().any()]
    pivot = pivot.drop(columns=incomplete_columns)
    one = next((c for c in pivot if c in {"가구_1인", "가구_1인가구"}), None)
    total = next((c for c in pivot if c in {"가구_계", "가구_합계"}), None)
    if one and total:
        pivot["1인가구비율"] = pivot[one] / pivot[total]
    validate_panel(pivot, "household-size")
    pivot.to_csv(OUT / "서울시_자치구_가구원수별_피처.csv", index=False, encoding="utf-8-sig")
    return pivot


def prepare_one_person() -> pd.DataFrame:
    frame = district_rows(read_raw("서울시_1인가구_성별연령별_원본.csv"))
    frame["가구수"] = numeric(frame["가구수"])
    # Age detail is stored on male/female rows while the all-sex rows are blank
    # for many age bands, so aggregate the two detailed sex rows.
    sexes = frame["성별명"].dropna().astype(str).unique().tolist()
    total_labels = {label for label in sexes if label in {"계", "전체", "합계"}}
    selected = frame[~frame["성별명"].isin(total_labels)].copy()
    selected["나이대명"] = selected["나이대명"].astype(str).str.replace(" ", "", regex=False)
    selected["나이대명"] = selected["나이대명"].str.replace(r"^(\d+~\d+)세$", r"\1", regex=True)
    grouped = selected.groupby(["통계연도", "구명", "나이대명"], as_index=False)["가구수"].sum(min_count=1)
    pivot = grouped.pivot(index=["통계연도", "구명"], columns="나이대명", values="가구수")
    pivot.columns = [f"1인가구_{column}" for column in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"통계연도": "연도", "구명": "자치구"})
    validate_panel(pivot, "one-person-household")
    pivot.to_csv(OUT / "서울시_자치구_1인가구_연령피처.csv", index=False, encoding="utf-8-sig")
    return pivot


def prepare_divorce() -> pd.DataFrame:
    frame = district_rows(read_raw("서울시_이혼건수_조이혼율_원본.csv"))
    frame["이혼건수"] = numeric(frame["이혼건수"])
    frame["조이혼율"] = numeric(frame["조이혼율"])
    result = frame[["통계연도", "구명", "이혼건수", "조이혼율"]].rename(
        columns={"통계연도": "연도", "구명": "자치구"}
    )
    result = result.drop_duplicates(["연도", "자치구"]).sort_values(["자치구", "연도"])
    result["이혼건수_전년변화율"] = result.groupby("자치구")["이혼건수"].pct_change(fill_method=None)
    validate_panel(result, "divorce")
    result.to_csv(OUT / "서울시_자치구_이혼_피처.csv", index=False, encoding="utf-8-sig")
    return result


def prepare_household_growth() -> pd.DataFrame:
    frame = district_rows(read_raw("서울시_총가구증가율_원본.csv"))
    frame["가구증가율"] = numeric(frame["가구증가율"])
    result = frame[["통계연도", "구명", "가구증가율"]].rename(
        columns={"통계연도": "연도", "구명": "자치구"}
    )
    result = result.drop_duplicates(["연도", "자치구"])
    validate_panel(result, "household-growth")
    result.to_csv(OUT / "서울시_자치구_가구증가율_피처.csv", index=False, encoding="utf-8-sig")
    return result


def prepare_apartment() -> pd.DataFrame:
    frame = district_rows(read_raw("서울시_공동주택관리_유형별_원본.csv"))
    for column in ("단지수", "동수", "세대수"):
        frame[column] = numeric(frame[column])
    grouped = frame.groupby(["통계연도", "구명", "주택형태명"], as_index=False)[["단지수", "동수", "세대수"]].sum(min_count=1)
    pieces = []
    for value in ("단지수", "동수", "세대수"):
        pivot = grouped.pivot(index=["통계연도", "구명"], columns="주택형태명", values=value)
        pivot.columns = [f"공동주택_{column}_{value}" for column in pivot.columns]
        pieces.append(pivot)
    result = pd.concat(pieces, axis=1).reset_index().rename(columns={"통계연도": "연도", "구명": "자치구"})
    result.to_csv(OUT / "서울시_자치구_공동주택관리_피처.csv", index=False, encoding="utf-8-sig")
    return result


def audit_occupancy() -> dict[str, object]:
    frame = district_rows(read_raw("서울시_점유형태별주택_원본.csv"))
    years = sorted(frame["통계연도"].dropna().astype(int).unique().tolist())
    counts = frame.groupby("통계연도")["구명"].nunique()
    return {
        "dataset": "점유형태별 주택",
        "years": ",".join(map(str, years)),
        "year_count": len(years),
        "all_years_25_districts": bool(counts.eq(25).all()),
        "status": "excluded_sparse_5year_cycle",
    }


def main() -> None:
    household = prepare_household_size()
    one_person = prepare_one_person()
    divorce = prepare_divorce()
    household_growth = prepare_household_growth()
    apartment = prepare_apartment()
    occupancy = audit_occupancy()
    for label, frame in (
        ("household_size", household),
        ("one_person", one_person),
        ("divorce", divorce),
        ("household_growth", household_growth),
    ):
        print(
            f"{label}: rows={len(frame)} years={int(frame['연도'].min())}-{int(frame['연도'].max())} "
            f"year_count={frame['연도'].nunique()} columns={len(frame.columns)}"
        )
    print(f"occupancy: {occupancy}")
    print(
        f"apartment: excluded_short_period rows={len(apartment)} "
        f"years={int(apartment['연도'].min())}-{int(apartment['연도'].max())} "
        f"year_count={apartment['연도'].nunique()}"
    )


if __name__ == "__main__":
    main()
