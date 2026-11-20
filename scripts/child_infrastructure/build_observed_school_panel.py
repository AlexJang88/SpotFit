"""학교별 실제 관측자료를 도로명주소 기준으로 행정동에 매핑한다.

이 스크립트의 핵심 원칙은 과거 값을 추정하거나 2026년 값을 과거로 복사하지 않는 것이다.
2021~2026년 각 4월 1일 학교 일람표에서 실제로 관측된 값만 사용한다.

행정동이 없는 학교 원자료는 행정안전부 도로명주소 파일
(`rnaddrkor_seoul.txt`)에 포함된 행정동 코드/명칭을 이용해 보완한다. 학교 주소와
도로명주소 기준표의 주소가 정확히 일치할 때만 매핑하며, 애매하거나 일치하지 않는 주소는
억지로 추정하지 않고 미매핑 파일에 남긴다.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd


# 연도별로 동일한 계절(4월 1일)을 비교해야 학생 수의 학기 효과가 섞이지 않는다.
# 하반기 파일도 존재하지만, 2026년은 4월 자료만 있으므로 주 패널에는 상반기만 사용한다.
APRIL_FILES = {
    2021: "2021년 상반기 학교별 일람표.xlsx",
    2022: "2022년 상반기 학교별 일람표.xlsx",
    2023: "2023년 상반기 학교별 일람표.xlsx",
    2024: "20240401기준_학교일람표_졸업후상황추가.xlsx",
    2025: "20250401기준_학교일람표_졸업후상황추가.xlsx",
    2026: "20260401기준_학교일람표.xlsx",
}


def normalize_road_address(value: object) -> str | None:
    """주소에서 `서울특별시 자치구 도로명 건물번호`만 표준화해 반환한다.

    학교 파일에는 `(법정동.학교명)` 같은 상세 설명이 붙어 있다. 이 설명은 공식 주소
    기준표의 키와 다르므로 제거한다. 반면 도로명과 건물 본번/부번은 실제 위치를 식별하는
    핵심이므로 보존한다. 문자열 유사도나 동 이름 추측은 다른 행정동으로 잘못 연결할 수
    있어 사용하지 않는다.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    # 주소 맨 앞의 축약 시도명만 바꾼다. 전체 문자열을 치환하면
    # `서울시립대로`가 `서울특별시립대로`로 훼손되는 문제가 생긴다.
    text = re.sub(r"^서울시(?=\s)", "서울특별시", text)
    text = re.sub(r"^서울(?=\s)", "서울특별시", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 도로명에는 숫자와 '길'이 들어갈 수 있어, 자치구 다음부터 마지막 건물번호 앞까지를
    # 도로명으로 잡는다. 예: 서울특별시 은평구 연서로 117
    match = re.search(
        # 학교 원본에는 `고산자로 197무학여자고등학교`처럼 건물번호 뒤에 공백이 없는
        # 행도 있다. 도로명이 '로/길'로 끝난다는 공식 규칙을 이용하면 그 경우도 정확히
        # 분리할 수 있다.
        r"서울특별시\s+(?P<gu>\S+구)\s+(?P<road>\S+(?:로|길))\s*(?P<main>\d+)(?:-(?P<sub>\d+))?",
        text,
    )
    if not match:
        return None
    sub = match.group("sub")
    number = match.group("main") if not sub or int(sub) == 0 else f"{match.group('main')}-{int(sub)}"
    road = re.sub(r"\s+", "", match.group("road"))
    return f"서울특별시 {match.group('gu')} {road} {number}"


def normalize_admin_name(value: object) -> str:
    """`화곡제1동`/`화곡1동`, `상계3.4동`/`상계3·4동` 표기 차이를 통일한다."""
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"제(?=\d)", "", text)
    return text.replace(".", "·").replace("ㆍ", "·")


def load_road_address_index(path: Path, boundaries: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """78MB 도로명주소 원본에서 매핑에 필요한 열만 읽어 메모리 사용량을 줄인다."""
    # 행정안전부 도로명주소 한글 파일의 필요한 필드 위치:
    # 2 시도명, 3 시군구명, 10 도로명, 11 지하여부, 12 건물본번,
    # 13 건물부번, 14 행정동코드, 15 행정동명.
    # 주의: 이 파일은 0부터 세는 위치 인덱스를 사용하므로 명세서상의 필드 순서와 혼동하면
    # 우편번호를 행정동명으로 읽게 된다.
    cols = [2, 3, 10, 11, 12, 13, 14, 15]
    names = ["sido", "sigungu", "road", "underground", "main_no", "sub_no", "admin_code", "admin_name"]
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=cols,
        names=names,
        dtype=str,
        encoding="cp949",
        keep_default_na=False,
        low_memory=False,
    )
    frame = frame[(frame["sido"] == "서울특별시") & frame["admin_code"].ne("") & frame["admin_name"].ne("")].copy()
    # 도로명주소의 10자리 행정동 코드와 프로젝트 경계의 8자리 통계 코드는 번호체계가
    # 다르므로 자치구명+행정동명으로 공식 경계 코드를 연결한다.
    boundary_codes = boundaries[["adm_cd", "adm_nm"]].drop_duplicates().copy()
    boundary_codes["sigungu"] = boundary_codes["adm_nm"].str.split().str[1]
    boundary_codes["admin_name"] = boundary_codes["adm_nm"].str.split().str[-1]
    boundary_codes["admin_key"] = boundary_codes["admin_name"].map(normalize_admin_name)
    boundary_codes = boundary_codes.rename(columns={"adm_cd": "boundary_admin_code"})
    frame["admin_key"] = frame["admin_name"].map(normalize_admin_name)
    frame = frame.merge(
        boundary_codes[["sigungu", "admin_key", "boundary_admin_code"]],
        on=["sigungu", "admin_key"], how="left", validate="many_to_one",
    )
    frame["admin_code"] = frame["boundary_admin_code"]
    frame = frame[frame["admin_code"].notna()].copy()
    frame["main_no"] = pd.to_numeric(frame["main_no"], errors="coerce").astype("Int64")
    frame["sub_no"] = pd.to_numeric(frame["sub_no"], errors="coerce").fillna(0).astype("Int64")
    frame = frame[frame["main_no"].notna()].copy()
    frame["road_key"] = (
        frame["sido"] + " " + frame["sigungu"] + " " + frame["road"].str.replace(" ", "", regex=False)
        + " " + frame["main_no"].astype(str)
        + frame["sub_no"].map(lambda x: "" if x == 0 else f"-{x}")
    )

    # 하나의 도로명 건물번호가 복수 행정동으로 나타나는 예외는 자동 선택하지 않는다.
    # 잘못된 정답을 만드는 것보다 미매핑으로 남겨 검토하는 편이 학습 데이터에 안전하다.
    unique_pairs = frame[["road_key", "admin_code", "admin_name"]].drop_duplicates()
    # 코드만 같고 명칭이 서로 다른 레코드도 기준정보 충돌로 본다. 코드 하나만 보고 임의로
    # 명칭을 선택하면 행정동 개편 전후 명칭이 섞일 수 있기 때문이다.
    counts = unique_pairs.groupby("road_key").size()
    ambiguous_keys = counts[counts > 1].index
    ambiguous = unique_pairs[unique_pairs["road_key"].isin(ambiguous_keys)].copy()
    index = unique_pairs[~unique_pairs["road_key"].isin(ambiguous_keys)].copy()
    return index, ambiguous


def read_school_file(path: Path, year: int) -> pd.DataFrame:
    """서식이 연도별로 조금 달라도 `학교명` 헤더 행을 찾아 안전하게 읽는다."""
    excel = pd.ExcelFile(path)
    sheet = next(name for name in excel.sheet_names if "주요" in name)
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_row = next(
        index
        for index, row in raw.iterrows()
        if any(str(value).strip() == "학교명" for value in row if pd.notna(value))
    )
    columns = [
        str(value).replace("\n", "").strip() if pd.notna(value) else f"unnamed_{position}"
        for position, value in enumerate(raw.iloc[header_row])
    ]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = columns
    data = data[data["시도"].astype(str).str.contains("서울", na=False)].copy()
    data["year"] = year
    data["road_key"] = data["주소"].map(normalize_road_address)
    return data


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """연도별 표기 차이가 있는 통계 열에서 첫 번째 사용 가능한 열을 고른다."""
    normalized = {column.replace(" ", ""): column for column in columns}
    for candidate in candidates:
        if candidate.replace(" ", "") in normalized:
            return normalized[candidate.replace(" ", "")]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="실제 관측 학교자료를 행정동 패널로 변환")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/child_school_observed"))
    parser.add_argument("--boundaries", type=Path, default=Path("frontend/public/seoul-dong.geojson"))
    args = parser.parse_args()

    address_path = args.raw_dir / "rnaddrkor_seoul.txt"
    school_dir = args.raw_dir / "school"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    boundaries = gpd.read_file(args.boundaries).to_crs("EPSG:4326")
    address_index, ambiguous = load_road_address_index(address_path, boundaries)
    school_frames = [read_school_file(school_dir / filename, year) for year, filename in APRIL_FILES.items()]
    schools = pd.concat(school_frames, ignore_index=True)
    schools = schools.merge(address_index, how="left", on="road_key", validate="many_to_one")
    schools["mapping_status"] = "MATCHED"
    schools.loc[schools["road_key"].isna(), "mapping_status"] = "ADDRESS_PARSE_FAILED"
    schools.loc[schools["road_key"].isin(set(ambiguous["road_key"])), "mapping_status"] = "AMBIGUOUS_ADDRESS"
    schools.loc[schools["admin_code"].isna() & schools["mapping_status"].eq("MATCHED"), "mapping_status"] = "NOT_FOUND"

    # 폐교는 해당 시점의 이용 가능한 학교 인프라가 아니므로 집계에서 제외한다.
    active = schools[schools["admin_code"].notna() & ~schools["상태"].astype(str).str.contains("폐", na=False)].copy()
    student_col = find_column(list(active.columns), ["학생수_총계_계", "학생수_계"])
    teacher_col = find_column(list(active.columns), ["교원수_총계_계", "교원수_계"])
    class_col = find_column(list(active.columns), ["편성학급수_계", "일반학급_학급수"])
    for output, source in {
        "student_count": student_col,
        "teacher_count": teacher_col,
        "class_count": class_col,
    }.items():
        active[output] = pd.to_numeric(active[source], errors="coerce") if source else pd.NA

    # 학교급별 열과 전체 열을 함께 제공하면 모델이 초등학교 공급만 선택하거나 전체 교육
    # 인프라를 선택하는 두 실험을 모두 수행할 수 있다.
    grouped = (
        active.groupby(["year", "admin_code", "admin_name", "학교급"], dropna=False)
        .agg(
            school_count=("학교명", "nunique"),
            student_count=("student_count", "sum"),
            teacher_count=("teacher_count", "sum"),
            class_count=("class_count", "sum"),
        )
        .reset_index()
    )
    wide = grouped.pivot_table(
        index=["year", "admin_code", "admin_name"],
        columns="학교급",
        values=["school_count", "student_count", "teacher_count", "class_count"],
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{metric}_{school_type}" for metric, school_type in wide.columns]
    panel = wide.reset_index()
    metric_columns = [column for column in panel.columns if column not in {"year", "admin_code", "admin_name"}]
    for metric in ("school_count", "student_count", "teacher_count", "class_count"):
        panel[f"{metric}_all"] = panel[[c for c in metric_columns if c.startswith(f"{metric}_")]].sum(axis=1)

    detail_columns = [
        "year", "조사기준일", "행정구", "학교급", "학교명", "상태", "주소", "road_key",
        "admin_code", "admin_name", "mapping_status", "student_count", "teacher_count", "class_count",
    ]
    schools_with_metrics = schools.merge(
        active[["year", "학교명", "주소", "student_count", "teacher_count", "class_count"]],
        how="left", on=["year", "학교명", "주소"], validate="one_to_one",
    )
    schools_with_metrics[detail_columns].to_csv(
        args.output_dir / "school_address_mapping_2021_2026.csv", index=False, encoding="utf-8-sig"
    )
    panel.sort_values(["year", "admin_code"]).to_csv(
        args.output_dir / "observed_school_panel_2021_2026.csv", index=False, encoding="utf-8-sig"
    )
    schools_with_metrics[schools_with_metrics["mapping_status"] != "MATCHED"][detail_columns].to_csv(
        args.output_dir / "unmatched_school_addresses.csv", index=False, encoding="utf-8-sig"
    )
    summary = (
        schools.groupby(["year", "mapping_status"]).size().rename("row_count").reset_index()
    )
    summary.to_csv(args.output_dir / "school_mapping_quality.csv", index=False, encoding="utf-8-sig")

    print(f"학교 원본 행: {len(schools):,}")
    print(f"주소 매핑 성공: {schools['admin_code'].notna().sum():,} ({schools['admin_code'].notna().mean():.2%})")
    print(f"행정동-연도 패널: {len(panel):,}")
    print(f"결과 위치: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
