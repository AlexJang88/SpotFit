"""현재 시설목록의 사건 날짜를 이용해 2021~2026년 행정동 패널을 복원한다.

이 패널은 실제 연도별 스냅샷과 동일하지 않다. 현재 원본에 남아 있는 시설의 인가·설치·
폐업 날짜로 과거 존재 여부를 계산하므로 결과 컬럼에는 ``reconstructed``를 붙인다.
정원, 의료인 수, 진료과목처럼 시점에 따라 바뀌는 현재 속성은 과거 값으로 사용하지 않는다.
단, 현재 진료과목을 과거에도 같다고 가정한 소아청소년과 수는 비교 실험용 컬럼으로만 만들고
이름에 ``current_attribute_backfill``을 명시한다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
YEARS = range(2021, 2027)


def read_csv(path: Path) -> pd.DataFrame:
    """서울시 파일에서 주로 사용하는 인코딩을 순서대로 검사한다."""
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"CSV 인코딩을 판별할 수 없습니다: {path}")


def parse_mixed_date(series: pd.Series) -> pd.Series:
    """`YYYY-MM-DD`, `YYYYMMDD`, 시간 포함 문자열이 섞인 공공데이터 날짜를 파싱한다.

    pandas 2.x는 첫 값에서 추론한 한 가지 형식을 전체 열에 적용할 수 있다. 공공데이터처럼
    형식이 혼재된 열에서는 정상 날짜가 대량으로 NaT가 되므로 각 값을 혼합 형식으로 읽는다.
    """
    return pd.to_datetime(series.replace(r"^\s*$", pd.NA, regex=True), errors="coerce", format="mixed")


def normalize_road_address(value: object) -> str | None:
    """상세주소를 제거하고 공식 도로명 건물번호 키를 만든다."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"^서울시(?=\s)", "서울특별시", text)
    text = re.sub(r"^서울(?=\s)", "서울특별시", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(
        r"서울특별시\s+(?P<gu>\S+구)\s+(?P<road>\S+(?:로|길))\s*(?P<main>\d+)(?:-(?P<sub>\d+))?",
        text,
    )
    if not match:
        return None
    sub = match.group("sub")
    number = match.group("main") if not sub or int(sub) == 0 else f"{match.group('main')}-{int(sub)}"
    return f"서울특별시 {match.group('gu')} {match.group('road').replace(' ', '')} {number}"


def normalize_admin_name(value: object) -> str:
    """도로명주소와 통계경계의 행정동 표기 차이를 결합용으로 통일한다."""
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"제(?=\d)", "", text)
    return text.replace(".", "·").replace("ㆍ", "·")


def load_address_index(path: Path, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    """도로명주소 원본에서 주소키와 행정동 코드만 읽고 충돌 주소를 제외한다."""
    frame = pd.read_csv(
        path, sep="|", header=None, usecols=[2, 3, 10, 12, 13, 14, 15],
        names=["sido", "sigungu", "road", "main_no", "sub_no", "admin_code", "admin_name"],
        dtype=str, encoding="cp949", keep_default_na=False, low_memory=False,
    )
    frame = frame[(frame["sido"] == "서울특별시") & frame["admin_code"].ne("")].copy()
    # 행정안전부 10자리 행정동 코드와 통계경계 8자리 코드는 단순 절단 관계가 아니다.
    # 자치구+행정동 명칭 대응표로 프로젝트 경계 코드를 명시적으로 연결한다.
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
        + frame["sub_no"].map(lambda value: "" if value == 0 else f"-{value}")
    )
    pairs = frame[["road_key", "admin_code", "admin_name"]].drop_duplicates()
    valid = pairs.groupby("road_key").filter(lambda group: len(group) == 1)
    return valid.drop_duplicates("road_key")


def attach_admin_dong(
    frame: pd.DataFrame,
    address_index: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    address_column: str,
    x_column: str | None = None,
    y_column: str | None = None,
    source_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """공식 도로명주소를 우선 사용하고, 실패한 행만 유효 좌표로 공간조인한다."""
    work = frame.copy()
    work["road_key"] = work[address_column].map(normalize_road_address)
    work = work.merge(address_index, how="left", on="road_key", validate="many_to_one")
    work["mapping_method"] = work["admin_code"].notna().map({True: "ROAD_ADDRESS", False: "UNMATCHED"})

    if x_column and y_column:
        missing = work["admin_code"].isna()
        x = pd.to_numeric(work.loc[missing, x_column], errors="coerce")
        y = pd.to_numeric(work.loc[missing, y_column], errors="coerce")
        coordinate_rows = work.loc[missing & x.reindex(work.index).notna() & y.reindex(work.index).notna()].copy()
        if not coordinate_rows.empty:
            coordinate_rows[x_column] = pd.to_numeric(coordinate_rows[x_column], errors="coerce")
            coordinate_rows[y_column] = pd.to_numeric(coordinate_rows[y_column], errors="coerce")
            points = gpd.GeoDataFrame(
                coordinate_rows,
                geometry=gpd.points_from_xy(coordinate_rows[x_column], coordinate_rows[y_column]),
                crs=source_crs,
            ).to_crs("EPSG:4326")
            joined = gpd.sjoin(points, boundaries[["adm_cd", "adm_nm", "geometry"]], how="left", predicate="within")
            valid = joined[joined["adm_cd"].notna()]
            work.loc[valid.index, "admin_code"] = valid["adm_cd"].astype(str)
            work.loc[valid.index, "admin_name"] = valid["adm_nm"].astype(str).str.split().str[-1]
            work.loc[valid.index, "mapping_method"] = "COORDINATE"
    return work


def active_on(frame: pd.DataFrame, year: int, start: str, end: str | None = None,
              pause_start: str | None = None, pause_end: str | None = None) -> pd.Series:
    """해당 연도 12월 31일에 시설이 운영 가능한 상태였는지 계산한다."""
    cutoff = pd.Timestamp(year=year, month=12, day=31)
    active = frame[start].notna() & frame[start].le(cutoff)
    if end:
        active &= frame[end].isna() | frame[end].gt(cutoff)
    if pause_start and pause_end:
        paused = frame[pause_start].notna() & frame[pause_start].le(cutoff)
        paused &= frame[pause_end].isna() | frame[pause_end].gt(cutoff)
        active &= ~paused
    return active


def count_panel(frame: pd.DataFrame, prefix: str, start: str, end: str | None = None,
                pause_start: str | None = None, pause_end: str | None = None,
                extra_masks: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """연도·행정동별 운영 시설 수와 연도별 신규·종료 사건 수를 만든다."""
    mapped = frame[frame["admin_code"].notna()].copy()
    rows: list[pd.DataFrame] = []
    for year in YEARS:
        year_frame = mapped[active_on(mapped, year, start, end, pause_start, pause_end)].copy()
        grouped = year_frame.groupby(["admin_code", "admin_name"]).size().rename(f"{prefix}_count_reconstructed")
        result = grouped.reset_index()
        opened = mapped[mapped[start].dt.year.eq(year)].groupby(["admin_code", "admin_name"]).size()
        result = result.merge(opened.rename(f"{prefix}_opened_observed_event").reset_index(), how="outer")
        if end:
            closed = mapped[mapped[end].dt.year.eq(year)].groupby(["admin_code", "admin_name"]).size()
            result = result.merge(closed.rename(f"{prefix}_closed_observed_event").reset_index(), how="outer")
        if extra_masks:
            for label, mask in extra_masks.items():
                subset = year_frame[mask.reindex(year_frame.index, fill_value=False)]
                metric = subset.groupby(["admin_code", "admin_name"]).size().rename(label).reset_index()
                result = result.merge(metric, how="outer")
        result["year"] = year
        rows.append(result)
    panel = pd.concat(rows, ignore_index=True)
    metrics = [column for column in panel if column not in {"year", "admin_code", "admin_name"}]
    # 행정동 개편으로 동일 8자리 코드에 과거·현재 명칭이 함께 남을 수 있다. 모델 결합키는
    # 코드이므로 코드 단위로 다시 합쳐 한 연도·행정동당 정확히 한 행을 보장한다.
    panel = panel.groupby(["year", "admin_code"], as_index=False)[metrics].sum()
    name_by_code = mapped.dropna(subset=["admin_code"]).drop_duplicates("admin_code").set_index("admin_code")["admin_name"]
    panel["admin_name"] = panel["admin_code"].map(name_by_code)
    panel[metrics] = panel[metrics].fillna(0).astype(int)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=ROOT / "data/raw/child_infrastructure_collected/20261023")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/child_infrastructure_reconstructed")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    boundaries = gpd.read_file(ROOT / "frontend/public/seoul-dong.geojson").to_crs("EPSG:4326")
    address_index = load_address_index(ROOT / "data/raw/rnaddrkor_seoul.txt", boundaries)
    panels: list[pd.DataFrame] = []
    quality: list[dict] = []

    daycare_payload = json.loads((args.collected_dir / "daycare.json").read_text(encoding="utf-8-sig"))
    daycare = pd.DataFrame(daycare_payload["rows"])
    daycare = attach_admin_dong(daycare, address_index, boundaries, "CRADDR", "LO", "LA")
    for column in ("CRCNFMDT", "CRABLDT", "CRPAUSEBEGINDT", "CRPAUSEENDDT"):
        daycare[column] = parse_mixed_date(daycare[column])
    panels.append(count_panel(daycare, "daycare", "CRCNFMDT", "CRABLDT", "CRPAUSEBEGINDT", "CRPAUSEENDDT"))
    quality.append({"dataset": "daycare", "rows": len(daycare), "mapped": daycare["admin_code"].notna().sum(), "history_basis": "인가/휴지/폐지일"})

    hospital = read_csv(ROOT / "data/raw/아동/병원/서울특별시_병원.csv")
    hospital = attach_admin_dong(hospital, address_index, boundaries, "도로명주소", "좌표정보(X)", "좌표정보(Y)", "EPSG:5174")
    for column in ("인허가일자", "인허가취소일자", "폐업일자", "휴업시작일자", "휴업종료일자"):
        hospital[column] = parse_mixed_date(hospital[column])
    # 현재 진료과목을 과거에도 동일했다고 보는 약한 가정이므로 이름으로 위험을 드러낸다.
    pediatric = hospital["진료과목내용명"].fillna("").str.contains("소아청소년과")
    panels.append(count_panel(
        hospital, "hospital", "인허가일자", "폐업일자", "휴업시작일자", "휴업종료일자",
        {"pediatric_hospital_count_current_attribute_backfill": pediatric},
    ))
    quality.append({"dataset": "hospital", "rows": len(hospital), "mapped": hospital["admin_code"].notna().sum(), "history_basis": "인허가/휴업/폐업일"})

    park = read_csv(ROOT / "data/raw/아동/기타/전국도시공원정보표준데이터.csv")
    park_address = park["소재지도로명주소"].fillna(park["소재지지번주소"])
    park = park[park_address.fillna("").str.contains("서울")].copy()
    park["address_for_mapping"] = park_address.loc[park.index]
    park = attach_admin_dong(park, address_index, boundaries, "address_for_mapping", "경도", "위도")
    park["지정고시일"] = pd.to_datetime(park["지정고시일"], errors="coerce")
    child_park = park["공원구분"].fillna("").str.contains("어린이")
    panels.append(count_panel(park, "park", "지정고시일", extra_masks={"children_park_count_reconstructed": child_park}))
    quality.append({"dataset": "park", "rows": len(park), "mapped": park["admin_code"].notna().sum(), "history_basis": "지정고시일; 폐지이력 없음"})

    cctv = read_csv(ROOT / "data/raw/아동/기타/CCTV정보_서울특별시.csv")
    cctv["address_for_mapping"] = cctv["소재지도로명주소"].fillna(cctv["소재지지번주소"])
    cctv = attach_admin_dong(cctv, address_index, boundaries, "address_for_mapping", "WGS84경도", "WGS84위도")
    cctv["installed_date"] = pd.to_datetime(cctv["설치연월"].astype(str).str.replace(r"\.0$", "", regex=True) + "01", format="%Y%m%d", errors="coerce")
    panels.append(count_panel(cctv, "cctv_location", "installed_date"))
    quality.append({"dataset": "cctv", "rows": len(cctv), "mapped": cctv["admin_code"].notna().sum(), "history_basis": "설치연월; 철거이력 없음"})

    bell = read_csv(ROOT / "data/raw/아동/기타/안전비상벨위치정보_서울특별시.csv")
    bell["address_for_mapping"] = bell["소재지도로명주소"].fillna(bell["소재지지번주소"])
    bell = attach_admin_dong(bell, address_index, boundaries, "address_for_mapping", "WGS84경도", "WGS84위도")
    bell["installed_date"] = pd.to_datetime(bell["안전비상벨설치연도"].astype(str).str.extract(r"(\d{4})")[0] + "-01-01", errors="coerce")
    panels.append(count_panel(bell, "emergency_bell", "installed_date"))
    quality.append({"dataset": "emergency_bell", "rows": len(bell), "mapped": bell["admin_code"].notna().sum(), "history_basis": "설치연도; 철거이력 없음"})

    # 모든 시설 패널을 연도·행정동으로 외부결합한다. 어떤 시설이 0개인지와 원본 누락을
    # 구분하기 위해 전체 서울 행정동 격자를 먼저 만들고 결측을 0으로 바꾼다.
    base_dongs = boundaries[["adm_cd", "adm_nm"]].drop_duplicates().rename(columns={"adm_cd": "admin_code"})
    base_dongs["admin_name"] = base_dongs["adm_nm"].str.split().str[-1]
    base = pd.MultiIndex.from_product([YEARS, base_dongs["admin_code"]], names=["year", "admin_code"]).to_frame(index=False)
    base = base.merge(base_dongs[["admin_code", "admin_name"]], on="admin_code", how="left")
    for panel in panels:
        base = base.merge(panel.drop(columns="admin_name"), on=["year", "admin_code"], how="left", validate="one_to_one")
    metrics = [column for column in base if column not in {"year", "admin_code", "admin_name"}]
    base[metrics] = base[metrics].fillna(0).astype(int)
    base.to_csv(args.output / "reconstructed_infrastructure_panel_2021_2026.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality).to_csv(args.output / "reconstruction_quality.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(quality).to_string(index=False))
    print(f"패널: {len(base):,}행, 특징 {len(metrics)}개")
    print(f"저장 위치: {args.output.resolve()}")


if __name__ == "__main__":
    main()
