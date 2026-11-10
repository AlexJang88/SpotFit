from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "지역아동센터"
OUT = ROOT / "data" / "interim" / "안전" / "자치구별_지역아동센터_2017_2025_최종.xlsx"
OFFICIAL_STATS = RAW / "서울시_지역아동센터_자치구별통계_2017_2025_원본.xlsx"

DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]


def clean_text(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else pd.NA


def normalize_phone(value):
    value = clean_text(value)
    if pd.isna(value):
        return pd.NA
    return re.sub(r"\.0$", "", str(value))


def find_header_row(path):
    preview = pd.read_excel(path, sheet_name=0, header=None, nrows=15)
    for idx, row in preview.iterrows():
        values = {str(v).strip().replace(" ", "") for v in row.dropna()}
        if "시설명" in values and any(v in values for v in {"시도", "시도명"}):
            return idx
    raise ValueError(f"헤더 행을 찾지 못했습니다: {path.name}")


def choose_column(columns, candidates, required=False):
    normalized = {str(col).strip().replace(" ", ""): col for col in columns}
    for candidate in candidates:
        key = candidate.replace(" ", "")
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"필수 컬럼 없음: {candidates}; 실제 컬럼={list(columns)}")
    return None


def read_address_book(path, year):
    header_row = find_header_row(path)
    raw = pd.read_excel(path, sheet_name=0, header=header_row)
    sido_col = choose_column(raw.columns, ["시도", "시도명"], required=True)
    district_col = choose_column(raw.columns, ["시군구", "시군구명"], required=True)
    name_col = choose_column(raw.columns, ["시설명"], required=True)
    address_col = choose_column(raw.columns, ["주소", "시설소재지", "시설 소재지"])
    phone_col = choose_column(raw.columns, ["시설전화번호", "시설 전화번호", "대표 전화번호"])
    fax_col = choose_column(raw.columns, ["팩스번호", "팩스 번호"])
    homepage_col = choose_column(raw.columns, ["홈페이지", "홈페이지 주소"])
    capacity_col = choose_column(raw.columns, ["입소정원"])
    children_col = choose_column(raw.columns, ["아동현원"])

    sido = raw[sido_col].map(clean_text)
    seoul_mask = sido.astype("string").str.contains("서울", na=False)
    raw = raw.loc[seoul_mask].copy()

    result = pd.DataFrame({
        "연도": year,
        "서울시": "서울특별시",
        "자치구": raw[district_col].map(clean_text),
        "시설명": raw[name_col].map(clean_text),
        "주소": raw[address_col].map(clean_text) if address_col else pd.NA,
        "입소정원": pd.to_numeric(raw[capacity_col], errors="coerce") if capacity_col else np.nan,
        "아동현원": pd.to_numeric(raw[children_col], errors="coerce") if children_col else np.nan,
        "전화번호": raw[phone_col].map(normalize_phone) if phone_col else pd.NA,
        "팩스번호": raw[fax_col].map(normalize_phone) if fax_col else pd.NA,
        "홈페이지": raw[homepage_col].map(clean_text) if homepage_col else pd.NA,
        "원본파일": path.name,
    })
    result = result[result["자치구"].isin(DISTRICTS) & result["시설명"].notna()].copy()
    result["중복판정키"] = (
        result["연도"].astype(str) + "|" + result["자치구"].astype(str) + "|"
        + result["시설명"].astype(str).str.replace(r"\s+", "", regex=True) + "|"
        + result["주소"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
    )
    result = result.drop_duplicates("중복판정키", keep="first").drop(columns="중복판정키")
    return result


def read_official_statistics(path):
    raw = pd.read_excel(path, sheet_name="데이터", header=None)
    headers = raw.iloc[:4].copy()
    rows = raw.iloc[5:30].copy()
    records = []
    for col in range(2, raw.shape[1]):
        year = pd.to_numeric(headers.iloc[0, col], errors="coerce")
        category = str(headers.iloc[1, col]).strip()
        measure = str(headers.iloc[2, col]).strip()
        subtotal = str(headers.iloc[3, col]).strip()
        if pd.isna(year) or category != "지역아동센터" or subtotal != "소계":
            continue
        if measure not in {"시설수 (개소)", "이용자수 (명)"}:
            continue
        value_name = "지역아동센터수" if measure.startswith("시설수") else "지역아동센터 이용아동수"
        for row_idx in rows.index:
            records.append({
                "연도": int(year),
                "서울시": "서울특별시",
                "자치구": clean_text(raw.iloc[row_idx, 1]),
                "지표": value_name,
                "값": pd.to_numeric(raw.iloc[row_idx, col], errors="raise"),
            })
    long = pd.DataFrame(records)
    data = long.pivot(index=["연도", "서울시", "자치구"], columns="지표", values="값").reset_index()
    data.columns.name = None
    data = data[data["연도"].between(2017, 2025)].copy()
    for col in ["지역아동센터수", "지역아동센터 이용아동수"]:
        data[col] = pd.to_numeric(data[col], errors="raise").astype("int64")
    if len(data) != 225 or data.duplicated(["연도", "자치구"]).any():
        raise ValueError("공식 통계가 2017~2025년 × 서울 25개 자치구 구조가 아닙니다.")
    if set(data["자치구"]) != set(DISTRICTS):
        raise ValueError("공식 통계의 자치구 목록이 서울 25개 자치구와 다릅니다.")
    return data.sort_values(["연도", "자치구"]).reset_index(drop=True)


def main():
    files = sorted(RAW.glob("*.xlsx"))
    year_files = {int(path.name[:4]): path for path in files if re.match(r"^20\d{2}_", path.name)}
    missing = sorted(set(range(2018, 2026)) - set(year_files))
    if missing:
        raise FileNotFoundError(f"누락 연도: {missing}")
    if not OFFICIAL_STATS.exists():
        raise FileNotFoundError(OFFICIAL_STATS)

    official = read_official_statistics(OFFICIAL_STATS)
    details = pd.concat(
        [read_address_book(year_files[year], year) for year in range(2018, 2026)],
        ignore_index=True,
    ).sort_values(["연도", "자치구", "시설명"]).reset_index(drop=True)

    counts = details.groupby(["연도", "자치구"], as_index=False).agg(
        주소록_시설수=("시설명", "size"),
        지역아동센터_입소정원=("입소정원", lambda s: s.sum(min_count=1)),
        지역아동센터_아동현원=("아동현원", lambda s: s.sum(min_count=1)),
    )
    summary = official.merge(counts, on=["연도", "자치구"], how="left", validate="one_to_one")
    summary["지역아동센터수"] = pd.to_numeric(summary["지역아동센터수"], errors="raise").astype("int64")
    summary["센터당_이용아동수"] = summary["지역아동센터 이용아동수"] / summary["지역아동센터수"]
    summary["시설수_주소록차이"] = summary["주소록_시설수"] - summary["지역아동센터수"]
    summary["이용아동수_아동현원차이"] = summary["지역아동센터_아동현원"] - summary["지역아동센터 이용아동수"]
    summary["시설수_집계기준"] = "서울시 공식 자치구 집계"
    summary = summary.sort_values(["연도", "자치구"]).reset_index(drop=True)

    year_check = summary.groupby("연도", as_index=False).agg(
        자치구수=("자치구", "nunique"),
        지역아동센터수=("지역아동센터수", "sum"),
        이용아동수=("지역아동센터 이용아동수", "sum"),
        주소록_시설수=("주소록_시설수", lambda s: s.sum(min_count=1)),
        입소정원=("지역아동센터_입소정원", lambda s: s.sum(min_count=1)),
        아동현원=("지역아동센터_아동현원", lambda s: s.sum(min_count=1)),
    )
    year_check["연도자치구_기대행수"] = 25
    year_check["자치구완전성"] = np.where(year_check["자치구수"].eq(25), "정상", "확인필요")

    quality = pd.DataFrame([
        ["대상 기간", "2017~2025", "정상"],
        ["모델용 행 수", len(summary), "정상" if len(summary) == 225 else "확인필요"],
        ["연도×자치구 중복", int(summary.duplicated(["연도", "자치구"]).sum()), "정상" if not summary.duplicated(["연도", "자치구"]).any() else "확인필요"],
        ["서울 외 시설", int((details["서울시"] != "서울특별시").sum()), "정상"],
        ["비표준 자치구", int((~details["자치구"].isin(DISTRICTS)).sum()), "정상"],
        ["시설 상세 중복 제거 후 행 수", len(details), "참고"],
        ["공식 이용아동수 결측", int(summary["지역아동센터 이용아동수"].isna().sum()), "정상"],
        ["공식 이용아동수 제공", "2017~2025년 25개 자치구", "정상"],
        ["2025 주소록 아동현원", "공식 이용아동수와 별도 보존", "참고"],
    ], columns=["검사항목", "결과", "판정"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="자치구별_모델용", index=False)
        details.to_excel(writer, sheet_name="시설별_상세", index=False)
        year_check.to_excel(writer, sheet_name="연도별_검산", index=False)
        quality.to_excel(writer, sheet_name="품질검사", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

    print(OUT)
    print(year_check.to_string(index=False))


if __name__ == "__main__":
    main()
