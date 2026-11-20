"""Build a Seoul gu-year child-environment training panel from the supplied raw files.

The scoring follows Lee & Koo (2015): demand-standardised indicators, four
relative classes (normal-theory or Jenks), and an AHP-weighted sum.  Because
the paper's AHP weights came from a survey that is not part of these raw data,
its published category weights are retained and renormalised over indicators
that pass the >=5-year panel rule.
"""

from __future__ import annotations

import json
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import jenkspy
import numpy as np
import pandas as pd
from scipy.stats import shapiro


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "아동인프라"
OUT = ROOT / "data" / "processed" / "child_environment_scores"

SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
]

CATEGORY_WEIGHT = {
    "education": 0.336,
    "parenting_support": 0.224,
    "park_green": 0.105,
    "culture": 0.099,
    "commercial": 0.068,
    "medical": 0.169,
}

INDICATORS = {
    "kindergarten_per_1000_age3_5": {
        "label": "3~5세 1,000명당 유치원 수", "category": "education"
    },
    "park_area_m2_per_capita": {
        "label": "1인당 공원면적(㎡)", "category": "park_green"
    },
    "children_park_area_m2_per_1000_age0_5": {
        "label": "0~5세 1,000명당 어린이공원 면적(㎡)", "category": "park_green"
    },
    "cultural_welfare_per_10000_age0_5": {
        "label": "0~5세 10,000명당 지역문화복지시설 수", "category": "culture"
    },
    "public_health_staff_per_capita": {
        "label": "1인당 보건소 인력 수", "category": "medical"
    },
    "general_hospital_per_100000_capita": {
        "label": "인구 10만명당 종합병원급 수", "category": "medical"
    },
    "hospital_clinic_per_1000_capita": {
        "label": "인구 1,000명당 병원·의원 수", "category": "medical"
    },
    "pediatric_provider_per_1000_age0_5": {
        "label": "0~5세 1,000명당 소아청소년과 진료기관 수", "category": "medical"
    },
}


def number(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "—", "nan", "None"}:
        return 0.0 if text in {"-", "—"} else np.nan
    return pd.to_numeric(text, errors="coerce")


def write_csv_resilient(frame: pd.DataFrame, path: Path) -> Path:
    """Write the canonical file, or an explicit updated copy when Excel locks it."""
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        frame.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"Locked file preserved; wrote updated result to {fallback}")
        return fallback


def extract_gu(value: object) -> str | None:
    text = str(value)
    for gu in SEOUL_GU:
        if gu in text:
            return gu
    return None


def load_population() -> pd.DataFrame:
    records = []
    folder = RAW / "연령별인구현황 2010~2025"
    for path in sorted(folder.glob("*.csv")):
        year = int(path.name[:4])
        df = pd.read_csv(path, encoding="cp949", dtype=str)
        for _, row in df.iterrows():
            gu = extract_gu(row.iloc[0])
            if gu is None:
                continue
            total = number(row.iloc[1])
            ages = [number(row.iloc[i]) for i in range(3, 9)]
            records.append({
                "year": year, "gu": gu, "population": total,
                "population_age0_5": np.nansum(ages),
                "population_age3_5": np.nansum(ages[3:6]),
            })
    return pd.DataFrame(records).drop_duplicates(["year", "gu"])


def load_kindergarten() -> pd.DataFrame:
    records = []
    folder = RAW / "3~5세 1000명당 유치원수"
    for path in sorted(folder.glob("*.xlsx")):
        match = re.search(r"(20\d{2})", path.name)
        if not match:
            continue
        year = int(match.group(1))
        df = pd.read_excel(path, header=2)
        address_col = next((c for c in df.columns if "주소" in str(c)), df.columns[6])
        gu = df[address_col].map(extract_gu)
        counts = gu.value_counts()
        for name, count in counts.items():
            if name in SEOUL_GU:
                records.append({"year": year, "gu": name, "kindergarten_count": int(count)})
    return pd.DataFrame(records)


def load_wide_kosis(path: Path, mode: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    years = pd.to_numeric(raw.iloc[0], errors="coerce")
    level = raw.iloc[1].astype(str)
    metric = raw.iloc[2].astype(str)
    records = []
    for r in range(3, len(raw)):
        gu = extract_gu(raw.iat[r, 1] if raw.shape[1] > 1 else None)
        if gu is None:
            continue
        for year in sorted({int(y) for y in years.dropna()}):
            cols = [c for c in range(raw.shape[1]) if years.iat[c] == year]
            if mode == "park":
                chosen = [c for c in cols if level.iat[c] == "공원" and "면적" in metric.iat[c]]
                if not chosen:
                    chosen = [c for c in cols if "공원" == level.iat[c] and "면적" in metric.iat[c]]
                value = number(raw.iat[r, chosen[0]]) * 1000 if chosen else np.nan
                key = "park_area_m2"
            else:
                values = [number(raw.iat[r, c]) for c in cols]
                value = float(np.nansum(values)) if any(pd.notna(v) for v in values) else np.nan
                key = "cultural_welfare_count"
            records.append({"year": year, "gu": gu, key: value})
    return pd.DataFrame(records)


def xml_spreadsheet_rows(path: Path) -> list[list[str | None]]:
    text = path.read_bytes().decode("euc-kr").lstrip()
    text = text.replace('encoding="EUC-KR"', 'encoding="UTF-8"', 1)
    end = text.find("</Worksheet>") + len("</Worksheet>")
    text = text[:end] + "</Workbook>"
    root = ET.fromstring(text.encode("utf-8"))
    ns = {"s": "urn:schemas-microsoft-com:office:spreadsheet"}
    result = []
    for row in root.findall(".//s:Worksheet[1]/s:Table/s:Row", ns):
        values: list[str | None] = []
        cursor = 1
        for cell in row.findall("s:Cell", ns):
            idx = cell.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Index")
            if idx:
                cursor = int(idx)
            values.extend([None] * (cursor - 1 - len(values)))
            data = cell.find("s:Data", ns)
            values.append(data.text if data is not None else None)
            cursor += 1
        result.append(values)
    return result


def load_xml_panel(path: Path, mode: str) -> pd.DataFrame:
    rows = xml_spreadsheet_rows(path)
    years = [int(re.search(r"\d{4}", x).group()) if x and re.search(r"\d{4}", x) else None for x in rows[1]]
    records = []
    for row in rows[2:]:
        row += [None] * (len(years) - len(row))
        gu = extract_gu(row[0] if row else None)
        if gu is None:
            continue
        if mode == "children_park":
            keep = len(row) > 3 and row[1] == "어린이공원" and row[2] == "면적"
            key = "children_park_area_m2"
        else:
            keep = len(row) > 2 and row[1] == "합계" and "보건소 인력" in str(row[2])
            key = "public_health_staff"
        if not keep:
            continue
        for c, year in enumerate(years):
            if year is not None:
                value = number(row[c])
                if mode == "children_park" and row[3] == "천㎡":
                    value *= 1000
                records.append({"year": year, "gu": gu, key: value})
    return pd.DataFrame(records)


def load_reconstructed_hospitals(years: list[int]) -> pd.DataFrame:
    """Backcast currently operating providers from establishment year.

    The source has no closure date.  Per the requested assumption, each unique
    provider is treated as continuously operating from estbDd through the
    latest panel year.  These are reconstructed survivor counts, not official
    historical stocks.
    """
    folder = RAW / "병원"
    all_facilities = pd.read_csv(folder / "서울시_의료기관_기본정보.csv", encoding="utf-8-sig", dtype=str)
    pediatric = pd.read_csv(
        folder / "서울시_소아청소년과_진료기관_설립일자포함.csv",
        encoding="utf-8-sig", dtype=str,
    )
    all_facilities = all_facilities.drop_duplicates("ykiho").copy()
    pediatric = pediatric.drop_duplicates("ykiho").copy()
    for frame in (all_facilities, pediatric):
        frame["establishment_year"] = pd.to_numeric(frame["estbDd"].str[:4], errors="coerce")
        frame["gu"] = frame["sgguCdNm"].map(extract_gu)

    records = []
    for year in years:
        active_all = all_facilities[all_facilities["establishment_year"] <= year]
        active_ped = pediatric[pediatric["establishment_year"] <= year]
        for gu in SEOUL_GU:
            gu_all = active_all[active_all["gu"] == gu]
            general = gu_all["clCdNm"].isin(["상급종합", "종합병원"]).sum()
            hospital_clinic = gu_all["clCdNm"].isin(["병원", "의원"]).sum()
            pediatric_count = (active_ped["gu"] == gu).sum()
            records.append({
                "year": year, "gu": gu,
                "general_hospital_count_reconstructed": int(general),
                "hospital_clinic_count_reconstructed": int(hospital_clinic),
                "pediatric_provider_count_reconstructed": int(pediatric_count),
            })
    return pd.DataFrame(records)


def four_class_scores(values: pd.Series) -> tuple[pd.Series, str, list[float], float | None]:
    arr = values.astype(float).to_numpy()
    p_value = float(shapiro(arr).pvalue) if len(arr) >= 3 else None
    if p_value is not None and p_value >= 0.05 and np.std(arr) > 0:
        mean, sd = float(np.mean(arr)), float(np.std(arr, ddof=0))
        boundaries = [mean - sd, mean, mean + sd]
        classes = np.digitize(arr, boundaries, right=False)
        method = "normal_mean_sd"
    else:
        unique = len(np.unique(arr))
        n_classes = min(4, unique)
        if n_classes < 2:
            return pd.Series(1.0, index=values.index), "constant", [], p_value
        breaks = jenkspy.jenks_breaks(arr, n_classes=n_classes)
        boundaries = list(map(float, breaks[1:-1]))
        classes = np.digitize(arr, boundaries, right=True)
        if n_classes < 4:
            classes = np.rint(classes * 3 / (n_classes - 1)).astype(int)
        method = "jenks_natural_breaks"
    score = np.array([0.25, 0.50, 0.75, 1.00])[classes]
    return pd.Series(score, index=values.index), method, boundaries, p_value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    population = load_population()
    kindergarten = load_kindergarten()
    park = load_wide_kosis(RAW / "1인당 공원면적" / "공원(1인당+공원면적)_20261104061630.xlsx", "park")
    culture = load_wide_kosis(RAW / "지역문화복지시설수" / "문화공간(지역문화복지시설)_20261104064427.xlsx", "culture")
    child_park_path = next((RAW / "0~5세 1000명당 어린이공원 면적").glob("*.xls"))
    health_path = next((RAW / "보건소 인력수").glob("*.xls"))
    child_park = load_xml_panel(child_park_path, "children_park")
    health = load_xml_panel(health_path, "health")
    hospitals = load_reconstructed_hospitals(sorted(population["year"].unique().tolist()))
    write_csv_resilient(hospitals, OUT / "hospital_annual_reconstructed_counts.csv")

    panel = population.merge(kindergarten, on=["year", "gu"], how="left")
    for frame in (park, culture, child_park, health, hospitals):
        panel = panel.merge(frame, on=["year", "gu"], how="left")

    panel["kindergarten_per_1000_age3_5"] = panel["kindergarten_count"] / panel["population_age3_5"] * 1000
    panel["park_area_m2_per_capita"] = panel["park_area_m2"] / panel["population"]
    panel["children_park_area_m2_per_1000_age0_5"] = panel["children_park_area_m2"] / panel["population_age0_5"] * 1000
    panel["cultural_welfare_per_10000_age0_5"] = panel["cultural_welfare_count"] / panel["population_age0_5"] * 10000
    panel["public_health_staff_per_capita"] = panel["public_health_staff"] / panel["population"]
    panel["general_hospital_per_100000_capita"] = panel["general_hospital_count_reconstructed"] / panel["population"] * 100000
    panel["hospital_clinic_per_1000_capita"] = panel["hospital_clinic_count_reconstructed"] / panel["population"] * 1000
    panel["pediatric_provider_per_1000_age0_5"] = panel["pediatric_provider_count_reconstructed"] / panel["population_age0_5"] * 1000

    coverage = []
    eligible = []
    for key, meta in INDICATORS.items():
        complete_years = []
        for year, group in panel.groupby("year"):
            if len(group) == 25 and group[key].notna().all():
                complete_years.append(int(year))
        passed = len(complete_years) >= 5
        if passed:
            eligible.append(key)
        coverage.append({
            "indicator": key, "label": meta["label"], "category": meta["category"],
            "complete_year_count": len(complete_years),
            "complete_years": ",".join(map(str, complete_years)), "included": passed,
        })

    common_years = sorted(set.intersection(*[
        set(map(int, next(x["complete_years"] for x in coverage if x["indicator"] == key).split(",")))
        for key in eligible
    ]))
    final = panel[panel["year"].isin(common_years)].copy()
    final = final[final["gu"].isin(SEOUL_GU)].sort_values(["year", "gu"]).reset_index(drop=True)

    by_category: dict[str, list[str]] = {}
    for key in eligible:
        by_category.setdefault(INDICATORS[key]["category"], []).append(key)
    raw_weights = {
        key: CATEGORY_WEIGHT[INDICATORS[key]["category"]] / len(by_category[INDICATORS[key]["category"]])
        for key in eligible
    }
    denominator = sum(raw_weights.values())
    weights = {key: value / denominator for key, value in raw_weights.items()}

    thresholds = []
    for year in common_years:
        mask = final["year"] == year
        for key in eligible:
            scored, method, boundaries, p_value = four_class_scores(final.loc[mask, key])
            score_col = f"score_{key}"
            final.loc[mask, score_col] = scored
            thresholds.append({
                "year": year, "indicator": key, "method": method,
                "shapiro_p": p_value, "boundaries": json.dumps(boundaries, ensure_ascii=False),
            })

    final["child_environment_score"] = sum(final[f"score_{key}"] * weight for key, weight in weights.items())
    final["child_environment_score_100"] = final["child_environment_score"] * 100
    # Exact quintiles: 25 gu -> five gu per A/B/C/D/E.  Gu name is the
    # deterministic tie-breaker because equal scores cannot be split otherwise.
    final = final.sort_values(["year", "child_environment_score", "gu"], ascending=[True, False, True])
    final["rank_within_year"] = final.groupby("year").cumcount() + 1
    final["grade"] = pd.cut(
        final["rank_within_year"], bins=[0, 5, 10, 15, 20, 25],
        labels=["A", "B", "C", "D", "E"], include_lowest=True,
    ).astype(str)

    weight_rows = [{
        "indicator": key, "label": INDICATORS[key]["label"],
        "category": INDICATORS[key]["category"],
        "published_category_weight": CATEGORY_WEIGHT[INDICATORS[key]["category"]],
        "pre_normalization_weight": raw_weights[key], "final_weight": weights[key],
    } for key in eligible]

    raw_cols = ["year", "gu", "population", "population_age0_5", "population_age3_5",
                "kindergarten_count", "park_area_m2", "children_park_area_m2",
                "cultural_welfare_count", "public_health_staff",
                "general_hospital_count_reconstructed", "hospital_clinic_count_reconstructed",
                "pediatric_provider_count_reconstructed"]
    write_csv_resilient(final[raw_cols + eligible], OUT / "preprocessed_merged_panel.csv")
    write_csv_resilient(final, OUT / "scored_training_panel.csv")
    write_csv_resilient(pd.DataFrame(weight_rows), OUT / "weights.csv")
    write_csv_resilient(pd.DataFrame(thresholds), OUT / "yearly_classification_thresholds.csv")
    write_csv_resilient(pd.DataFrame(coverage), OUT / "indicator_coverage.csv")

    exclusions = pd.DataFrame([
        {"source": "서울시 어린이집 정보(표준 데이터).csv", "candidate_indicator": "어린이집 수",
         "observed_years": "2026", "reason": "단일 최신 스냅샷으로 5개년 기준 미충족"},
        {"source": "서울시_도서관_표준데이터.csv/json", "candidate_indicator": "도서관 수",
         "observed_years": "2026", "reason": "단일 최신 스냅샷으로 5개년 기준 미충족"},
        {"source": "서울시 및 자치구 장난감도서관 현황*.xlsx", "candidate_indicator": "장난감도서관 수",
         "observed_years": "2022,2024,2025", "reason": "서로 다른 연도 3개로 5개년 기준 미충족"},
        {"source": "문화공간(문화보급전수시설/전시시설).xlsx", "candidate_indicator": "별도 문화시설 수",
         "observed_years": "1993~2025", "reason": "논문의 지역문화복지시설 지표와 중복되므로 별도 가산하지 않음"},
    ])
    write_csv_resilient(exclusions, OUT / "excluded_sources.csv")

    formula = " + ".join(f"{weights[k]:.12f} × score_{k}" for k in eligible)
    (OUT / "FORMULA.md").write_text(
        "# 아동 물리환경 점수 계산식\n\n"
        f"공통기간: {common_years[0]}–{common_years[-1]} ({len(common_years)}개년), 자치구 25개\n\n"
        "항목값은 논문처럼 대상 인구로 표준화하고, 각 연도 25개 구 안에서 정규성 검정 후 "
        "정규분포면 평균±모집단표준편차, 아니면 Jenks natural breaks로 4등급 "
        "0.25/0.50/0.75/1.00을 부여한다.\n\n"
        f"`child_environment_score = {formula}`\n\n"
        "`child_environment_score_100 = child_environment_score × 100`\n\n"
        "원 논문의 세부항목별 AHP 설문 원행렬은 공개 원자료에 없으므로, 공개된 대분류 가중치를 "
        "5개년 기준 통과 항목에 대분류 내 균등배분한 뒤 전체 합이 1이 되도록 재정규화했다.\n\n"
        "병원 관련 3개 지표는 폐업일이 없는 현재 운영기관 자료를 사용했다. 기관별 설립연도부터 "
        "계속 운영 중이라고 가정한 역산치이므로, 과거 폐업기관이 누락되는 생존편향이 있으며 "
        "공식 과거 시점 재고와 동일하지 않다.\n",
        encoding="utf-8",
    )
    assert len(final) == len(common_years) * 25
    assert final[["year", "gu"]].duplicated().sum() == 0
    assert final[eligible].notna().all().all()
    assert np.isclose(sum(weights.values()), 1.0)
    print(f"Wrote {len(final)} rows, {len(common_years)} years, {len(eligible)} indicators to {OUT}")
    print("Years:", common_years)
    print("Weights:", weights)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    main()
