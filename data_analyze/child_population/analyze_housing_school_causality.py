"""2024년 주택 입주와 아동 인구 변화의 준인과 근거를 자치구 단위로 점검한다."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml/training"))
from train_child_population_growth import CORE, GROUPS, load_population  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def number(value: object) -> float:
    return pd.to_numeric(str(value).replace(",", "").strip(), errors="coerce")


def housing_2024() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = ROOT / "data/raw/아동/주택공급"
    latest = next(folder.glob("*25~26년*.xlsx"))
    sheet = next(name for name in pd.ExcelFile(latest).sheet_names if "별첨" in name and "24" in name)
    raw = pd.read_excel(latest, sheet_name=sheet, header=None)
    rows = []
    for _, row in raw.iloc[5:].iterrows():
        year = number(row.iloc[2])
        gu = str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else ""
        if year != 2024 or not gu.endswith("구"):
            continue
        rows.append({"year": 2024, "month": number(row.iloc[3]), "gu": gu,
                     "address": str(row.iloc[8]).strip(), "project": str(row.iloc[9]).strip(),
                     "complex": str(row.iloc[10]).strip(), "supply_total": number(row.iloc[11]),
                     "supply_sale": number(row.iloc[12]), "supply_rental": number(row.iloc[13]),
                     "business_type": str(row.iloc[15]).strip()})
    projects = pd.DataFrame(rows).dropna(subset=["supply_total"])
    projects = projects.drop_duplicates(["gu", "address", "project", "complex", "supply_total"])
    audit = pd.DataFrame([{"source": str(path.relative_to(ROOT)), "sha256": sha256(path),
                           "role": "included_2024_actual_appendix" if path == latest else "excluded_overlapping_vintage",
                           "reason": "최신 2024년 말 발표본의 2024 별첨 우선" if path == latest else "동일 사업 중복·수정 가능"}
                          for path in sorted(folder.glob("*.xlsx"))])
    return projects, audit


def school_openings() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "data/raw/school/20250401기준_학교일람표_졸업후상황추가.xlsx"
    excel = pd.ExcelFile(path)
    sheet = next(name for name in excel.sheet_names if "주요" in name)
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_row = next(i for i, row in raw.iterrows() if any(str(v).strip() == "학교명" for v in row if pd.notna(v)))
    columns = [str(v).replace("\n", "").strip() if pd.notna(v) else f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_row])]
    schools = raw.iloc[header_row + 1:].copy(); schools.columns = columns
    schools = schools[schools["시도"].astype(str).str.contains("서울", na=False)].copy()
    schools["opening_date"] = pd.to_datetime(schools["개교일"], errors="coerce")
    schools["opening_year"] = schools["opening_date"].dt.year
    schools = schools[schools["opening_year"].between(2021, 2025)].copy()
    detail = schools[["행정구", "학교급", "학교명", "상태", "개교일", "opening_date", "opening_year", "주소"]]
    counts = detail.groupby(["행정구", "opening_year"]).size().rename("school_openings").reset_index().rename(columns={"행정구": "gu"})
    return detail, counts


def births_by_district() -> tuple[pd.DataFrame, pd.DataFrame]:
    """서울 통계 XLSX의 자치구 소계행에서 신고기준 출생아 수를 연도별로 합산한다."""
    folder = ROOT / "data/raw/아동/출생인구"
    frames, audits = [], []
    for path in sorted(folder.glob("*.xlsx")):
        raw = pd.read_excel(path, sheet_name=0, header=None)
        year_values = raw.iloc[0].astype(str).str.extract(r"(20\d{2})", expand=False).dropna()
        if year_values.empty:
            continue
        year = int(year_values.iloc[0])
        birth_columns = [i for i, value in enumerate(raw.iloc[1]) if str(value).strip() == "출생자수"]
        district_rows = raw.iloc[3:].copy()
        district_rows = district_rows[
            district_rows.iloc[:, 1].astype(str).str.endswith("구", na=False)
            & district_rows.iloc[:, 2].astype(str).str.strip().eq("소계")
        ]
        values = district_rows.iloc[:, birth_columns].apply(
            lambda column: pd.to_numeric(column.astype(str).str.replace(",", "", regex=False), errors="coerce")
        )
        frame = pd.DataFrame({
            "gu": district_rows.iloc[:, 1].astype(str).str.strip(),
            "year": year,
            "births": values.sum(axis=1, min_count=1),
        })
        frames.append(frame)
        audits.append({
            "source": str(path.relative_to(ROOT)), "sha256": sha256(path), "year": year,
            "district_rows": len(frame), "role": "included_birth_control",
            "reason": "자치구 소계행의 월별 출생자수 합산; 신고기준",
        })
    births = pd.concat(frames, ignore_index=True).sort_values(["gu", "year"])
    if births.duplicated(["gu", "year"]).any():
        raise ValueError("출생 자료에 자치구·연도 중복이 있습니다.")
    return births, pd.DataFrame(audits)


def school_pressure_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """기존 학교 관측 패널을 자치구·연도별 학교급 수용압력 대체지표로 집계한다."""
    path = ROOT / "data/interim/child_school_observed/observed_school_panel_2021_2026.csv"
    boundaries = json.loads((ROOT / "frontend/public/seoul-dong.geojson").read_text(encoding="utf-8"))
    names = pd.DataFrame([
        {"admin_code": str(feature["properties"]["adm_cd"]), "gu": feature["properties"]["sggnm"]}
        for feature in boundaries["features"]
    ]).drop_duplicates("admin_code")
    school = pd.read_csv(path, dtype={"admin_code": str}).merge(
        names, on="admin_code", how="left", validate="many_to_one"
    )
    if school["gu"].isna().any():
        raise ValueError(f"학교 패널의 행정경계 미매핑 {school['gu'].isna().sum()}행")
    levels = {
        "infant": "유치원", "elementary": "초등학교", "middle": "중학교", "high": "고등학교",
    }
    pieces = []
    for group, level in levels.items():
        cols = [f"student_count_{level}", f"class_count_{level}", f"school_count_{level}"]
        part = school.groupby(["gu", "year"], as_index=False)[cols].sum()
        part[f"school_class_size_{group}"] = part[cols[0]].div(part[cols[1]].replace(0, np.nan))
        part = part.rename(columns={cols[0]: f"school_students_{group}", cols[1]: f"school_classes_{group}",
                                    cols[2]: f"school_count_{group}"})
        pieces.append(part)
    panel = pieces[0]
    for part in pieces[1:]:
        panel = panel.merge(part, on=["gu", "year"], how="outer", validate="one_to_one")
    audit = pd.DataFrame([{
        "source": str(path.relative_to(ROOT)), "sha256": sha256(path), "years": "2021-2026",
        "district_year_rows": len(panel), "role": "included_school_pressure_control",
        "reason": "학생수/학급수로 학교급별 학급당 학생 수 산출; 공식 정원·과밀 판정의 대체지표",
    }])
    return panel, audit


def housing_starts_audit() -> pd.DataFrame:
    """추가 착공자료의 분석 가능 공간 단위를 감사한다."""
    path = ROOT / "data/raw/아동/주택공급/주택유형별 주택건설 착공실적(월계).csv"
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    years = pd.to_numeric(frame["기준연월"].str[:4], errors="coerce")
    regions = sorted(frame["지역"].dropna().astype(str).unique())
    return pd.DataFrame([{
        "source": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "years": f"{int(years.min())}-{int(years.max())}", "regions": ", ".join(regions),
        "role": "excluded_from_district_effect_estimation",
        "reason": "서울 전체 월계만 있어 같은 연도의 자치구 간 차이가 없으며 자치구 고정효과와 분리 불가",
    }])


def city_mobility_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """서울 전체 연령별 월간 이동을 연도로 합산하되 자치구 효과에는 사용하지 않는다."""
    folder = ROOT / "data/raw/아동/아동,청소년,청년 인구수"
    candidates = sorted(folder.glob("인구이동(연령별_월별)_*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    path = candidates[-1]
    raw = pd.read_excel(path, sheet_name=0, header=None)
    records = []
    for column in range(2, raw.shape[1]):
        period = pd.to_datetime(str(raw.iloc[0, column]).replace(" ", ""), format="%Y.%m", errors="coerce")
        sex = str(raw.iloc[4, column]).strip()
        if pd.isna(period) or sex != "계":
            continue
        level1, level2 = str(raw.iloc[1, column]).strip(), str(raw.iloc[2, column]).strip()
        if level1 == "총이동" and level2.startswith("총전입"):
            metric = "inflow"
        elif level1 == "총이동" and level2.startswith("총전출"):
            metric = "outflow"
        else:
            continue
        for row in range(6, raw.shape[0]):
            age_band = str(raw.iloc[row, 1]).strip()
            if not re.fullmatch(r"\d+-\d+세", age_band):
                continue
            records.append({"year": int(period.year), "month": int(period.month), "age_band": age_band,
                            "metric": metric, "value": number(raw.iloc[row, column])})
    monthly_long = pd.DataFrame(records)
    monthly = monthly_long.pivot_table(index=["year", "month", "age_band"], columns="metric",
                                       values="value", aggfunc="first").reset_index()
    monthly.columns.name = None
    monthly["net_migration"] = monthly["inflow"] - monthly["outflow"]
    annual = monthly.groupby(["year", "age_band"], as_index=False)[["inflow", "outflow", "net_migration"]].sum(min_count=1)
    coverage = monthly.groupby("year")["month"].nunique().rename("months_covered").reset_index()
    annual = annual.merge(coverage, on="year", how="left", validate="many_to_one")
    annual["complete_year"] = annual["months_covered"].eq(12)
    audit = pd.DataFrame([{
        "source": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "periods": f"{monthly.year.min()}-{monthly.month[monthly.year.eq(monthly.year.min())].min():02d}~"
                   f"{monthly.year.max()}-{monthly.month[monthly.year.eq(monthly.year.max())].max():02d}",
        "age_bands": ", ".join(sorted(monthly["age_band"].unique())),
        "geography": "서울 전체 합계(지역 차원 없음)",
        "role": "included_descriptive_context_only",
        "reason": "연도별 연령 순이동 설명에는 사용; 자치구별 주택공급 인과계수에는 공간 변이가 없어 제외",
    }])
    return annual, audit


def district_age_mobility_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """KOSIS 시군구·성·5세 연령별 월자료를 서울 자치구 연간 이동으로 집계한다."""
    folder = ROOT / "data/raw/아동/인구이동_자치구_연령별"
    age_codes = {"020": "0-4세", "050": "5-9세", "070": "10-14세", "100": "15-19세"}
    frames, audits = [], []
    for path in sorted(folder.glob("101_DT_1B26001_M_20*.csv")):
        raw = pd.read_csv(path, encoding="cp949", skiprows=2, dtype=str, low_memory=False)
        region_code = raw.iloc[:, 0].str.replace("'", "", regex=False).str.strip()
        sex_code = raw.iloc[:, 2].str.replace("'", "", regex=False).str.strip()
        age_code = raw.iloc[:, 4].str.replace("'", "", regex=False).str.strip()
        selected = raw[region_code.str.fullmatch(r"11\d{3}", na=False)
                       & sex_code.eq("0") & age_code.isin(age_codes)].copy()
        selected["gu"] = selected.iloc[:, 1].str.strip()
        selected["age_band"] = age_code[selected.index].map(age_codes)
        selected["year"] = pd.to_numeric(selected.iloc[:, 6].str[:4], errors="coerce")
        selected["inflow"] = pd.to_numeric(selected["총전입 (명)"], errors="coerce")
        selected["outflow"] = pd.to_numeric(selected["총전출 (명)"], errors="coerce")
        selected["net_migration"] = selected["inflow"] - selected["outflow"]
        annual = selected.groupby(["gu", "year", "age_band"], as_index=False)[
            ["inflow", "outflow", "net_migration"]
        ].sum(min_count=1)
        frames.append(annual)
        audits.append({
            "source": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "year": int(annual["year"].iloc[0]), "districts": annual["gu"].nunique(),
            "age_bands": annual["age_band"].nunique(), "role": "included_district_mobility_outcome",
            "reason": "서울 25개 자치구·5세 연령구간의 월별 총전입/총전출을 연간 합산",
        })
    panel = pd.concat(frames, ignore_index=True).sort_values(["gu", "year", "age_band"])
    if panel.duplicated(["gu", "year", "age_band"]).any():
        raise ValueError("자치구 인구이동 패널에 자치구·연도·연령 중복이 있습니다.")
    coverage = panel.groupby("year").agg(districts=("gu", "nunique"), age_bands=("age_band", "nunique"))
    if not (coverage["districts"].eq(25) & coverage["age_bands"].eq(4)).all():
        raise ValueError(f"자치구 인구이동 자료 범위가 불완전합니다.\n{coverage}")
    return panel, pd.DataFrame(audits)


def apartment_prices() -> pd.DataFrame:
    folder = ROOT / "data/raw/부동산/서울시_전월세가"
    frames = []
    for year in (2023, 2024):
        path = folder / f"서울특별시_전월세가_{year}.csv"
        for encoding in ("utf-8-sig", "cp949", "utf-8"):
            try:
                frame = pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        frame["contract_year"] = pd.to_numeric(frame["계약일"].str[:4], errors="coerce")
        frame["area"] = pd.to_numeric(frame["임대면적"], errors="coerce")
        frame["deposit"] = pd.to_numeric(frame["보증금(만원)"].str.replace(",", "", regex=False), errors="coerce")
        frame = frame[frame["contract_year"].eq(year) & frame["전월세구분"].eq("전세")
                      & frame["건물용도"].eq("아파트") & frame["area"].ge(20) & frame["deposit"].gt(0)].copy()
        frame["deposit_per_m2"] = frame["deposit"] / frame["area"]
        frames.append(frame[["자치구명", "contract_year", "deposit_per_m2"]])
    rent = pd.concat(frames, ignore_index=True)
    annual = rent.groupby(["자치구명", "contract_year"])["deposit_per_m2"].median().unstack()
    annual["apartment_jeonse_log_change_2024"] = np.log(annual[2024]) - np.log(annual[2023])
    return annual[["apartment_jeonse_log_change_2024"]].reset_index().rename(columns={"자치구명": "gu"})


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return values - controls @ np.linalg.lstsq(controls, values, rcond=None)[0]


def adjusted_effect(frame: pd.DataFrame, outcome: str, treatment: str, controls: list[str], seed: int = 42) -> dict:
    work = frame[[outcome, treatment, *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    usable_controls = [c for c in controls if work[c].std(ddof=0) > 1e-12]
    standardized = [(work[c].to_numpy(float) - work[c].mean()) / work[c].std(ddof=0) for c in usable_controls]
    z = np.column_stack([np.ones(len(work)), *standardized])
    x = residualize(work[treatment].to_numpy(float), z)
    y = residualize(work[outcome].to_numpy(float), z)
    denominator = float(x @ x)
    beta = float(x @ y / denominator)
    residual = y - beta * x
    se = float(np.sqrt((residual @ residual) / (len(work) - z.shape[1] - 1) / denominator))
    rng = np.random.default_rng(seed)
    permuted = np.array([x @ rng.permutation(y) / denominator for _ in range(5000)])
    p_perm = float((np.count_nonzero(np.abs(permuted) >= abs(beta)) + 1) / (len(permuted) + 1))
    return {"n": len(work), "adjusted_beta": beta, "standard_error_classical": se,
            "ci95_low_classical": beta - 1.96 * se, "ci95_high_classical": beta + 1.96 * se,
            "permutation_p": p_perm, "controls_used": ", ".join(usable_controls),
            "controls_dropped_constant": ", ".join(c for c in controls if c not in usable_controls)}


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "artifacts/child_population_causality" / stamp
    out.mkdir(parents=True)
    population, population_audit = load_population(CORE)
    projects, housing_audit = housing_2024()
    school_detail, openings = school_openings()
    births, birth_audit = births_by_district()
    school_panel, school_audit = school_pressure_panel()
    starts_audit = housing_starts_audit()
    city_mobility, mobility_audit = city_mobility_panel()
    district_mobility, district_mobility_audit = district_age_mobility_panel()
    prices = apartment_prices()

    gu_pop = population.groupby(["gu", "year"], as_index=False)[["total_population", *[f"children_{g}" for g in GROUPS]]].sum()
    gu_pop = gu_pop.sort_values(["gu", "year"])
    for group in GROUPS:
        gu_pop[f"log_growth_{group}"] = gu_pop.groupby("gu")[f"children_{group}"].transform(lambda s: np.log1p(s).diff())
    wide = gu_pop.pivot(index="gu", columns="year")
    analysis = pd.DataFrame(index=wide.index).reset_index()
    analysis["population_2023"] = wide["total_population"][2023].to_numpy()
    analysis["population_2024"] = wide["total_population"][2024].to_numpy()
    for group in GROUPS:
        analysis[f"pretrend_2022_2023_{group}"] = wide[f"log_growth_{group}"][2023].to_numpy()
        analysis[f"growth_2023_2024_{group}"] = wide[f"log_growth_{group}"][2024].to_numpy()
        analysis[f"growth_2024_2025_{group}"] = wide[f"log_growth_{group}"][2025].to_numpy()
        analysis[f"growth_acceleration_{group}"] = analysis[f"growth_2024_2025_{group}"] - analysis[f"growth_2023_2024_{group}"]

    supply = projects.groupby("gu", as_index=False)[["supply_total", "supply_sale", "supply_rental"]].sum()
    analysis = analysis.merge(supply, on="gu", how="left").fillna({"supply_total": 0, "supply_sale": 0, "supply_rental": 0})
    analysis["supply_per_1000"] = analysis["supply_total"] / analysis["population_2024"] * 1000
    opening_2024 = openings[openings.opening_year.eq(2024)][["gu", "school_openings"]]
    analysis = analysis.merge(opening_2024, on="gu", how="left").fillna({"school_openings": 0})
    analysis["school_openings_per_100k"] = analysis["school_openings"] / analysis["population_2024"] * 100000
    analysis = analysis.merge(prices, on="gu", how="left")

    birth_wide = births.pivot(index="gu", columns="year", values="births")
    birth_features = birth_wide[[2023, 2024, 2025]].rename(columns=lambda year: f"births_{year}").reset_index()
    analysis = analysis.merge(birth_features, on="gu", how="left", validate="one_to_one")
    analysis["birth_rate_2023_per_1000"] = analysis["births_2023"] / analysis["population_2023"] * 1000
    analysis["birth_log_change_2023_2024"] = np.log1p(analysis["births_2024"]) - np.log1p(analysis["births_2023"])

    school_wide = school_panel[school_panel["year"].isin([2023, 2024, 2025])].pivot(index="gu", columns="year")
    school_features = pd.DataFrame(index=school_wide.index).reset_index()
    for group in GROUPS:
        for year in (2023, 2024, 2025):
            school_features[f"school_class_size_{group}_{year}"] = school_wide[f"school_class_size_{group}"][year].to_numpy()
            school_features[f"school_students_{group}_{year}"] = school_wide[f"school_students_{group}"][year].to_numpy()
        school_features[f"school_class_size_change_{group}_2023_2024"] = (
            school_features[f"school_class_size_{group}_2024"] - school_features[f"school_class_size_{group}_2023"]
        )
    analysis = analysis.merge(school_features, on="gu", how="left", validate="one_to_one")

    results = []
    for group in GROUPS:
        outcome = f"growth_acceleration_{group}"
        specifications = {
            "baseline": [f"growth_2023_2024_{group}", "apartment_jeonse_log_change_2024"],
            "birth_school_adjusted": [f"growth_2023_2024_{group}", "apartment_jeonse_log_change_2024",
                                      "birth_rate_2023_per_1000", "birth_log_change_2023_2024",
                                      f"school_class_size_{group}_2023"],
            "conditional_on_2024_school_opening": [f"growth_2023_2024_{group}", "apartment_jeonse_log_change_2024",
                                                    "birth_rate_2023_per_1000", "birth_log_change_2023_2024",
                                                    f"school_class_size_{group}_2023", "school_openings_per_100k"],
        }
        rho, p = stats.spearmanr(analysis["supply_per_1000"], analysis[f"growth_2024_2025_{group}"], nan_policy="omit")
        placebo_rho, placebo_p = stats.spearmanr(analysis["supply_per_1000"], analysis[f"pretrend_2022_2023_{group}"], nan_policy="omit")
        for specification, controls in specifications.items():
            effect = adjusted_effect(analysis, outcome, "supply_per_1000", controls)
            results.append({"age_group": group, "specification": specification, **effect,
                            "spearman_post": rho, "spearman_post_p": p,
                            "placebo_pretrend_spearman": placebo_rho, "placebo_pretrend_p": placebo_p,
                            "passes_pretrend_placebo": bool(placebo_p >= 0.05),
                            "robust_causal_candidate": bool(effect["permutation_p"] < 0.05 and placebo_p >= 0.05),
                            "interpretation": "exploratory_not_identified_causal_effect"})
    result = pd.DataFrame(results)

    school_results = []
    for group in GROUPS:
        controls = [f"growth_2023_2024_{group}", "birth_rate_2023_per_1000",
                    f"school_class_size_{group}_2023"]
        effect = adjusted_effect(analysis, f"growth_acceleration_{group}", "school_openings_per_100k", controls)
        pressure_rho, pressure_p = stats.spearmanr(
            analysis[f"school_class_size_{group}_2023"], analysis["school_openings"], nan_policy="omit"
        )
        school_results.append({"age_group": group, **effect,
                               "pre_pressure_vs_opening_spearman": pressure_rho,
                               "pre_pressure_vs_opening_p": pressure_p,
                               "interpretation": "school_opening_association_not_causal_effect"})
    school_result = pd.DataFrame(school_results)

    age_band_groups = {"0-4세": "infant", "5-9세": "elementary", "10-14세": "middle", "15-19세": "high"}
    migration_wide = district_mobility.pivot(index="gu", columns=["age_band", "year"], values="net_migration")
    migration_features = pd.DataFrame(index=migration_wide.index).reset_index()
    migration_results = []
    for age_band, group in age_band_groups.items():
        for year in (2023, 2024, 2025):
            migration_features[f"net_migration_{group}_{year}"] = migration_wide[(age_band, year)].to_numpy()
    analysis = analysis.merge(migration_features, on="gu", how="left", validate="one_to_one")
    for age_band, group in age_band_groups.items():
        analysis[f"net_migration_rate_{group}_2023"] = analysis[f"net_migration_{group}_2023"] / analysis["population_2023"] * 1000
        analysis[f"net_migration_rate_{group}_2024"] = analysis[f"net_migration_{group}_2024"] / analysis["population_2024"] * 1000
        analysis[f"net_migration_rate_{group}_2025"] = analysis[f"net_migration_{group}_2025"] / analysis["population_2024"] * 1000
        controls = [f"net_migration_rate_{group}_2023", "apartment_jeonse_log_change_2024",
                    "birth_rate_2023_per_1000", f"school_class_size_{group}_2023"]
        effect = adjusted_effect(analysis, f"net_migration_rate_{group}_2025", "supply_per_1000", controls)
        placebo_rho, placebo_p = stats.spearmanr(
            analysis["supply_per_1000"], analysis[f"net_migration_rate_{group}_2023"], nan_policy="omit"
        )
        migration_results.append({
            "age_band": age_band, "mapped_population_group": group, **effect,
            "placebo_2023_spearman": placebo_rho, "placebo_2023_p": placebo_p,
            "passes_pretrend_placebo": bool(placebo_p >= 0.05),
            "robust_causal_candidate": bool(effect["permutation_p"] < 0.05 and placebo_p >= 0.05),
            "interpretation": "post_supply_net_migration_association_not_randomized_causal_effect",
        })
    migration_result = pd.DataFrame(migration_results)

    projects.to_csv(out / "housing_projects_2024.csv", index=False, encoding="utf-8-sig")
    housing_audit.to_csv(out / "housing_source_audit.csv", index=False, encoding="utf-8-sig")
    population_audit.to_csv(out / "population_source_audit.csv", index=False, encoding="utf-8-sig")
    birth_audit.to_csv(out / "birth_source_audit.csv", index=False, encoding="utf-8-sig")
    school_audit.to_csv(out / "school_source_audit.csv", index=False, encoding="utf-8-sig")
    starts_audit.to_csv(out / "housing_starts_source_audit.csv", index=False, encoding="utf-8-sig")
    mobility_audit.to_csv(out / "mobility_source_audit.csv", index=False, encoding="utf-8-sig")
    city_mobility.to_csv(out / "seoul_age_mobility_annual_2021_2026.csv", index=False, encoding="utf-8-sig")
    district_mobility_audit.to_csv(out / "district_mobility_source_audit.csv", index=False, encoding="utf-8-sig")
    district_mobility.to_csv(out / "district_age_mobility_annual_2021_2025.csv", index=False, encoding="utf-8-sig")
    births.to_csv(out / "district_births_2021_2026.csv", index=False, encoding="utf-8-sig")
    school_panel.to_csv(out / "district_school_pressure_2021_2026.csv", index=False, encoding="utf-8-sig")
    school_detail.to_csv(out / "school_openings_2021_2025.csv", index=False, encoding="utf-8-sig")
    analysis.to_csv(out / "gu_causal_analysis_panel.csv", index=False, encoding="utf-8-sig")
    result.to_csv(out / "housing_effect_estimates.csv", index=False, encoding="utf-8-sig")
    school_result.to_csv(out / "school_opening_effect_estimates.csv", index=False, encoding="utf-8-sig")
    migration_result.to_csv(out / "housing_to_net_migration_estimates.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", font="Malgun Gothic"); plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, group in zip(axes.flat, GROUPS):
        sns.regplot(data=analysis, x="supply_per_1000", y=f"growth_acceleration_{group}", ax=ax,
                    scatter_kws={"s": 45}, line_kws={"color": "crimson"})
        ax.set_title(f"{GROUPS[group][2]}: 2024 입주물량과 증감 가속")
        ax.set_xlabel("2024 입주 세대 / 주민 1,000명"); ax.set_ylabel("로그증감률 가속")
    plt.tight_layout(); plt.savefig(out / "housing_supply_vs_child_growth.png", dpi=170); plt.close()

    adjusted = result[result["specification"].eq("birth_school_adjusted")].copy()
    significant_housing = adjusted[adjusted["permutation_p"].lt(0.05)]
    robust_housing = adjusted[adjusted["robust_causal_candidate"]]
    significant_school = school_result[school_result["permutation_p"].lt(0.05)]
    if significant_housing.empty:
        housing_summary = "출생·학교 압력을 보정한 규격에서 순열검정 5% 기준의 연령군이 없습니다."
    elif robust_housing.empty:
        housing_summary = ("초등·중등 연령군의 조건부 계수는 5% 기준이지만 공급 이전 추세 위약검정도 유의하여 "
                           "인과 후보로 채택하지 않습니다. 원래 성장하던 자치구에 공급이 집중된 선택효과와 구분되지 않습니다.")
    else:
        housing_summary = "위약검정까지 통과한 탐색적 후보 연령군: " + ", ".join(robust_housing["age_group"])
    school_summary = ("학교 신설과 이후 아동 인구 가속도의 관계도 순열검정 5% 기준에서 확인되지 않았습니다."
                      if significant_school.empty else
                      "학교 신설 관계가 5% 기준인 연령군: " + ", ".join(significant_school["age_group"]))
    migration_candidates = migration_result[migration_result["robust_causal_candidate"]]
    migration_summary = ("공급 이후 2025년 연령별 순이동률에서 효과검정과 2023년 위약검정을 함께 통과한 구간은 없습니다."
                         if migration_candidates.empty else
                         "순이동의 탐색적 인과 후보 구간: " + ", ".join(migration_candidates["age_band"]))
    report = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>주택공급·학교와 아동인구 준인과 분석</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;max-width:1200px;margin:35px auto;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px}}.warn{{background:#fff3cd;padding:14px;border-left:5px solid #d39e00}}img{{max-width:100%}}</style>
<h1>주택공급·학교 신설과 아동 인구 변화: 준인과 가능성 점검</h1>
<div class='warn'><b>결론 수준:</b> 출생과 학교 수용압력을 추가했지만 인과효과는 아직 식별되지 않습니다. 입주지역의 비무작위 선택, 자치구 25개, 전입·전출 부재, 1개 처치연도라는 한계가 남습니다.</div>
<h2>1. 추가 데이터와 설계</h2><p>2021~2025 자치구별 신고기준 출생아 수와 2021~2025 학교급별 학생·학급 패널을 추가했습니다. 학교 공식 정원 대신 전년도 학급당 학생 수를 수용압력 대체지표로 사용했습니다. 처치는 주민 1,000명당 2024 입주 세대, 결과는 2023→2024 대비 2024→2025 아동 로그증감률의 변화입니다.</p>
<p>월별 착공실적은 서울 전체 합계만 제공되어 자치구 간 차이를 설명할 수 없으므로 효과 추정에서 제외했습니다. 착공을 준공 또는 실제 입주로 해석하지 않았습니다.</p>
<p>추가 인구이동 파일은 2021년 1월~2026년 8월 서울 전체 연령별 월자료입니다. 완전연도인 2021~2025년의 전입·전출·순이동을 보조 설명에 사용했지만, 자치구 컬럼이 없어 자치구별 주택공급 효과 추정에는 투입하지 않았습니다.</p>
<h2>2. 서울 전체 연령별 이동 맥락</h2><p>2026년은 8개월 누계이므로 완전연도 비교에서 제외해야 합니다. 이 표는 서울 전체 이동 방향을 보여줄 뿐 어느 자치구의 주택공급과 연결됐는지는 보여주지 않습니다.</p>{city_mobility.to_html(index=False, border=0, float_format=lambda x:f'{x:.0f}')}
<h2>3. 주택공급과 자치구별 연령 순이동</h2><p><b>{migration_summary}</b> 2024년 공급 뒤 완전한 사후연도인 2025년 순이동을 주민 1,000명당 값으로 분석했습니다. 2023년 순이동률, 주거비, 출생률, 해당 학교급 학급당 학생 수를 통제했습니다. 연령구간은 인구 결과의 연령군과 완전히 일치하지 않으므로 별도 결과로 해석합니다.</p>{migration_result.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}
<h2>4. 주택공급과 아동 인구 결과</h2><p><b>{housing_summary}</b> adjusted_beta는 주민 1,000명당 입주 1세대와 로그증감률 가속의 조건부 관계입니다. baseline, 출생·학교 사전압력 보정, 2024 학교개교까지 조건화한 규격을 나란히 제시합니다.</p>{result.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}<img src='housing_supply_vs_child_growth.png'>
<h2>5. 학교 신설 결과</h2><p><b>{school_summary}</b> 개교일은 실제 학교 원본을 사용했고, 신설 이전인 2023년 학급당 학생 수와 직전 인구추세·출생률을 통제했습니다. 학교가 예상 인구 증가에 대응해 신설되는 역인과 가능성은 남습니다.</p>{school_result.to_html(index=False, border=0, float_format=lambda x:f'{x:.5f}')}
<h2>6. 해석 범위</h2><table><tr><th>질문</th><th>현재 판정</th></tr><tr><td>신규 주택공급이 아동 인구를 증가시켰는가?</td><td>출생·주거비·학교압력을 보정한 2024년 자치구 단면 연관성만 평가; 인과 확정 불가</td></tr><tr><td>자연증가인가 다른 지역 유입인가?</td><td>자치구·연령별 순이동을 별도 결과로 직접 평가 가능</td></tr><tr><td>학교 부족과 신설의 선후관계는?</td><td>신설 전 학급당 학생 수와 개교일의 선후 비교 가능; 공식 정원·과밀 판정은 아님</td></tr></table>
<h2>7. 남은 핵심 자료</h2><ol><li>자치구별 2021~2023 실제 준공·입주 실적을 추가해 여러 처치연도의 이벤트 스터디를 구성합니다.</li><li>예정일·사용승인일·착공일을 단지별로 연결해 지연 충격을 검토합니다.</li></ol></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")
    (out / "methodology.json").write_text(json.dumps({"unit": "seoul_district", "treatment": "2024 completed-or-scheduled move-in units per 1000 residents",
        "primary_outcome": "2024_to_2025 log child growth minus 2023_to_2024 log child growth",
        "mechanism_outcome": "2025 district net migration per 1000 residents by 5-year age band",
        "mechanism_placebo": "association between 2024 supply and 2023 pre-supply net migration",
        "controls": ["prior growth or 2023 migration", "apartment price log change", "2023 birth rate", "2023_to_2024 birth change", "2023 school class size by level"],
        "inference": "5000 permutation exploratory", "causal_identification": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(school_result.to_string(index=False))
    print(f"2024 사업 {len(projects):,}개, 공급 {projects.supply_total.sum():,.0f}세대")
    print(f"결과: {out.resolve()}")


if __name__ == "__main__":
    main()
