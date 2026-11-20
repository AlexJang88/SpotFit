"""어린이보호구역 실제 스냅샷과 다음 연도 아동 인구 증가율의 관계를 분석한다.

분석 단위는 ``기준연도 × 서울 행정동``이다. 2023·2024·2025년 12월 자료를 연간
스냅샷으로 사용하고, 2026년 6월 자료는 반기 자료이므로 학습에서 제외해 현재 현황으로만
저장한다. 목표값은 같은 연도 인구가 아니라 다음 연도 인구 증가율이다. 이 시차는 인구가
많아서 보호구역이 많아진 동시성 문제를 조금 줄이지만 인과관계를 증명하지는 않는다.
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/아동"
DEFAULT_POPULATION = (
    ROOT / "data/interim/child_population/"
    "facility_child_counts_20261023_093757_426625/analysis_panel.csv"
)
AGE_GROUPS = ["infant", "elementary", "middle", "high"]
# 현재 아동 수가 지나치게 작은 행정동은 한두 명의 이동만으로도 증가율이 크게 변한다.
# 이러한 불안정한 비율이 상관계수와 회귀 손실을 지배하지 않도록 학습 목표의 최소 분모를 둔다.
# 행 자체와 원래 인구·시설 값은 삭제하지 않고, 해당 연령군의 목표값만 결측 처리한다.
MIN_BASE_CHILDREN = 30
ZONE_COLUMNS_KO = ["계", "초등학교", "유치원", "어린이집", "특수학교", "학원", "외국인학교", "대안학교"]
ZONE_RENAME = {
    "계": "safety_zone_total",
    "초등학교": "safety_zone_elementary",
    "유치원": "safety_zone_kindergarten",
    "어린이집": "safety_zone_daycare",
    "특수학교": "safety_zone_special_school",
    "학원": "safety_zone_academy",
    "외국인학교": "safety_zone_foreign_school",
    "대안학교": "safety_zone_alternative_school",
}


def normalize_dong(value: object) -> str:
    """파일 간 `제1동/1동`, `1.2동/1·2동` 표기 차이를 결합 전에 통일한다."""
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"제(?=\d)", "", text)
    return text.replace(".", "·").replace("ㆍ", "·")


def find_snapshot_files(raw_dir: Path) -> list[tuple[Path, pd.Timestamp]]:
    """제목 행의 기준일을 읽어 파일명 표기 차이에 의존하지 않고 스냅샷을 찾는다."""
    found: list[tuple[Path, pd.Timestamp]] = []
    for path in sorted(raw_dir.glob("*보호구역*.xlsx")):
        excel = pd.ExcelFile(path)
        if "1. 지정 통계" not in excel.sheet_names:
            # 2023년 6월 파일은 세부목록만 있으므로 별도 집계 함수에서 처리한다.
            preview = pd.read_excel(path, sheet_name=0, header=None, nrows=3)
        else:
            preview = pd.read_excel(path, sheet_name="1. 지정 통계", header=None, nrows=3)
        joined = " ".join(preview.fillna("").astype(str).to_numpy().ravel())
        match = re.search(r"(20\d{2})[-_.년\s]+(\d{1,2})(?:[-_.월\s]+(\d{1,2}))?", joined)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3) or 30)
            found.append((path, pd.Timestamp(year, month, day)))
    # 같은 2026 파일이 두 폴더에 있을 수 있으므로 기준일과 파일크기가 같은 중복을 제거한다.
    unique: dict[tuple[pd.Timestamp, int], Path] = {}
    for path, date in found:
        unique.setdefault((date, path.stat().st_size), path)
    return [(path, date) for (date, _), path in sorted(unique.items())]


def read_snapshot(path: Path, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """연도별 서식 차이를 흡수하여 행정동별 보호구역 유형 수를 읽는다."""
    excel = pd.ExcelFile(path)
    if "1. 지정 통계" in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name="1. 지정 통계", header=2)
        frame = frame.rename(columns={frame.columns[1]: "gu", frame.columns[2]: "dong"})
        selected = frame[["gu", "dong", *[c for c in ZONE_COLUMNS_KO if c in frame.columns]]].copy()
    else:
        # 2023년 6월 자료는 시설별 세부행만 존재한다. 시설유형별 건수를 직접 집계한다.
        raw = pd.read_excel(path, sheet_name=0, header=None)
        detail = raw.iloc[4:, [2, 3, 6]].copy()
        detail.columns = ["gu", "dong", "facility_type"]
        detail = detail[detail["gu"].astype(str).str.endswith("구") & detail["dong"].notna()]
        grouped = detail.groupby(["gu", "dong", "facility_type"]).size().unstack(fill_value=0).reset_index()
        grouped["계"] = grouped.drop(columns=["gu", "dong"]).sum(axis=1)
        selected = grouped

    selected = selected[selected["gu"].astype(str).str.endswith("구")].copy()
    selected["gu"] = selected["gu"].astype(str).str.strip()
    selected["dong"] = selected["dong"].map(normalize_dong)
    for column in ZONE_COLUMNS_KO:
        if column not in selected:
            selected[column] = 0
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0)
    selected = selected.rename(columns=ZONE_RENAME)
    selected["snapshot_date"] = snapshot_date
    selected["year"] = snapshot_date.year
    selected["month"] = snapshot_date.month
    selected["region_key"] = selected["gu"] + "|" + selected["dong"]
    return selected[["snapshot_date", "year", "month", "region_key", "gu", "dong", *ZONE_RENAME.values()]]


def prepare_population(path: Path) -> pd.DataFrame:
    """연도별 실제 아동 수에서 다음 연도 증가율을 직접 계산한다."""
    population = pd.read_csv(path, dtype={"dong_code": str})
    population["gu"] = population["gu"].astype(str).str.strip()
    population["dong"] = population["dong"].map(normalize_dong)
    population["region_key"] = population["gu"] + "|" + population["dong"]
    population = population.sort_values(["region_key", "year"])
    for group in AGE_GROUPS:
        current = population[f"children_{group}"]
        future = population.groupby("region_key")[f"children_{group}"].shift(-1)
        population[f"future_children_{group}"] = future
        population[f"target_next_year_growth_{group}"] = (future - current).div(current.where(current.gt(0)))
        # 일반 증감률은 시작 인구가 0~몇 명인 행정동에서 수십 배가 되어 회귀 손실을
        # 지배한다. +1 평활 후 로그 비율로 증가·감소를 대칭적으로 표현하고, 현재 인구가
        # 30명 미만이면 해당 연령군의 목표만 제외한다. 양수는 증가, 0은 유지, 음수는 감소다.
        log_growth = np.log1p(future) - np.log1p(current)
        population[f"target_next_year_log_growth_{group}"] = log_growth.where(
            current.ge(MIN_BASE_CHILDREN)
        )
    return population


def partial_spearman(data: pd.DataFrame, feature: str, target: str) -> tuple[float, int]:
    """초기 인구 규모·전체 인구·자치구·연도를 통제한 순위 잔차 상관을 계산한다."""
    columns = [feature, target, "total_population", "current_children", "gu", "year"]
    work = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 30 or work[feature].nunique() < 2 or work[target].nunique() < 2:
        return np.nan, len(work)
    work["feature_rank"] = work[feature].rank(method="average")
    work["target_rank"] = work[target].rank(method="average")
    work["log_total_population"] = np.log1p(work["total_population"])
    work["log_current_children"] = np.log1p(work["current_children"])
    design = ColumnTransformer(
        [("district", OneHotEncoder(handle_unknown="ignore", drop="first"), ["gu"])],
        remainder="passthrough",
    ).fit_transform(work[["gu", "year", "log_total_population", "log_current_children"]])
    model = LinearRegression()
    feature_residual = work["feature_rank"] - model.fit(design, work["feature_rank"]).predict(design)
    target_residual = work["target_rank"] - model.fit(design, work["target_rank"]).predict(design)
    return float(np.corrcoef(feature_residual, target_residual)[0, 1]), len(work)


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    """선형·순위·통제 후 순위 관계를 모두 계산해 방법 의존성을 확인한다."""
    rows: list[dict] = []
    features = list(ZONE_RENAME.values())
    for group in AGE_GROUPS:
        target = f"target_next_year_log_growth_{group}"
        group_data = panel.copy()
        group_data["current_children"] = group_data[f"children_{group}"]
        for feature in features:
            pair = group_data[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair[feature].nunique() < 2:
                continue
            pearson, pearson_p = pearsonr(pair[feature], pair[target])
            spearman, spearman_p = spearmanr(pair[feature], pair[target])
            partial, partial_n = partial_spearman(group_data, feature, target)
            rows.append({
                "age_group": group, "feature": feature, "n": len(pair),
                "pearson_r": pearson, "pearson_p": pearson_p,
                "spearman_rho": spearman, "spearman_p": spearman_p,
                "partial_spearman_rho": partial, "partial_n": partial_n,
            })
    result = pd.DataFrame(rows)
    result["abs_partial_spearman"] = result["partial_spearman_rho"].abs()
    return result.sort_values(["age_group", "abs_partial_spearman"], ascending=[True, False])


def configure_plot() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def create_visualizations(snapshots: pd.DataFrame, correlations: pd.DataFrame, out: Path) -> None:
    configure_plot()
    annual = snapshots.groupby("snapshot_date")[list(ZONE_RENAME.values())].sum().reset_index()
    plt.figure(figsize=(12, 6))
    for feature in ["safety_zone_total", "safety_zone_elementary", "safety_zone_kindergarten", "safety_zone_daycare"]:
        plt.plot(annual["snapshot_date"], annual[feature], marker="o", label=feature)
    plt.title("서울시 어린이보호구역 실제 스냅샷 추이")
    plt.xlabel("기준일"); plt.ylabel("보호구역 수"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "01_snapshot_trend.png", dpi=160); plt.close()

    heat = correlations.pivot(index="feature", columns="age_group", values="partial_spearman_rho")
    plt.figure(figsize=(9, 7))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="vlag", center=0, vmin=-0.3, vmax=0.3)
    plt.title("통제 후 보호구역 특징과 다음 연도 아동 증가율의 순위상관")
    plt.tight_layout(); plt.savefig(out / "02_partial_correlation_heatmap.png", dpi=160); plt.close()


def write_report(snapshots: pd.DataFrame, panel: pd.DataFrame, correlations: pd.DataFrame, out: Path) -> None:
    top = correlations.groupby("age_group", group_keys=False).head(3)
    totals = snapshots.groupby("snapshot_date")["safety_zone_total"].sum().reset_index()
    report = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>어린이보호구역 분석</title>
<style>body{{font-family:Malgun Gothic,sans-serif;max-width:1150px;margin:36px auto;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#f3f3f3;padding:2px 5px}}.warn{{background:#fff3cd;padding:14px;border-left:5px solid #d39e00}}</style>
<h1>어린이보호구역과 다음 연도 아동 인구 증가율</h1>
<p>생성시각: {datetime.now().isoformat(timespec='seconds')}</p>
<h2>분석 설계</h2><p>2023·2024·2025년 12월 실제 스냅샷 중 미래 인구가 존재하는 2023·2024년을 분석했습니다. 설명변수는 연도 t의 보호구역 유형별 개수이고, 목표는 t+1년의 영유아·초등·중등·고등 연령대 로그 증감률입니다. 현재 연령군 인구가 30명 미만이면 작은 분모로 인한 왜곡을 막기 위해 그 연령군의 목표만 분석에서 제외했습니다.</p>
<div class='warn'><b>해석 제한:</b> 보호구역 수는 시설 수와 인구 규모의 영향을 함께 받습니다. 부분상관은 시작 아동 수, 전체 인구, 자치구, 연도를 통제하지만 주택 입주·출생·전입 같은 모든 혼란요인을 제거하지 못합니다. 상관관계와 예측 중요도는 인과효과가 아닙니다.</div>
<h2>스냅샷 총계</h2>{totals.to_html(index=False, border=0)}
<img src='01_snapshot_trend.png' alt='스냅샷 추이'>
<h2>분석 표본</h2><p>결합 행: {len(panel):,}개, 행정동: {panel['region_key'].nunique():,}개, 설명 연도: {', '.join(map(str, sorted(panel['year'].unique())))}</p>
<h2>통제 후 상관</h2><img src='02_partial_correlation_heatmap.png' alt='부분상관 히트맵'>
<p>아래는 연령대별 절대 부분상관 상위 3개입니다. 방향과 크기가 연도별·모델별로 안정적인지 추가 확인해야 합니다.</p>{top.to_html(index=False, border=0, float_format=lambda x:f'{x:.4f}')}
<h2>방법 차이</h2><ul><li>Pearson: 개수와 증가율의 직선 관계에 민감합니다.</li><li>Spearman: 순위 기반이라 극단값과 비선형 단조 관계에 상대적으로 강합니다.</li><li>부분 Spearman: 두 변수를 순위화한 뒤 시작 인구·자치구·연도 효과를 제거한 잔차 관계입니다.</li></ul>
<p>모델 성능과 선정 근거는 학습 스크립트가 생성하는 별도 <code>report.html</code>에서 확인합니다.</p></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--output", type=Path, default=ROOT / "data_analyze/child_safety_zone/results")
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim/child_safety_zone")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / stamp; out.mkdir(parents=True, exist_ok=False)
    args.interim.mkdir(parents=True, exist_ok=True)

    snapshots = pd.concat([read_snapshot(path, date) for path, date in find_snapshot_files(args.raw_dir)], ignore_index=True)
    # 12월 말 스냅샷만 연간 대표값으로 사용한다. 6월과 12월을 같은 연도로 중복 투입하면
    # 동일 행정동이 두 번 나타나고 반기 변화가 연간 변화처럼 학습되는 문제가 생긴다.
    annual = snapshots[snapshots["month"].eq(12)].copy()
    population = prepare_population(args.population)
    panel = annual.merge(population, on=["year", "region_key", "gu", "dong"], how="left", validate="one_to_one", indicator=True)
    panel[panel["_merge"].ne("both")].to_csv(out / "unmatched_regions.csv", index=False, encoding="utf-8-sig")
    panel = panel[panel["_merge"].eq("both")].drop(columns="_merge")
    target_columns = [f"target_next_year_log_growth_{group}" for group in AGE_GROUPS]
    analysis_panel = panel[panel[target_columns].notna().any(axis=1)].copy()

    snapshots.to_csv(args.interim / "safety_zone_snapshots.csv", index=False, encoding="utf-8-sig")
    analysis_panel.to_csv(args.interim / "safety_zone_analysis_panel.csv", index=False, encoding="utf-8-sig")
    correlations = correlation_table(analysis_panel)
    correlations.to_csv(out / "correlations.csv", index=False, encoding="utf-8-sig")
    create_visualizations(snapshots, correlations, out)
    write_report(snapshots, analysis_panel, correlations, out)
    print(f"스냅샷: {len(snapshots):,}행 / 분석 패널: {len(analysis_panel):,}행")
    print(f"분석 연도: {sorted(analysis_panel['year'].unique())}")
    print(f"전처리: {args.interim.resolve()}")
    print(f"보고서: {(out / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
