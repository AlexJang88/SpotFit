"""아동 원자료의 행정동 매핑·인구 증가율·탐색적 상관 비교.

실행: python data_analyze/child_population/analyze_child_population.py
현재 시설을 과거 시설로 소급하지 않는다. 결과는 예측/인과/가중치가 아닌 탐색 자료다.
모든 출력은 날짜별 새 폴더에 생성하고 원본은 수정하지 않는다.
"""
from pathlib import Path
from datetime import datetime
import json
import re
import html
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/아동"
OUT = ROOT / "data/processed/child_population" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
AUDIT = []


def save(df, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read(path):
    # 인코딩 오류를 대체 문자로 숨기지 않고 엄격하게 복호화한다.
    for encoding in ("utf-8-sig", "cp949", "utf-16"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str), encoding
        except UnicodeError:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")


def number(series):
    # 코드에는 적용하지 않는다. 수량의 쉼표만 제거하며 미상은 NaN으로 남긴다.
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def norm(value):
    # 제1동/1동, 가운데점/마침표 등의 표기 차이만 정규화한다. 지리적 통합은 하지 않는다.
    return re.sub(r"[\s.·ㆍ]", "", str(value)).replace("제", "")


features = json.loads((ROOT / "frontend/public/seoul-dong.geojson").read_text(encoding="utf-8-sig"))["features"]
districts = pd.DataFrame([{
    "dong_code": f["properties"]["adm_cd2"],
    "gu": f["properties"]["sggnm"],
    "dong": f["properties"]["adm_nm"].split()[-1],
} for f in features])
assert districts.dong_code.is_unique
geoms = [shape(f["geometry"]) for f in features]
tree = STRtree(geoms)
name_lookup = {(norm(r.gu), norm(r.dong)): r.dong_code for r in districts.itertuples()}
code_set = set(districts.dong_code)
save(districts, "dong_reference_20261001.csv")

# 법정동 하나가 행정동 여러 개에 대응하면 자동 배분하지 않는다.
crosswalk = pd.read_excel(ROOT / "data/raw/행정동_법정동_매핑.xlsx", dtype=str)
crosswalk = crosswalk[crosswalk["행정동코드"].isin(code_set) & crosswalk["말소일자"].isna()]
legal_lookup = {}
for (gu, legal), group in crosswalk.groupby(["시군구명", "동리명"]):
    legal_lookup[(norm(gu), norm(legal))] = sorted(set(group["행정동코드"]))


def address_map(text):
    gu = re.search(r"([가-힣]+구)\s", text + " ")
    legal = re.search(r"\s([가-힣0-9.·]+(?:동|가))(?=[\s,)\d-]|$)", text)
    if not gu or not legal:
        return None, "address_insufficient"
    matches = legal_lookup.get((norm(gu[1]), norm(legal[1])), [])
    if len(matches) == 1:
        return matches[0], "unique_legal_crosswalk"
    return None, "ambiguous_legal_dong" if matches else "address_unmatched"


def map_row(row):
    # 전국 데이터에서 서울 밖 주소를 서울의 동명 지역으로 잘못 연결하지 않는다.
    province = str(row.get("시도명", ""))
    if province and province != "nan" and "서울" not in province:
        return None, "outside_seoul"
    # 좌표가 있으면 point-in-polygon. 경계선에 걸쳐 여러 동이 잡히면 모호함으로 남긴다.
    lat_col = next((c for c in ("WGS84위도", "위도") if c in row.index), None)
    lon_col = next((c for c in ("WGS84경도", "경도") if c in row.index), None)
    address = " ".join(str(row.get(c, "")) for c in ("소재지지번주소","지번주소","소재지주소","주소","소재지도로명주소","도로명주소","신주소"))
    if "서울" not in address and re.search(r"(부산|인천|대구|대전|광주|울산|세종|경기도|강원|충청|전라|경상|제주)", address):
        return None, "outside_seoul"
    gu = re.search(r"([가-힣]+구)\s", address + " ")
    try:
        lat, lon = float(row[lat_col]), float(row[lon_col])
        if np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            matches = tree.query(Point(lon, lat), predicate="intersects")
            if len(matches) == 1:
                r = districts.iloc[matches[0]]
                if gu and norm(gu[1]) != norm(r.gu):
                    return None, "coordinate_address_gu_conflict"
                return r.dong_code, "coordinate"
            if len(matches) > 1:
                return None, "coordinate_boundary_ambiguous"
    except (TypeError, ValueError, KeyError):
        pass
    # 병원 X/Y는 투영좌표계 메타데이터가 없어 추정 변환하지 않는다.
    return address_map(address)


population = []
aggregates = districts.copy()
feature_notes = []
for path in sorted(RAW.rglob("*")):
    if path.suffix not in (".csv", ".xlsx"):
        continue
    key = path.stem
    print("처리:", key, flush=True)
    record = {"file": str(path.relative_to(RAW)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix == ".xlsx":
        book = pd.ExcelFile(path)
        # 나머지 시트도 원형 보존용 UTF-8 CSV로 변환하되 중복 시설 집계에는 사용하지 않는다.
        for sheet in book.sheet_names:
            save(pd.read_excel(path, sheet, header=None), f"normalized/{key}/{sheet.replace('/', '_')}.csv")
        d = pd.read_excel(path, "2. 지정 세부현황", header=None).iloc[5:].copy()
        d = d[d[1].notna()]
        d.columns = ["unused","sequence","gu_raw","dong_raw","address","facility","type","designation_year","note"]
        d["gu_raw"] = d["gu_raw"].ffill()
        d["dong_code"] = [name_lookup.get((norm(g), norm(n))) for g,n in zip(d.gu_raw, d.dong_raw)]
        d["mapping_status"] = np.where(d.dong_code.notna(), "explicit_dong_name", "unmatched_name")
        encoding = "xlsx"
        eligible = pd.Series(True, index=d.index)
        metric = "school_zone_count"
    else:
        d, encoding = read(path)
        d["source_row"] = np.arange(2, len(d)+2)
        record["input_rows"] = len(d)
        if "행정구역" in d:
            d["dong_code"] = d["행정구역"].str.extract(r"\((\d{10})\)", expand=False)
            d["mapping_status"] = np.where(d.dong_code.isin(code_set), "code_exact", "not_current_seoul_dong")
            if path.parent.name == "핵심":
                year = int(path.name[:4])
                p = d[d.dong_code.isin(code_set)][["dong_code"]].copy()
                p["year"] = year
                p["total_population"] = number(d.loc[p.index, f"{year}년_계_총인구수"])
                for end in (5,12,17):
                    values = d.loc[p.index, [f"{year}년_계_{age}세" for age in range(end+1)]].apply(number)
                    p[f"children_0_{end}"] = values.sum(axis=1, min_count=end+1)
                assert p.dong_code.is_unique
                population.append(p)
            save(d, f"mapped/{key}.csv")
            record.update(encoding=encoding, mapped=int(d.dong_code.isin(code_set).sum()), role="population_auxiliary_or_core")
            AUDIT.append(record)
            continue
        if "행정기관코드" in d:
            d["dong_code"] = d["행정기관코드"]
            d["mapping_status"] = np.where(d.dong_code.isin(code_set), "code_exact", "not_current_seoul_dong")
            save(d, f"mapped/{key}.csv")
            record.update(encoding=encoding, mapped=int(d.dong_code.isin(code_set).sum()), role="monthly_population_auxiliary")
            AUDIT.append(record)
            continue
        if "학구ID" in d:
            # 학구 코드/구 코드만으로 동을 결정할 수 없다. 공간객체 ID는 좌표가 아니다.
            d["dong_code"] = None
            d["mapping_status"] = "missing_school_zone_geometry"
            save(d, f"mapped/{key}.csv")
            record.update(encoding=encoding, mapped=0, role="blocked_no_geometry")
            AUDIT.append(record)
            continue
        mapped = d.apply(map_row, axis=1)
        d["dong_code"] = [v[0] for v in mapped]
        d["mapping_status"] = [v[1] for v in mapped]
        eligible = pd.Series(True, index=d.index)
        metric = {"CCTV정보_서울특별시":"cctv_locations", "안전비상벨위치정보_서울특별시":"emergency_bells",
                  "모범음식점정보_서울특별시":"designated_restaurants", "서울시 관광 명소":"tourism_records",
                  "서울특별시_병원":"hospitals", "전국지역아동센터표준데이터":"child_centers",
                  "전국초중등학교위치표준데이터":"schools", "전국도시공원정보표준데이터":"parks"}[key]
        if "영업상태명" in d:
            eligible &= d["영업상태명"].fillna("").str.contains("영업|정상") & ~d["영업상태명"].fillna("").str.contains("폐업|취소|휴업")
        if "지정취소일자" in d:
            eligible &= d["지정취소일자"].isna()
        if "운영상태" in d:
            eligible &= d["운영상태"].fillna("").str.contains("운영")
        if "언어" in d:
            eligible &= d["언어"].str.lower().isin(["ko","kr","kor"])
        identity = next((c for c in ("관리번호","학교ID","고유번호") if c in d), None)
        # 중복 식별자는 집계에서 제외하지만 원행과 제외 사유는 파일에 남긴다.
        dup = d.duplicated(subset=[identity], keep="first") & d[identity].notna() if identity else d.drop(columns=["source_row"]).duplicated()
        d["duplicate"] = dup
        eligible &= ~dup
    d["eligible_for_count"] = eligible
    d["included_in_count"] = eligible & d.dong_code.notna()
    save(d, f"mapped/{key}.csv")
    statuses = d.mapping_status.value_counts().to_dict()
    record.update(encoding=encoding, input_rows=len(d), mapped=int(d.dong_code.notna().sum()),
                  counted=int(d.included_in_count.sum()), statuses=json.dumps(statuses, ensure_ascii=False), role="current_snapshot")
    datecols = [c for c in d if c in ("데이터기준일자","데이터갱신시점","최종수정시점")]
    record["date_samples"] = "; ".join(f"{c}: {d[c].dropna().min()} ~ {d[c].dropna().max()}" for c in datecols if d[c].notna().any())
    AUDIT.append(record)
    counts = d[d.included_in_count].groupby("dong_code").size()
    # 0은 '이 파일에서 매핑한 적격 기록이 없음'. 실제 시설 부재라고 보장하지 않는다.
    aggregates[metric] = aggregates.dong_code.map(counts).fillna(0)
    feature_notes.append({"feature":metric, "source":record["file"], "meaning":"매핑 성공한 현재 적격 기록 수", "limitation":"미매핑/누락으로 과소집계 가능; 과거 예측용 사용 금지"})
    for rawcol, suffix in [("정원수","capacity"), ("카메라대수","cameras"), ("공원면적","listed_area")]:
        if rawcol in d:
            valid = d[d.included_in_count].copy()
            valid["amount"] = number(valid[rawcol]).where(lambda x:x>=0)
            totals = valid.groupby("dong_code").amount.sum(min_count=1)
            col = metric+"_"+suffix
            aggregates[col] = aggregates.dong_code.map(totals)
            feature_notes.append({"feature":col,"source":record["file"],"meaning":rawcol+" 합계", "limitation":"공원면적은 대표좌표 소재지 귀속 면적, 동내 실제 면적 아님; 누락은 NaN"})

save(pd.DataFrame(AUDIT), "source_audit.csv")
save(pd.DataFrame(feature_notes), "feature_dictionary.csv")
save(aggregates, "current_facilities_by_dong.csv")
pop = pd.concat(population, ignore_index=True).merge(districts, on="dong_code", validate="many_to_one")
save(pop, "population_panel_2021_2025.csv")
# 모든 5개 연도에 동일 코드로 관측된 지역만 비교한다. 같은 코드의 경계 조정까지 해결되지는 않는다.
stable_codes = pop.groupby("dong_code").year.nunique().loc[lambda s:s==5].index
base = pop[(pop.year==2024) & pop.dong_code.isin(stable_codes)].copy()
end = pop[pop.year==2025]
base = base.merge(end[["dong_code"]+[f"children_0_{a}" for a in (5,12,17)]], on="dong_code", suffixes=("_base","_end"), validate="one_to_one")
for age in (5,12,17):
    denominator = base[f"children_0_{age}_base"].where(base[f"children_0_{age}_base"]>0)
    base[f"growth_0_{age}"] = (base[f"children_0_{age}_end"]-denominator)/denominator*100
data = base.merge(aggregates.drop(columns=["gu","dong"]), on="dong_code", validate="one_to_one")
predictors = [r["feature"] for r in feature_notes]
for col in predictors.copy():
    data[col+"_per1000"] = data[col]/data.total_population.where(data.total_population>0)*1000
    predictors.append(col+"_per1000")
save(data, "analysis_current_facilities_vs_2024_2025_growth.csv")
# 연간/3년 증감률 전체를 별도 보존한다. 현재 시설을 모든 연도에 복제하여 n을 부풀리지 않는다.
transitions=[]
for horizon in (1,3):
    for year in range(2021,2026-horizon):
        a=pop[(pop.year==year)&pop.dong_code.isin(stable_codes)]
        b=pop[pop.year==year+horizon]
        merged=a.merge(b,on="dong_code",suffixes=("_start","_end"),validate="one_to_one")
        for age in (5,12,17):
            s=merged[f"children_0_{age}_start"]
            transitions.append(pd.DataFrame({"dong_code":merged.dong_code,"start":year,"end":year+horizon,"age":f"0-{age}",
                "growth_pct":(merged[f"children_0_{age}_end"]-s)/s.where(s>0)*100}))
save(pd.concat(transitions), "population_growth_all_periods.csv")


def residual(y, controls):
    return y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]


results=[]
for age in (5,12,17):
    target=f"growth_0_{age}"
    for feature in predictors:
        subset=data[[feature,target,"total_population",f"children_0_{age}_base","gu"]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(subset)<30 or subset[feature].nunique()<2:
            continue
        x,y=subset[feature].to_numpy(),subset[target].to_numpy()
        # 지역 규모와 초기 연령 구성을 보정한 순위 잔차의 상관: 인과효과가 아닌 민감도 비교.
        controls=np.column_stack([np.ones(len(subset)),stats.rankdata(subset.total_population),
            stats.rankdata(subset[f"children_0_{age}_base"]/subset.total_population),
            pd.get_dummies(subset.gu,drop_first=True).to_numpy(dtype=float)])
        partial=np.corrcoef(residual(stats.rankdata(x),controls),residual(stats.rankdata(y),controls))[0,1]
        for method, res in [("spearman",stats.spearmanr(x,y)),("pearson",stats.pearsonr(x,y)),("kendall",stats.kendalltau(x,y))]:
            results.append({"age":f"0-{age}","feature":feature,"method":method,"coefficient":res.statistic,
                            "p_nominal":res.pvalue,"n":len(subset),"partial_rank":partial,
                            "quality_flag":"low_mapping_coverage_do_not_select" if feature.startswith(("hospitals","designated_restaurants","tourism_records")) else "snapshot_exploratory"})
result=pd.DataFrame(results)
# 다중 비교 보정은 연령×방법별 적용. 공간 의존 때문에 p/q는 탐색 참고값에 한정한다.
result["q_bh"]=np.nan
for _,group in result.groupby(["age","method"]):
    order=group.sort_values("p_nominal")
    q=np.minimum.accumulate((order.p_nominal.to_numpy()*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
    result.loc[order.index,"q_bh"]=np.minimum(q,1)
result["abs_coefficient"]=result.coefficient.abs()
result=result.sort_values(["age","method","abs_coefficient"],ascending=[True,True,False])
save(result,"method_comparison.csv")
primary=result[(result.age=="0-5")&(result.method=="spearman")].head(12)
plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(figsize=(12,7))
ax.barh(primary.feature[::-1],primary.coefficient[::-1])
ax.axvline(0,color="gray");ax.set_xlim(-1,1)
ax.set_title("현재 시설 vs 2024→2025 영유아 증감률: 탐색 Spearman")
for i,v in enumerate(primary.coefficient[::-1]):ax.text(v,i,f" {v:.3f}",va="center")
fig.tight_layout();fig.savefig(OUT/"correlations.png",dpi=160);plt.close(fig)
comparison=result[result.feature.isin(primary.feature)&(result.age=="0-5")].pivot(index="feature",columns="method",values="coefficient")
comparison["partial_rank"]=primary.set_index("feature").partial_rank
report = """<!doctype html><meta charset="utf-8"><title>아동 인구·시설 탐색 분석</title>
<style>body{font-family:Malgun Gothic,sans-serif;max-width:1300px;margin:35px auto;line-height:1.8}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}.warning{background:#fff0d2;padding:20px}</style>
<h1>행정동별 아동 인구 변화와 시설 현황</h1>
<div class="warning">시설은 현재(주로 2026년), 인구 변화는 2024→2025년입니다.
현재 시설과 과거 변화의 탐색적 비교일 뿐 미래 예측·인과관계·점수 가중치가 아닙니다.
어린이집·학원 전체 자료는 이 폴더에 없으며 보호구역 지정 시설이나 지역아동센터로 대체하지 않았습니다.</div>
<h2>전처리</h2><p>CP949/UTF-8을 엄격하게 읽고 UTF-8 BOM으로 저장. 코드 문자열 보존.
시설 좌표를 2026-10-01 행정동 경계와 공간 결합. 주소의 구와 좌표 구가 다르면 제외.
좌표가 없으면 법정동 대응이 유일할 때만 매핑. 법정동과 행정동 이름이 비슷하다고 임의 배정하지 않음.
병원 투영좌표 CRS 미확정, 통학구역 도형 부재, 영문 관광 주소 등은 미매핑 사유로 보존.
영업·운영 상태, 관광 한국어 기록, 중복 식별자를 필터링하며 원행은 보존.
시설 수 0은 파일 내 적격 매핑 기록 없음이며 실제 부재를 보장하지 않음.</p>
<h2>목표변수</h2><p>1세 단위 인구를 합산해 0~5세(주 분석), 0~12세, 0~17세를 계산.
증감률=(종료인구-시작인구)/시작인구×100. 분모 0은 결측. 출생·연령 이동·전출입 모두 포함되므로 순유입률 아님.
핵심 0~19세 파일만 인구 산출에 사용하며 성별·아동전체·주민등록 파일은 이중 합산하지 않음.
2021~2025년 동일 코드 관측 지역으로 제한. 과거 경계 변동은 완전히 보정하지 못했으므로 결과에 잔여 오차 가능.</p>
<h2>분석 방법을 비교하는 이유</h2><ul>
<li>Spearman(주 분석): 시설 분포의 치우침과 극단값 영향을 줄여 순위의 단조 관계 확인.</li>
<li>Pearson: 실제 값의 선형 관계. 극단값에 더 민감하며 Spearman과 차이가 크면 분포 점검 필요.</li>
<li>Kendall tau-b: 두 지역 순서의 일치 정도, 동점 반영. 계수 크기를 Spearman과 직접 동등 비교하지 않음.</li>
<li>부분 순위 상관: 전체 인구 규모·기초 아동 비율·자치구를 보정한 잔차 비교. 누락 변수·역인과는 해결하지 못함.</li>
</ul><p>전국 파일의 서울 밖 기록은 미매핑 내역에 포함되어 있어 원본 대비 매핑률만으로 품질을 판단하면 안 됩니다.
p값과 BH q값은 지역 간 독립성 가정이 있어 공간적 의존이 존재하는 이 자료에서는 참고용입니다.
검색된 높은 상관만 골라 점수 가중치로 사용하지 않습니다. 1천명당 지표의 분모는 모두 2024 전체 인구로 통일했습니다.</p>"""
report += f"<h2>분석 대상: {len(data)}개 행정동</h2><h2>대표 결과 비교 (0~5세)</h2>"+comparison.round(4).to_html()
report += '<img src="correlations.png"><h2>전체 연령·방법 결과</h2>'+result.round(5).to_html(index=False)
report += "<h2>파일별 매핑·시점 감사</h2>"+pd.DataFrame(AUDIT).to_html(index=False,escape=True)
report += "<h2>후속 작업</h2><p>미매핑 주소의 지오코딩 권한/CRS 확인, 과거 경계 대응, 2021~2025 시설 스냅샷 확보가 우선입니다. 현재 자료로 3년 예측모델의 성능이나 최적 가중치는 산출하지 않았습니다.</p>"
(OUT/"report.html").write_text(report,encoding="utf-8")
(OUT/"README.md").write_text("# 결과 안내\n\nreport.html에서 전처리·시점 제약·분석 비교를 확인하세요.\n원행 매핑: mapped/\n인구 원천은 핵심 0~19세 파일이며 보조자료는 별도 보존했습니다.\n",encoding="utf-8")
print("OUTPUT", OUT)
print(primary[["feature","coefficient","n","partial_rank"]].to_string(index=False))
