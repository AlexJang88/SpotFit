"""Build district/month/year feature tables from collected and existing raw data."""
from __future__ import annotations

import json
import re
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "안전" / "교통사고_모델보강"
OUT.mkdir(parents=True, exist_ok=True)

CODE_TO_GU = {
    11110:"종로구",11140:"중구",11170:"용산구",11200:"성동구",11215:"광진구",
    11230:"동대문구",11260:"중랑구",11290:"성북구",11305:"강북구",11320:"도봉구",
    11350:"노원구",11380:"은평구",11410:"서대문구",11440:"마포구",11470:"양천구",
    11500:"강서구",11530:"구로구",11545:"금천구",11560:"영등포구",11590:"동작구",
    11620:"관악구",11650:"서초구",11680:"강남구",11710:"송파구",11740:"강동구"
}
GUS = list(CODE_TO_GU.values())


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    print(name, len(df))
    return df


def read_csv_auto(path, **kwargs):
    if hasattr(path, "read"):
        raw = path.read()
        for enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise ValueError("unable to detect in-memory CSV encoding")
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def build_living_population():
    monthly_parts = []
    zips = sorted((RAW / "모델보강" / "생활인구").glob("*.zip"))
    for zp in zips:
        year = int(re.search(r"(20\d{2})", zp.name).group(1))
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with z.open(member) as f:
                    d = read_csv_auto(f)
                # The 2018-2019 files contain one more field than their header;
                # pandas consequently places the date in the row index.
                first_col = pd.to_numeric(d.iloc[:, 0], errors="coerce")
                idx_num = pd.to_numeric(pd.Series(d.index, index=d.index), errors="coerce")
                if first_col.max(skipna=True) < 100 and idx_num.max(skipna=True) > 20_000_000:
                    original_columns = list(d.columns)
                    d = d.reset_index().iloc[:, :len(original_columns)]
                    d.columns = original_columns
                aliases = {
                    "기준일ID":"date", "시간대구분":"hour", "자치구코드":"code", "총생활인구수":"value",
                    "stdr_de_id":"date", "tmzon_pd_se":"hour", "adstrd_code_se":"code", "tot_lvpop_co":"value"
                }
                d = d.rename(columns=aliases)
                if not {"date", "hour", "code", "value"}.issubset(d.columns):
                    raise ValueError(f"unexpected living-population columns: {zp} {d.columns.tolist()[:5]}")
                d["date"] = d["date"].astype(str).str.replace(r"\.0$", "", regex=True)
                d["month"] = pd.to_numeric(d["date"].str[4:6], errors="coerce")
                code5 = d["code"].astype(str).str.replace(r"\.0$", "", regex=True).str[:5]
                d["district"] = pd.to_numeric(code5, errors="coerce").map(CODE_TO_GU)
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
                # Average simultaneous population across all date-hour observations.
                g = d.dropna(subset=["district", "month"]).groupby(["month", "district"])["value"].agg(["mean", "max", "std"]).reset_index()
                g.insert(0, "year", year)
                g = g.rename(columns={"mean":"생활인구_시간평균", "max":"생활인구_시간최대", "std":"생활인구_시간표준편차"})
                monthly_parts.append(g)
    monthly = pd.concat(monthly_parts).groupby(["year","month","district"], as_index=False).mean(numeric_only=True)
    annual = monthly.groupby(["year","district"], as_index=False).agg(
        생활인구_연평균=("생활인구_시간평균","mean"),
        생활인구_연중시간최대=("생활인구_시간최대","max"),
        생활인구_월변동성=("생활인구_시간평균","std"))
    return save(monthly, "자치구별_월간_생활인구_2017_2025.csv"), save(annual, "자치구별_연간_생활인구_2017_2025.csv")


def build_commercial_population():
    base = RAW / "부동산" / "서울시 상권분석서비스" / "행정동"
    outputs = []
    specs = [("*상주인구*.csv", "총_상주인구_수", "상주인구"), ("*길단위인구*.csv", "총_유동인구_수", "유동인구")]
    for pattern, col, label in specs:
        paths = list(base.glob(pattern))
        if not paths:
            continue
        d = read_csv_auto(paths[0])
        qcol = "기준_년분기_코드"
        codecol = "행정동_코드"
        d["year"] = pd.to_numeric(d[qcol].astype(str).str[:4], errors="coerce")
        d["quarter"] = pd.to_numeric(d[qcol].astype(str).str[-1], errors="coerce")
        d["district"] = pd.to_numeric(d[codecol].astype(str).str[:5], errors="coerce").map(CODE_TO_GU)
        d[col] = pd.to_numeric(d[col], errors="coerce")
        q = d.dropna(subset=["district"]).groupby(["year","quarter","district"],as_index=False)[col].sum()
        a = q.groupby(["year","district"],as_index=False)[col].mean().rename(columns={col:f"{label}_연평균"})
        outputs.append(a)
        save(q.rename(columns={col:f"{label}_분기값"}), f"자치구별_분기_{label}.csv")
    merged = outputs[0]
    for d in outputs[1:]: merged = merged.merge(d,on=["year","district"],how="outer")
    return save(merged, "자치구별_연간_상주유동인구.csv")


def build_weather():
    p = RAW / "모델보강" / "기상" / "OpenMeteo_ERA5Land_자치구일별_2010_2025.csv.gz"
    if p.exists():
        d = pd.read_csv(p)
    else:
        cached = sorted((RAW / "모델보강" / "기상" / "district_cache").glob("*.csv.gz"))
        d = pd.concat([pd.read_csv(x) for x in cached], ignore_index=True)
    d["time"] = pd.to_datetime(d["time"]); d["year"] = d.time.dt.year; d["month"] = d.time.dt.month
    d["강수일"] = (d.precipitation_sum >= 1).astype(int)
    d["폭우일"] = (d.precipitation_sum >= 80).astype(int)
    d["적설일"] = (d.snowfall_sum > 0).astype(int)
    d["폭염일"] = (d.temperature_2m_max >= 33).astype(int)
    d["결빙일"] = (d.temperature_2m_min <= 0).astype(int)
    agg = dict(강수량합계=("precipitation_sum","sum"),강수일수=("강수일","sum"),폭우일수=("폭우일","sum"),
               적설량합계=("snowfall_sum","sum"),적설일수=("적설일","sum"),폭염일수=("폭염일","sum"),결빙일수=("결빙일","sum"),
               최고기온평균=("temperature_2m_max","mean"),최저기온평균=("temperature_2m_min","mean"))
    monthly=d.groupby(["year","month","district"],as_index=False).agg(**agg)
    annual=d.groupby(["year","district"],as_index=False).agg(**agg)
    return save(monthly,"자치구별_월간_기상_2010_2025.csv"),save(annual,"자치구별_연간_기상_2010_2025.csv")


def build_transit():
    p=RAW/"모델보강"/"대중교통"/"서울데이터허브_자치구월별대중교통_2024_2025.csv"
    d=pd.read_csv(p)
    cols={"all_psgr_cnt":"대중교통_승객수","bus_psgr_cnt":"버스_승객수","sbwy_psgr_cnt":"지하철_승객수",
          "bus_sttn_cnt":"버스정류장수","sbwstt_cnt":"지하철역수","bus_route_cnt":"버스노선수","sbwy_route_cnt":"지하철노선수"}
    keep=[c for c in cols if c in d.columns]
    m=d[["year","month","district"]+keep].rename(columns=cols)
    for c in cols.values():
        if c in m: m[c]=pd.to_numeric(m[c],errors="coerce")
    sums=[c for c in ["대중교통_승객수","버스_승객수","지하철_승객수"] if c in m]
    means=[c for c in cols.values() if c in m and c not in sums]
    annual=m.groupby(["year","district"],as_index=False).agg(**{**{f"{c}_Q4합계":(c,"sum") for c in sums},**{f"{c}_Q4평균":(c,"mean") for c in means}})
    return save(m,"자치구별_월간_대중교통_2024_2025.csv"),save(annual,"자치구별_연간_대중교통_2024_2025.csv")


def build_intersections():
    p=next((RAW/"모델보강"/"도로시설").glob("*.xlsx"))
    d=pd.read_excel(p)
    out=d.groupby("자치구",as_index=False).agg(
        교차로수=("교차로관리번호",lambda x:x.dropna().nunique()),
        횡단보도수=("횡단보도관리번호",lambda x:x.dropna().nunique()),
        신호횡단보도수=("보행등유무",lambda x:x.astype(str).str.contains("유|있",regex=True).sum()))
    return save(out.rename(columns={"자치구":"district"}),"자치구별_교차로횡단보도_2026현황.csv")


def district_from_text(series):
    pat="("+"|".join(map(re.escape,GUS))+")"
    return series.fillna("").astype(str).str.extract(pat,expand=False)


def build_cctv():
    p=next((RAW/"안전"/"교통사고").glob("*CCTV*.csv"))
    d=read_csv_auto(p)
    text=d["소재지도로명주소"].fillna("")+" "+d["소재지지번주소"].fillna("")+" "+d["관리기관명"].fillna("")
    d["district"]=district_from_text(text)
    d["install_year"]=pd.to_numeric(d["설치연월"].astype(str).str[:4],errors="coerce")
    d["camera_count"]=pd.to_numeric(d["카메라대수"],errors="coerce").fillna(1)
    purpose=d["설치목적구분"].fillna("").astype(str)
    d["speed_signal"]=purpose.str.contains("과속|신호").astype(int)*d.camera_count
    d["traffic_all"]=purpose.str.contains("교통|단속|과속|신호").astype(int)*d.camera_count
    rows=[]
    for year in range(2010,2027):
        x=d[(d.install_year<=year)&d.district.notna()]
        g=x.groupby("district",as_index=False).agg(CCTV전체누적=("camera_count","sum"),과속신호카메라누적=("speed_signal","sum"),교통단속CCTV누적=("traffic_all","sum"))
        g.insert(0,"year",year); rows.append(g)
    return save(pd.concat(rows,ignore_index=True),"자치구별_CCTV_설치연도기준누적_2010_2026.csv")


def build_child_accidents():
    p=next((RAW/"안전"/"교통사고").glob("어린이*.xlsx"))
    d=pd.read_excel(p,sheet_name="데이터")
    d["district"]=district_from_text(d["지역명"])
    nums=["사고건수","사망자수","부상자수"]
    for c in nums:d[c]=pd.to_numeric(d[c],errors="coerce")
    out=d.dropna(subset=["district"]).groupby(["연도","district"],as_index=False)[nums].sum().rename(columns={"연도":"year",**{c:"어린이사고_"+c for c in nums}})
    return save(out,"자치구별_어린이교통사고_2010_2025.csv")


def build_drunk_enforcement():
    p=next((RAW/"모델보강"/"음주단속").glob("*.csv")); d=read_csv_auto(p)
    police={"중부":"중구","남대문":"중구","종로":"종로구","혜화":"종로구","용산":"용산구","성동":"성동구","광진":"광진구","동대문":"동대문구","중랑":"중랑구","성북":"성북구","종암":"성북구","강북":"강북구","도봉":"도봉구","노원":"노원구","은평":"은평구","서부":"은평구","서대문":"서대문구","마포":"마포구","양천":"양천구","강서":"강서구","구로":"구로구","금천":"금천구","영등포":"영등포구","동작":"동작구","관악":"관악구","서초":"서초구","방배":"서초구","강남":"강남구","수서":"강남구","송파":"송파구","강동":"강동구"}
    def map_station(s):
        s=str(s)
        for k,v in police.items():
            if k in s:return v
        return np.nan
    d=d[d["지역"].astype(str).str.contains("서울")].copy(); d["district"]=d["경찰서"].map(map_station)
    d["건수"]=pd.to_numeric(d["건수"],errors="coerce")
    out=d.groupby("district",as_index=False)["건수"].sum().rename(columns={"건수":"음주단속적발건수"}); out.insert(0,"year",2025)
    return save(out,"자치구별_음주단속적발_2025.csv")


def build_combined(tables):
    base=pd.MultiIndex.from_product([range(2010,2026),GUS],names=["year","district"]).to_frame(index=False)
    for d in tables:
        if {"year","district"}.issubset(d.columns):
            base=base.merge(d,on=["year","district"],how="left")
    return save(base,"자치구별_연간_모델보강특성_2010_2025.csv")


def main():
    _,living=build_living_population(); commercial=build_commercial_population()
    _,weather=build_weather(); _,transit=build_transit(); intersections=build_intersections()
    cctv=build_cctv(); child=build_child_accidents(); drunk=build_drunk_enforcement()
    combined=build_combined([living,commercial,weather,transit,cctv,child,drunk])
    coverage=[]
    for c in combined.columns[2:]:
        z=combined.loc[combined[c].notna(),"year"]
        coverage.append({"feature":c,"first_year":int(z.min()) if len(z) else None,"last_year":int(z.max()) if len(z) else None,"non_null_rows":int(z.size),"coverage_pct":round(z.size/len(combined)*100,1)})
    save(pd.DataFrame(coverage),"특성별_연도커버리지.csv")
    manifest=[
      ["생활인구","확보","2017-2025","서울 열린데이터광장 OA-15439"],
      ["상주·유동인구","확보","원천자료 수록기간","기존 서울 상권분석서비스 원천"],
      ["기상","대체자료 확보","2010-2025","Open-Meteo ERA5-Land; 관측소 실측 아님"],
      ["버스·지하철 승객","부분 확보","2024-2025년 10-12월","서울 데이터허브 자치구 월별 API; 다른 월은 빈 응답이므로 Q4 특성으로 표기"],
      ["교차로·횡단보도","현황 확보","2026","서울 열린데이터광장 OA-23081; 역사자료 아님"],
      ["CCTV","부분 확보","2010-2026 누적","기존 CCTV 원천; 목적 문자열 기반, 경찰 무인단속장비와 다를 수 있음"],
      ["음주단속 적발","확보","2025","경찰청 공공데이터; 경찰서 관할을 자치구로 매핑"],
      ["어린이 사고","확보","2010-2025","기존 TAAS 원천"],
      ["실제 교통량","미확보","-","서울 OpenAPI 인증키 또는 TOPIS 상세 내려받기 필요"],
      ["도로연장·차로수","미확보","-","서울 통계표/API 인증 또는 KOSIS 내려받기 필요"],
      ["보행자·고령자·이륜차 사고","미확보","-","TAAS 조건별 사고 내려받기 필요"],
      ["고령운전자·면허보유자","미확보","-","경찰청 운전면허 통계의 경찰서/자치구 매핑 자료 필요"],
    ]
    save(pd.DataFrame(manifest,columns=["category","status","period","note"]),"데이터_확보현황.csv")
    (OUT/"README.md").write_text("# 교통사고 모델 보강 데이터\n\n`자치구별_연간_모델보강특성_2010_2025.csv`가 모델 결합용 주 테이블입니다. 정적 2026 교차로 현황은 미래정보 누출 방지를 위해 주 테이블에 합치지 않았습니다. 결측치는 0이 아니라 미확보이며, 모델 학습 시 기간별 가용성을 고려해야 합니다.\n",encoding="utf-8")


if __name__=="__main__":main()
