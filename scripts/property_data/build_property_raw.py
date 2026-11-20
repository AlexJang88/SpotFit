"""Turn the months downloaded from the APIs into the project's raw files and merge them in.

  data/raw/부동산/refresh/trade/YYYYMM.csv  -> data/raw/부동산/아파트_매매_2010_2026.csv   (same columns as before)
  data/raw/부동산/refresh/rent/YYYYMM.csv   -> data/raw/부동산/아파트_전월세_2010_2026.csv
  data/raw/부동산/refresh/mortgage_rate.json -> data/raw/부동산/예금은행_대출금리_신규취급액_기준__*.csv

Same rules as the existing files (checked against them):
  - 행정동은 법정동+지번을 도로명주소 DB(rnaddrkor_seoul.txt)에 대어 찾음 (ml/etl/ConvertSale.py와 같은 방식)
  - 평 = ㎡ ÷ 3.3058 (소수 2자리), 평당 금액 = 금액 ÷ 평 (소수 1자리), 건물 나이 = 계약연도 − 건축년도
  - 한 달을 새로 받으면 그 달의 기존 행은 모두 새 행으로 바꿈 (늦게 신고된 건·취소 반영). 다른 달의 행은 그대로.
  - 취소된 매매(해제)는 넣지 않음. 행정동을 못 찾은 행은 넣지 않고 개수를 보고함.
Before a file is replaced the old one is kept in data/raw/부동산/_backup/ (the latest copy only).
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.property_data.common import (  # noqa: F401  (raw_end_months is re-exported)
    BACKUP, RAW, REFRESH, RENT_RAW, SALE_RAW, SEOUL_SGG, month_index, raw_end_months)

JUSO_FILE = RAW / "rnaddrkor_seoul.txt"
AREA_FILE = RAW / "서울시 상권분석서비스(영역-행정동).csv"
PYEONG = 3.3058
MIN_KEEP_RATIO = 0.5          # 새로 받은 달이 기존 달의 절반도 안 되면 API가 덜 준 것으로 보고 중단


# ---------------------------------------------------------------- 행정동 연결
class DongMapper:
    """법정동 + 지번 -> 행정동 코드(상권분석 기준), 행정동 이름."""

    def __init__(self, lot: pd.DataFrame, code_map: dict[int, int], names: dict[int, str]):
        self.lot = lot
        self.one = lot.groupby(["구", "_동"])["행정동_코드"].agg(["nunique", "first"])
        self.near = lot.dropna(subset=["본번"]).drop_duplicates(["구", "_동", "본번"]).sort_values("본번")
        self.code_map, self.names = code_map, names

    @classmethod
    def load(cls, sale_path: Path = SALE_RAW) -> "DongMapper":
        a = pd.read_csv(JUSO_FILE, sep="|", encoding="cp949", header=None, dtype=str, usecols=[1, 4, 6, 7, 8, 14, 15])
        a.columns = ["법정동코드", "동", "산", "본번", "부번", "행정동코드", "행정동명"]
        a["구"] = a["법정동코드"].str[:5]
        a["jibun"] = (np.where(a["산"] == "1", "산", "") + a["본번"].astype(int).astype(str)
                      + np.where(a["부번"].astype(int) > 0, "-" + a["부번"].astype(int).astype(str), ""))
        lot = a.drop_duplicates(["구", "동", "jibun"])[["구", "동", "jibun", "행정동코드"]].rename(columns={"동": "_동"})
        lot["행정동_코드"] = lot["행정동코드"].str[:8].astype(int)
        lot["본번"] = lot["jibun"].str.extract(r"^(\d+)")[0].astype(float)
        # 상권분석 영역 파일에 없는 새 행정동 코드는 기존 데이터에서 쓰던 대응(예: 개포3동 11680675 -> 11680740)을 따름
        base = pd.read_csv(sale_path, usecols=["행정동_코드", "행정동_코드(상권분석)", "행정동"], encoding="utf-8-sig")
        pairs = base.drop_duplicates(["행정동_코드", "행정동_코드(상권분석)"])
        code_map = dict(zip(pairs["행정동_코드"].astype(int), pairs["행정동_코드(상권분석)"].astype(int)))
        names = base.groupby("행정동_코드(상권분석)")["행정동"].agg(lambda s: s.value_counts().index[0]).to_dict()
        area = pd.read_csv(AREA_FILE, encoding="cp949").drop_duplicates("행정동_코드")
        for code, name in zip(area["행정동_코드"], area["행정동_명"]):
            names.setdefault(int(code), name)
            code_map.setdefault(int(code), int(code))
        return cls(lot[["구", "_동", "jibun", "행정동_코드", "본번"]], code_map, {int(k): v for k, v in names.items()})

    def attach(self, d: pd.DataFrame) -> pd.DataFrame:
        """d needs sggCd, umdNm, jibun. Adds 행정동_코드 (Int64, NaN when not found)."""
        n = len(d)
        d = d.copy()
        d["구"], d["jibun"] = d["sggCd"].astype(str), d["jibun"].astype(str).str.strip()
        d = d.merge(self.lot[["구", "_동", "jibun", "행정동_코드"]], left_on=["구", "umdNm", "jibun"],
                    right_on=["구", "_동", "jibun"], how="left").drop(columns="_동")
        assert len(d) == n, "지번이 겹쳐 행이 늘어남"
        miss = d["행정동_코드"].isna()
        key = pd.MultiIndex.from_frame(d.loc[miss, ["구", "umdNm"]])
        single = self.one.reindex(key)
        d.loc[miss, "행정동_코드"] = np.where(single["nunique"].eq(1), single["first"], np.nan)
        miss = d["행정동_코드"].isna()
        q = (d.loc[miss, ["구", "umdNm", "jibun"]].drop_duplicates()
             .assign(본번=lambda x: x["jibun"].str.extract(r"^(\d+)")[0].astype(float))
             .dropna(subset=["본번"]).sort_values("본번"))
        if len(q):
            kw = dict(left_on="본번", right_on="본번", left_by=["구", "umdNm"], right_by=["구", "_동"])
            cols = ["구", "_동", "본번", "행정동_코드"]
            prev = pd.merge_asof(q, self.near[cols], direction="backward", **kw)
            nxt = pd.merge_asof(q, self.near[cols], direction="forward", **kw)
            q = q.assign(앞=prev["행정동_코드"].values, 뒤=nxt["행정동_코드"].values)
            q = q[q["앞"].notna() & (q["앞"] == q["뒤"])].assign(보완=lambda x: x["앞"])
            d = d.merge(q[["구", "umdNm", "jibun", "보완"]], on=["구", "umdNm", "jibun"], how="left")
            fill = d["행정동_코드"].isna() & d["보완"].notna()
            d.loc[fill, "행정동_코드"] = d.loc[fill, "보완"]
            d = d.drop(columns="보완")
        d["행정동_코드"] = d["행정동_코드"].astype("Int64")
        return d.drop(columns="구")

    def finish(self, d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Add 행정동_코드(상권분석), 자치구, 행정동 and drop rows without a dong. Returns (frame, dropped)."""
        d = self.attach(d)
        found = d["행정동_코드"].notna()
        dropped = int((~found).sum())
        d = d[found].copy()
        d["행정동_코드(상권분석)"] = d["행정동_코드"].astype(int).map(self.code_map)
        unknown = d["행정동_코드(상권분석)"].isna()
        dropped += int(unknown.sum())
        d = d[~unknown].copy()
        d["행정동_코드"] = d["행정동_코드"].astype(int)
        d["행정동_코드(상권분석)"] = d["행정동_코드(상권분석)"].astype(int)
        d["자치구"] = d["sggCd"].astype(str).map(SEOUL_SGG)
        d["행정동"] = d["행정동_코드(상권분석)"].map(self.names)
        return d, dropped


# ---------------------------------------------------------------- API 열 -> 기존 원본 열
def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _date(frame: pd.DataFrame) -> pd.Series:
    return (frame["dealYear"].astype(str) + "-" + frame["dealMonth"].astype(str).str.zfill(2)
            + "-" + frame["dealDay"].astype(str).str.zfill(2))


SALE_COLUMNS = ["행정동_코드", "행정동_코드(상권분석)", "자치구", "행정동", "법정동", "지번", "단지명", "동", "계약일", "계약연도",
                "거래금액(만원)", "전용면적(㎡)", "전용면적(평)", "평당 금액(만원)", "층", "건축년도", "건물 나이(년)",
                "토지임대부", "거래유형", "매수자", "매도자", "등기일"]
RENT_COLUMNS = ["행정동_코드", "행정동_코드(상권분석)", "자치구", "행정동", "법정동", "지번", "계약일", "계약연도", "전월세구분",
                "보증금(만원)", "월세(만원)", "임대면적(㎡)", "임대면적(평)", "평당 보증금(만원)", "평당 월세(만원)",
                "층", "건축년도", "건물 나이(년)", "신규계약구분"]


def trade_to_raw(api: pd.DataFrame, mapper: DongMapper) -> tuple[pd.DataFrame, dict]:
    d = api.copy()
    cancelled = d.get("cdealType", pd.Series("", index=d.index)).astype(str).str.strip().ne("")
    d = d[~cancelled].copy()
    area = _number(d["excluUseAr"])
    amount = _number(d["dealAmount"])
    d["계약일"], d["계약연도"] = _date(d), _number(d["dealYear"]).astype(int)
    d["거래금액(만원)"], d["전용면적(㎡)"] = amount, area
    d["전용면적(평)"] = (area / PYEONG).round(2)
    d["평당 금액(만원)"] = (amount / (area / PYEONG)).round(1)
    d["층"], d["건축년도"] = _number(d["floor"]), _number(d["buildYear"])
    d["건물 나이(년)"] = d["계약연도"] - d["건축년도"]
    d = d.rename(columns={"umdNm": "법정동", "aptNm": "단지명", "aptDong": "동", "landLeaseholdGbn": "토지임대부",
                          "dealingGbn": "거래유형", "buyerGbn": "매수자", "slerGbn": "매도자", "rgstDate": "등기일"})
    d["umdNm"] = d["법정동"]
    d, no_dong = mapper.finish(d)
    d["지번"] = d["jibun"]
    d = d[SALE_COLUMNS]
    for column in ("층", "건축년도", "건물 나이(년)"):
        d[column] = d[column].round().astype("Int64")
    for column in ("거래유형", "매수자", "매도자", "등기일", "동"):
        d[column] = d[column].replace("", np.nan)
    return d, {"cancelled": int(cancelled.sum()), "no_dong": no_dong}


def rent_to_raw(api: pd.DataFrame, mapper: DongMapper) -> tuple[pd.DataFrame, dict]:
    d = api.copy()
    area = _number(d["excluUseAr"])
    deposit, monthly = _number(d["deposit"]), _number(d["monthlyRent"]).fillna(0)
    d["계약일"], d["계약연도"] = _date(d), _number(d["dealYear"]).astype(int)
    d["전월세구분"] = np.where(monthly > 0, "월세", "전세")
    d["보증금(만원)"], d["월세(만원)"], d["임대면적(㎡)"] = deposit, monthly, area
    d["임대면적(평)"] = (area / PYEONG).round(2)
    d["평당 보증금(만원)"] = (deposit / (area / PYEONG)).round(1)
    d["평당 월세(만원)"] = (monthly / (area / PYEONG)).round(2).where(monthly > 0)
    d["층"], d["건축년도"] = _number(d["floor"]), _number(d["buildYear"])
    d["건물 나이(년)"] = d["계약연도"] - d["건축년도"]
    d["신규계약구분"] = d.get("contractType", pd.Series("", index=d.index)).replace("", np.nan)
    d["법정동"] = d["umdNm"]
    d, no_dong = mapper.finish(d)
    d["지번"] = d["jibun"]
    return d[RENT_COLUMNS], {"no_dong": no_dong}


# ---------------------------------------------------------------- 병합
def read_refresh(kind: str, months: list[str], root: Path | None = None) -> pd.DataFrame:
    root = root or REFRESH
    frames = [pd.read_csv(root / kind / f"{ym}.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
              for ym in months]
    return pd.concat(frames, ignore_index=True)


def merge_months(base: pd.DataFrame, new: pd.DataFrame, months: list[str]) -> tuple[pd.DataFrame, dict]:
    """Replace every base row of `months` with the new rows (other months stay untouched, in order)."""
    key = base["계약일"].astype(str).str[:7].str.replace("-", "", regex=False)
    new_key = new["계약일"].astype(str).str[:7].str.replace("-", "", regex=False)
    stats = {}
    for ym in months:
        old_n, new_n = int((key == ym).sum()), int((new_key == ym).sum())
        stats[ym] = {"old": old_n, "new": new_n}
        if old_n >= 1000 and new_n < old_n * MIN_KEEP_RATIO:
            raise RuntimeError(f"{ym}: 새로 받은 거래({new_n:,})가 기존({old_n:,})의 절반 미만입니다. API가 덜 준 것 같아 중단합니다.")
    out = pd.concat([base[~key.isin(months)], new[new_key.isin(months)]], ignore_index=True)
    return out, stats


def _replace_file(path: Path, frame: pd.DataFrame) -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, BACKUP / path.name)
    temporary = path.with_suffix(".part")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _read_base(path: Path, kind: str) -> pd.DataFrame:
    base = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    if kind == "trade":
        for column in ("층", "건축년도", "건물 나이(년)"):
            base[column] = base[column].astype("Int64")
    return base


def merge_sales_and_rents(months: list[str], mapper: DongMapper | None = None) -> dict:
    mapper = mapper or DongMapper.load()
    report: dict = {}
    for kind, path, convert in (("trade", SALE_RAW, trade_to_raw), ("rent", RENT_RAW, rent_to_raw)):
        raw, info = convert(read_refresh(kind, months), mapper)
        merged, stats = merge_months(_read_base(path, kind), raw, months)
        _replace_file(path, merged)
        report[kind] = {**info, "months": stats, "rows_after": len(merged)}
        print(f"{kind}: {len(merged):,}행 (행정동 못 찾아 제외 {info['no_dong']:,}건)", flush=True)
    return report


def update_rate_file(rates: dict[str, float]) -> dict:
    files = sorted(glob.glob(str(RAW / "예금은행_대출금리_신규취급액_기준__*.csv")))
    if not files:
        raise FileNotFoundError("기존 금리 파일이 없습니다.")
    path = Path(files[0])
    table = pd.read_csv(path, encoding="cp949", dtype=str)
    label = table["계정항목별"].str.strip() == "주택담보대출 (연리%)"
    if label.sum() != 1:
        raise ValueError("금리 파일에서 '주택담보대출 (연리%)' 행을 찾지 못했습니다.")
    for month, value in rates.items():
        if month not in table.columns:
            table[month] = "-"                                    # 이 파이프라인은 주택담보대출 행만 새로 채움
        table.loc[label, month] = f"{value:.2f}"
    months = sorted(c for c in table.columns if c != "계정항목별")
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, BACKUP / path.name)
    table[["계정항목별"] + months].to_csv(path, index=False, encoding="cp949")
    return {"file": path.name, "last_month": months[-1]}


def base_month(ends: dict[str, str]) -> str:
    """Month the sale / jeonse models predict from: the last month that has sale, jeonse and rate data."""
    return min(ends["sale"], ends["rent"], ends["rate"], key=month_index)
