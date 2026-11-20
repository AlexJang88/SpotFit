"""Round-trip check against the real raw files: turn existing rows back into API-shaped rows, convert them again
and compare with the original rows. Needs data/raw/부동산 (skipped when it is missing).

  python -m pytest scripts/property_data/test_build_property_raw.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.property_data import build_property_raw as raw
from scripts.property_data import collect_molit_refresh as molit
from scripts.property_data import collect_rate_refresh as rate
from scripts.property_data.common import RENT_RAW, SALE_RAW, last_complete_month, month_end_date, month_range

needs_data = pytest.mark.skipif(not (SALE_RAW.is_file() and raw.JUSO_FILE.is_file()), reason="원본 데이터 없음")
MONTH = "2026-08"


@pytest.fixture(scope="module")
def mapper():
    return raw.DongMapper.load()


def _api_sale(base: pd.DataFrame) -> pd.DataFrame:
    day = pd.to_datetime(base["계약일"])
    return pd.DataFrame({
        "sggCd": base["행정동_코드"].astype(str).str[:5], "umdNm": base["법정동"], "jibun": base["지번"].astype(str),
        "aptNm": base["단지명"], "aptDong": base["동"].replace(" ", ""),
        "dealYear": day.dt.year.astype(str), "dealMonth": day.dt.month.astype(str), "dealDay": day.dt.day.astype(str),
        "dealAmount": base["거래금액(만원)"].map("{:,}".format), "excluUseAr": base["전용면적(㎡)"].astype(str),
        "floor": base["층"].astype(str), "buildYear": base["건축년도"].astype(str),
        "landLeaseholdGbn": base["토지임대부"], "dealingGbn": base["거래유형"].fillna(""),
        "buyerGbn": base["매수자"].fillna(""), "slerGbn": base["매도자"].fillna(""),
        "rgstDate": base["등기일"].fillna("").str.strip(), "cdealType": "", "cdealDay": ""})


@needs_data
def test_sale_rows_come_back_the_same(mapper):
    base = pd.read_csv(SALE_RAW, low_memory=False, encoding="utf-8-sig")
    base = base[base["계약일"].str.startswith(MONTH)].reset_index(drop=True)
    converted, info = raw.trade_to_raw(_api_sale(base), mapper)
    assert info["no_dong"] / len(base) < 0.01
    same = base.merge(converted, on=["단지명", "계약일", "거래금액(만원)", "전용면적(㎡)", "층"], suffixes=("", "_new"))
    assert len(same) >= len(converted)
    for column in ["행정동_코드(상권분석)", "자치구", "행정동", "법정동", "계약연도", "전용면적(평)", "평당 금액(만원)",
                   "건축년도", "건물 나이(년)", "토지임대부", "지번"]:
        match = (same[column].astype(str) == same[column + "_new"].astype(str)).mean()
        assert match > 0.995, (column, match)
    assert list(converted.columns) == raw.SALE_COLUMNS


@needs_data
def test_rent_rows_come_back_the_same(mapper):
    base = pd.read_csv(RENT_RAW, low_memory=False, encoding="utf-8-sig")
    base = base[base["계약일"].str.startswith(MONTH)].reset_index(drop=True)
    day = pd.to_datetime(base["계약일"])
    api = pd.DataFrame({
        "sggCd": base["행정동_코드"].astype(str).str[:5], "umdNm": base["법정동"], "jibun": base["지번"].astype(str),
        "dealYear": day.dt.year.astype(str), "dealMonth": day.dt.month.astype(str), "dealDay": day.dt.day.astype(str),
        "deposit": base["보증금(만원)"].map("{:,.0f}".format), "monthlyRent": base["월세(만원)"].astype(int).astype(str),
        "excluUseAr": base["임대면적(㎡)"].astype(str), "floor": base["층"].astype(str),
        "buildYear": base["건축년도"].astype(str), "contractType": base["신규계약구분"].fillna("")})
    converted, info = raw.rent_to_raw(api, mapper)
    assert info["no_dong"] / len(base) < 0.01
    assert (converted["전월세구분"].value_counts(normalize=True).round(2)
            .equals(base["전월세구분"].value_counts(normalize=True).round(2)))
    for column in ["평당 보증금(만원)", "임대면적(평)", "건물 나이(년)", "행정동_코드(상권분석)"]:
        old = base[column].sort_values().reset_index(drop=True)
        new = converted[column].sort_values().reset_index(drop=True)
        assert abs(old.mean() - new.mean()) <= 0.005 * abs(old.mean()), column
    assert list(converted.columns) == raw.RENT_COLUMNS


def test_merge_replaces_only_the_refreshed_months():
    base = pd.DataFrame({"계약일": ["2026-06-01", "2026-07-02", "2026-07-03", "2026-08-01"], "값": [1, 2, 3, 4]})
    new = pd.DataFrame({"계약일": ["2026-07-05", "2026-08-02", "2026-08-03"], "값": [20, 40, 41]})
    merged, stats = raw.merge_months(base, new, ["202607", "202608"])
    assert merged["값"].tolist() == [1, 20, 40, 41]
    assert stats["202607"] == {"old": 2, "new": 1}


def test_merge_refuses_a_month_the_api_returned_only_partly():
    base = pd.DataFrame({"계약일": ["2026-07-01"] * 2000, "값": 1})
    new = pd.DataFrame({"계약일": ["2026-07-01"] * 100, "값": 2})
    with pytest.raises(RuntimeError, match="절반 미만"):
        raw.merge_months(base, new, ["202607"])


def test_months_to_collect_covers_gap_and_recent_months():
    from datetime import date
    assert molit.months_to_collect("202609", date(2026, 11, 17)) == ["202607", "202608", "202609"]
    assert molit.months_to_collect("202609", date(2027, 2, 3)) == ["202610", "202611", "202612"]
    assert molit.months_to_collect("202606", date(2026, 11, 17)) == ["202607", "202608", "202609"]
    assert last_complete_month(date(2027, 1, 5)) == "202611"
    assert month_end_date("202602") == "2026-02-28" and month_range("202611", "202702")[-1] == "202702"


def test_api_errors_are_not_mistaken_for_empty_months():
    ok = b"<response><header><resultCode>000</resultCode></header><body><items><item><umdNm>a</umdNm></item></items><totalCount>1</totalCount></body></response>"
    assert molit.parse_page(ok) == ([{"umdNm": "a"}], 1)
    bad_key = b"<OpenAPI_ServiceResponse><cmmMsgHeader><returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg><returnReasonCode>30</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>"
    with pytest.raises(molit.ApiError, match="게이트웨이"):
        molit.parse_page(bad_key)
    quota = b"<response><header><resultCode>22</resultCode><resultMsg>LIMITED</resultMsg></header></response>"
    with pytest.raises(molit.ApiError, match="API 오류 22"):
        molit.parse_page(quota)


def test_rate_item_must_be_mortgage_loan():
    row = {"TIME": "202609", "DATA_VALUE": "4.88", "ITEM_NAME1": "주택담보대출 (신규취급액 기준)"}
    assert rate.parse({"StatisticSearch": {"row": [row]}}) == {"2026.09": 4.88}
    with pytest.raises(RuntimeError, match="주택담보대출이 아닙니다"):
        rate.parse({"StatisticSearch": {"row": [{**row, "ITEM_NAME1": "기업대출"}]}})
    with pytest.raises(RuntimeError, match="ECOS 오류"):
        rate.parse({"RESULT": {"CODE": "INFO-200"}})


def test_rate_file_gets_new_months_only_on_the_mortgage_row(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "RAW", tmp_path)
    monkeypatch.setattr(raw, "BACKUP", tmp_path / "b")
    (tmp_path / "예금은행_대출금리_신규취급액_기준__1.csv").write_text(
        '"계정항목별",2026.08,2026.09\n"기업대출 (연리%)",4.3,4.49\n"주택담보대출 (연리%)",4.66,4.88\n', encoding="cp949")
    info = raw.update_rate_file({"2026.09": 4.9, "2026.10": 5.0})
    table = pd.read_csv(tmp_path / info["file"], encoding="cp949", dtype=str).set_index("계정항목별")
    assert table.loc["주택담보대출 (연리%)", ["2026.09", "2026.10"]].tolist() == ["4.90", "5.00"]
    assert table.loc["기업대출 (연리%)", "2026.10"] == "-" and info["last_month"] == "2026.10"


def _ready(monkeypatch, tmp_path, ends, served="202609", **files):
    from datetime import date

    from scripts.property_data import run_property_refresh as run
    monkeypatch.setattr(run, "raw_end_months", lambda: dict(ends))
    for kind, months in files.items():
        (tmp_path / kind).mkdir(exist_ok=True)
        for ym in months:
            (tmp_path / kind / f"{ym}.csv").write_text("x")
    return run.readiness(date(2026, 12, 5), refresh=tmp_path, served=served)   # target month = 202610


def test_refresh_waits_until_every_source_reaches_the_target_month(monkeypatch, tmp_path):
    ends = {"sale": "202609", "rent": "202610", "rate": "202609"}
    result = _ready(monkeypatch, tmp_path, ends, trade=["202610"])
    assert not result["ready"] and len(result["reasons"]) == 1 and "rate" in result["reasons"][0]
    (tmp_path / "mortgage_rate.json").write_text('{"2026.10": 4.9}')
    from datetime import date

    from scripts.property_data import run_property_refresh as run
    result = run.readiness(date(2026, 12, 5), refresh=tmp_path, served="202609")
    assert result["ready"] and result["base_month"] == "202610"


def test_refresh_is_skipped_when_there_is_no_new_month(monkeypatch, tmp_path):
    ends = {"sale": "202609", "rent": "202610", "rate": "202609"}
    from datetime import date

    from scripts.property_data import run_property_refresh as run
    monkeypatch.setattr(run, "raw_end_months", lambda: dict(ends))
    result = run.readiness(date(2026, 11, 17), refresh=tmp_path, served="202609")
    assert not result["ready"] and "새로운 달이 없습니다" in result["reasons"][0]
    assert run.stage_process(date(2026, 11, 17))["skipped"] is True
