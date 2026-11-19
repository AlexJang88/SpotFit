"""Monthly refresh of the apartment sale / jeonse / wolse models, from API download to the served model.

  collect : download the newest months (국토교통부 실거래가 API, 한국은행 금리 API) into data/raw/부동산/refresh/
            needs DATA_GO_KR_API_KEY and ECOS_API_KEY                         (light: standard library only)
  process : merge the months into the raw files -> rebuild the three input files -> retrain -> validate
            -> replace models/recommender/v1/apart_*_final                       (heavy: pandas / xgboost / sklearn)
  all     : collect, then process

Run from the repository root:  python -m scripts.property_data.run_property_refresh --stage all
The served models are only replaced when the new ones pass validation; otherwise the old ones stay and the run exits 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from scripts.property_data.common import (
    REFRESH, ROOT, last_complete_month, month_end_date, month_index, month_str, raw_end_months, write_report)


def stage_collect(today: date | None = None) -> dict:
    from scripts.property_data import collect_molit_refresh, collect_rate_refresh

    ends = raw_end_months()
    end = last_complete_month(today)
    months = collect_molit_refresh.months_to_collect(min(ends["sale"], ends["rent"], key=month_index), today)
    report = {"target_end": end, "months": months}
    if months:
        report["molit"] = {"records": [(r["kind"], r["month"], r["rows"])
                                       for r in collect_molit_refresh.collect(months)["records"]]}
    rate_start = month_str(month_index(end) - 11)
    report["rate_months"] = len(collect_rate_refresh.collect(rate_start, end))
    return report


SERVED_BASE_FALLBACK = "202609"      # base month of the models delivered before the pipeline wrote pipeline_meta.json


def served_base_month(root=ROOT) -> str:
    try:
        meta = json.loads((root / "models" / "recommender" / "v1" / "apart_sale_final" / "pipeline_meta.json")
                          .read_text(encoding="utf-8"))
        return str(meta["base_year_month"]).replace("-", "")
    except (OSError, ValueError, KeyError, TypeError):
        return SERVED_BASE_FALLBACK


def readiness(today: date | None = None, refresh=REFRESH, served: str | None = None) -> dict:
    """Decide before touching any file whether a refresh may run.

    Ready only when sale, jeonse/wolse and the mortgage rate all reach the last confirmed month
    (raw file or freshly downloaded month) AND that month is newer than the month the served models start from.
    """
    target = last_complete_month(today)
    ends = raw_end_months()
    for kind, name in (("trade", "sale"), ("rent", "rent")):
        downloaded = [p.stem for p in (refresh / kind).glob("??????.csv")]
        ends[name] = max([ends[name], *downloaded], key=month_index)
    rate_file = refresh / "mortgage_rate.json"
    if rate_file.is_file():
        months = [m.replace(".", "") for m in json.loads(rate_file.read_text(encoding="utf-8"))]
        ends["rate"] = max([ends["rate"], *months], key=month_index)
    reasons = [f"{name} 자료가 {ends[name]}까지만 있습니다 (필요: {target})"
               for name in ("sale", "rent", "rate") if month_index(ends[name]) < month_index(target)]
    base = min(ends.values(), key=month_index)
    served = served or served_base_month()
    if not reasons and month_index(base) <= month_index(served):
        reasons.append(f"새로운 달이 없습니다 (서비스 모델 기준 {served}, 이번 기준 {base})")
    return {"ready": not reasons, "target_month": target, "ends": ends, "base_month": base,
            "served_base_month": served, "reasons": reasons}


def stage_process(today: date | None = None) -> dict:
    ready = readiness(today)
    if not ready["ready"]:                       # a normal outcome, not an error: keep the served models and say why
        return {"skipped": True, **ready}
    sys.path.insert(0, str(ROOT / "backend"))
    from app.services.property_model_bootstrap import refresh_models
    from scripts.property_data import build_property_raw as raw

    report: dict = {"readiness": ready}
    months = sorted(p.stem for p in (REFRESH / "trade").glob("??????.csv")
                    if (REFRESH / "rent" / p.name).is_file())
    if not months:
        raise RuntimeError("받아 둔 월 데이터가 없습니다. 먼저 --stage collect를 실행하세요.")
    report["merge"] = raw.merge_sales_and_rents(months)
    rate_file = REFRESH / "mortgage_rate.json"
    if rate_file.is_file():
        report["rate"] = raw.update_rate_file(json.loads(rate_file.read_text(encoding="utf-8")))
    ends = raw.raw_end_months()
    base = raw.base_month(ends)
    report.update(raw_end_months=ends, base_month=base)
    from scripts.property_data.build_price_history import build as build_price_history
    report["price_history"] = build_price_history()
    report["models"] = refresh_models(ROOT, data_end=month_end_date(base), base_month=f"{base[:4]}-{base[4:]}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("collect", "process", "all"), default="all")
    args = parser.parse_args()
    report: dict = {"stage": args.stage}
    try:
        if args.stage in ("collect", "all"):
            report["collect"] = stage_collect()
        if args.stage in ("process", "all"):
            report["process"] = stage_process()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write_report("last_run.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 1
    failed = [name for name, value in report.get("process", {}).get("models", {}).items() if value != "refreshed"]
    report["ok"] = not failed
    write_report("last_run.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
