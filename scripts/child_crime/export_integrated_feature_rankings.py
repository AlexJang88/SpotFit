"""Export borough rankings for every integrated-influence feature."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "child_crime_exposure_rank_3y_v1"


def main() -> int:
    weights = pd.read_csv(OUT / "integrated_feature_weights.csv")
    panel = pd.read_csv(ROOT / "artifacts" / "child_crime_borough_v2" / "panel.csv")
    latest = panel[panel.year == 2025].sort_values("gu")
    rows = []
    summary = []
    for item in weights.itertuples():
        values = pd.to_numeric(latest[item.feature], errors="coerce")
        ascending = item.direction == "lower_risk"
        risk_rank = values.rank(method="average", ascending=ascending)
        points = np.select(
            [risk_rank <= 5, risk_rank <= 10, risk_rank <= 15, risk_rank <= 20],
            [100, 80, 60, 40],
            default=20,
        )
        summary.append({
            "feature": item.feature, "direction": item.direction,
            "model_weight_all_pct": item.model_weight_all_pct,
            "score_eligible": item.score_eligible, "score_weight_pct": item.score_weight_pct,
            "borough_count": len(values), "non_null_count": int(values.notna().sum()),
            "unique_value_count": int(values.nunique()), "has_ties": bool(values.nunique() < len(values)),
            "rankable": bool(values.notna().all() and values.nunique() > 1),
        })
        for gu, value, rank, point in zip(latest.gu, values, risk_rank, points):
            rows.append({
                "feature": item.feature, "direction": item.direction,
                "borough": gu, "reference_year": 2025, "raw_value": value,
                "risk_rank_average": rank, "quintile_points": int(point),
                "model_weight_all_pct": item.model_weight_all_pct,
                "score_eligible": item.score_eligible, "score_weight_pct": item.score_weight_pct,
            })
    detail = pd.DataFrame(rows)
    audit = pd.DataFrame(summary)
    detail.to_csv(OUT / "feature_borough_rankings_2025.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "feature_rankability_audit.csv", index=False, encoding="utf-8-sig")
    workbook = OUT / "exposure_rank_model_and_weights.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        detail.to_excel(writer, sheet_name="feature_borough_ranks", index=False)
        audit.to_excel(writer, sheet_name="feature_rank_audit", index=False)
    print({"feature_count": len(audit), "rankable_count": int(audit.rankable.sum()), "rows": len(detail), "all_weight_sum": float(weights.model_weight_all_pct.sum()), "score_weight_sum": float(weights.loc[weights.score_eligible, 'score_weight_pct'].sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
