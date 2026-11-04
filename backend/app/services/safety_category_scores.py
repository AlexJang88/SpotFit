import pandas as pd

from app.services.recommender_v1 import get_district_forecasts


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True).mul(100.0)


def get_safety_category_scores() -> dict:
    payload = get_district_forecasts()
    combined = pd.DataFrame(payload["forecasts"])[
        ["district", "traffic_score", "crime_score", "child_environment_1y"]
    ].copy()
    combined["traffic_score"] = _percentile(combined["traffic_score"])
    combined["crime_score"] = _percentile(combined["crime_score"])
    combined["infrastructure_score"] = _percentile(combined["child_environment_1y"])
    combined = combined.sort_values("district")
    scores = [
        {
            "district": row.district,
            "traffic_score": round(float(row.traffic_score), 4),
            "crime_score": round(float(row.crime_score), 4),
            "infrastructure_score": round(float(row.infrastructure_score), 4),
        }
        for row in combined.itertuples(index=False)
    ]
    return {
        "district_count": len(scores),
        "source_years": {
            "traffic": int(payload["models"]["traffic_accident"]["prediction_year"]),
            "crime": int(payload["models"]["crime_safety"]["prediction_year"]),
            "infrastructure": int(payload["models"]["child_environment"]["prediction_years"][0]),
        },
        "placeholders": {"traffic": False, "crime": False, "infrastructure": False},
        "scores": scores,
    }
