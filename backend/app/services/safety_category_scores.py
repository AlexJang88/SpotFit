import pandas as pd

from app.services.recommender_v1 import _child_predictions, _traffic_predictions
from app.services.child_crime_exposure import forecast_payload


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True).mul(100.0)


def get_safety_category_scores(horizon: int = 1) -> dict:
    child, child_meta = _child_predictions()
    traffic, traffic_meta = _traffic_predictions()
    combined = traffic[["district", "traffic_score"]].merge(
        child[["district", "child_environment_1y"]], on="district", validate="one_to_one"
    )
    crime = pd.DataFrame(forecast_payload(horizon)["forecasts"])[["district", "predicted_score"]].rename(columns={"predicted_score": "crime_score"})
    combined = combined.merge(crime, on="district", validate="one_to_one")
    combined["traffic_score"] = _percentile(combined["traffic_score"])
    # 범죄 모델은 이미 5분위별 100·80·60·40·20점으로 환산되어 있다.
    combined["crime_score"] = pd.to_numeric(combined["crime_score"], errors="raise")
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
            "traffic": int(
                traffic_meta.get(
                    "prediction_year",
                    max(traffic_meta.get("prediction_years", [2027])),
                )
            ),
            "crime": int(forecast_payload(horizon)["prediction_year"]),
            "infrastructure": int(child_meta["prediction_years"][0]),
        },
        "placeholders": {"traffic": False, "crime": False, "infrastructure": False},
        "scores": scores,
    }
