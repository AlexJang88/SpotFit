from pathlib import Path
from hashlib import sha256

import pandas as pd

from app.services.crime_safety_forecast import predict_all_districts


ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_DATA = ROOT / "data/interim/안전/교통사고안전지수/서울시_자치구_교통안전_대체지수_2017_2025.csv"


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True).mul(100.0)


def get_safety_category_scores() -> dict:
    crime_payload = predict_all_districts()
    crime = pd.DataFrame(crime_payload["forecasts"])[["district", "predicted_score"]]
    crime = crime.rename(columns={"predicted_score": "raw_crime"})
    crime["crime_score"] = _percentile(crime["raw_crime"])

    traffic = pd.read_csv(TRAFFIC_DATA)
    traffic_year = int(traffic["연도"].max())
    traffic = traffic.loc[traffic["연도"].eq(traffic_year), ["자치구", "교통안전_대체점수"]].copy()
    traffic = traffic.rename(columns={"자치구": "district", "교통안전_대체점수": "raw_traffic"})
    traffic["traffic_score"] = _percentile(traffic["raw_traffic"])

    # 인프라 예측모델 연결 전까지 UI 동작 검증에만 쓰는 재현 가능한 더미 점수다.
    infra = crime[["district"]].copy()
    infra["infrastructure_score"] = infra["district"].map(
        lambda name: 20.0 + int.from_bytes(sha256(name.encode("utf-8")).digest()[:2], "big") % 8001 / 100.0
    )
    infra["infrastructure_score"] = _percentile(infra["infrastructure_score"])

    combined = crime.merge(traffic[["district", "traffic_score"]], on="district", how="inner")
    combined = combined.merge(infra, on="district", how="inner").sort_values("district")
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
            "traffic": traffic_year,
            "crime": int(crime_payload["prediction_year"]),
            "infrastructure": 2026,
        },
        "placeholders": {"traffic": False, "crime": False, "infrastructure": True},
        "scores": scores,
    }
