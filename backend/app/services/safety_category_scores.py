"""Expose final weighted safety domains without deleted legacy models."""
from app.services.neighborhood_recommendations import district_scores


def get_safety_category_scores(horizon: int = 2) -> dict:
    result = district_scores(horizon)
    return {
        'district_count': len(result['districts']),
        'source_years': {'crime': result['crime_prediction_year'], 'traffic': result['traffic_prediction_year']},
        'placeholders': {'crime': False, 'traffic': False, 'infrastructure': True},
        'scores': [{'district': row['district'], 'crime_score': row['crime_score'],
                    'traffic_score': row['traffic_score'], 'infrastructure_score': None}
                   for row in result['districts']],
    }
