import pandas as pd

from app.services import historical_safety_reference as reference


def test_historical_reference_respects_selected_weights(monkeypatch):
    frame = pd.DataFrame([
        {'district': '중구', 'year': 2024, 'crime_reference_score': 100, 'traffic_reference_score': 20},
        {'district': '중구', 'year': 2025, 'crime_reference_score': 80, 'traffic_reference_score': 40},
    ])
    monkeypatch.setattr(reference, 'reference_frame', lambda: frame)
    weighted = reference.reference_series_by_district(.75, .25)['중구']
    assert weighted == [
        {'year': 2024, 'score': 80.0, 'kind': 'historical_reference'},
        {'year': 2025, 'score': 70.0, 'kind': 'historical_reference'},
    ]


def test_historical_reference_missing_data_is_optional(monkeypatch):
    monkeypatch.setattr(reference, 'reference_frame', lambda: pd.DataFrame())
    assert reference.reference_series_by_district(.5, .5) == {}
