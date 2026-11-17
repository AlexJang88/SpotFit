"""A release must not combine forecasts for different target years."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import neighborhood_recommendations as service
from app.services.recommender_v1 import RecommenderV1Error


class NeighborhoodYearGateTests(unittest.TestCase):
    def test_mismatched_years_fail_before_property_loading(self):
        scores = {"crime_prediction_year": 2028, "traffic_prediction_year": 2027}
        with patch.object(service, "district_scores", return_value=scores), patch.object(
            service, "property_catalog"
        ) as property_catalog:
            with self.assertRaisesRegex(RecommenderV1Error, "예측연도가 달라"):
                service.service_catalog(horizon=2)
            property_catalog.assert_not_called()

    def test_matching_years_can_reach_property_loading(self):
        scores = {"crime_prediction_year": 2028, "traffic_prediction_year": 2028}
        property_result = {"neighborhoods": [], "property_models": {
            "sale": {"prediction_year": 2028}, "rent": {"prediction_year": 2028}}}
        with patch.object(service, "district_scores", return_value=scores), patch.object(
            service, "property_catalog", return_value=property_result
        ) as property_catalog:
            result = service.service_catalog(horizon=2)
            self.assertEqual(result["crime_prediction_year"], 2028)
            property_catalog.assert_called_once_with(2, target_year=2028)

    def test_property_year_mismatch_fails(self):
        scores = {"crime_prediction_year": 2028, "traffic_prediction_year": 2028}
        property_result = {"neighborhoods": [], "property_models": {
            "sale": {"prediction_year": 2027}, "rent": {"prediction_year": 2028}}}
        with patch.object(service, "district_scores", return_value=scores), patch.object(
            service, "property_catalog", return_value=property_result
        ):
            with self.assertRaises(RecommenderV1Error):
                service.service_catalog(horizon=2)

    def test_catalog_endpoint_returns_503_for_mismatched_years(self):
        scores = {"crime_prediction_year": 2028, "traffic_prediction_year": 2027}
        with patch.object(service, "district_scores", return_value=scores):
            response = TestClient(app).get("/api/neighborhoods/catalog?horizon=2")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
