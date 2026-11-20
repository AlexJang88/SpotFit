from __future__ import annotations

import numpy as np
import pandas as pd


class TrafficSafetyPersistenceModel:
    """Keep the current safety state unchanged for a future horizon."""

    def __init__(self, horizon: int, base_year: int = 2025):
        self.horizon = int(horizon)
        self.base_year = int(base_year)
        self.forecast_year = self.base_year + self.horizon
        self.version = 50
        self.kind = "persistence"
        self.grade_definition = "안전 1등급=가장 안전, 안전 5등급=가장 위험"

    def fit(self, frame: pd.DataFrame | None = None, y=None):
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the current safety grade as the future safety grade."""
        return 6 - frame["현재연도_위험등급"].to_numpy(dtype=int)

    def predict_safety_score(self, frame: pd.DataFrame) -> np.ndarray:
        return 120.0 - frame["현재연도_종합위험점수"].to_numpy(dtype=float)

    def predict_safety_rank(self, frame: pd.DataFrame) -> np.ndarray:
        risk = frame["현재연도_종합위험점수"].to_numpy(dtype=float)
        order = np.argsort(risk, kind="stable")
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks
