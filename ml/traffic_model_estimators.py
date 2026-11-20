from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class CurrentAccidentPersistenceRegressor(BaseEstimator, RegressorMixin):
    """Predict next year's accidents as the current year's accident count."""

    def __init__(self, feature: str = "현재연도_교통사고건수"):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(X[self.feature], errors="raise").to_numpy(dtype=float)
