from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressionModel(BaseEstimator, RegressorMixin):
    """Random-forest regressor with ensemble variance as predictive uncertainty."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float | str = 1.0,
        bootstrap: bool = True,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.model: RandomForestRegressor | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X2 = np.asarray(X, dtype=float)
        y1 = np.asarray(y, dtype=float).reshape(-1)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X2, y1)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        X2 = np.asarray(X, dtype=float)
        preds = np.vstack([est.predict(X2) for est in self.model.estimators_]).astype(float)
        mu = preds.mean(axis=0)

        if not return_std:
            return mu

        std = preds.std(axis=0, ddof=1) if preds.shape[0] >= 2 else np.zeros_like(mu)
        return mu, np.clip(std, 0.0, np.inf)
