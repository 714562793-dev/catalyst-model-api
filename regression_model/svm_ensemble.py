from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVMEnsembleRegressionModel(BaseEstimator, RegressorMixin):
    """Bootstrap SVR ensemble with predictive mean and model-spread uncertainty."""

    def __init__(
        self,
        *,
        ensemble_size: int = 15,
        bootstrap_ratio: float = 1.0,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
    ):
        self.ensemble_size = ensemble_size
        self.bootstrap_ratio = bootstrap_ratio
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self._models: List[Pipeline] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X2 = np.asarray(X, dtype=float)
        y1 = np.asarray(y, dtype=float).reshape(-1)
        n_samples = X2.shape[0]

        self._models = []
        for _ in range(self.ensemble_size):
            m = int(np.clip(round(self.bootstrap_ratio * n_samples), 1, n_samples))
            idx = np.random.randint(0, n_samples, size=m)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon)),
                ]
            )
            model.fit(X2[idx], y1[idx])
            self._models.append(model)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        if not self._is_fitted or not self._models:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        X2 = np.asarray(X, dtype=float)
        preds = np.vstack([model.predict(X2) for model in self._models]).astype(float)
        mu = preds.mean(axis=0)

        if not return_std:
            return mu

        std = preds.std(axis=0, ddof=1) if preds.shape[0] >= 2 else np.zeros_like(mu)
        return mu, np.clip(std, 0.0, np.inf)
