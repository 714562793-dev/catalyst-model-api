from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError as e:
    XGBRegressor = None
    XGB_AVAILABLE = False


@dataclass(frozen=True)
class XGBEnsembleConfig:
    """Compatibility shell for loading older serialized XGBoost models."""

    n_models: int = 25
    bootstrap: bool = True
    bootstrap_ratio: float = 1.0
    random_state: int | None = None
    xgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_weight=1.0,
            gamma=0.0,
            objective="reg:squarederror",
            n_jobs=1,
        )
    )


class XGBEnsembleRegressionModel(BaseEstimator, RegressorMixin):
    """Bootstrap XGBoost ensemble that returns mean and model-spread uncertainty."""

    def __init__(
        self,
        *,
        ensemble_size: int = 15,
        bootstrap: bool = True,
        bootstrap_ratio: float = 1.0,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        n_jobs: int = 1,
    ):
        self.ensemble_size = ensemble_size
        self.bootstrap = bootstrap
        self.bootstrap_ratio = bootstrap_ratio
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_jobs = n_jobs
        self._models: List[XGBRegressor] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not XGB_AVAILABLE or XGBRegressor is None:
            raise RuntimeError("XGBoost is not available. Install with: pip install xgboost")
        X2 = np.asarray(X, dtype=float)
        y1 = np.asarray(y, dtype=float).reshape(-1)
        n_samples = X2.shape[0]

        self._models = []
        for _ in range(self.ensemble_size):
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                n_jobs=self.n_jobs,
            )

            if self.bootstrap:
                m = int(np.clip(round(self.bootstrap_ratio * n_samples), 1, n_samples))
                idx = np.random.randint(0, n_samples, size=m)
                X_fit = X2[idx]
                y_fit = y1[idx]
            else:
                X_fit = X2
                y_fit = y1

            model.fit(X_fit, y_fit)
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
